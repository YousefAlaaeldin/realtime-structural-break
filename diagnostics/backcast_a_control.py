"""Is A_mean (prepending a backcast pseudo-past) anything more than re-weighting the first online points?
(HANDOFF section 32; follows the backcast gate of section 31.)

In the gate, only the SIGNED A_mean_e2 / A_mean_rough beat online-only statistics (+0.003 overall, +0.007-0.009 on
steps 0-100), and the incremental gain (+0.0037, 1.9 SE, 3/4 folds) vanished when B was added. The conditional-mean
backcast is a smoothed copy of the first ~10 online points, so prepending it (a) up-weights those points and (b)
changes the "seam": the first online residuals get pseudo lags instead of the real history lags. Two controls copy
those effects WITHOUT any backcast:
  W_k_c   weighted running mean of q over online points, the first k get weight 1 + c   (k in 5/10/20, c in 1/2/4/8)
  A_zero  30 zeros prepended instead of the backcast (same seam structure, same sample sizes, no information)
q in {e2, rough} (signed, as defined in backcast_gate.py: (mean q - history mean) / history sd).

Protocol: the control grid is tuned on the FIT series (best signed TS-AUC overall per q), then everything is judged
on the EVAL series.

DECISION RULE (written before any run):
  A goes on to a paired cv.py run only if BOTH
  (a) for e2 or rough: A_mean - max(best W, A_zero) >= +0.002 AND > 3 bootstrap SE, overall or on steps 0-100;
  (b) incremental LightGBM (4 folds, 3 seeds averaged per fold): base = bo + vol + nb + step + A_plain + best W +
      A_zero; test = + A_mean (e2, rough): the project bar (>= +0.003, > 2 SE, >= 3/4 folds).
  Otherwise A is closed.

Run (CPU; my estimate 15-25 min, of which the reference features are about 3 min unless cached by another script):
    python diagnostics/backcast_a_control.py
    python diagnostics/backcast_a_control.py --fit 0:400 --eval 2000:2600 --folds 2 --seeds 1   # smoke run
Out: cv_out/backcast_a_control.txt
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "diagnostics"))
os.chdir(HERE)

import _backcast_common as C   # noqa: E402
import backcast_gate as G      # noqa: E402
from sbrt import features as F  # noqa: E402
from sbrt import volstate as VS  # noqa: E402

AR = G.AR
L = G.L
QN = ("e2", "rough")
WK = (5, 10, 20)
WC = (1.0, 2.0, 4.0, 8.0)
W_NAMES = [f"W_{k}_{int(c)}" for k in WK for c in WC]


def a_series(hist, online, tau, seed):
    """columns: plain, mean, zero (q = e2, rough), then the W grid for e2, then for rough."""
    _, _, Am, _, _ = G.series_work(hist, online, tau, seed)
    i_plain = [G.A_NAMES.index(f"A_plain_{q}") for q in QN]
    i_mean = [G.A_NAMES.index(f"A_mean_{q}") for q in QN]
    H = VS.fit_history(hist)
    P, zh, coef, se = H["P"], H["z"], H["coef"], H["se"]
    zo = (np.asarray(online, dtype=np.float64) - P[F.P_MU]) / P[F.P_SD]
    T, n = len(zh), len(zo)
    z = np.r_[zh, zo]
    ef = G.fwd_resid(z, coef, se, AR)
    qf = np.minimum(ef * ef, G.QCAP)
    rough = np.minimum(np.diff(z) ** 2, G.QCAP)
    qh = np.column_stack([qf[AR:T], rough[AR - 1:T - 1]])
    qmu, qsd = qh.mean(0), qh.std(0) + 1e-9
    qo = np.column_stack([qf[T:], rough[T - 1:]])
    co = np.vstack([np.zeros(2), np.cumsum(qo, 0)])
    rows = np.arange(0, n, G.SUB)
    out = np.zeros((len(rows), 6 + 2 * len(W_NAMES)))
    out[:, 0:2] = Am[:, i_plain]
    out[:, 2:4] = Am[:, i_mean]
    seq = np.zeros(L + AR + 1)
    sums = np.zeros(5)
    for r, t in enumerate(rows):
        nn = t + 1
        k_on = min(nn, AR)
        seq[:L] = 0.0
        seq[L:L + k_on] = zo[:k_on]
        G.seam_sums(seq[:L + k_on], coef, se, L + k_on, sums)
        tail = co[nn] - co[k_on]
        tn = nn - k_on
        out[r, 4] = ((sums[0] + tail[0]) / (sums[3] + tn) - qmu[0]) / qsd[0]
        out[r, 5] = ((sums[1] + tail[1]) / (sums[4] + tn) - qmu[1]) / qsd[1]
        j = 6
        for qi in range(2):
            for k in WK:
                kk = min(nn, k)
                for c in WC:
                    m = (co[nn, qi] + c * co[kk, qi]) / (nn + c * kk)
                    out[r, j] = (m - qmu[qi]) / qsd[qi]
                    j += 1
    return out


def _chunk(items):
    return [a_series(*it) for it in items]


def build(S_all, tau_all, lo, hi, threads):
    from joblib import Parallel, delayed
    items = [(S_all[i][1], S_all[i][2], int(tau_all[i]), int(i)) for i in range(lo, hi)]
    chunks = [items[k:k + 50] for k in range(0, len(items), 50)]
    return np.concatenate([x for ch in Parallel(n_jobs=threads)(delayed(_chunk)(c) for c in chunks) for x in ch])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", default="0:2000")
    ap.add_argument("--eval", default="2000:10000")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default="backcast_a_control")
    args = ap.parse_args()
    from sbrt.data import load_train
    from sbrt.metric import ts_auc

    say = C.Log(os.path.join(HERE, "cv_out", f"{args.out}.txt"))
    fit = tuple(map(int, args.fit.split(":")))
    ev = tuple(map(int, args.eval.split(":")))
    S_all, tau_all = load_train()
    t0 = time.time()
    say(f"A control | fit {fit[0]}-{fit[1] - 1} | eval {ev[0]}-{ev[1] - 1} | folds {args.folds} | lgb seeds {args.seeds}")

    # ---- tune the W grid on the fit series ----
    Xf = build(S_all, tau_all, fit[0], fit[1], args.threads)
    stf, sidf, yf, slf = C.layout(S_all, tau_all, fit)
    best_w = {}
    say("\nfit set: signed TS-AUC (overall / 0-100) of plain, mean, zero and the W grid")
    for qi, q in enumerate(QN):
        say(f"  {q}: plain {ts_auc(Xf[:, qi], yf, stf):.4f}  mean {ts_auc(Xf[:, 2 + qi], yf, stf):.4f}  "
            f"zero {ts_auc(Xf[:, 4 + qi], yf, stf):.4f}")
        sc = []
        for j, nm in enumerate(W_NAMES):
            col = 6 + qi * len(W_NAMES) + j
            sc.append(ts_auc(Xf[:, col], yf, stf))
        say("    " + "  ".join(f"{nm} {s:.4f}" for nm, s in zip(W_NAMES, sc)))
        j = int(np.argmax(sc))
        best_w[q] = 6 + qi * len(W_NAMES) + j
        say(f"    picked {W_NAMES[j]} for {q}")

    # ---- judge on the eval series ----
    X = build(S_all, tau_all, ev[0], ev[1], args.threads)
    step, sid, y, in_slice = C.layout(S_all, tau_all, ev)
    say(f"\neval: rows {len(y):,}, series {ev[1] - ev[0]}  ({time.time() - t0:.0f}s)")
    say("single-feature signed TS-AUC        " + C.HDR)
    for qi, q in enumerate(QN):
        for nm, col in (("plain", qi), ("mean", 2 + qi), ("zero", 4 + qi), (W_NAMES[(best_w[q] - 6) % len(W_NAMES)], best_w[q])):
            say(f"  {q:5s} {nm:12s}                " + C.fmt(C.auc_row(X[:, col], y, step, in_slice)))
    early = step < 100
    say("\nA_mean minus control (TS-AUC difference +- series-bootstrap SE, 10 draws)")
    rule_a = []
    for qi, q in enumerate(QN):
        for nm, col in (("best W", best_w[q]), ("zero", 4 + qi), ("plain", qi)):
            a = C.boot_diff(X[:, 2 + qi], X[:, col], y, step, sid)
            b = C.boot_diff(X[:, 2 + qi], X[:, col], y, step, sid, mask=early)
            say(f"  {q:5s} mean - {nm:7s}  all {a[0]:+.4f} +- {a[1]:.4f}   0-100 {b[0]:+.4f} +- {b[1]:.4f}")
            if nm != "plain":
                rule_a.append((q, nm, a, b))
    # (a): vs the stronger control, per q
    pass_a = False
    for q in QN:
        rr = [r for r in rule_a if r[0] == q]
        for part in (2, 3):
            g = min(r[part][0] for r in rr)
            s = max(r[part][1] for r in rr)
            if g >= 0.002 and g > 3 * s:
                pass_a = True

    # ---- incremental ----
    Bo, V, NBF = C.references(S_all, tau_all, fit, ev, args.threads, say)
    say(f"\nIncremental: LightGBM {args.folds}-fold by series, {args.seeds} seeds averaged per fold")
    ctrl = np.column_stack([X[:, 0:2], X[:, [best_w[q] for q in QN]], X[:, 4:6]])
    base = np.column_stack([Bo, V, NBF, step, ctrl]).astype(np.float32)
    sets = {"base": base, "base+A_mean": np.column_stack([base, X[:, 2:4]]).astype(np.float32)}
    res = C.incremental(sets, y, step, sid, args.folds, args.seeds, args.threads, say)
    pass_b = C.passes_project_bar(res["base+A_mean"], args.folds)

    say("\nDECISION (rule written in the docstring before any run)")
    if ev[1] - ev[0] < 4000:
        say("  SMOKE RUN: pipeline check only; do not act on it.")
    say(f"  (a) A_mean beats both controls by >= +0.002 and > 3 SE: {'yes' if pass_a else 'no'}")
    m, se, npos = res["base+A_mean"]
    say(f"  (b) incremental {m:+.4f} +- {se:.4f}, {npos}/{args.folds}: {'yes' if pass_b else 'no'}")
    say(f"  -> {'A goes to a paired cv.py run' if pass_a and pass_b else 'A is CLOSED'}")
    say(f"\ntotal {time.time() - t0:.0f}s; log {say.path}")


if __name__ == "__main__":
    main()
