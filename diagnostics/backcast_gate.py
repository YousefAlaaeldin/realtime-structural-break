"""Backcasting gate (owner's idea, 2026-10-15): predict the PAST from the PRESENT and use it two ways.

  B  "compare the predicted past with the real past": a model of normal (history) dynamics run BACKWARD in time; if
     the recent points come from a different process, backcasting the earlier points from them goes wrong.
  A  "feed the predicted past in as extra data": the pseudo-past is built from the new points, so it carries the new
     regime's signature; prepending it to the online data should make the existing statistics react sooner.

Everything is linear first (backward AR(10) / AR(2) fitted by least squares); a backward NBM is only worth building
if the linear version passes. History is z-scored exactly, z = (x - mu_h) / sd_h, as everywhere else.

Backward AR(10), fitted on the series' own history:  z_s = sum_j b_j z_{s+j} + r_s  (j = 1..10)
  backward residual e_b(s) = (z_s - sum_j b_j z_{s+j}) / se_b, known at online step s + 10 (10 steps late).
  forward residual e_f(s) = the usual AR(10) residual (sbrt/bocpd.py's e), known at step s.

---- B features (all causal) ----
  bk_logodds, bk_tsb, bk_lvr, bk_mu   e_b whitened by the history's own e_b ECDF (as bocpd.py whitens e_f), then the
                                     same one-break BOCPD posterior with an empirical-Bayes prior fitted on the
                                     fit-set breaks. Delayed by 10 steps.
  bk_e2_W / fw_e2_W  (W = 100)       mean of min(e^2, 25) over the same delayed window, backward / forward, z-scored
                                     by the history's rolling-W means. Shown side by side: for a time-reversible
                                     process these two are nearly the same number (the redundancy argument).
  irr_W  (W = 30, 100)               irreversibility shift: mean of min(e_f^2,25) - min(e_b^2,25) over the window,
                                     z-scored by the same quantity over the history. ~0 for a reversible process,
                                     and history gives each series its own baseline; a break that changes the
                                     forward/backward asymmetry (volatility clustering, skew) moves it.
  bkwin, bkwin_near                  the literal version: window = last 100 points of history+online; backcast the
                                     first 30 from the last 70 (recursively, backward AR(10)); log of the mean over
                                     depths of err_d^2 / ref_d, where ref_d is the same error over the history's own
                                     windows (so 0 = "as good as on normal data"). _near uses depths 1-5 only.
  Also reported: backcast skill by depth R2_d = 1 - ref_d / var(z_h) (how far back the past is predictable at all,
  i.e. what "valid accuracy" can mean here), and err_d^2 / ref_d on windows lying wholly after a break.

---- A features (pseudo-past prepended to the online data) ----
Per-point quantities q: e2 = min(e_f^2, 25), rough = min((dz)^2, 25), tail = 1{|e_f| > 2}. Statistic at step t:
mean of q over the sample, minus the history mean of q, divided by the history sd of q; reported signed and as |.|
(variance can go up or down at a break). Samples:
  A_plain_q   online points 0..t (real history lags)                   -> the baseline every A variant must beat
  A_real_q    the last 30 REAL history points + online 0..t            -> control: real past instead of pseudo-past
  A_mean_q    30 pseudo-past points + online 0..t; pseudo-past = conditional-mean backcast of the history's backward
              AR(10) from the first online points (unseen ones set to 0); the online points' lags at the seam come
              from the pseudo-past ("if the break is at the first point, the pseudo-past is built from the new regime")
  A_samp_q    30 pseudo-past points SAMPLED from a backward AR(2) refitted on the online points so far (ridge toward
              the history's AR(2), lambda 20 points; innovation sd shrunk the same way; fixed noise per series, so
              deterministic). This is the "skewed toward the now" version: its scale and dependence follow the online
              fit, not the history.
  A_win_q     (q = z2, rough) the last-100 window with its first 30 points replaced by the backcast from the last 70,
              vs A_w100_q (real 100) and A_w70_q (real last 70). z-scored by the same statistic on history windows.

---- evaluation ----
Prior fits on --fit series (default 0-1999); everything is scored on --eval (default 2000-9999, 8,000 series; the
section-30 lesson: small runs pass on noise). Rows every 3rd online step, as cv.py. References: bo_* (bocpd.py), vol_*
(volstate.py), nb_* (the NBM gate model data/nbm_v1.pt, trained on histories 0-4499 only, no labels).
  1. single-feature TS-AUC: all, 0-100, 100-300, 300-700, 700+, var-same (break series whose whitened post-break
     variance is within 15% of the history's, plus all no-break series), and max |rank corr| with any reference.
  2. A: paired TS-AUC difference augmented - plain, per q, overall and 0-100, with a series-bootstrap SE.
  3. incremental: LightGBM, K-fold by series on the eval set. The base holds bo + vol + nb + step AND every
     real-data twin computed here (fw_e2_100, A_plain_*, A_real_*, A_w100_*, A_w70_*), so a gain can only come from
     the backward / pseudo-past part: B = bk_*, irr_*, bkwin* (backward only); A = A_mean_*, A_samp_*, A_win_*.
     base vs base + B, base + A, base + B + A. Per-fold paired differences, mean +- SE, folds positive.
     (changed after the smoke run, before any full run: the first version left the twins out of the base, so ordinary
     forward / real-window statistics could have passed as "backcasting")

KILL CRITERIA (written before any run; the script prints the verdicts):
  KB1 (B redundant): bk_logodds max |rank corr| with bo_/nb_ >= 0.90 AND its var-same AUC <= nb_logodds' + 0.005.
  KB2 (irreversibility dead): every irr_W has single-feature AUC (best of signed / |.|) < 0.53 overall AND < 0.52 on
      var-same.
  KB3 (B adds nothing): base + B - base < +0.002 mean OR positive in fewer than (K - 1) of K folds.
  KA1 (A dead as a statistic): no A_mean / A_samp / A_win beats its real counterpart (plain; win vs BOTH w100 and w70),
      same transform (|.| or signed), by >= +0.003 AND > 3 bootstrap SE, overall or on steps 0-100.
      (tightened after the 600-series smoke run, before any full run: one noisy +0.021 +- 0.012 had passed the
      first version, which had no SE condition; then 2 -> 3 SE, since about 52 tests are run and 2 SE gives about
      one false pass; KA2 is the deciding gate)
  KA2 (A adds nothing): base + A - base < +0.002 mean OR positive in fewer than (K - 1) of K folds.
  B goes on to a cv.py flag (and a backward NBM) only if KB3 passes; A only if KA1 AND KA2 pass.

Run (my estimate 25-40 min on 8 processes, CPU only; the bootstrap SEs and the 16 LightGBM fits are the slow part):
    python diagnostics/backcast_gate.py                         # full gate
    python diagnostics/backcast_gate.py --eval 2000:2600 --folds 2   # smoke run (pipeline check only, no verdict)
Out: cv_out/backcast_gate.txt, cv_out/backcast_gate.npz
"""
import argparse
import os
import sys
import time

import numpy as np
from numba import njit

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
os.chdir(HERE)

from sbrt import bocpd as BO      # noqa: E402
from sbrt import features as F    # noqa: E402
from sbrt import nbm as NB        # noqa: E402
from sbrt import volstate as VS   # noqa: E402

AR = F.AR
SUB = 3
QCAP = 25.0
L = 30            # pseudo-past length / backcast depth
WIN, LATE = 100, 70
IRR_WS = (30, 100)
LAM = 20.0
NEAR = 5
NBOOT = 10
QS = ("e2", "rough", "tail")
WQS = ("z2", "rough")
B_NAMES = (["bk_logodds", "bk_tsb", "bk_lvr", "bk_mu", "bk_e2_100", "fw_e2_100"] + [f"irr_{w}" for w in IRR_WS]
           + ["bkwin", "bkwin_near"])
A_NAMES = ([f"A_{k}_{q}" for k in ("plain", "real", "mean", "samp") for q in QS]
           + [f"A_{k}_{q}" for k in ("w100", "w70", "win") for q in WQS])
N_B0 = 6          # bk_logodds..fw_e2_100 come first; posterior columns are filled after the prior fit


# ---------------------------------------------------------------------------------------------
# kernels

@njit(cache=True)
def fwd_resid(z, coef, se, start):
    N, p = len(z), len(coef)
    out = np.full(N, np.nan)
    for i in range(max(start, p), N):
        s = 0.0
        for k in range(p):
            s += coef[k] * z[i - 1 - k]
        e = (z[i] - s) / se
        out[i] = min(max(e, -10.0), 10.0)
    return out


@njit(cache=True)
def bwd_resid(z, b, se):
    N, p = len(z), len(b)
    out = np.full(N, np.nan)
    for i in range(N - p):
        s = 0.0
        for j in range(p):
            s += b[j] * z[i + 1 + j]
        e = (z[i] - s) / se
        out[i] = min(max(e, -10.0), 10.0)
    return out


@njit(cache=True)
def backcast_window(z, c, b, tmp):
    """tmp[i] = value at position c - WIN + 1 + i; the first WIN - LATE are backcast from the rest. Returns err^2 by
    depth d = 1..WIN-LATE (d = 1 is the point right before the late part)."""
    p = len(b)
    first = c - WIN + 1
    nd = WIN - LATE
    for i in range(nd, WIN):
        tmp[i] = z[first + i]
    err2 = np.empty(nd)
    for i in range(nd - 1, -1, -1):
        s = 0.0
        for j in range(p):
            s += b[j] * tmp[i + 1 + j]
        tmp[i] = s
        d = z[first + i] - s
        err2[nd - 1 - i] = d * d
    return err2


@njit(cache=True)
def win_stats(v, out):
    """mean z^2 and mean (dz)^2 (capped) over v."""
    s1 = 0.0
    s2 = 0.0
    for i in range(len(v)):
        s1 += v[i] * v[i]
        if i > 0:
            d = v[i] - v[i - 1]
            s2 += min(d * d, QCAP)
    out[0] = s1 / len(v)
    out[1] = s2 / (len(v) - 1)


@njit(cache=True)
def seam_sums(seq, coef, se, upto, sums):
    """q sums over seq positions [AR, upto) for e2 / tail (lags from seq) and [1, upto) for rough.
    Returns counts in sums[3], sums[4]."""
    for k in range(5):
        sums[k] = 0.0
    p = len(coef)
    for i in range(1, upto):
        d = seq[i] - seq[i - 1]
        sums[1] += min(d * d, QCAP)
        sums[4] += 1.0
        if i >= p:
            s = 0.0
            for k in range(p):
                s += coef[k] * seq[i - 1 - k]
            e = min(max((seq[i] - s) / se, -10.0), 10.0)
            sums[0] += min(e * e, QCAP)
            sums[2] += 1.0 if abs(e) > 2.0 else 0.0
            sums[3] += 1.0


@njit(cache=True)
def samp_pseudo(zo, t, bh2, sh2, eps, out):
    """Backward AR(2) refitted on zo[0..t] (ridge toward bh2 with LAM points), sampled L points into out
    (out[L-1] is the point right before online 0)."""
    m = t - 1          # rows s = 0..t-2
    a11 = LAM
    a12 = 0.0
    a22 = LAM
    r1 = LAM * bh2[0]
    r2 = LAM * bh2[1]
    if m >= 18:
        for s in range(m):
            x1, x2, y = zo[s + 1], zo[s + 2], zo[s]
            a11 += x1 * x1
            a12 += x1 * x2
            a22 += x2 * x2
            r1 += x1 * y
            r2 += x2 * y
    det = a11 * a22 - a12 * a12
    c1 = (a22 * r1 - a12 * r2) / det
    c2 = (a11 * r2 - a12 * r1) / det
    tot = abs(c1) + abs(c2)
    if tot > 0.98:
        c1 *= 0.98 / tot
        c2 *= 0.98 / tot
    s2 = LAM * sh2
    n2 = LAM
    if m >= 18:
        for s in range(m):
            r = zo[s] - c1 * zo[s + 1] - c2 * zo[s + 2]
            s2 += r * r
            n2 += 1.0
    sd = np.sqrt(s2 / n2)
    nxt1 = zo[0] if t >= 0 else 0.0
    nxt2 = zo[1] if t >= 1 else 0.0
    for k in range(L - 1, -1, -1):
        v = c1 * nxt1 + c2 * nxt2 + sd * eps[L - 1 - k]
        out[k] = v
        nxt2 = nxt1
        nxt1 = v


# ---------------------------------------------------------------------------------------------
# per series

def _rolling_ref(x, w):
    """mean and sd of rolling-w means of x (history reference for a window statistic)."""
    if len(x) < w + 2:
        return float(x.mean()), float(x.std() / np.sqrt(w) + 1e-9)
    c = np.r_[0.0, np.cumsum(x)]
    m = (c[w:] - c[:-w]) / w
    return float(m.mean()), float(m.std() + 1e-9)


def whiten(e, ref):
    from scipy.special import ndtri
    n = len(ref)
    k = np.searchsorted(ref, e)
    w = ndtri((k + 0.5) / (n + 1.0))
    hi, lo = e > ref[-1], e < ref[0]
    w[hi] = ndtri((n + 0.5) / (n + 1.0)) + (e[hi] - ref[-1])
    w[lo] = ndtri(0.5 / (n + 1.0)) - (ref[0] - e[lo])
    return np.clip(w, -8.0, 8.0)


def back_fit(zh, p):
    n = len(zh)
    X = np.column_stack([zh[1 + j: n - p + 1 + j] for j in range(p)])
    y = zh[:n - p]
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    return b, float((y - X @ b).std())


def series_work(hist, online, tau, seed):
    H = VS.fit_history(hist)
    P, zh, coef, se = H["P"], H["z"], H["coef"], H["se"]
    zo = (np.asarray(online, dtype=np.float64) - P[F.P_MU]) / P[F.P_SD]
    T, n = len(zh), len(zo)
    z = np.r_[zh, zo]
    b, seb = back_fit(zh, AR)
    b2, sb2 = back_fit(zh, 2)
    ef = fwd_resid(z, coef, se, AR)
    eb = bwd_resid(z, b, seb)
    rows = np.arange(0, n, SUB)
    nr = len(rows)
    Bm = np.zeros((nr, len(B_NAMES)), dtype=np.float64)
    Am = np.zeros((nr, len(A_NAMES)), dtype=np.float64)

    # --- B1: whitened backward stream (online s = 0..n-1-AR) ---
    ref_b = np.sort(eb[:T - AR])
    wb = whiten(eb[T:T + max(n - AR, 0)], ref_b) if n > AR else np.zeros(0)

    # --- B2: windowed backward / forward energy and irreversibility, over the delayed range [hi-W+1, hi] ---
    qf, qb = np.minimum(ef * ef, QCAP), np.minimum(eb * eb, QCAP)
    hs = slice(AR, T - AR)
    dh = qf[hs] - qb[hs]
    # qb (and so qf - qb) is nan for online s > n-1-AR; those points are never inside a delayed window
    cf, cb, cd = (np.r_[0.0, np.cumsum(np.nan_to_num(v[T:T + n]))] for v in (qf, qb, qf - qb))
    refs = {("bk", 100): _rolling_ref(qb[hs], 100), ("fw", 100): _rolling_ref(qf[hs], 100)}
    for w in IRR_WS:
        refs[("irr", w)] = _rolling_ref(dh, w)
    for r, t in enumerate(rows):
        hi = t - AR
        if hi < 0:
            continue
        for c, (key, w, csum) in enumerate([(("bk", 100), 100, cb), (("fw", 100), 100, cf)]):
            lo = max(0, hi - w + 1)
            k = hi - lo + 1
            mu, sd = refs[key]
            Bm[r, 4 + c] = ((csum[hi + 1] - csum[lo]) / k - mu) / (sd * np.sqrt(w / k))
        for c, w in enumerate(IRR_WS):
            lo = max(0, hi - w + 1)
            k = hi - lo + 1
            mu, sd = refs[("irr", w)]
            Bm[r, 6 + c] = ((cd[hi + 1] - cd[lo]) / k - mu) / (sd * np.sqrt(w / k))

    # --- B3 + A_win: literal window backcast; history reference first ---
    tmp = np.empty(WIN)
    st = np.empty(2)
    hc = np.arange(WIN - 1, T, SUB)
    E2h = np.empty((len(hc), WIN - LATE))
    Wh = np.empty((len(hc), 3, 2))                       # real100 / real70 / aug  x  (z2, rough)
    for i, c in enumerate(hc):
        E2h[i] = backcast_window(z, c, b, tmp)
        win_stats(tmp, st)
        Wh[i, 2] = st
        win_stats(z[c - WIN + 1:c + 1], st)
        Wh[i, 0] = st
        win_stats(z[c - LATE + 1:c + 1], st)
        Wh[i, 1] = st
    ref_d = E2h.mean(0) + 1e-9
    wmu, wsd = Wh.mean(0), Wh.std(0) + 1e-9
    skill = 1.0 - ref_d / max(float(zh.var()), 1e-9)
    post = np.zeros(WIN - LATE)
    npost = 0
    for r, t in enumerate(rows):
        c = T + t
        e2 = backcast_window(z, c, b, tmp)
        ratio = e2 / ref_d
        Bm[r, 8] = np.log(ratio.mean())
        Bm[r, 9] = np.log(ratio[:NEAR].mean())
        if tau >= 0 and t - WIN + 1 >= tau:
            post += ratio
            npost += 1
        win_stats(tmp, st)
        Am[r, A_NAMES.index("A_win_z2"):A_NAMES.index("A_win_rough") + 1] = (st - wmu[2]) / wsd[2]
        win_stats(z[c - WIN + 1:c + 1], st)
        Am[r, A_NAMES.index("A_w100_z2"):A_NAMES.index("A_w100_rough") + 1] = (st - wmu[0]) / wsd[0]
        win_stats(z[c - LATE + 1:c + 1], st)
        Am[r, A_NAMES.index("A_w70_z2"):A_NAMES.index("A_w70_rough") + 1] = (st - wmu[1]) / wsd[1]

    # --- A: start-of-online augmentation ---
    rough = np.minimum(np.diff(z) ** 2, QCAP)            # rough[i-1] = (z_i - z_{i-1})^2
    qh = np.column_stack([qf[AR:T], rough[AR - 1:T - 1], (np.abs(ef[AR:T]) > 2.0).astype(float)])
    qmu, qsd = qh.mean(0), qh.std(0) + 1e-9
    qo = np.column_stack([qf[T:], rough[T - 1:], (np.abs(ef[T:]) > 2.0).astype(float)])   # real lags
    co = np.vstack([np.zeros(3), np.cumsum(qo, 0)])
    creal = qh[-L:].sum(0)
    sums = np.zeros(5)
    pseudo_mean = np.zeros(L)
    seq = np.empty(L + AR + 1)
    eps = np.random.default_rng(seed).standard_normal(L)
    pseudo_samp = np.empty(L)
    fixed_mean = None
    for r, t in enumerate(rows):
        nn = t + 1
        plain = co[nn] / nn
        Am[r, 0:3] = (plain - qmu) / qsd
        Am[r, 3:6] = ((creal + co[nn]) / (L + nn) - qmu) / qsd
        k_on = min(nn, AR)                               # online points whose lags reach into the pseudo-past
        tail_s = co[nn] - co[k_on]                       # online points with all-online lags
        tail_n = nn - k_on
        for kind, col in (("mean", 6), ("samp", 9)):
            if kind == "mean":
                if fixed_mean is None or t < AR:
                    zfill = np.zeros(AR)
                    zfill[:min(nn, AR)] = zo[:min(nn, AR)]
                    ext = np.r_[np.zeros(L), zfill]
                    for i in range(L - 1, -1, -1):
                        ext[i] = b @ ext[i + 1:i + 1 + AR]
                    pseudo_mean = ext[:L].copy()
                    if t >= AR:
                        fixed_mean = pseudo_mean
                ps = pseudo_mean
            else:
                samp_pseudo(zo, t, b2, sb2 * sb2, eps, pseudo_samp)
                ps = pseudo_samp
            seq[:L] = ps
            seq[L:L + k_on] = zo[:k_on]
            seam_sums(seq[:L + k_on], coef, se, L + k_on, sums)
            # rough of the first online point is (zo0 - pseudo_last); seam_sums covers it. e2/tail counts
            # differ from rough counts because e needs AR lags.
            tot_e = sums[0] + tail_s[0]
            tot_r = sums[1] + tail_s[1]
            tot_t = sums[2] + tail_s[2]
            ne, nr_ = sums[3] + tail_n, sums[4] + tail_n
            Am[r, col:col + 3] = (np.array([tot_e / ne, tot_r / nr_, tot_t / ne]) - qmu) / qsd
    return wb, Bm, Am, skill, (post / npost if npost else None)


def _work_chunk(items):
    return [series_work(*it) for it in items]


# ---------------------------------------------------------------------------------------------
# evaluation helpers

BUCKETS = ((0, 100), (100, 300), (300, 700), (700, 10 ** 6))


def auc_row(ts_auc, x, y, step, in_slice):
    r = [ts_auc(x, y, step)]
    for lo, hi in BUCKETS:
        m = (step >= lo) & (step < hi)
        r.append(ts_auc(x[m], y[m], step[m]))
    r.append(ts_auc(x[in_slice], y[in_slice], step[in_slice]))
    return np.array(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", default="0:2000")
    ap.add_argument("--eval", default="2000:10000")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default="backcast_gate")
    args = ap.parse_args()
    from joblib import Parallel, delayed
    from scipy.stats import rankdata

    from sbrt.data import load_train
    from sbrt.metric import ts_auc

    os.makedirs("cv_out", exist_ok=True)
    log_path = os.path.join("cv_out", f"{args.out}.txt")
    logf = open(log_path, "w", encoding="utf-8")

    def say(s=""):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    f0, f1 = map(int, args.fit.split(":"))
    e0, e1 = map(int, args.eval.split(":"))
    S_all, tau_all = load_train()
    idx = np.r_[np.arange(f0, f1), np.arange(e0, e1)]
    t0 = time.time()
    say(f"backcast gate | fit {f0}-{f1 - 1} | eval {e0}-{e1 - 1} | folds {args.folds}")

    items = [(S_all[i][1], S_all[i][2], int(tau_all[i]), int(i)) for i in idx]
    chunks = [items[k:k + 50] for k in range(0, len(items), 50)]
    res = [x for ch in Parallel(n_jobs=args.threads)(delayed(_work_chunk)(c) for c in chunks) for x in ch]
    say(f"per-series streams: {time.time() - t0:.0f}s")
    nfit = f1 - f0
    fit_res, ev_res = res[:nfit], res[nfit:]
    tau_f, tau = tau_all[f0:f1], tau_all[e0:e1]

    prior, nfb = NB.fit_prior_w([q[0] for q in fit_res], tau_f)
    nob = np.concatenate([q[0] for q, t in zip(ev_res, tau) if t < 0])
    say(f"bk EB prior from {nfb} breaks: kappa0 {prior[0]:.0f} alpha0 {prior[1]:.2f} beta0 {prior[2]:.2f}; "
        f"eval no-break whitened backward stream mean {nob.mean():+.3f} var {nob.var():.3f}")
    for wb, Bm, *_ in ev_res:
        m = len(wb)
        if m == 0:
            continue
        C1 = np.r_[0.0, np.cumsum(wb)][None, :]
        C2 = np.r_[0.0, np.cumsum(wb * wb)][None, :]
        fe = NB.features_from_sums((C1, C2, np.zeros(m + 1)), prior, 1)[:, :4]
        rows = np.arange(0, m + AR, SUB)
        ok = rows >= AR
        Bm[ok, 0:4] = fe[rows[ok] - AR]              # stream count t - AR + 1 at online step t

    S = [S_all[i] for i in range(e0, e1)]
    pairs = [(s[1], s[2]) for s in S]
    t1 = time.time()
    Bo = np.concatenate(Parallel(n_jobs=args.threads, backend="threading")(
        delayed(BO.run_series_bocpd)(h, o, SUB) for h, o in pairs))
    V = np.concatenate([VS.run_series_vol(h, o)[::SUB] for h, o in pairs])
    M = NB.load_model()
    sf = NB.sums_parallel([(S_all[i][1], S_all[i][2]) for i in range(f0, f1)], M, args.threads)
    nb_prior, _ = NB.fit_prior_w([NB.w_of(q) for q in sf], tau_f)
    se_ = NB.sums_parallel(pairs, M, args.threads)
    NBF = np.concatenate([NB.features_from_sums(q, nb_prior, SUB) for q in se_])
    say(f"reference features (bo, vol, nb): {time.time() - t1:.0f}s")

    Bf = np.concatenate([r[1] for r in ev_res])
    Af = np.concatenate([r[2] for r in ev_res])
    step = np.concatenate([np.arange(0, len(o), SUB) for _, o in pairs])
    sid = np.repeat(np.arange(len(S)), [(len(o) + SUB - 1) // SUB for _, o in pairs])
    y = ((tau[sid] >= 0) & (step >= tau[sid])).astype(np.int8)
    vr = np.full(len(S), np.nan)
    for i, ((h, o), t) in enumerate(zip(pairs, tau)):
        if t >= 0 and len(o) - t >= 30:
            vr[i] = BO.whitened_online(h, o)[0][t:t + 300].var()
    same = (vr > 0.85) & (vr < 1.15)
    in_slice = (tau[sid] < 0) | same[sid]
    say(f"rows {len(y):,} | series {len(S)} | breaks {(tau >= 0).sum()} | var-same breaks {int(same.sum())}")

    # ---- skill ----
    skill = np.array([r[3] for r in ev_res])
    posts = np.array([r[4] for r in ev_res if r[4] is not None])
    say("\nBackcast skill on the history (R2 by depth d = points before the late 70; 0 = no better than the mean)")
    dsel = [0, 1, 2, 4, 9, 19, 29]
    say("  depth      " + "  ".join(f"{d + 1:6d}" for d in dsel))
    for qq in (10, 50, 90):
        say(f"  R2 q{qq:<2d}     " + "  ".join(f"{np.percentile(skill[:, d], qq):6.3f}" for d in dsel))
    say(f"  share of series with R2 > 0.10 at depth 1 / 5 / 10: "
        + " / ".join(f"{(skill[:, d] > 0.10).mean():.2f}" for d in (0, 4, 9)))
    if len(posts):
        say(f"  windows wholly after a break ({len(posts)} series): mean err^2 / normal err^2 by depth "
            + "  ".join(f"{np.median(posts[:, d]):.3f}" for d in dsel) + "  (median over series)")

    # ---- single-feature gate ----
    rng = np.random.default_rng(0)
    sel = rng.choice(len(y), min(len(y), 150000), replace=False)
    refZ = np.column_stack([Bo[sel], V[sel], NBF[sel]])
    ref_names = list(BO.FEATURE_NAMES) + list(VS.FEATURE_NAMES) + list(NB.FEATURE_NAMES)
    R = np.column_stack([rankdata(refZ[:, j]) for j in range(refZ.shape[1])])
    R = (R - R.mean(0)) / (R.std(0) + 1e-12)

    def corr_best(x):
        r = rankdata(x[sel])
        r = (r - r.mean()) / (r.std() + 1e-12)
        cc = np.abs(R.T @ r) / len(r)
        j = int(np.argmax(cc))
        return cc[j], ref_names[j], cc

    hdr = "                          all    0-100  100-300  300-700    700+  var-same   corr"
    G = {}
    say("\nREFERENCES" + hdr[10:])
    for nm, x in (("bo_w_logodds", Bo[:, 0]), ("vol_scanmax", V[:, VS.FEATURE_NAMES.index("vol_scanmax")]),
                  ("nb_logodds", NBF[:, 0]), ("nb_lvr", NBF[:, 2])):
        G[nm] = auc_row(ts_auc, x, y, step, in_slice)
        say(f"  {nm:22s}" + "  ".join(f"{v:7.4f}" for v in G[nm]))
    say("\nB (signed, then |.|)" + hdr[19:])
    corrB = {}
    for c, nm in enumerate(B_NAMES):
        x = Bf[:, c]
        G[nm] = auc_row(ts_auc, x, y, step, in_slice)
        G[nm + "|"] = auc_row(ts_auc, np.abs(x), y, step, in_slice)
        cb, cn, call = corr_best(x)
        corrB[nm] = (cb, cn, call)
        say(f"  {nm:22s}" + "  ".join(f"{v:7.4f}" for v in G[nm]) + f"   {cb:.2f} {cn}")
        say(f"  {'  |' + nm + '|':22s}" + "  ".join(f"{v:7.4f}" for v in G[nm + '|']))
    say("\nA (|.|, then signed)" + hdr[19:])
    for c, nm in enumerate(A_NAMES):
        x = Af[:, c]
        G[nm + "|"] = auc_row(ts_auc, np.abs(x), y, step, in_slice)
        G[nm] = auc_row(ts_auc, x, y, step, in_slice)
        cb, cn, _ = corr_best(np.abs(x))
        say(f"  {'|' + nm + '|':22s}" + "  ".join(f"{v:7.4f}" for v in G[nm + '|']) + f"   {cb:.2f} {cn}")
        say(f"  {nm:22s}" + "  ".join(f"{v:7.4f}" for v in G[nm]))

    # ---- A paired differences (series bootstrap SE) ----
    say(f"\nA: augmented minus real counterpart, same transform (|.| and signed), TS-AUC difference +- "
        f"series-bootstrap SE ({NBOOT} draws)")
    nS = len(S)
    starts = np.r_[0, np.cumsum([(len(o) + SUB - 1) // SUB for _, o in pairs])]
    boot_sets = []
    for bi in range(NBOOT):
        pick = np.random.default_rng(100 + bi).integers(0, nS, nS)
        boot_sets.append(np.concatenate([np.arange(starts[i], starts[i + 1]) for i in pick]))
    early = step < 100
    comps = [(f"A_mean_{q}", f"A_plain_{q}") for q in QS] + [(f"A_samp_{q}", f"A_plain_{q}") for q in QS] \
        + [(f"A_real_{q}", f"A_plain_{q}") for q in QS] \
        + [(f"A_win_{q}", f"A_w100_{q}") for q in WQS] + [(f"A_win_{q}", f"A_w70_{q}") for q in WQS]
    adiff = {}
    for tr in ("abs", "signed"):
        Acol = {nm: (np.abs(Af[:, c]) if tr == "abs" else Af[:, c]) for c, nm in enumerate(A_NAMES)}
        for a, bnm in comps:
            out = []
            for m in (slice(None), early):
                xa, xb, yy, ss = Acol[a][m], Acol[bnm][m], y[m], step[m]
                d = ts_auc(xa, yy, ss) - ts_auc(xb, yy, ss)
                bs = []
                for bset in boot_sets:
                    bm = bset if isinstance(m, slice) else bset[early[bset]]
                    bs.append(ts_auc(Acol[a][bm], y[bm], step[bm]) - ts_auc(Acol[bnm][bm], y[bm], step[bm]))
                out.append((d, float(np.std(bs))))
            adiff[(a, bnm, tr)] = out
            say(f"  {tr:6s} {a:14s} - {bnm:14s}  all {out[0][0]:+.4f} +- {out[0][1]:.4f}   "
                f"0-100 {out[1][0]:+.4f} +- {out[1][1]:.4f}")

    # ---- incremental LightGBM ----
    import lightgbm as lgb
    say(f"\nIncremental test: LightGBM, {args.folds}-fold by series, metric-matched weights")
    twin_b = [B_NAMES.index("fw_e2_100")]
    test_b = [i for i in range(len(B_NAMES)) if i not in twin_b]
    twin_a = [i for i, nm in enumerate(A_NAMES) if nm.split("_")[1] in ("plain", "real", "w100", "w70")]
    test_a = [i for i in range(len(A_NAMES)) if i not in twin_a]
    say(f"  base = bo + vol + nb + step + twins {[B_NAMES[i] for i in twin_b] + [A_NAMES[i] for i in twin_a]}")
    say(f"  B = {[B_NAMES[i] for i in test_b]}")
    say(f"  A = {[A_NAMES[i] for i in test_a]}")
    base = np.column_stack([Bo, V, NBF, step, Bf[:, twin_b], Af[:, twin_a]]).astype(np.float32)
    sets = {"base": base,
            "base+B": np.column_stack([base, Bf[:, test_b]]).astype(np.float32),
            "base+A": np.column_stack([base, Af[:, test_a]]).astype(np.float32),
            "base+B+A": np.column_stack([base, Bf[:, test_b], Af[:, test_a]]).astype(np.float32)}
    wt = np.zeros(len(y))
    sb = step // 30
    for g in np.unique(sb):
        m = sb == g
        npos, nneg = int(y[m].sum()), int(m.sum() - y[m].sum())
        if npos and nneg:
            wt[m & (y == 1)] = nneg / 2.0
            wt[m & (y == 0)] = npos / 2.0
    wt /= wt.mean()
    fold_of = np.random.default_rng(7).permutation(nS) % args.folds
    params = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=500, feature_fraction=0.7,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=args.threads, verbose=-1, seed=0)
    scores = {k: [] for k in sets}
    for f in range(args.folds):
        va = fold_of[sid] == f
        for k, X in sets.items():
            bst = lgb.train(params, lgb.Dataset(X[~va], y[~va], weight=wt[~va]), num_boost_round=300)
            scores[k].append(ts_auc(bst.predict(X[va]), y[va], step[va]))
        say(f"  fold {f}: " + "  ".join(f"{k} {scores[k][-1]:.4f}" for k in sets))
    inc = {}
    for k in ("base+B", "base+A", "base+B+A"):
        d = np.array(scores[k]) - np.array(scores["base"])
        inc[k] = d
        say(f"  {k:9s} - base: {d.mean():+.4f} +- {d.std(ddof=1) / np.sqrt(len(d)):.4f}  ({(d > 0).sum()}/{len(d)} folds)"
            f"   [base {np.mean(scores['base']):.4f}]")

    # ---- verdicts ----
    K = args.folds
    say("\nVERDICTS (criteria written in the docstring before the run)")
    if e1 - e0 < 4000:
        say("  SMOKE RUN (< 4,000 eval series): verdicts printed for the pipeline check only; do not act on them.")
    cb, cn, call = corrB["bk_logodds"]
    j_bonb = [i for i, n_ in enumerate(ref_names) if n_.startswith("bo_") or n_.startswith("nb_")]
    cbn = float(np.max(call[j_bonb]))
    kb1 = cbn >= 0.90 and G["bk_logodds"][5] <= G["nb_logodds"][5] + 0.005
    say(f"  KB1 B redundant: corr with bo_/nb_ {cbn:.2f}, var-same {G['bk_logodds'][5]:.4f} vs nb {G['nb_logodds'][5]:.4f}"
        f" -> {'KILLED' if kb1 else 'passes'}")
    irr_best = [(max(G[f'irr_{w}'][0], G[f'irr_{w}|'][0], 1 - G[f'irr_{w}'][0]),
                 max(G[f'irr_{w}'][5], G[f'irr_{w}|'][5], 1 - G[f'irr_{w}'][5])) for w in IRR_WS]
    kb2 = all(a < 0.53 and v < 0.52 for a, v in irr_best)
    say(f"  KB2 irreversibility dead: best (all, var-same) per W {[(round(a, 4), round(v, 4)) for a, v in irr_best]}"
        f" -> {'KILLED' if kb2 else 'passes'}")
    d = inc["base+B"]
    kb3 = d.mean() < 0.002 or (d > 0).sum() < K - 1
    say(f"  KB3 B adds nothing: {d.mean():+.4f}, {(d > 0).sum()}/{K} -> {'KILLED' if kb3 else 'PASSES: build cv.py flag'}")
    cand = []                                            # (label, [(gain, se) all, (gain, se) 0-100])
    for tr in ("abs", "signed"):
        for q in QS:
            for a in ("A_mean_", "A_samp_"):
                cand.append((f"{tr} {a}{q}", adiff[(a + q, "A_plain_" + q, tr)]))
        for q in WQS:      # the window version must beat BOTH real windows (100 and the last 70)
            o100, o70 = adiff[("A_win_" + q, "A_w100_" + q, tr)], adiff[("A_win_" + q, "A_w70_" + q, tr)]
            cand.append((f"{tr} A_win_{q}", [(min(x[0], z_[0]), max(x[1], z_[1])) for x, z_ in zip(o100, o70)]))
    passing = [lab for lab, o in cand if any(g >= 0.003 and g > 3 * s_ for g, s_ in o)]
    ka1 = not passing
    top = max(cand, key=lambda c: max(c[1][0][0], c[1][1][0]))
    say(f"  KA1 A dead as a statistic (needs >= +0.003 AND > 3 SE, all or 0-100): best {top[0]} all "
        f"{top[1][0][0]:+.4f}+-{top[1][0][1]:.4f}, 0-100 {top[1][1][0]:+.4f}+-{top[1][1][1]:.4f}; passing {passing} "
        f"-> {'KILLED' if ka1 else 'passes'}")
    d = inc["base+A"]
    ka2 = d.mean() < 0.002 or (d > 0).sum() < K - 1
    say(f"  KA2 A adds nothing: {d.mean():+.4f}, {(d > 0).sum()}/{K} -> {'KILLED' if ka2 else 'passes'}")
    say(f"\ntotal {time.time() - t0:.0f}s; log {log_path}")
    np.savez(os.path.join("cv_out", f"{args.out}.npz"), B=Bf.astype(np.float32), A=Af.astype(np.float32),
             B_names=np.array(B_NAMES), A_names=np.array(A_NAMES), step=step, y=y, sid=sid, skill=skill)
    logf.close()


if __name__ == "__main__":
    main()
