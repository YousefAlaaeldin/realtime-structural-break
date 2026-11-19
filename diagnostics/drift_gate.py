"""Gate for option 2 of HANDOFF section 32: is the NO-BREAK drift of our detectors predictable from history
information the trees do not already have?

The trees see, per row, the detector value D(t), the step and the per-series history descriptors (ctx_*, the volatility
constants). A model of "expected D(t) without a break" built from those same columns adds nothing the trees cannot
learn. It is new information only if RICHER history inputs explain more of the no-break D(t). So, on no-break series
only, at fixed online steps, this compares the out-of-fold R^2 of
    A: LightGBM on [ctx_* (5), vol constants (4)]                         (what the trees have)
    B: A + end-of-history NBM hidden state (8 PCA channels of the fold's NBM, state after the first online point)
         + end-of-history window statistics (last 100 / 200 / 500 history points: log var ratio to the whole
           history, kurtosis, lag-1 autocorrelation, share |z| > 2 and > 3, log ratio of diff variance)   (new)
for 8 detectors at steps 99 / 300 / 600. Folds = cv.py's folds (the NBM files are per fold, so validation series
use the NBM of the fold that never trained on them). The models train on the no-break series of the fold's
training part and predict its no-break validation series. Paired per-series SE of the R^2 gain.

ONE-SIDED KILL TEST (written before any run, HANDOFF section 32): CLOSE option 2 unless at least one of the four top
detectors (nb_lvr, nb_logodds, bo_wu_lvr, win10_tail3) gains dR^2 >= 0.02 and > 2 SE at >= 2 of the 3 steps. A pass
only means "build the cv.py columns"; the paired cv.py run decides adoption as usual.

Run (owner, CPU, my estimate 3-6 min):  python diagnostics/drift_gate.py  -> cv_out/drift_gate.txt
"""
import os
import sys
import time

import numpy as np
from scipy.stats import kurtosis
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
os.chdir(HERE)
import lightgbm as lgb
from sbrt import bocpd as BO
from sbrt import features as F
from sbrt import nbm as NBM
from sbrt import volstate as VS
from sbrt.data import load_train

NBM_CFG = "e25_h64_lr0.002_b24_wd0.0001_lf50"
STEPS = (99, 300, 600)
TOP = ("nb_lvr", "nb_logodds", "bo_wu_lvr", "win10_tail3")
TARGETS = TOP + ("win100_e10sq", "vol_win100", "vol_lr100", "bo_wu_logodds")
KILL_DR2, KILL_SE, KILL_STEPS = 0.02, 2.0, 2
out_lines = []
QUICK = "--quick" in sys.argv   # smoke test: fold 0 only, history statistics of the first 2,000 series


def say(s=""):
    print(s, flush=True)
    out_lines.append(s)


def hist_stats(h):
    h = np.asarray(h, np.float64)
    sd_all = h.var()
    dv_all = np.diff(h).var()
    r = []
    for W in (100, 200, 500):
        x = h[-W:]
        r += [np.log(x.var() / sd_all), kurtosis(x), np.corrcoef(x[:-1], x[1:])[0, 1],
              np.mean(np.abs(x) > 2), np.mean(np.abs(x) > 3), np.log(np.diff(x).var() / dv_all)]
    return r


def r2_pair(y, pa, pb):
    """R^2 of A and B and the paired SE of (R2_B - R2_A) over rows (one row per series at a fixed step)."""
    v = y.var()
    ea, eb = (y - pa) ** 2, (y - pb) ** 2
    d = (ea - eb) / v
    return 1 - ea.mean() / v, 1 - eb.mean() / v, d.mean(), d.std(ddof=1) / np.sqrt(len(d))


def main():
    t0 = time.time()
    S, tau = load_train()
    n = len(S)
    z = np.load(os.path.join("data", "feat_270.npz"))
    sid, step = z["sid"], z["step"]
    names = list(F.FEATURE_NAMES)
    ctx = [names.index(c) for c in ("ctx_logn", "ctx_skew", "ctx_kurt", "ctx_rho1", "ctx_phi")]
    kcols = {c: names.index(c) for c in TARGETS if c in names}
    Xk = z["X"][:, ctx + list(kcols.values())]
    del z
    V = np.load(os.path.join("data", f"vol{VS.VERSION}_orig.npz"))["V"]
    vn = list(VS.FEATURE_NAMES)
    Bz = np.load(os.path.join("data", f"bo{BO.VERSION}_orig.npz"))["B"]
    bn = list(BO.FEATURE_NAMES)
    nn_ = list(NBM.FEATURE_NAMES)
    vconst = [vn.index(c) for c in ("vol_lvh200", "vol_lvh1000", "vol_logs2T", "vol_persist")]
    say(f"loaded features ({time.time() - t0:.0f}s); computing end-of-history window statistics ...")
    HS = np.array([hist_stats(s[1]) if (not QUICK or i < 2000) else [0.0] * 18 for i, s in enumerate(S)], np.float32)
    say(f"window statistics for {n} histories ({time.time() - t0:.0f}s)")

    # one row per (series, step) for the steps of interest; series must be long enough
    row_of = {}
    for st in STEPS:
        m = step == st
        r = np.full(n, -1, np.int64)
        r[sid[m]] = np.flatnonzero(m)
        row_of[st] = r
    has_break = (tau >= 0).astype(int)
    skf = StratifiedKFold(5, shuffle=True, random_state=0)   # identical to cv.py
    res = {(t, st): [[], [], []] for t in TARGETS for st in STEPS}   # y, pA, pB pooled over folds
    params = dict(objective="regression", learning_rate=0.03, num_leaves=15, min_data_in_leaf=50,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1,
                  num_threads=8, seed=0)
    for f, (tr, va) in enumerate(skf.split(np.arange(n), has_break)):
        if QUICK and f > 0:
            break
        NB = np.load(os.path.join("data", f"nbm{NBM.VERSION}_f{f}_{NBM_CFG}_orig.npz"))["B"]
        hz = np.load(os.path.join("data", f"gru_nbmh8v1_f{f}_{NBM_CFG}_aug2_s0_m20.npz"))
        HP = hz["flat"][hz["off"][:n]]           # state after the first online point of every original series
        del hz
        tr0 = tr[tau[tr] < 0]
        va0 = va[tau[va] < 0]
        for st in STEPS:
            rt, rv = row_of[st][tr0], row_of[st][va0]
            kt, kv = rt >= 0, rv >= 0
            st_tr, st_va = tr0[kt], va0[kv]
            rt, rv = rt[kt], rv[kv]

            def inputs(rows, ser):
                A = np.hstack([Xk[rows][:, :5], V[rows][:, vconst]])
                B = np.hstack([A, HP[ser], HS[ser]])
                return A, B

            At, Bt = inputs(rt, st_tr)
            Av, Bv = inputs(rv, st_va)
            for t in TARGETS:
                if t in kcols:
                    j = 5 + list(kcols).index(t)
                    yt, yv = Xk[rt, j], Xk[rv, j]
                elif t in vn:
                    yt, yv = V[rt, vn.index(t)], V[rv, vn.index(t)]
                elif t in bn:
                    yt, yv = Bz[rt, bn.index(t)], Bz[rv, bn.index(t)]
                else:
                    yt, yv = NB[rt, nn_.index(t)], NB[rv, nn_.index(t)]
                ok_t, ok_v = np.isfinite(yt), np.isfinite(yv)
                pa = lgb.train(params, lgb.Dataset(At[ok_t], yt[ok_t]), 300).predict(Av[ok_v])
                pb = lgb.train(params, lgb.Dataset(Bt[ok_t], yt[ok_t]), 300).predict(Bv[ok_v])
                res[(t, st)][0].append(yv[ok_v])
                res[(t, st)][1].append(pa)
                res[(t, st)][2].append(pb)
        say(f"fold {f} done: {len(tr0)} / {len(va0)} no-break training / validation series ({time.time() - t0:.0f}s)")

    say("\nOut-of-fold R^2 of the NO-BREAK detector value, A = what the trees have, B = A + richer history inputs")
    say(f"{'detector':16s} {'step':>5s} {'n':>5s} {'R2 A':>7s} {'R2 B':>7s} {'dR2':>8s} {'SE':>7s}")
    passed = []
    for t in TARGETS:
        good = 0
        for st in STEPS:
            y, pa, pb = (np.concatenate(v) for v in res[(t, st)])
            ra, rb, d, se = r2_pair(y, pa, pb)
            ok = d >= KILL_DR2 and d > KILL_SE * se
            good += ok
            say(f"{t:16s} {st:5d} {len(y):5d} {ra:7.3f} {rb:7.3f} {d:+8.4f} {se:7.4f}{'  *' if ok else ''}")
        if t in TOP and good >= KILL_STEPS:
            passed.append(t)
    say(f"\nKILL TEST (top detectors {', '.join(TOP)}: dR2 >= {KILL_DR2} and > {KILL_SE:g} SE at >= {KILL_STEPS} of "
        f"{len(STEPS)} steps): " + (f"PASS on {', '.join(passed)} -> build the cv.py columns" if passed
                                     else "FAIL -> option 2 CLOSED"))
    say(f"total {time.time() - t0:.0f}s")
    if QUICK:
        return
    with open(os.path.join("cv_out", "drift_gate.txt"), "w") as fh:
        fh.write("\n".join(out_lines) + "\n")


if __name__ == "__main__":
    main()
