"""Learned per-point evidence, accumulated exactly over the unknown break time ("learned CUSUM", direction 4).

bocpd.py scores each online point with a hand-written model (normal before, normal with new mean/variance after).
Here the per-point score is LEARNED: a classifier is trained on single online points, "after the break" (1) vs
"before the break" (0). Its log-odds then estimate the
log density ratio r(x_t) = log p_after(x_t) / p_before(x_t) for the point's inputs x_t. That uses every labelled
POINT (about 4M per fold) instead of one label pattern per series.

Point inputs (all causal; no online position, so the ratio carries no break-timing prior):
    w_t, w_{t-1}, w_{t-2}       history-ECDF-whitened AR residual (as bocpd.py / evidence.py)
    wu_t, wu_{t-1}              the same for the volatility-adjusted residual
    series context (constant):  lag-1 autocorrelation, excess kurtosis, volatility persistence, end-of-history
                                volatility level (log L / vbar), log history length
The context lets the ratio depend on the kind of series, e.g. "large |w| means more in a calm, persistent series".

Accumulation, with R_n = sum of r over the first n points and the generator's hazard prior (as bocpd.py):
    lr_logodds   log posterior odds of "break by t": logsumexp_j(log pi_j + R_n - R_j) - log P(no break by t)
    lr_logodds_h the same with the ratio halved (a tempered version, in case the classifier is over-confident)
    lr_cusum     max_j (R_n - R_j)                              (CUSUM of the learned ratio)
    lr_tsb       posterior mean log(1 + time since the break)
    lr_mean100   mean r over the last 100 points
Mis-specification is accepted: the learned ratio averages over all break types, like evidence.py's mixture.

Honest by cross-fitting: the classifier is trained once per CV fold (the same StratifiedKFold(5, shuffle,
random_state=0) split by has-break as cv.py), and a series' features always come from the classifier that did not
see it (originals: the fold where they are validation series; re-cut variants: their parent's fold).

Build the feature caches (needed before cv.py --lratio):
    python -m sbrt.lratio build --aug 2 --threads 6
-> data/lr1_orig.npz, data/lr1_aug2_s0_m20.npz, cv_out/lratio_build.txt (point AUCs, single-feature TS-AUCs)
"""
import argparse
import math
import os
import sys
import time

import numpy as np
from numba import njit

from sbrt import bocpd as BO
from sbrt import bocpd2 as B2
from sbrt import volstate as VS

VERSION = 1
SUB = 3
FEATURE_NAMES = ["lr_logodds", "lr_logodds_h", "lr_cusum", "lr_tsb", "lr_mean100"]
N_LR = len(FEATURE_NAMES)
INPUT_NAMES = ["w", "w1", "w2", "wu", "wu1", "c_rho1", "c_kurt", "c_persist", "c_lvh", "c_logn"]
CLIP = 5.0
LGB_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=15, min_data_in_leaf=5000,
                  feature_fraction=1.0, bagging_fraction=0.5, bagging_freq=1, lambda_l2=10.0, seed=0,
                  deterministic=True, force_row_wise=True, verbose=-1)
ROUNDS = 300


def point_inputs(hist, online):
    """(n, 10) float32 point inputs for one series (causal)."""
    H = VS.fit_history(hist)
    W = B2.streams(hist, online)
    n = W.shape[1]
    X = np.zeros((n, len(INPUT_NAMES)), dtype=np.float32)
    w, wu = W[0], W[1]
    X[:, 0] = w
    X[1:, 1] = w[:-1]
    X[2:, 2] = w[:-2]
    X[:, 3] = wu
    X[1:, 4] = wu[:-1]
    z = H["z"]
    zc = z - z.mean()
    X[:, 5] = np.dot(zc[:-1], zc[1:]) / max(np.dot(zc, zc), 1e-12)
    X[:, 6] = np.mean(zc ** 4) / max(np.mean(zc ** 2) ** 2, 1e-12) - 3.0
    X[:, 7] = H["p"]
    X[:, 8] = math.log(H["L"] / H["vbar"])
    X[:, 9] = math.log(len(hist))
    return X


@njit(cache=True, nogil=True)
def _accumulate(r, LOGH, LP, sub, out):
    n_all = len(r)
    R = np.zeros(n_all + 1)
    for i in range(n_all):
        R[i + 1] = R[i] + r[i]
    buf = np.zeros(n_all + 1)
    o = 0
    for i in range(0, n_all, sub):
        n = i + 1
        for half in range(2):
            c = 1.0 if half == 0 else 0.5
            mx = -1e300
            for j in range(n):
                a = LP[j] + LOGH[j] + c * (R[n] - R[j])
                buf[j] = a
                if a > mx:
                    mx = a
            s = 0.0
            e_tsb = 0.0
            for j in range(n):
                wgt = math.exp(buf[j] - mx)
                s += wgt
                e_tsb += wgt * math.log(1.0 + n - j)
            out[o, half] = mx + math.log(s) - LP[n]
            if half == 0:
                out[o, 3] = e_tsb / s
        cm = 0.0
        for j in range(n):
            d = R[n] - R[j]
            if d > cm:
                cm = d
        out[o, 2] = cm
        k = n - 100 if n > 100 else 0
        out[o, 4] = (R[n] - R[k]) / (n - k)
        o += 1


def accumulate(r, sub=SUB):
    """Feature rows (online steps 0, sub, 2 sub, ...) from one series' per-point log ratios."""
    r = np.clip(np.asarray(r, dtype=np.float64), -CLIP, CLIP)
    out = np.zeros(((len(r) + sub - 1) // sub, N_LR))
    _accumulate(r, BO.LOGH, BO.LP, sub, out)
    return out


def _train(X, y, g, threads):
    """Break series only, every series balanced: its pre-break points and its post-break points each get total
    weight 1/2. The constant series context then cannot predict the label on its own (no 'this kind of series has
    long post-break stretches' shortcut); it can only change how the point inputs are read."""
    import lightgbm as lgb
    key = g * 2 + y
    cnt = np.bincount(key, minlength=2 * (int(g.max()) + 1)).astype(float)
    ok = (cnt[g * 2] > 0) & (cnt[g * 2 + 1] > 0)
    X, y, key = X[ok], y[ok], key[ok]
    w = 0.5 / cnt[key]
    w = w / w.mean()
    p = dict(LGB_PARAMS, num_threads=threads)
    return lgb.train(p, lgb.Dataset(X, y, weight=w), ROUNDS)


def _ratio(m, X, prior_logit):
    return m.predict(X, raw_score=True) - prior_logit   # balanced training -> prior logit is ~0; kept explicit


def build(args):
    from joblib import Parallel, delayed
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from sbrt import augment as A
    from sbrt.data import load_train
    from sbrt.metric import ts_auc

    os.makedirs("cv_out", exist_ok=True)
    log = open(os.path.join("cv_out", "lratio_build.txt"), "w", encoding="utf-8")

    def say(s=""):
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    S, tau = load_train()
    dbg = "_dbg" if args.n_series else ""     # debug runs never overwrite the real caches
    if args.n_series:
        S, tau = S[:args.n_series], tau[:args.n_series]
    N = len(S)
    t0 = time.time()
    XI = Parallel(n_jobs=args.threads, backend="threading")(delayed(point_inputs)(s[1], s[2]) for s in S)
    lens = np.array([len(x) for x in XI])
    say(f"point inputs for {N:,} series ({lens.sum():,} points) in {time.time() - t0:.0f}s")
    Xall = np.concatenate(XI)
    del XI
    pid = np.repeat(np.arange(N), lens)
    pstep = np.concatenate([np.arange(L) for L in lens])
    py = ((tau[pid] >= 0) & (pstep >= tau[pid])).astype(np.int8)
    offs = np.r_[0, np.cumsum(lens)]
    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(np.arange(N), (tau >= 0).astype(int)))
    fold_of = np.zeros(N, dtype=np.int64)
    for f, (_, va) in enumerate(folds):
        fold_of[va] = f
    models = []
    r_all = np.zeros(len(py))
    for f, (tr, va) in enumerate(folds):
        t0 = time.time()
        trm = np.isin(pid, tr)
        fit_m = trm & (tau[pid] >= 0) & (pstep >= 2)   # steps 0-1 have zero-filled lags and are always pre-break:
                                                        # training on them would teach "lag == 0 -> no break"
        m = _train(Xall[fit_m], py[fit_m], pid[fit_m], args.threads)
        models.append(m)
        vam = ~trm
        r_all[vam] = _ratio(m, Xall[vam], 0.0)
        auc = roc_auc_score(py[vam], r_all[vam])
        say(f"fold {f}: point classifier trained on {int(fit_m.sum()):,} points of break series ({time.time() - t0:.0f}s); "
            f"validation point AUC {auc:.4f}; mean r on pre/post points {r_all[vam][py[vam] == 0].mean():+.4f} / "
            f"{r_all[vam][py[vam] == 1].mean():+.4f}")
    imp = np.mean([mm.feature_importance("gain") / mm.feature_importance("gain").sum() for mm in models], axis=0)
    say("point-classifier gain share: " + ", ".join(f"{n} {100 * v:.1f}%" for n, v in sorted(
        zip(INPUT_NAMES, imp), key=lambda kv: -kv[1])))
    del Xall
    t0 = time.time()
    L = np.concatenate(Parallel(n_jobs=args.threads, backend="threading")(
        delayed(accumulate)(r_all[offs[i]:offs[i + 1]]) for i in range(N))).astype(np.float32)
    np.savez(os.path.join("data", f"lr{VERSION}_orig{dbg}.npz"), L=L, n=N)
    say(f"accumulated features for the originals in {time.time() - t0:.0f}s -> data/lr{VERSION}_orig.npz")

    # single-feature TS-AUC on the (cross-fitted) originals, vs bocpd.py's best feature
    step = np.concatenate([np.arange(0, Lx, SUB) for Lx in lens])
    sid = np.repeat(np.arange(N), (lens + SUB - 1) // SUB)
    y = ((tau[sid] >= 0) & (step >= tau[sid])).astype(np.int8)
    buckets = ((0, 100), (100, 300), (300, 700), (700, 10000))

    def row(nm, x):
        r = [ts_auc(x, y, step)] + [ts_auc(x[(step >= lo) & (step < hi)], y[(step >= lo) & (step < hi)],
                                          step[(step >= lo) & (step < hi)]) for lo, hi in buckets]
        say(f"  {nm:16s} " + "  ".join(f"{v:7.4f}" for v in r))

    say("\nsingle-feature TS-AUC, all 10,000 series (out-of-fold)   all     0-100  100-300  300-700    700+")
    bo = os.path.join("data", f"bo{BO.VERSION}_orig.npz")
    if os.path.exists(bo):
        Bo = np.load(bo)["B"]
        if len(Bo) == len(y):
            for nm in ("bo_w_logodds", "bo_wu_logodds"):
                row(nm, Bo[:, BO.FEATURE_NAMES.index(nm)])
            from scipy.stats import spearmanr
            sel = np.random.default_rng(0).choice(len(y), min(len(y), 200000), replace=False)
            rc = spearmanr(L[sel, 0], Bo[sel, 0]).statistic
    for c, nm in enumerate(FEATURE_NAMES):
        row(nm, L[:, c])
    if os.path.exists(bo) and len(Bo) == len(y):
        say(f"rank correlation lr_logodds vs bo_w_logodds: {rc:.3f}")

    if args.aug > 0:
        t0 = time.time()
        V = A.make_variants(S, tau, args.aug, seed=args.aug_seed, min_shift=args.aug_min_shift)

        def one(v):
            Xv = point_inputs(v[2], v[3])
            return accumulate(_ratio(models[fold_of[v[0]]], Xv, 0.0))

        LA = np.concatenate(Parallel(n_jobs=args.threads, backend="threading")(delayed(one)(v) for v in V))
        stub = f"aug{args.aug}_s{args.aug_seed}_m{args.aug_min_shift}"
        np.savez(os.path.join("data", f"lr{VERSION}_{stub}{dbg}.npz"), L=LA.astype(np.float32), n=len(V))
        say(f"re-cut variants: {len(V):,} series in {time.time() - t0:.0f}s -> data/lr{VERSION}_{stub}.npz")
    say("\ndone")


def _selftest():
    rng = np.random.default_rng(0)
    r = rng.standard_normal(301) * 0.1
    a1 = accumulate(r, 1)
    a3 = accumulate(r, 3)
    assert np.allclose(a1[::3], a3), "sub-sampled != full"
    r2 = r.copy()
    r2[200:] = 3.0
    assert np.array_equal(accumulate(r2, 1)[:200], a1[:200]), "future points changed earlier features"
    b = accumulate(r2, 1)
    print(f"accumulate: logodds at 199 {b[199, 0]:+.1f}, 230 {b[230, 0]:+.1f}; tsb at 300 {np.expm1(b[300, 3]):.0f} "
          f"(true 100); cusum {b[300, 2]:.1f}")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "test"], nargs="?", default="test")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--aug", type=int, default=2)
    ap.add_argument("--aug-seed", type=int, default=0)
    ap.add_argument("--aug-min-shift", type=int, default=20)
    ap.add_argument("--n-series", type=int, default=0, help="debug: first N series only (writes *_dbg caches)")
    a = ap.parse_args()
    build(a) if a.cmd == "build" else _selftest()
