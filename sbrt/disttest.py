"""Whole-distribution tests, online vs history: Kolmogorov-Smirnov, Anderson-Darling, Cramer-von Mises, Wasserstein-1
(streaming, causal).

The question: two segments can share mean and variance and still differ in distribution (shape, tails). The
270-feature kernel covers the distribution only through a decile chi-square, a rank mean, tail counts at fixed
thresholds and skew/kurtosis terms. This module adds the classical two-sample distances between the online points
and the history (the history is break-free by construction, so it is the reference sample).

Separate from sbrt/features.py like sbrt/evidence.py and sbrt/volstate.py; columns are appended only with
cv.py --dist.

1. Three variables, each compared with its own history reference:
       z   the standardised value (marginal distribution; inflated by autocorrelation, see the null check)
       e   the history-fitted AR(10) residual (= kernel e10): the distribution after removing linear dependence
       u   e divided by the frozen volatility forecast of sbrt/volstate.py; history reference: e divided by the
           in-sample GARCH forecast. The distribution after accounting for the volatility state.
2. Streaming ECDF on 19 fixed history-quantile edges (probabilities 0.001 ... 0.999, denser in the tails, so
   Anderson-Darling's tail weighting has something to work with). G_k = share of the online points <= edge k,
   p_k = share of the history <= edge k. With n online and m history points and f = m / (n + m):
       KS  = sqrt(n f) max_k |G_k - p_k|
       AD  = n f sum_k (G_k - p_k)^2 / (p_k (1 - p_k)) dp_k        (trapezoid weights dp_k)
       CvM = n f sum_k (G_k - p_k)^2 dp_k
       W1  = sum_k |G_k - p_k| dx_k                                (Wasserstein-1 between the two ECDFs, value units)
3. Per variable, 8 features: KS, AD, CvM, W1 over all online points; KS and AD over the last 100 points; KS and AD
   scanned over every start j of the online segment (segment >= 20 points; the max over j). 24 features in all.
   p-values are not computed: the score is a ranking across series per step (TS-AUC), so only the statistic matters;
   the trees learn how much to trust it per kind of series.

Cost: the scan is O(t x 19 x 3) per step. In batch mode (training) the scan is only evaluated on the rows cv.py keeps
(every 3rd step); the cumulative counts are updated at every step, so those rows equal the streaming values exactly.

Self-test (go/no-go oracle + redundancy check):  python -m sbrt.disttest [offset]      (offset 5000 = held out)
"""
import math

import numpy as np
from numba import njit

from sbrt import features as F
from sbrt import volstate as VS

AR = F.AR
CAP = F.CAP
PROBS = np.array([0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99,
                  0.995, 0.999])
NE = len(PROBS)
NB = NE + 1
NV = 3
VARS = ("z", "e", "u")
STATS = ("ks", "ad", "cvm", "w1", "ks100", "ad100", "ksscan", "adscan")
NS = len(STATS)
FEATURE_NAMES = [f"dist_{v}_{s}" for v in VARS for s in STATS]
N_DIST = len(FEATURE_NAMES)
# compact set (cv.py --dist compact), chosen from the held-out self-test (series 5000-6999): z is inflated by
# autocorrelation and weak; Anderson-Darling beats KS and CvM on every variable; W1 is the other strong one
COMPACT = [f"dist_{v}_{s}" for v in ("e", "u") for s in ("ad", "w1", "ad100", "adscan")]
COMPACT_COLS = [FEATURE_NAMES.index(nm) for nm in COMPACT]
MINSEG = 20
VERSION = 1   # bump when the kernel changes (part of the cache names)


def _ref_tables(ref):
    """Edges, reference CDF at the edges, trapezoid weights in p and in x, for one history reference sample."""
    s = np.sort(ref)
    edges = np.quantile(s, PROBS)
    p = np.searchsorted(s, edges, side="right") / len(s)
    pp = np.r_[0.0, p, 1.0]
    dp = (pp[2:] - pp[:-2]) / 2.0
    ex = np.r_[edges[0] - (edges[1] - edges[0]), edges, edges[-1] + (edges[-1] - edges[-2])]
    dx = (ex[2:] - ex[:-2]) / 2.0
    return edges, p, dp, dx


# parameter vector Q and state ST
Q_MU, Q_SD, Q_SE, Q_L, Q_P, Q_AR = 0, 1, 2, 3, 4, 5
Q_LEN = Q_AR + AR
S_T, S_D, S_LAG = 0, 1, 2
S_LEN = S_LAG + AR


def make_dist_params(hist):
    H = VS.fit_history(hist)
    z = H["z"]
    refs = (z, H["eh"], H["eh"] / np.sqrt(H["s2h"]))
    E = np.zeros((NV, NE))
    PR = np.zeros((NV, NE))
    W = np.zeros((NV, NE))
    DX = np.zeros((NV, NE))
    M = np.zeros(NV)
    for v, r in enumerate(refs):
        E[v], PR[v], W[v], DX[v] = _ref_tables(r)
        M[v] = len(r)
    Q = np.zeros(Q_LEN)
    Q[Q_MU], Q[Q_SD], Q[Q_SE] = H["P"][F.P_MU], H["P"][F.P_SD], H["se"]
    Q[Q_L], Q[Q_P] = H["L"], H["p"]
    Q[Q_AR:] = H["coef"]
    ST = np.zeros(S_LEN)
    ST[S_D] = H["s2T1"] - H["L"]
    for i in range(AR):
        ST[S_LAG + i] = z[-1 - i]
    return Q, (E, PR, W, DX, M), ST, np.zeros((NV, NB, CAP + 1))


@njit(cache=True, nogil=True)
def _seg(C, v, j, n, PR, W, DX, M, out, o, full):
    """Statistics of online points j..n-1 of variable v, written to out[o:]. full=False: only KS and AD."""
    L = n - j
    f = M[v] / (L + M[v])
    cum = 0.0
    ks = 0.0
    ad = 0.0
    cvm = 0.0
    w1 = 0.0
    for k in range(NE):
        cum += C[v, k, n] - C[v, k, j]
        d = cum / L - PR[v, k]
        ad_ = d * d * W[v, k]
        pk = PR[v, k] * (1.0 - PR[v, k])
        ad += ad_ / (pk if pk > 1e-4 else 1e-4)
        if abs(d) > ks:
            ks = abs(d)
        if full:
            cvm += ad_
            w1 += abs(d) * DX[v, k]
    out[o] = math.sqrt(L * f) * ks
    out[o + 1] = L * f * ad
    if full:
        out[o + 2] = L * f * cvm
        out[o + 3] = w1


@njit(cache=True, nogil=True)
def dist_step(x, Q, E, PR, W, DX, M, C, ST, out, do_scan):
    t = int(ST[S_T])
    z = (x - Q[Q_MU]) / Q[Q_SD]
    pred = 0.0
    for i in range(AR):
        pred += Q[Q_AR + i] * ST[S_LAG + i]
    e = (z - pred) / Q[Q_SE]
    if e > 10.0:
        e = 10.0
    elif e < -10.0:
        e = -10.0
    u = e / math.sqrt(Q[Q_L] + ST[S_D])
    n = t + 1
    tmp = np.zeros(2)
    for v in range(NV):
        val = z if v == 0 else (e if v == 1 else u)
        b = np.searchsorted(E[v], val)
        for k in range(NB):
            C[v, k, n] = C[v, k, t]
        C[v, b, n] += 1.0
        o = v * NS
        _seg(C, v, 0, n, PR, W, DX, M, out, o, True)
        w = 100 if n > 100 else n
        _seg(C, v, n - w, n, PR, W, DX, M, out, o + 4, False)
        if do_scan:
            kmax = 0.0
            amax = 0.0
            for j in range(0, n - MINSEG + 1):
                _seg(C, v, j, n, PR, W, DX, M, tmp, 0, False)
                if tmp[0] > kmax:
                    kmax = tmp[0]
                if tmp[1] > amax:
                    amax = tmp[1]
            out[o + 6] = kmax
            out[o + 7] = amax
    ST[S_D] = ST[S_D] * Q[Q_P]
    for i in range(AR - 1, 0, -1):
        ST[S_LAG + i] = ST[S_LAG + i - 1]
    ST[S_LAG] = z
    ST[S_T] = n * 1.0


@njit(cache=True, nogil=True)
def _dist_run(online, Q, E, PR, W, DX, M, C, ST, out, sub):
    r = 0
    row = np.zeros(out.shape[1])
    for i in range(len(online)):
        keep = i % sub == 0
        dist_step(online[i], Q, E, PR, W, DX, M, C, ST, row, keep)
        if keep:
            for c in range(out.shape[1]):
                out[r, c] = row[c]
            r += 1


def run_series_dist(hist, online, sub=1):
    """Batch matrix (rows for online steps 0, sub, 2 sub, ..., N_DIST) for one series -- identical to streaming."""
    Q, (E, PR, W, DX, M), ST, C = make_dist_params(hist)
    online = np.asarray(online, dtype=np.float64)
    if len(online) >= CAP:
        raise ValueError(f"online segment longer than the supported {CAP - 1} points")
    out = np.zeros(((len(online) + sub - 1) // sub, N_DIST))
    _dist_run(online, Q, E, PR, W, DX, M, C, ST, out, sub)
    return out


class DistStream:
    """One series' streaming state for use inside infer()."""

    def __init__(self, hist):
        self.Q, self.T, self.ST, self.C = make_dist_params(hist)
        self.out = np.zeros(N_DIST)

    def push(self, x):
        if self.ST[S_T] >= CAP - 1:
            return self.out
        E, PR, W, DX, M = self.T
        dist_step(float(x), self.Q, E, PR, W, DX, M, self.C, self.ST, self.out, True)
        return self.out


# ---------------------------------------------------------------------------------------------
def _selftest(n_series=2000, offset=0, threads=8):
    import os
    import sys
    import time
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from joblib import Parallel, delayed
    from scipy.stats import ks_2samp, rankdata

    from sbrt.data import load_train
    from sbrt.metric import ts_auc

    S, tau = load_train()
    S, tau = S[offset:offset + n_series], tau[offset:offset + n_series]
    print(f"series {offset}..{offset + n_series - 1}")
    rng = np.random.default_rng(0)

    # 1. streaming == batch (all rows and every 3rd row), causal, finite, deterministic; KS vs scipy on a fine grid
    for i in rng.choice(n_series, 30, replace=False):
        _, h, o = S[i]
        B = run_series_dist(h, o)
        assert np.array_equal(B, run_series_dist(h, o)), "two runs differ"
        st = DistStream(h)
        Sm = np.array([st.push(x).copy() for x in o])
        assert np.array_equal(B, Sm), "streaming != batch"
        assert np.array_equal(run_series_dist(h, o, 3), Sm[::3]), "sub-sampled batch != streaming"
        assert np.isfinite(B).all()
        if len(o) > 30:
            o2 = o.copy()
            o2[25:] = rng.standard_normal(len(o) - 25) * 3
            assert np.array_equal(run_series_dist(h, o2)[:25], B[:25]), "future points changed earlier features"
    print("streaming == batch (also every 3rd row), causal, finite, deterministic: OK")
    _, h, o = S[0]
    z = (h - h.mean()) / h.std()
    zo = (o - h.mean()) / h.std()
    d_exact = ks_2samp(zo, z).statistic
    d_grid = run_series_dist(h, o)[-1, 0] / math.sqrt(len(o) * len(z) / (len(o) + len(z)))
    print(f"KS distance of series {offset}, z: 19-edge grid {d_grid:.4f} vs exact scipy {d_exact:.4f} (grid <= exact)")

    # 2. features + all kernel and volatility-state features on every 3rd step (like cv.py)
    t0 = time.time()
    D = Parallel(n_jobs=threads, backend="threading")(delayed(run_series_dist)(s[1], s[2], 3) for s in S)
    print(f"distribution features for {n_series} series in {time.time() - t0:.0f}s")
    K = Parallel(n_jobs=threads, backend="threading")(delayed(lambda s: F.run_series(s[1], s[2])[::3])(s) for s in S)
    V = Parallel(n_jobs=threads, backend="threading")(delayed(lambda s: VS.run_series_vol(s[1], s[2])[::3])(s) for s in S)
    D, K, V = np.concatenate(D), np.concatenate(K), np.concatenate(V)
    step = np.concatenate([np.arange(0, len(s[2]), 3) for s in S])
    sid = np.repeat(np.arange(n_series), [(len(s[2]) + 2) // 3 for s in S])
    y = ((tau[sid] >= 0) & (step >= tau[sid])).astype(np.int8)

    m99 = (step == 99) & (tau[sid] < 0)
    print(f"\nnull check, no-break series at step 99 ({int(m99.sum())}): mean / sd of KS (an i.i.d. sample would give "
          f"about 0.87 / 0.26 on an exact grid, less on 19 edges)")
    for v, nm in enumerate(VARS):
        c = FEATURE_NAMES.index(f"dist_{nm}_ks")
        print(f"  {nm}: {D[m99, c].mean():.3f} / {D[m99, c].std():.3f}")

    buckets = ((0, 100), (100, 300), (300, 700), (700, 10000))

    def row(nm, x, extra=""):
        r = [ts_auc(x, y, step)] + [ts_auc(x[(step >= lo) & (step < hi)], y[(step >= lo) & (step < hi)],
                                          step[(step >= lo) & (step < hi)]) for lo, hi in buckets]
        print(f"  {nm:20s} " + "  ".join(f"{v:7.4f}" for v in r) + extra)

    print("\nsingle-feature TS-AUC (0.5 = no signal)          all     0-100  100-300  300-700    700+")
    print("  -- existing references:")
    ref = ["chi2_full", "chi2_decay", "full_u", "full_usq", "full_tail3", "glrmax_usq", "glrmax_e10sq", "glrmax_abse10"]
    for nm in ref:
        row(nm, K[:, F.FEATURE_NAMES.index(nm)])
    for nm in ("vol_lr_full", "vol_scanmax"):
        row(nm, V[:, VS.FEATURE_NAMES.index(nm)])

    # redundancy: max |Spearman| with any existing feature (kernel 270 + volatility state 13), rows sub-sampled
    sel = rng.choice(len(y), min(len(y), 150000), replace=False)
    Z = np.column_stack([K[sel], V[sel]])
    names_old = list(F.FEATURE_NAMES) + list(VS.FEATURE_NAMES)
    R = np.column_stack([rankdata(Z[:, j]) for j in range(Z.shape[1])])
    R = (R - R.mean(0)) / (R.std(0) + 1e-12)
    print("  -- new (last columns: max |rank correlation| with any existing kernel/vol feature, and which):")
    for c, nm in enumerate(FEATURE_NAMES):
        r = rankdata(D[sel, c])
        r = (r - r.mean()) / (r.std() + 1e-12)
        cc = np.abs(R.T @ r) / len(r)
        j = int(np.argmax(cc))
        row(nm, D[:, c], f"   {cc[j]:.2f} {names_old[j]}")
    print("\nGo/no-go: a new statistic is worth a cross-validation run only if it is competitive with the best existing "
          "single features (about 0.59-0.60 in 100-300) AND not a near-copy of one (rank correlation below ~0.9).")
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    import sys
    _selftest(offset=int(sys.argv[1]) if len(sys.argv) > 1 else 0)
