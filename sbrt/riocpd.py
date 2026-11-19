"""Single-series Rio-CPD: Riemannian (affine-invariant) geodesic distance between lag-covariance matrices of the online
part and the history, plus a history-calibrated CUSUM (HANDOFF section 21; owner's request 2026-10-19 after reading
Deng et al., "Correlation-aware Online Change Point Detection", CIKM 2025). Columns only with cv.py --rio.

Adaptation to one series. Rio-CPD tracks the correlation matrix of a MULTIVARIATE stream on the manifold of symmetric
positive definite (SPD) matrices and runs a CUSUM on the geodesic distance to the Frechet mean of the past. A univariate
series has no cross-correlations, so the matrix here is the K x K lag-covariance (Toeplitz) matrix of the whitened
residual w (bocpd.py stream: history-ECDF-whitened AR(10) residual, ~N(0, 1) and ~uncorrelated before a break), lags
0..K-1, K = 5. The reference (the Frechet mean of the "past") is the HISTORY's matrix; its sliding windows give the
null distribution of the distance, so the CUSUM is calibrated per series on its own break-free data.
    distance d(A, B) = || log(A^-1/2 B A^-1/2) ||_F = sqrt(sum_i log^2 lambda_i), lambda = eigenvalues of A^-1/2 B A^-1/2
    covariance version: variance and dependence together; correlation version (each matrix divided by its lag-0
    entry): dependence only, the part a variance detector cannot see.
10 features at online step t (all causal):
    rio_cov_full, rio_cov_100, rio_cov_30    covariance distance, all online points / last 100 / last 30
    rio_cor_full, rio_cor_100                correlation distance, all online points / last 100
    rio_cov_100_u                            covariance distance of the volatility-adjusted stream wu, last 100
    rio_z30_cov                              (d30 - mu0) / sd0: the last-30 distance against the history's own
                                             sliding 30-point windows (step 5)
    rio_cusum_cov, rio_cusum_cor             CUSUM S = max(0, S + z - 0.5) of that z (covariance / correlation)
    rio_age_cov                              (t - last reset of the covariance CUSUM) / (t + 1)
Lag products reach back into the last K-1 history points (same whitening), windows shorter than 10 points give 0.
Prior art in the project (section 8): variogram lags 1-16, lag products (kernel), AR(1)-change stream (bocpd2,
-0.0002), MMD on residual pairs/triples (-0.0004), --shape dependence agreement (section 20). What is new is the
joint SPD geometry and the per-series sliding-window null; the information (second-order structure) is not.

DECISION RULE (written before any run): one family, one flag, the standing rule (HANDOFF section 5):
  1. cv.py trees seed 0 with --rio vs the current best seed-0 trees. Early stop: folds 0 and 1 both <= 0 -> CLOSED.
  2. 5-fold paired mean > 0: seed 1, the seed-average comparison, the trees-only blend.
  3. ADOPT if the blend gains >= +0.0005, >= 4/5 folds, > 2 SE. Production then needs a streaming class (the kernel is
     causal) and the GRU/SVM inputs.
  Reported, not a gate: single-feature TS-AUC (self-test) and the gain share of the 10 columns.
EXPECTATION (written before any run): ~10% chance to pass (a new algorithm on information the trees already have,
section 9); blend +0.000 to +0.001.

Self-test (causality, geodesic sanity, single-feature TS-AUC on series 5000-6999):  python -m sbrt.riocpd
"""
import math

import numpy as np
from numba import njit

from sbrt import bocpd as BO
from sbrt import evidence as EV
from sbrt import volstate as VS

VERSION = 1
K = 5
RIDGE = 1e-3
MINWIN = 10
NULL_W, NULL_STEP = 30, 5
CUSUM_K = 0.5
FEATURE_NAMES = ["rio_cov_full", "rio_cov_100", "rio_cov_30", "rio_cor_full", "rio_cor_100", "rio_cov_100_u",
                 "rio_z30_cov", "rio_cusum_cov", "rio_cusum_cor", "rio_age_cov"]
N_RIO = len(FEATURE_NAMES)


@njit(cache=True, nogil=True)
def _cum_lags(x):
    """P[l, i + 1] = sum_{s <= i} x[s] x[s - l] over s >= K - 1 (x starts with K - 1 history points)."""
    n = len(x)
    P = np.zeros((K, n + 1))
    for i in range(n):
        for l in range(K):
            v = x[i] * x[i - l] if (i >= K - 1) else 0.0
            P[l, i + 1] = P[l, i] + v
    return P


@njit(cache=True, nogil=True)
def _toeplitz(P, a, b, cor):
    """Ridge-regularised K x K lag-covariance (or correlation) matrix of window x[a..b] (inclusive, x indices)."""
    m = b - a + 1
    g = np.zeros(K)
    for l in range(K):
        g[l] = (P[l, b + 1] - P[l, a]) / m
    if cor:
        g0 = max(g[0], 1e-8)
        for l in range(K):
            g[l] /= g0
    M = np.empty((K, K))
    for i in range(K):
        for j in range(K):
            M[i, j] = g[abs(i - j)]
        M[i, i] += RIDGE
    return M


@njit(cache=True, nogil=True)
def _dist(Ais, B):
    """Affine-invariant geodesic distance between A (given as A^-1/2) and B."""
    C = Ais @ B @ Ais
    C = 0.5 * (C + C.T)
    lam = np.linalg.eigvalsh(C)
    s = 0.0
    for v in lam:
        if v < 1e-10:
            v = 1e-10
        s += math.log(v) ** 2
    return math.sqrt(s)


def _inv_sqrt(A):
    lam, V = np.linalg.eigh(0.5 * (A + A.T))
    return (V * (1.0 / np.sqrt(np.maximum(lam, 1e-10)))) @ V.T


@njit(cache=True, nogil=True)
def _run(x, xu, Ais_c, Ais_r, Ais_u, mu_c, sd_c, mu_r, sd_r, sub, out):
    """x / xu: K - 1 history points followed by the online stream; rows for online steps 0, sub, ..."""
    off = K - 1
    n = len(x) - off
    P = _cum_lags(x)
    Pu = _cum_lags(xu)
    Sc, Sr, reset = 0.0, 0.0, 0
    r = 0
    for t in range(n):
        b = off + t                       # x index of the current point
        zc = 0.0
        if t + 1 >= NULL_W:
            a = b - NULL_W + 1
            dc = _dist(Ais_c, _toeplitz(P, a, b, False))
            dr = _dist(Ais_r, _toeplitz(P, a, b, True))
            zc = (dc - mu_c) / sd_c
            zr = (dr - mu_r) / sd_r
            Sc = max(0.0, Sc + zc - CUSUM_K)
            Sr = max(0.0, Sr + zr - CUSUM_K)
            if Sc == 0.0:
                reset = t
        else:
            reset = t
        if t % sub != 0:
            continue
        o = out[r]
        r += 1
        for c in range(out.shape[1]):
            o[c] = 0.0
        if t + 1 < MINWIN:
            continue
        a0 = off
        a100 = max(off, b - 99)
        a30 = max(off, b - 29)
        o[0] = _dist(Ais_c, _toeplitz(P, a0, b, False))
        o[1] = _dist(Ais_c, _toeplitz(P, a100, b, False))
        o[2] = _dist(Ais_c, _toeplitz(P, a30, b, False))
        o[3] = _dist(Ais_r, _toeplitz(P, a0, b, True))
        o[4] = _dist(Ais_r, _toeplitz(P, a100, b, True))
        o[5] = _dist(Ais_u, _toeplitz(Pu, a100, b, False))
        o[6] = zc
        o[7] = Sc
        o[8] = Sr
        o[9] = (t - reset) / (t + 1.0)


@njit(cache=True, nogil=True)
def _null(P, n_hist, Ais, cor):
    """Mean and sd of the 30-point window distance over the history's own sliding windows."""
    vals = []
    for b in range(K - 1 + NULL_W - 1, n_hist, NULL_STEP):
        vals.append(_dist(Ais, _toeplitz(P, b - NULL_W + 1, b, cor)))
    v = np.array(vals)
    return v.mean(), max(v.std(), 1e-3)


def _hist_streams(hist):
    """History whitened residuals on the same scale as the online w / wu (in-sample ECDF, as bocpd.py builds refs)."""
    H = VS.fit_history(hist)
    eh = np.asarray(H["eh"], dtype=np.float64)
    eu = eh / np.sqrt(np.asarray(H["s2h"], dtype=np.float64))
    rw, ru = np.sort(eh), np.sort(eu)
    w = np.clip(np.array([EV._whiten(e, rw) for e in eh]), -8, 8)
    u = np.clip(np.array([EV._whiten(e, ru) for e in eu]), -8, 8)
    return w, u


def run_series_rio(hist, online, sub=1):
    """Rows for online steps 0, sub, 2 sub, ... (N_RIO columns); causal: row t uses online[:t + 1] only."""
    online = np.asarray(online, dtype=np.float64)
    W = BO.whitened_online(hist, online)
    hw, hu = _hist_streams(hist)
    Ph, Phu = _cum_lags(hw), _cum_lags(hu)
    nh = len(hw)
    Ac = _toeplitz(Ph, K - 1, nh - 1, False)
    Ar = _toeplitz(Ph, K - 1, nh - 1, True)
    Au = _toeplitz(Phu, K - 1, nh - 1, False)
    Ais_c, Ais_r, Ais_u = _inv_sqrt(Ac), _inv_sqrt(Ar), _inv_sqrt(Au)
    mu_c, sd_c = _null(Ph, nh, Ais_c, False)
    mu_r, sd_r = _null(Ph, nh, Ais_r, True)
    x = np.r_[hw[-(K - 1):], W[0]]
    xu = np.r_[hu[-(K - 1):], W[1]]
    out = np.zeros(((len(online) + sub - 1) // sub, N_RIO))
    _run(x, xu, Ais_c, Ais_r, Ais_u, mu_c, sd_c, mu_r, sd_r, sub, out)
    return out


def _selftest(n_series=2000, offset=5000, threads=8):
    import time
    from joblib import Parallel, delayed
    from sbrt.data import load_train
    from sbrt.metric import ts_auc
    # geodesic sanity: d(A, A) = 0, d(A, cA) = sqrt(K) |log c|, symmetric
    rng = np.random.default_rng(0)
    G = rng.normal(size=(K, K))
    A = G @ G.T + K * np.eye(K)
    G2 = rng.normal(size=(K, K))
    B = G2 @ G2.T + K * np.eye(K)
    assert abs(_dist(_inv_sqrt(A), A)) < 1e-6
    assert abs(_dist(_inv_sqrt(A), 2.0 * A) - math.sqrt(K) * math.log(2.0)) < 1e-6
    assert abs(_dist(_inv_sqrt(A), B) - _dist(_inv_sqrt(B), A)) < 1e-6
    print("geodesic sanity OK (identity, scaling, symmetry)")
    S, tau = load_train()
    for i in rng.choice(len(S), 20, replace=False):
        _, h, o = S[i]
        full = run_series_rio(h, o, 1)
        cut = int(rng.integers(1, len(o) + 1))
        assert np.array_equal(run_series_rio(h, o[:cut], 1), full[:cut]), f"series {i}: prefix rows differ"
        alt = np.r_[o[:cut], rng.normal(0, 3, len(o) - cut)]
        assert np.array_equal(run_series_rio(h, alt, 1)[:cut], full[:cut]), f"series {i}: future leaked"
    print("causality OK on 20 series (prefix and altered-future checks)")
    t0 = time.time()
    idx = range(offset, offset + n_series)
    mats = Parallel(n_jobs=threads, backend="threading")(delayed(run_series_rio)(S[i][1], S[i][2], 3) for i in idx)
    print(f"{n_series} series in {time.time() - t0:.0f}s")
    X = np.concatenate(mats)
    step = np.concatenate([np.arange(0, len(S[i][2]), 3) for i in idx])
    y = np.concatenate([((np.arange(0, len(S[i][2]), 3) >= tau[i]) & (tau[i] >= 0)) for i in idx]).astype(np.int8)
    print("single-feature TS-AUC (0.5 = none):")
    for j, nm in enumerate(FEATURE_NAMES):
        a = ts_auc(X[:, j], y, step)
        print(f"  {nm:15s} {a:.4f}   |0.5 distance| {abs(a - 0.5):.4f}")
    print("self-test OK")


if __name__ == "__main__":
    _selftest()
