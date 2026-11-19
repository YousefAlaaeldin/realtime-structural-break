"""Streaming kernel two-sample test (MMD) between the online and the history JOINT behaviour of consecutive residuals.

Why (HANDOFF section 20): in the oracle study, MMD on residual pairs/triples was the only new information family
with an incremental gain (+0.0066 +- 0.0022 at 300 post-break points), and pairs/triples beat single residuals:
the joint behaviour of consecutive points adds to the marginal distribution (which sbrt/disttest.py covers).

Separate from sbrt/features.py like evidence/volstate/disttest; columns are appended only with cv.py --mmd.

1. Embedded samples, one per online step (all causal; the first ones reach back into the history):
       e2  (e_{t-1}, e_t)           history-fitted AR(10) residuals (= kernel e10)
       e3  (e_{t-2}, e_{t-1}, e_t)
       u2  (u_{t-1}, u_t)           u = e / frozen volatility forecast (sbrt/volstate.py); history: e / in-sample GARCH
2. Gaussian kernel exp(-|x - y|^2 / (2 d)) (d = 2 or 3, the oracle's kernel) approximated by D = 256 random Fourier
   features phi(x) = sqrt(2/D) cos(W x + b), with ONE fixed random W, b for all series (seeded; constants to ship).
   Then MMD^2 between two samples = |mean phi(sample 1) - mean phi(sample 2)|^2, and the history's mean embedding is a
   per-series constant: every online statistic is an O(D) update of cumulative sums.
3. Per variable, normalised as  L |mean_segment phi - mean_history phi|^2 / tr(Cov_history phi)  (about 1 under an
   i.i.d. null whatever L is, so steps are comparable):
       full (all online points), w100, w300 (last 100 / 300), scan (max over every start, segment >= 30 points).
   12 features. In batch mode (training) the scan runs only on the rows cv.py keeps (every 3rd step); those rows are
   identical to streaming (tested).

Self-test (go/no-go oracle + redundancy check):  python -m sbrt.mmdstream [offset]     (offset 5000 = held out)
"""
import math

import numpy as np
from numba import njit

from sbrt import features as F
from sbrt import volstate as VS

AR = F.AR
CAP = F.CAP
D = 256
MINSEG = 30
VARS = ("e2", "e3", "u2")
DIMS = np.array([2, 3, 2], dtype=np.int64)
NV = len(VARS)
STATS = ("full", "w100", "w300", "scan")
NS = len(STATS)
FEATURE_NAMES = [f"mmd_{v}_{s}" for v in VARS for s in STATS]
N_MMD = len(FEATURE_NAMES)
VERSION = 1   # bump when the kernel changes (part of the cache names)

_rng = np.random.default_rng(20261014)
W = np.zeros((NV, 3, D))
for _k in range(NV):
    W[_k, :DIMS[_k]] = _rng.standard_normal((DIMS[_k], D)) / math.sqrt(DIMS[_k])
B = _rng.uniform(0.0, 2.0 * math.pi, (NV, D))
SC = math.sqrt(2.0 / D)


def _phi(X, k):
    return SC * np.cos(X @ W[k, :X.shape[1]] + B[k])


Q_MU, Q_SD, Q_SE, Q_L, Q_P, Q_AR = 0, 1, 2, 3, 4, 5
Q_LEN = Q_AR + AR
S_T, S_D, S_E1, S_E2, S_U1, S_LAG = 0, 1, 2, 3, 4, 5
S_LEN = S_LAG + AR


def make_mmd_params(hist):
    H = VS.fit_history(hist)
    eh = H["eh"]
    vh = eh / np.sqrt(H["s2h"])
    samples = (np.column_stack([eh[:-1], eh[1:]]), np.column_stack([eh[:-2], eh[1:-1], eh[2:]]),
               np.column_stack([vh[:-1], vh[1:]]))
    MU = np.zeros((NV, D))
    TR = np.zeros(NV)
    for k, Xs in enumerate(samples):
        P = _phi(Xs, k)
        MU[k] = P.mean(0)
        TR[k] = max(float(P.var(0).sum()), 1e-9)
    Q = np.zeros(Q_LEN)
    Q[Q_MU], Q[Q_SD], Q[Q_SE] = H["P"][F.P_MU], H["P"][F.P_SD], H["se"]
    Q[Q_L], Q[Q_P] = H["L"], H["p"]
    Q[Q_AR:] = H["coef"]
    ST = np.zeros(S_LEN)
    ST[S_D] = H["s2T1"] - H["L"]
    ST[S_E1], ST[S_E2], ST[S_U1] = eh[-1], eh[-2], vh[-1]
    z = H["z"]
    for i in range(AR):
        ST[S_LAG + i] = z[-1 - i]
    return Q, MU, TR, ST, np.zeros((NV, D, CAP + 1))


@njit(cache=True, nogil=True)
def _seg(C, k, j, n, MU, TR):
    L = n - j
    s = 0.0
    for q in range(C.shape[1]):
        d = (C[k, q, n] - C[k, q, j]) / L - MU[k, q]
        s += d * d
    return L * s / TR[k]


@njit(cache=True, nogil=True)
def mmd_step(x, Q, Wt, Bt, sc, dims, MU, TR, C, ST, out, do_scan):
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
    vec = np.zeros(3)
    for k in range(C.shape[0]):
        if k == 0:
            vec[0] = ST[S_E1]
            vec[1] = e
        elif k == 1:
            vec[0] = ST[S_E2]
            vec[1] = ST[S_E1]
            vec[2] = e
        else:
            vec[0] = ST[S_U1]
            vec[1] = u
        for q in range(C.shape[1]):
            a = Bt[k, q]
            for i in range(dims[k]):
                a += Wt[k, i, q] * vec[i]
            C[k, q, n] = C[k, q, t] + sc * math.cos(a)
        o = k * 4
        out[o] = _seg(C, k, 0, n, MU, TR)
        w = 100 if n > 100 else n
        out[o + 1] = _seg(C, k, n - w, n, MU, TR)
        w = 300 if n > 300 else n
        out[o + 2] = _seg(C, k, n - w, n, MU, TR)
        if do_scan:
            m = 0.0
            for j in range(0, n - MINSEG + 1):
                s = _seg(C, k, j, n, MU, TR)
                if s > m:
                    m = s
            out[o + 3] = m
    ST[S_D] = ST[S_D] * Q[Q_P]
    ST[S_E2] = ST[S_E1]
    ST[S_E1] = e
    ST[S_U1] = u
    for i in range(AR - 1, 0, -1):
        ST[S_LAG + i] = ST[S_LAG + i - 1]
    ST[S_LAG] = z
    ST[S_T] = n * 1.0


@njit(cache=True, nogil=True)
def _mmd_run(online, Q, Wt, Bt, sc, dims, MU, TR, C, ST, out, sub):
    r = 0
    row = np.zeros(out.shape[1])
    for i in range(len(online)):
        keep = i % sub == 0
        mmd_step(online[i], Q, Wt, Bt, sc, dims, MU, TR, C, ST, row, keep)
        if keep:
            for c in range(out.shape[1]):
                out[r, c] = row[c]
            r += 1


def run_series_mmd(hist, online, sub=1):
    """Batch matrix (rows for online steps 0, sub, 2 sub, ..., N_MMD) for one series -- identical to streaming."""
    Q, MU, TR, ST, C = make_mmd_params(hist)
    online = np.asarray(online, dtype=np.float64)
    if len(online) >= CAP:
        raise ValueError(f"online segment longer than the supported {CAP - 1} points")
    out = np.zeros(((len(online) + sub - 1) // sub, N_MMD))
    _mmd_run(online, Q, W, B, SC, DIMS, MU, TR, C, ST, out, sub)
    return out


class MmdStream:
    """One series' streaming state for use inside infer()."""

    def __init__(self, hist):
        self.Q, self.MU, self.TR, self.ST, self.C = make_mmd_params(hist)
        self.out = np.zeros(N_MMD)

    def push(self, x):
        if self.ST[S_T] >= CAP - 1:
            return self.out
        mmd_step(float(x), self.Q, W, B, SC, DIMS, self.MU, self.TR, self.C, self.ST, self.out, True)
        return self.out


# ---------------------------------------------------------------------------------------------
def _selftest(n_series=2000, offset=0, threads=8):
    import os
    import sys
    import time
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from joblib import Parallel, delayed
    from scipy.stats import rankdata

    from sbrt import disttest as DT
    from sbrt.data import load_train
    from sbrt.metric import ts_auc

    S, tau = load_train()
    S, tau = S[offset:offset + n_series], tau[offset:offset + n_series]
    print(f"series {offset}..{offset + n_series - 1}; D = {D} random Fourier features")
    rng = np.random.default_rng(0)

    # 1. streaming == batch (all rows and every 3rd row), causal, finite, deterministic
    for i in rng.choice(n_series, 20, replace=False):
        _, h, o = S[i]
        Bm = run_series_mmd(h, o)
        assert np.array_equal(Bm, run_series_mmd(h, o)), "two runs differ"
        st = MmdStream(h)
        Sm = np.array([st.push(x).copy() for x in o])
        assert np.array_equal(Bm, Sm), "streaming != batch"
        assert np.array_equal(run_series_mmd(h, o, 3), Sm[::3]), "sub-sampled batch != streaming"
        assert np.isfinite(Bm).all()
        if len(o) > 40:
            o2 = o.copy()
            o2[35:] = rng.standard_normal(len(o) - 35) * 3
            assert np.array_equal(run_series_mmd(h, o2)[:35], Bm[:35]), "future points changed earlier features"
    print("streaming == batch (also every 3rd row), causal, finite, deterministic: OK")

    # 2. the random-feature MMD tracks the exact MMD (unbiased, same Gaussian kernel) on real segments
    def exact(Xa, Xb, d):
        def km(A, Bb, same):
            Dm = (A * A).sum(1)[:, None] + (Bb * Bb).sum(1)[None, :] - 2 * A @ Bb.T
            K = np.exp(-np.maximum(Dm, 0) / (2 * d))
            return (K.sum() - np.trace(K)) / (len(A) * (len(A) - 1)) if same else K.mean()
        return km(Xa, Xa, True) + km(Xb, Xb, True) - 2 * km(Xa, Xb, False)
    ex, ap = [], []
    for i in range(150):
        _, h, o = S[i]
        if len(o) < 200:
            continue
        H = VS.fit_history(h)
        eh = H["eh"]
        zz = np.r_[H["z"], (o - H["P"][F.P_MU]) / H["P"][F.P_SD]]
        nh = len(H["z"])
        Xo = np.column_stack([zz[nh - 1 - k: len(zz) - 1 - k] for k in range(AR)])
        eo = np.clip((zz[nh:] - Xo @ H["coef"]) / H["se"], -10, 10)[:200]
        Ph = np.column_stack([eh[:-1], eh[1:]])
        Po = np.column_stack([eo[:-1], eo[1:]])
        pick = rng.choice(len(Ph), min(800, len(Ph)), replace=False)
        ex.append(exact(Po, Ph[pick], 2))
        ap.append(float(((_phi(Po, 0).mean(0) - _phi(Ph, 0).mean(0)) ** 2).sum()))
    print(f"random-feature vs exact MMD^2 on residual pairs (200 online points, {len(ex)} series): "
          f"rank correlation {np.corrcoef(rankdata(ex), rankdata(ap))[0, 1]:.3f}")

    # 3. features, single-feature TS-AUC vs the strongest existing ones, redundancy
    t0 = time.time()
    M = Parallel(n_jobs=threads, backend="threading")(delayed(run_series_mmd)(s[1], s[2], 3) for s in S)
    print(f"MMD features for {n_series} series in {time.time() - t0:.0f}s")
    K = Parallel(n_jobs=threads, backend="threading")(delayed(lambda s: F.run_series(s[1], s[2])[::3])(s) for s in S)
    V = Parallel(n_jobs=threads, backend="threading")(delayed(lambda s: VS.run_series_vol(s[1], s[2])[::3])(s) for s in S)
    Dd = Parallel(n_jobs=threads, backend="threading")(delayed(DT.run_series_dist)(s[1], s[2], 3) for s in S)
    M, K, V, Dd = np.concatenate(M), np.concatenate(K), np.concatenate(V), np.concatenate(Dd)
    step = np.concatenate([np.arange(0, len(s[2]), 3) for s in S])
    sid = np.repeat(np.arange(n_series), [(len(s[2]) + 2) // 3 for s in S])
    y = ((tau[sid] >= 0) & (step >= tau[sid])).astype(np.int8)
    m99 = (step == 99) & (tau[sid] < 0)
    print(f"null check, no-break series at step 99: mean of the normalised full statistic "
          f"e2 {M[m99, 0].mean():.2f}, e3 {M[m99, 4].mean():.2f}, u2 {M[m99, 8].mean():.2f} (about 1 if i.i.d.)")

    buckets = ((0, 100), (100, 300), (300, 700), (700, 10000))

    def row(nm, x, extra=""):
        r = [ts_auc(x, y, step)] + [ts_auc(x[(step >= lo) & (step < hi)], y[(step >= lo) & (step < hi)],
                                          step[(step >= lo) & (step < hi)]) for lo, hi in buckets]
        print(f"  {nm:18s} " + "  ".join(f"{v:7.4f}" for v in r) + extra)

    print("\nsingle-feature TS-AUC (0.5 = no signal)        all     0-100  100-300  300-700    700+")
    print("  -- existing references:")
    for nm in ("glrmax_e10sq", "glrmax_abse10"):
        row(nm, K[:, F.FEATURE_NAMES.index(nm)])
    for nm in ("vol_scanmax", "vol_lr_full"):
        row(nm, V[:, VS.FEATURE_NAMES.index(nm)])
    for nm in ("dist_e_ad", "dist_e_w1", "dist_u_adscan"):
        row(nm, Dd[:, DT.FEATURE_NAMES.index(nm)])
    sel = rng.choice(len(y), min(len(y), 150000), replace=False)
    Z = np.column_stack([K[sel], V[sel], Dd[sel]])
    old = list(F.FEATURE_NAMES) + list(VS.FEATURE_NAMES) + list(DT.FEATURE_NAMES)
    R = np.column_stack([rankdata(Z[:, j]) for j in range(Z.shape[1])])
    R = (R - R.mean(0)) / (R.std(0) + 1e-12)
    print("  -- new (last columns: max |rank correlation| with any existing kernel/vol/dist feature, and which):")
    for c, nm in enumerate(FEATURE_NAMES):
        r = rankdata(M[sel, c])
        r = (r - r.mean()) / (r.std() + 1e-12)
        cc = np.abs(R.T @ r) / len(r)
        j = int(np.argmax(cc))
        row(nm, M[:, c], f"   {cc[j]:.2f} {old[j]}")
    print("\nGo/no-go: worth a cross-validation run if the best new statistics are competitive with the existing ones "
          "(about 0.58 overall) and not near-copies (rank correlation below ~0.9), especially at 300+ steps.")
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    import sys
    _selftest(offset=int(sys.argv[1]) if len(sys.argv) > 1 else 0)
