"""Causal streaming features for online structural-break detection (v2).

One compiled `step` kernel is used both for training (looping over the online segment) and for
inference (one call per arriving point), so train and test features are identical by construction.
Feature t uses only online points 0..t and the historical segment.

For each of K "increment series" g_k(t) (functions of z_t and a few lags, centred by their
historical mean) we keep cumulative sums C_k. From those, in O(1) or O(t):
  * full-online standardised mean, windowed standardised means (10/30/100 points),
  * a scan (GLR-style) maximum / minimum of the standardised segment-mean over all candidate
    change points j -- the natural statistic when the break position is unknown.
"""
import numpy as np
from numba import njit

K = 32
CAP = 2048
WINDOWS = (10, 30, 100)
NLAG = 16
AR = 10

# parameter vector layout
P_CEN, P_LRV, P_INVSD = 0, K, 2 * K
P_PHI, P_SE1, P_SE10, P_MU, P_SD = 3 * K, 3 * K + 1, 3 * K + 2, 3 * K + 3, 3 * K + 4
P_EDGE = 3 * K + 5          # 9 decile edges
P_AR = P_EDGE + 9           # AR coefficients
P_CTX = P_AR + AR           # 5 context features
P_LEN = P_CTX + 5

# state vector layout
S_T, S_EW1, S_EW2, S_E10 = 0, 1, 2, 3
S_CNT, S_DCNT, S_DN, S_LAG = 4, 14, 24, 25
S_LEN = S_LAG + NLAG

# rare-event increments need longer minimum segments in the scan
MINSEG = np.full(K, 8, dtype=np.int64)
for _k in (8, 23):
    MINSEG[_k] = 20
for _k in (9, 30):
    MINSEG[_k] = 40

INC_NAMES = [
    "z", "z2", "lag1prod", "e1", "e1sq", "u", "usq", "absz", "tail2", "tail3",
    "vg1", "vg2", "vg4", "vg8", "vg16", "prod2", "prod3", "prod5", "prod10", "abs_abs1",
    "e10", "e10sq", "abse10", "tail_e10", "z3", "z4", "tail1", "tail1p5", "signpers", "e10prod",
    "tail4", "centre",
]
assert len(INC_NAMES) == K

# lower bound on the long-run variance of very rare indicators (avoids exploding z-scores)
ABS_FLOOR = {9: 0.004, 30: 0.004}

N_FEAT = 5 + 2 + K + K * len(WINDOWS) + 2 * K + 2 + 2 + 2 + 1 + 2 * K
FEATURE_NAMES = (
    ["ctx_logn", "ctx_skew", "ctx_kurt", "ctx_rho1", "ctx_phi", "t_log", "z_now"]
    + [f"full_{n}" for n in INC_NAMES]
    + [f"win{w}_{n}" for w in WINDOWS for n in INC_NAMES]
    + [f"glrmax_{n}" for n in INC_NAMES] + [f"glrmin_{n}" for n in INC_NAMES]
    + ["glr_Lstar_mean", "glr_Lstar_var", "ewma_z", "ewma_v", "chi2_full", "chi2_decay", "abs_z_now"]
    # localisation of the scan: log-length of the most suspicious stretch and its per-point effect size
    + [f"scanlogL_{n}" for n in INC_NAMES] + [f"scaneff_{n}" for n in INC_NAMES]
)
assert len(FEATURE_NAMES) == N_FEAT, (len(FEATURE_NAMES), N_FEAT)


@njit(cache=True, nogil=True)
def _inc(z, lag, e10prev, P, ref, inc):
    """Fill inc[0..K) for the point z given lag[i] = z_{t-1-i}. Returns e10 (AR residual)."""
    zc = min(max(z, -10.0), 10.0)
    zp1 = min(max(lag[0], -10.0), 10.0)
    e1 = (zc - P[P_PHI] * zp1) / P[P_SE1]
    pred = 0.0
    for i in range(AR):
        pred += P[P_AR + i] * lag[i]
    e10 = min(max((z - pred) / P[P_SE10], -10.0), 10.0)
    u = (np.searchsorted(ref, z) + 0.5) / (len(ref) + 1.0) - 0.5
    az = abs(zc)
    inc[0] = zc
    inc[1] = zc * zc
    inc[2] = zp1 * zc
    inc[3] = e1
    inc[4] = e1 * e1
    inc[5] = u
    inc[6] = u * u
    inc[7] = az
    inc[8] = 1.0 if az > 2.0 else 0.0
    inc[9] = 1.0 if az > 3.0 else 0.0
    d = zc - zp1
    inc[10] = d * d
    d = zc - min(max(lag[1], -10.0), 10.0)
    inc[11] = d * d
    d = zc - min(max(lag[3], -10.0), 10.0)
    inc[12] = d * d
    d = zc - min(max(lag[7], -10.0), 10.0)
    inc[13] = d * d
    d = zc - min(max(lag[15], -10.0), 10.0)
    inc[14] = d * d
    inc[15] = zc * min(max(lag[1], -10.0), 10.0)
    inc[16] = zc * min(max(lag[2], -10.0), 10.0)
    inc[17] = zc * min(max(lag[4], -10.0), 10.0)
    inc[18] = zc * min(max(lag[9], -10.0), 10.0)
    inc[19] = az * abs(zp1)
    inc[20] = e10
    inc[21] = e10 * e10
    inc[22] = abs(e10)
    inc[23] = 1.0 if abs(e10) > 2.0 else 0.0
    zz = min(max(zc, -6.0), 6.0)
    inc[24] = zz * zz * zz
    inc[25] = zz * zz * zz * zz
    inc[26] = 1.0 if az > 1.0 else 0.0
    inc[27] = 1.0 if az > 1.5 else 0.0
    inc[28] = 1.0 if zc * zp1 > 0.0 else 0.0
    inc[29] = e10 * e10prev
    inc[30] = 1.0 if az > 4.0 else 0.0
    inc[31] = 1.0 if az < 0.5 else 0.0
    return e10


@njit(cache=True)
def _hist_inc(zh, P, ref, out):
    """Increment matrix (K, n-NLAG) over the historical segment, using the shared kernel."""
    n = len(zh)
    lag = np.zeros(NLAG)
    for i in range(NLAG):
        lag[i] = zh[NLAG - 1 - i]
    e10prev = 0.0
    inc = np.empty(K)
    for t in range(NLAG, n):
        # lag[i] = zh[t-1-i]
        for i in range(NLAG):
            lag[i] = zh[t - 1 - i]
        e10prev = _inc(zh[t], lag, e10prev, P, ref, inc)
        for k in range(K):
            out[k, t - NLAG] = inc[k]


def _lrv(x, floor_iid, abs_floor=1e-8):
    n = len(x)
    b = max(5, min(50, n // 20))
    nb = n // b
    bm = x[: nb * b].reshape(nb, b).sum(axis=1) / np.sqrt(b)
    return max(float(bm.var()), 0.25 * floor_iid, abs_floor)


def make_params(hist):
    h = np.asarray(hist, dtype=np.float64)
    n = len(h)
    mu = float(h.mean())
    sd = float(h.std())
    if not sd > 0:
        sd = 1.0
    z = (h - mu) / sd
    z0, z1 = z[:-1], z[1:]
    d0 = float(np.dot(z0, z0))
    phi = float(np.dot(z0, z1) / d0) if d0 > 0 else 0.0
    se1 = max(float((z1 - phi * z0).std()), 1e-8)
    # AR(AR) by least squares on the history
    Xl = np.column_stack([z[AR - 1 - i: n - 1 - i] for i in range(AR)])
    yl = z[AR:]
    coef, *_ = np.linalg.lstsq(Xl, yl, rcond=None)
    se10 = max(float((yl - Xl @ coef).std()), 1e-8)
    ref = np.sort(z)

    P = np.zeros(P_LEN)
    P[P_PHI], P[P_SE1], P[P_SE10], P[P_MU], P[P_SD] = phi, se1, se10, mu, sd
    P[P_EDGE:P_EDGE + 9] = np.quantile(z, np.arange(1, 10) / 10.0)
    P[P_AR:P_AR + AR] = coef
    P[P_CTX] = np.log(n)
    P[P_CTX + 1] = float((z ** 3).mean())
    P[P_CTX + 2] = float((z ** 4).mean()) - 3.0
    P[P_CTX + 3] = float(np.dot(z0, z1) / max(float(np.dot(z, z)), 1e-12))
    P[P_CTX + 4] = phi
    inc = np.empty((K, n - NLAG))
    _hist_inc(z, P, ref, inc)
    for k in range(K):
        c = float(inc[k].mean())
        v = float(inc[k].var())
        P[P_CEN + k] = c
        P[P_LRV + k] = _lrv(inc[k] - c, v, ABS_FLOOR.get(k, 1e-8))
        P[P_INVSD + k] = 1.0 / np.sqrt(P[P_LRV + k])
    return P, ref, z


def new_state(z_hist_tail):
    ST = np.zeros(S_LEN)
    for i in range(NLAG):
        ST[S_LAG + i] = z_hist_tail[-1 - i]
    return ST, np.zeros((K, CAP + 1))


@njit(cache=True, nogil=True)
def step(x, P, ref, C, ST, out):
    t = int(ST[S_T])
    z = (x - P[P_MU]) / P[P_SD]
    inc = np.empty(K)
    lag = ST[S_LAG:S_LAG + NLAG]
    e10 = _inc(z, lag, ST[S_E10], P, ref, inc)
    n = t + 1
    for k in range(K):
        C[k, n] = C[k, t] + inc[k] - P[P_CEN + k]

    b = 0
    for q in range(9):
        if z > P[P_EDGE + q]:
            b += 1
    for q in range(10):
        ST[S_DCNT + q] *= 0.97
    ST[S_CNT + b] += 1.0
    ST[S_DCNT + b] += 1.0
    ST[S_DN] = ST[S_DN] * 0.97 + 1.0
    chi_f = 0.0
    chi_d = 0.0
    for q in range(10):
        d = ST[S_CNT + q] - n / 10.0
        chi_f += d * d / (n / 10.0)
        d2 = ST[S_DCNT + q] - ST[S_DN] / 10.0
        chi_d += d2 * d2 / (ST[S_DN] / 10.0)
    zc = min(max(z, -10.0), 10.0)
    ST[S_EW1] = 0.95 * ST[S_EW1] + 0.05 * zc
    ST[S_EW2] = 0.95 * ST[S_EW2] + 0.05 * (zc * zc - 1.0)

    for q in range(5):
        out[q] = P[P_CTX + q]
    out[5] = np.log(n * 1.0)
    out[6] = z
    o = 7
    sq = np.sqrt(n * 1.0)
    for k in range(K):
        out[o + k] = C[k, n] * P[P_INVSD + k] / sq
    o += K
    for wi in range(3):
        w = 10 if wi == 0 else (30 if wi == 1 else 100)
        if w > n:
            w = n
        sw = np.sqrt(w * 1.0)
        for k in range(K):
            out[o + k] = (C[k, n] - C[k, n - w]) * P[P_INVSD + k] / sw
        o += K

    gmax = np.full(K, -1e9)
    gmin = np.full(K, 1e9)
    bestabs = np.zeros(K)
    bestL = np.zeros(K, dtype=np.int64)
    for j in range(0, n - 7):
        L = n - j
        il = 1.0 / np.sqrt(L * 1.0)
        for k in range(K):
            if L >= MINSEG[k]:
                s = (C[k, n] - C[k, j]) * il * P[P_INVSD + k]
                if s > gmax[k]:
                    gmax[k] = s
                if s < gmin[k]:
                    gmin[k] = s
                if abs(s) > bestabs[k]:
                    bestabs[k] = abs(s)
                    bestL[k] = L
    for k in range(K):
        out[o + k] = gmax[k] if gmax[k] > -1e8 else 0.0
        out[o + K + k] = gmin[k] if gmin[k] < 1e8 else 0.0
    o += 2 * K
    out[o] = bestL[0] / (n * 1.0)
    out[o + 1] = bestL[1] / (n * 1.0)
    out[o + 2] = ST[S_EW1] / 0.16
    out[o + 3] = ST[S_EW2] / 0.16
    out[o + 4] = (chi_f - 9.0) / 4.243
    out[o + 5] = (chi_d - 9.0) / 4.243
    out[o + 6] = abs(zc)
    for k in range(K):
        if bestL[k] > 0:
            out[o + 7 + k] = np.log(bestL[k] * 1.0)
            out[o + 7 + K + k] = bestabs[k] / np.sqrt(bestL[k] * 1.0)
        else:
            out[o + 7 + k] = 0.0
            out[o + 7 + K + k] = 0.0

    ST[S_E10] = e10
    for i in range(NLAG - 1, 0, -1):
        ST[S_LAG + i] = ST[S_LAG + i - 1]
    ST[S_LAG] = z
    ST[S_T] = n * 1.0


@njit(cache=True, nogil=True)
def _run(online, P, ref, C, ST, out_mat):
    for i in range(len(online)):
        step(online[i], P, ref, C, ST, out_mat[i])


def run_series(hist, online):
    """Batch feature matrix (T, N_FEAT) for one series -- identical to streaming."""
    P, ref, z = make_params(hist)
    ST, C = new_state(z)
    online = np.asarray(online, dtype=np.float64)
    if len(online) >= CAP:
        raise ValueError(f"online segment longer than the supported {CAP - 1} points")
    out = np.zeros((len(online), N_FEAT))
    _run(online, P, ref, C, ST, out)
    return out


class Stream:
    """One series' streaming state for use inside infer()."""

    def __init__(self, hist):
        self.P, self.ref, z = make_params(hist)
        self.ST, self.C = new_state(z)
        self.out = np.zeros(N_FEAT)

    def push(self, x):
        if self.ST[S_T] >= CAP - 1:  # buffer full: keep the last features rather than write out of bounds
            return self.out
        step(float(x), self.P, self.ref, self.C, self.ST, self.out)
        return self.out
