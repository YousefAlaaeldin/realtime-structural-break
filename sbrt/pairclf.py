"""Learned two-segment classifier run as a scan (HANDOFF section 38).

Question asked of a classifier: given a series' history (the known break-free reference) and a WINDOW of L
consecutive online points, is the WHOLE window post-break? One series gives many labelled (history, window) pairs
instead of one effective label pattern per series (the 2025 offline-edition task, solved there with segment
comparisons + gradient boosting). The classifier is then run on suffix windows at every online step (the scan) and
on non-overlapping blocks whose logits are accumulated over candidate break times.

1. Per online point (causal; the history fit is the one bocpd.py uses): z (history-standardised value), e (history
   AR(10) residual / se, clipped +-10), w (e whitened by the history's own residual ECDF, as bocpd.py), wu (the same
   for e / frozen volatility forecast, as bocpd.py), u (rank of z in the history's ECDF). From these, NCH = 20
   "channels" (CH_NAMES: w, w^2, |w|, lag products, volatility clustering, tail flags, cubes, centre mass, sign
   persistence, wu, wu^2, z, z^2, z z_1, roughness (dz)^2, |dz|, u, dz dz_1) and 2 x 8 equal-probability bin
   indicators (w bins at normal quantiles, u bins at k/8). Prefix sums of all 36 make every window statistic O(1).
2. Classifier inputs for a window of n points (NIN = 33, INPUT_NAMES; NO online position, NO history descriptors):
     t_<ch>    (window sum - n * history mean) / sqrt(n * history long-run variance)  (batch means, features._lrv)
     te_<ch>   the same against the mean of the LAST 256 history points (HANDOFF 15.2: no-break online variance
               drifts, and the drift is predictable from the end of the history) for w^2, |w|, tails, centre, wu^2,
               roughness
     w_/u_ chi2, KS, AD on the 8 bins (window frequencies vs the history's 1/8), and log2 n.
   History descriptors (length, skew, kurtosis, rho1, ...) were tried and REMOVED: they are constant per series, and
   inside a cell positives and negatives mostly come from different series, so the trees used them to memorise series
   (top gain shares) and held-out pair AUC fell from 0.619 to 0.595 (train 3,000 / held-out 1,000 series). The
   downstream trees have the descriptors anyway (ctx_ columns).
3. Pairs (`series_pairs`): full windows only, lengths L in LS = 16..256 ending at online step t (t + 1 >= L).
   Positive: window entirely at/after tau. Negative: window ends before tau, or no-break series. Straddling
   windows are dropped. Partial windows (t + 1 < L) are never used: a window starting at online step 0 is fully
   post-break only when tau = 0, so they would teach "short window = negative", i.e. the online position.
   Up to `k` windows per class per series per L, drawn uniformly over the admissible end steps.
   Weights (`pair_weights`): cells = (L, end-step bucket END_EDGES). Positives end late, pre-break negatives early,
   and no-break online variance drifts with the distance from the history: without matching, the classifier would
   learn "far from the history and drifted = break". Inside a cell every series' rows of a class sum to 1, then
   both classes get the same total P N / (P + N) (P, N = series counts), so the end step cannot predict the label.
   Every L gets the same total weight.
4. Model: LightGBM binary (deterministic, use_missing=False), exported to flat arrays (`export_model`) and evaluated
   in numba (`_score`; equal to booster.predict(raw_score=True), tested). Logits clipped to +-CLIP. With balanced
   classes the logit estimates log p(window | post) / p(window | pre). Small on purpose (15 leaves, 200 rounds): on
   the same 3,000 / 1,000 split, 31 leaves 0.619, 63 leaves 0.613, 600 rounds 0.611, 15 leaves 0.624, 7 leaves
   0.626, k = 12 windows per class 0.615 (k = 6 kept): more capacity only memorises series.
5. Scan -> FEATURE_NAMES (11 columns, `pc_`) at online step t, n = t + 1 points so far:
     pc_s<L>       suffix-window logit for each L (0 while n < L)
     pc_smax       max over the available L (0 while n < 16)
     non-overlapping blocks of LB = 32 points (blocks 0..B-1 complete, B = n // LB), logits l_b treated as
     log-likelihood ratios; candidate break in block j: S_j = sum_{b >= j} l_b, prior mass pi_j from the generator's
     hazard (evidence.HAZARD via bocpd.LP), the incomplete tail gets its prior mass with ratio 1:
     pc_logodds    log posterior odds of "break by t";  pc_logodds_h  the same with the logits halved
     pc_cusum      max(0, max_j S_j);  pc_tsb  posterior mean log(1 + time since the break)
     pc_recent     mean logit of the last (up to) 4 blocks
   Block logits are computed at every completed block, rows only where wanted (t % sub == 0), so a batch run on
   cv.py's kept rows equals streaming exactly (tested). Inference cost about 5 classifier calls per point.

Self-test (about a minute):  python -m sbrt.pairclf test
"""
import math
import os
import sys
import time

import numpy as np
from numba import njit

from sbrt import bocpd as BO
from sbrt import evidence as EV
from sbrt import features as F
from sbrt import volstate as VS

VERSION = 1   # bump when the statistics, the pair sampler or the scan change (part of the cache names)
AR = F.AR
CAP = F.CAP
LS = np.array([16, 32, 64, 128, 256], dtype=np.int64)
NL = len(LS)
LB = 32
NEND = 256
CLIP = 8.0
CH_NAMES = ["w", "w2", "aw", "ww1", "ww2", "vclust", "tail2", "tail3", "w3", "centre", "sgn", "wu", "wu2",
            "z", "z2", "zz1", "dz2", "adz", "u", "dzdz1"]
NCH = len(CH_NAMES)
NBIN = 8
NC = NCH + 2 * NBIN
END_CH = np.array([1, 2, 6, 9, 12, 16], dtype=np.int64)
NEC = len(END_CH)
INPUT_NAMES = ([f"t_{c}" for c in CH_NAMES] + [f"te_{CH_NAMES[g]}" for g in END_CH]
               + ["w_chi2", "w_ks", "w_ad", "u_chi2", "u_ks", "u_ad", "log2n"])
NIN = len(INPUT_NAMES)
FEATURE_NAMES = [f"pc_s{L}" for L in LS] + ["pc_smax", "pc_logodds", "pc_logodds_h", "pc_cusum", "pc_tsb",
                                             "pc_recent"]
N_PC = len(FEATURE_NAMES)
W_EDGES = np.array([EV.ndtri(k / NBIN) for k in range(1, NBIN)])
END_EDGES = np.array([0, 50, 100, 150, 200, 300, 400, 550, 700, 100000], dtype=np.int64)
LGB_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=15, min_data_in_leaf=200,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, seed=0,
                  deterministic=True, force_row_wise=True, use_missing=False, verbose=-1)
ROUNDS = 200
LEAVES = 15
CLF_THREADS = 6   # fixed LightGBM thread count: results (and the cv.py caches) do not depend on --threads
K_PER = 6
LP = BO.LP

# parameter vector Q and state ST layouts
Q_MU, Q_SD, Q_SE, Q_L, Q_P, Q_AR = 0, 1, 2, 3, 4, 5
Q_LEN = Q_AR + AR
S_T, S_D, S_W1, S_W2, S_DZ1, S_LAG = 0, 1, 2, 3, 4, 5
S_LEN = S_LAG + AR


# ---------------------------------------------------------------------------------------------
@njit(cache=True, nogil=True)
def _clip(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)


@njit(cache=True, nogil=True)
def _channels(w, w1, w2, wu, z, z1, dz, dz1, u, g):
    ww = w * w
    g[0] = w
    g[1] = min(ww, 25.0) - 1.0
    g[2] = abs(w) - 0.7978845608028654
    g[3] = w * w1
    g[4] = w * w2
    g[5] = (min(ww, 9.0) - 1.0) * (min(w1 * w1, 9.0) - 1.0)
    g[6] = 1.0 if abs(w) > 2.0 else 0.0
    g[7] = 1.0 if abs(w) > 3.0 else 0.0
    c = _clip(w, -3.0, 3.0)
    g[8] = c * c * c
    g[9] = 1.0 if abs(w) < 0.5 else 0.0
    g[10] = (1.0 if w > 0 else (-1.0 if w < 0 else 0.0)) * (1.0 if w1 > 0 else (-1.0 if w1 < 0 else 0.0))
    g[11] = wu
    g[12] = min(wu * wu, 25.0) - 1.0
    g[13] = _clip(z, -10.0, 10.0)
    g[14] = min(z * z, 25.0)
    g[15] = _clip(z * z1, -25.0, 25.0)
    g[16] = min(dz * dz, 25.0)
    g[17] = min(abs(dz), 5.0)
    g[18] = u - 0.5
    g[19] = _clip(dz * dz1, -25.0, 25.0)


@njit(cache=True, nogil=True)
def _bins(w, u, wedges, bw):
    b = 0
    for q in range(len(wedges)):
        if w > wedges[q]:
            b += 1
    bu = int(u * NBIN)
    if bu > NBIN - 1:
        bu = NBIN - 1
    bw[0] = b
    bw[1] = bu


@njit(cache=True, nogil=True)
def _rank_u(z, zref):
    return (np.searchsorted(zref, z) + 0.5) / (len(zref) + 1.0)


@njit(cache=True, nogil=True)
def _hist_channels(z, eh, s2h, eref, uref, zref, out):
    """Channels of the history itself (in-sample residuals), rows for z positions AR + 2 .. n - 1."""
    n = len(z)
    g = np.zeros(NCH)
    for i in range(AR + 2, n):
        k = i - AR
        w = _clip(EV._whiten(eh[k], eref), -8.0, 8.0)
        w1 = _clip(EV._whiten(eh[k - 1], eref), -8.0, 8.0)
        w2 = _clip(EV._whiten(eh[k - 2], eref), -8.0, 8.0)
        wu = _clip(EV._whiten(eh[k] / math.sqrt(s2h[k]), uref), -8.0, 8.0)
        _channels(w, w1, w2, wu, z[i], z[i - 1], z[i] - z[i - 1], z[i - 1] - z[i - 2], _rank_u(z[i], zref), g)
        for c in range(NCH):
            out[i - AR - 2, c] = g[c]


def make_pc_params(hist):
    """Per-series constants and a fresh state: Q, G (3, NCH: history mean, 1/sqrt(long-run var), end-of-history
    mean), eref, uref, zref, wedges, ST, C (prefix sums), blk (block logits)."""
    H = VS.fit_history(np.asarray(hist, dtype=np.float64))
    z, eh, s2h = H["z"], H["eh"], H["s2h"]
    eref = np.sort(eh)
    uref = np.sort(eh / np.sqrt(s2h))
    zref = np.sort(z)
    G_h = np.zeros((len(z) - AR - 2, NCH))
    _hist_channels(z, eh, s2h, eref, uref, zref, G_h)
    G = np.zeros((3, NCH))
    for c in range(NCH):
        x = G_h[:, c]
        m = float(x.mean())
        G[0, c] = m
        G[1, c] = 1.0 / math.sqrt(F._lrv(x - m, float(x.var()), 1e-6))
        G[2, c] = float(x[-NEND:].mean())
    P = H["P"]
    Q = np.zeros(Q_LEN)
    Q[Q_MU], Q[Q_SD], Q[Q_SE], Q[Q_L], Q[Q_P] = P[F.P_MU], P[F.P_SD], H["se"], H["L"], H["p"]
    Q[Q_AR:] = H["coef"]
    ST = np.zeros(S_LEN)
    ST[S_D] = H["s2T1"] - H["L"]
    ST[S_W1] = _clip(EV._whiten(eh[-1], eref), -8.0, 8.0)
    ST[S_W2] = _clip(EV._whiten(eh[-2], eref), -8.0, 8.0)
    ST[S_DZ1] = z[-1] - z[-2]
    for i in range(AR):
        ST[S_LAG + i] = z[-1 - i]
    return Q, G, eref, uref, zref, W_EDGES.copy(), ST, np.zeros((NC, CAP + 1)), np.zeros(CAP // LB + 2)


@njit(cache=True, nogil=True)
def _push(x, Q, eref, uref, zref, wedges, ST, C):
    """Consume one online point: channels into the prefix sums C[:, t + 1]."""
    t = int(ST[S_T])
    z = (x - Q[Q_MU]) / Q[Q_SD]
    pred = 0.0
    for i in range(AR):
        pred += Q[Q_AR + i] * ST[S_LAG + i]
    e = _clip((z - pred) / Q[Q_SE], -10.0, 10.0)
    uv = e / math.sqrt(Q[Q_L] + ST[S_D])
    w = _clip(EV._whiten(e, eref), -8.0, 8.0)
    wu = _clip(EV._whiten(uv, uref), -8.0, 8.0)
    z1 = ST[S_LAG]
    dz = z - z1
    u = _rank_u(z, zref)
    g = np.empty(NCH)
    _channels(w, ST[S_W1], ST[S_W2], wu, z, z1, dz, ST[S_DZ1], u, g)
    bw = np.empty(2, dtype=np.int64)
    _bins(w, u, wedges, bw)
    n = t + 1
    for k in range(NCH):
        C[k, n] = C[k, t] + g[k]
    for k in range(2 * NBIN):
        C[NCH + k, n] = C[NCH + k, t]
    C[NCH + bw[0], n] += 1.0
    C[NCH + NBIN + bw[1], n] += 1.0
    ST[S_D] = ST[S_D] * Q[Q_P]
    for i in range(AR - 1, 0, -1):
        ST[S_LAG + i] = ST[S_LAG + i - 1]
    ST[S_LAG] = z
    ST[S_W2] = ST[S_W1]
    ST[S_W1] = w
    ST[S_DZ1] = dz
    ST[S_T] = n * 1.0


@njit(cache=True, nogil=True)
def _window_inputs(C, a, b, G, x):
    """Classifier inputs of the window of online points a .. b - 1 (prefix indices)."""
    n = b - a
    sq = math.sqrt(n)
    for g in range(NCH):
        s = C[g, b] - C[g, a]
        x[g] = (s - n * G[0, g]) * G[1, g] / sq
    for i in range(NEC):
        g = END_CH[i]
        s = C[g, b] - C[g, a]
        x[NCH + i] = (s - n * G[2, g]) * G[1, g] / sq
    o = NCH + NEC
    for blk in range(2):
        base = NCH + blk * NBIN
        chi = 0.0
        ks = 0.0
        ad = 0.0
        cum = 0.0
        for k in range(NBIN):
            f = (C[base + k, b] - C[base + k, a]) / n
            d = f - 1.0 / NBIN
            chi += d * d * NBIN
            if k < NBIN - 1:
                cum += f
                p = (k + 1.0) / NBIN
                dd = cum - p
                if abs(dd) > ks:
                    ks = abs(dd)
                ad += dd * dd / (p * (1.0 - p))
        x[o] = n * chi
        x[o + 1] = sq * ks
        x[o + 2] = n * ad / (NBIN - 1)
        o += 3
    x[o] = math.log(n) / math.log(2.0)


@njit(cache=True, nogil=True)
def _score(x, roots, feat, thr, left, right, val):
    s = 0.0
    for r in range(len(roots)):
        nd = roots[r]
        while feat[nd] >= 0:
            if x[feat[nd]] <= thr[nd]:
                nd = left[nd]
            else:
                nd = right[nd]
        s += val[nd]
    return s


@njit(cache=True, nogil=True)
def _row(C, n, G, blk, LP, roots, feat, thr, left, right, val, xb, out):
    smax = -1e300
    for l in range(NL):
        L = LS[l]
        if n >= L:
            _window_inputs(C, n - L, n, G, xb)
            s = _clip(_score(xb, roots, feat, thr, left, right, val), -CLIP, CLIP)
            out[l] = s
            if s > smax:
                smax = s
        else:
            out[l] = 0.0
    out[NL] = smax if smax > -1e299 else 0.0
    B = n // LB
    rest = 0.0
    if n > B * LB:
        rest = math.exp(LP[B * LB]) - math.exp(LP[n])
    lrest = math.log(rest) if rest > 1e-300 else -1e300
    for h in range(2):
        c = 1.0 if h == 0 else 0.5
        mx = lrest
        S = 0.0
        for j in range(B - 1, -1, -1):
            S += blk[j]
            a = LP[j * LB] + math.log(-math.expm1(LP[(j + 1) * LB] - LP[j * LB])) + c * S
            if a > mx:
                mx = a
        tot = math.exp(lrest - mx) if lrest > -1e299 else 0.0
        tsb = tot * math.log(1.0 + 0.5 * (n - B * LB))
        cus = 0.0
        S = 0.0
        for j in range(B - 1, -1, -1):
            S += blk[j]
            if S > cus:
                cus = S
            a = LP[j * LB] + math.log(-math.expm1(LP[(j + 1) * LB] - LP[j * LB])) + c * S
            wgt = math.exp(a - mx)
            tot += wgt
            tsb += wgt * math.log(1.0 + n - (j + 0.5) * LB)
        out[NL + 1 + h] = mx + math.log(tot) - LP[n]
        if h == 0:
            out[NL + 3] = cus
            out[NL + 4] = tsb / tot
    k = B - 4 if B > 4 else 0
    m = 0.0
    for j in range(k, B):
        m += blk[j]
    out[NL + 5] = m / (B - k) if B > k else 0.0


@njit(cache=True, nogil=True)
def pc_step(x, Q, G, eref, uref, zref, wedges, ST, C, blk, LP, roots, feat, thr, left, right, val, xb, out, do_row):
    _push(x, Q, eref, uref, zref, wedges, ST, C)
    n = int(ST[S_T])
    if n % LB == 0:
        _window_inputs(C, n - LB, n, G, xb)
        blk[n // LB - 1] = _clip(_score(xb, roots, feat, thr, left, right, val), -CLIP, CLIP)
    if do_row:
        _row(C, n, G, blk, LP, roots, feat, thr, left, right, val, xb, out)


@njit(cache=True, nogil=True)
def _pc_run(online, Q, G, eref, uref, zref, wedges, ST, C, blk, LP, roots, feat, thr, left, right, val, sub, out):
    xb = np.zeros(NIN)
    row = np.zeros(N_PC)
    r = 0
    for i in range(len(online)):
        keep = i % sub == 0
        pc_step(online[i], Q, G, eref, uref, zref, wedges, ST, C, blk, LP, roots, feat, thr, left, right, val, xb,
                row, keep)
        if keep:
            for c in range(N_PC):
                out[r, c] = row[c]
            r += 1


@njit(cache=True, nogil=True)
def _prefix_run(online, Q, eref, uref, zref, wedges, ST, C):
    for i in range(len(online)):
        _push(online[i], Q, eref, uref, zref, wedges, ST, C)


# ---------------------------------------------------------------------------------------------
def export_model(booster):
    """LightGBM booster -> flat numba arrays (roots, feat, thr, left, right, val). Leaves have feat = -1."""
    dm = booster.dump_model()
    feat, thr, left, right, val, roots = [], [], [], [], [], []

    def add(nd):
        i = len(feat)
        feat.append(-1)
        thr.append(0.0)
        left.append(-1)
        right.append(-1)
        val.append(0.0)
        if "leaf_value" in nd:
            val[i] = float(nd["leaf_value"])
            return i
        if nd["decision_type"] != "<=" or nd["missing_type"] not in ("None",):
            raise ValueError(f"unsupported split {nd['decision_type']} / missing {nd['missing_type']}")
        feat[i] = int(nd["split_feature"])
        thr[i] = float(nd["threshold"])
        left[i] = add(nd["left_child"])
        right[i] = add(nd["right_child"])
        return i

    for t in dm["tree_info"]:
        roots.append(add(t["tree_structure"]))
    return (np.array(roots, np.int64), np.array(feat, np.int64), np.array(thr, np.float64),
            np.array(left, np.int64), np.array(right, np.int64), np.array(val, np.float64))


def run_series_pc(hist, online, M, sub=1):
    """Batch rows (online steps 0, sub, 2 sub, ...) x N_PC for one series -- identical to streaming."""
    Q, G, eref, uref, zref, wedges, ST, C, blk = make_pc_params(hist)
    online = np.asarray(online, dtype=np.float64)
    if len(online) >= CAP:
        raise ValueError(f"online segment longer than the supported {CAP - 1} points")
    out = np.zeros(((len(online) + sub - 1) // sub, N_PC))
    _pc_run(online, Q, G, eref, uref, zref, wedges, ST, C, blk, LP, *M, sub, out)
    return out


class PcStream:
    """One series' streaming state for use inside infer()."""

    def __init__(self, hist, M):
        self.Q, self.G, self.eref, self.uref, self.zref, self.wedges, self.ST, self.C, self.blk = make_pc_params(hist)
        self.M = M
        self.xb = np.zeros(NIN)
        self.out = np.zeros(N_PC)

    def push(self, x):
        if self.ST[S_T] >= CAP - 1:
            return self.out
        pc_step(float(x), self.Q, self.G, self.eref, self.uref, self.zref, self.wedges, self.ST, self.C, self.blk, LP,
                *self.M, self.xb, self.out, True)
        return self.out


# ---------------------------------------------------------------------------------------------
def series_pairs(hist, online, tau, k=K_PER, seed=(0,)):
    """Training pairs of one series: X (rows, NIN), y, L index, end step t. Deterministic for a given seed key."""
    Q, G, eref, uref, zref, wedges, ST, C, _ = make_pc_params(hist)
    online = np.asarray(online, dtype=np.float64)
    N = len(online)
    _prefix_run(online, Q, eref, uref, zref, wedges, ST, C)
    rng = np.random.default_rng(list(seed))
    rows, ys, ls, ts = [], [], [], []
    xb = np.zeros(NIN)
    for li, L in enumerate(LS):
        b_all = np.arange(L, N + 1)                       # prefix end index b = t + 1, window [b - L, b)
        if len(b_all) == 0:
            continue
        if tau < 0:
            groups = ((0, b_all),)
        else:
            groups = ((0, b_all[b_all <= tau]), (1, b_all[b_all - L >= tau]))
        for lab, cand in groups:
            if len(cand) == 0:
                continue
            pick = np.sort(rng.choice(cand, size=min(k, len(cand)), replace=False))
            for b in pick:
                _window_inputs(C, int(b) - int(L), int(b), G, xb)
                rows.append(xb.copy())
                ys.append(lab)
                ls.append(li)
                ts.append(int(b) - 1)
    if not rows:
        return np.zeros((0, NIN)), np.zeros(0, np.int8), np.zeros(0, np.int64), np.zeros(0, np.int64)
    return np.array(rows), np.array(ys, np.int8), np.array(ls, np.int64), np.array(ts, np.int64)


def build_pairs(items, taus, ids, k=K_PER, seed=0, threads=8):
    """Pairs of many series (items = (hist, online)); ids are the series' global indices (seed key and group id)."""
    from joblib import Parallel, delayed
    res = Parallel(n_jobs=threads, backend="threading")(
        delayed(series_pairs)(h, o, int(t), k, (seed, int(i))) for (h, o), t, i in zip(items, taus, ids))
    cnt = np.array([len(r[1]) for r in res])
    return dict(X=np.concatenate([r[0] for r in res]), y=np.concatenate([r[1] for r in res]),
                li=np.concatenate([r[2] for r in res]), tend=np.concatenate([r[3] for r in res]),
                sid=np.repeat(np.asarray(ids, np.int64), cnt))


def pair_weights(y, li, tend, sid):
    """Cells (L, end-step bucket): each series' rows of a class sum to 1, both classes get P N / (P + N) (P, N =
    series counts in the cell), every L the same total. Rows in one-class cells get weight 0."""
    y = np.asarray(y).astype(np.int64)
    bk = np.searchsorted(END_EDGES, tend, side="right") - 1
    cell = li * len(END_EDGES) + bk
    ncell = NL * len(END_EDGES)
    _, inv, cnt = np.unique(np.stack([cell, y, sid]), axis=1, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    w = 1.0 / cnt[inv]                                          # per (cell, class, series): sums to 1
    ser = np.zeros(len(y))
    ser[np.unique(inv, return_index=True)[1]] = 1.0             # one count per (cell, class, series)
    P = np.bincount(cell, weights=ser * (y == 1), minlength=ncell)
    N = np.bincount(cell, weights=ser * (y == 0), minlength=ncell)
    tgt = np.where((P > 0) & (N > 0), P * N / np.maximum(P + N, 1e-12), 0.0)
    cls_tot = np.where(y == 1, P[cell], N[cell])
    w = w * tgt[cell] / np.maximum(cls_tot, 1e-12)
    Lt = np.bincount(li, weights=w, minlength=NL)
    w = w / np.maximum(Lt[li], 1e-12)
    return w / max(w[w > 0].mean(), 1e-12)


def train_classifier(pairs, rows=None, rounds=ROUNDS, leaves=LEAVES, threads=CLF_THREADS, seed=0):
    import lightgbm as lgb
    sel = np.arange(len(pairs["y"])) if rows is None else rows
    w = pair_weights(pairs["y"][sel], pairs["li"][sel], pairs["tend"][sel], pairs["sid"][sel])
    ok = w > 0
    p = dict(LGB_PARAMS, num_leaves=leaves, num_threads=threads, seed=seed)
    return lgb.train(p, lgb.Dataset(pairs["X"][sel][ok], pairs["y"][sel][ok], weight=w[ok],
                                    free_raw_data=True), rounds)


def score_pairs(M, X):
    out = np.empty(len(X))
    _score_many(X, *M, out)
    return out


@njit(cache=True, nogil=True)
def _score_many(X, roots, feat, thr, left, right, val, out):
    for i in range(len(X)):
        out[i] = _score(X[i], roots, feat, thr, left, right, val)


def pair_auc_by_L(M, pairs, rows):
    """Held-out pair AUC per window length (weighted like training)."""
    from sklearn.metrics import roc_auc_score
    s = score_pairs(M, pairs["X"][rows])
    y, li = pairs["y"][rows], pairs["li"][rows]
    w = pair_weights(y, li, pairs["tend"][rows], pairs["sid"][rows])
    out = []
    for l in range(NL):
        m = (li == l) & (w > 0)
        out.append(roc_auc_score(y[m], s[m], sample_weight=w[m]) if len(np.unique(y[m])) == 2 else float("nan"))
    return out


def features_parallel(items, M, sub=3, threads=8):
    from joblib import Parallel, delayed
    return Parallel(n_jobs=threads, backend="threading")(delayed(run_series_pc)(h, o, M, sub) for h, o in items)


def fold_features(S, tau, tr_s, extra, k=K_PER, rounds=ROUNDS, leaves=LEAVES, inner=4, fold=0, sub=3, threads=8,
                  say=print, pairs=None):
    """One CV fold. Classifiers are trained on the fold's ORIGINAL training series only.
    Training originals: cross-fitted (inner StratifiedKFold over tr_s; a series is scored by the inner model that
    did not see it). Validation originals: the model trained on all of tr_s. Extra series (re-cuts etc.):
    extra = list of (items, parent array); a variant whose parent is a training series is scored by its parent's
    inner model, a parent < 0 (never in classifier training) by the full model, a validation parent -> zeros (those
    rows never train in this fold). Returns ([orig rows] + [rows per extra source], info dict)."""
    from sklearn.model_selection import StratifiedKFold
    N = len(S)
    if pairs is None:
        pairs = build_pairs([(s[1], s[2]) for s in S], tau, np.arange(N), k, 0, threads)
    t0 = time.time()
    tr_s = np.asarray(tr_s)
    grp = np.full(N, -1, np.int64)                 # inner group of a training original; -1 = validation
    iskf = StratifiedKFold(inner, shuffle=True, random_state=100 + fold)
    for g, (_, hv) in enumerate(iskf.split(tr_s, (tau[tr_s] >= 0).astype(int))):
        grp[tr_s[hv]] = g
    models, aucs = [], []
    for g in range(inner + 1):                     # g == inner: the full model
        tr_g = tr_s[grp[tr_s] != g] if g < inner else tr_s
        rows = np.flatnonzero(np.isin(pairs["sid"], tr_g))
        b = train_classifier(pairs, rows, rounds, leaves, CLF_THREADS, seed=fold)
        M = export_model(b)
        models.append(M)
        if g < inner:
            ho = np.flatnonzero(np.isin(pairs["sid"], tr_s[grp[tr_s] == g]))
            aucs.append(pair_auc_by_L(M, pairs, ho))
    t1 = time.time()
    a = np.nanmean(np.array(aucs), axis=0)
    say(f"pair classifiers ({inner} inner + 1 full, {rounds} rounds) trained in {t1 - t0:.0f}s; inner held-out pair "
        f"AUC by L " + " ".join(f"{L}:{v:.3f}" for L, v in zip(LS, a)))
    from joblib import Parallel, delayed

    def rows_of(o):
        return (len(o) + sub - 1) // sub

    def featurize(jobs):
        # jobs: list of (hist, online, model index or -1)
        res = Parallel(n_jobs=threads, backend="threading")(
            delayed(lambda h, o, m: run_series_pc(h, o, models[m], sub) if m >= 0 else np.zeros((rows_of(o), N_PC)))
            (h, o, m) for h, o, m in jobs)
        return np.concatenate(res).astype(np.float32) if res else np.zeros((0, N_PC), np.float32)

    out = [featurize([(s[1], s[2], int(grp[i]) if grp[i] >= 0 else inner) for i, s in enumerate(S)])]
    for items, parent in extra:
        mi = [inner if p < 0 else (int(grp[p]) if grp[p] >= 0 else -1) for p in parent]
        out.append(featurize([(h, o, m) for (h, o), m in zip(items, mi)]))
    say(f"pc features for {N + sum(len(e[0]) for e in extra):,} series in {time.time() - t1:.0f}s")
    return out, dict(pair_auc=a, n_pairs=len(pairs["y"]))


# ---------------------------------------------------------------------------------------------
def _selftest(n_train=800, n_hold=400, threads=8):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from sbrt.data import load_train

    S, tau = load_train()
    t0 = time.time()
    tr = np.arange(n_train)
    pairs = build_pairs([(S[i][1], S[i][2]) for i in tr], tau[tr], tr, K_PER, 0, threads)
    ho = np.arange(n_train, n_train + n_hold)
    ph = build_pairs([(S[i][1], S[i][2]) for i in ho], tau[ho], ho, K_PER, 0, threads)
    print(f"pairs: {len(pairs['y']):,} training rows from {n_train} series ({time.time() - t0:.0f}s), "
          f"positive share {pairs['y'].mean():.3f}; {len(ph['y']):,} held-out rows")
    p2 = build_pairs([(S[i][1], S[i][2]) for i in tr[:50]], tau[tr[:50]], tr[:50], K_PER, 0, threads)
    assert all(np.array_equal(p2[key], pairs[key][:len(p2["y"])]) for key in p2), "pair sampling not deterministic"
    w = pair_weights(pairs["y"], pairs["li"], pairs["tend"], pairs["sid"])
    bk = np.searchsorted(END_EDGES, pairs["tend"], side="right") - 1
    for l in range(NL):
        for c in range(len(END_EDGES)):
            m = (pairs["li"] == l) & (bk == c)
            if m.any():
                a, b = w[m & (pairs["y"] == 1)].sum(), w[m & (pairs["y"] == 0)].sum()
                assert abs(a - b) <= 1e-9 * max(a + b, 1.0), "classes not balanced inside a cell"
    print("pair sampling deterministic; classes balanced in every (L, end-step bucket) cell: OK")
    t0 = time.time()
    b = train_classifier(pairs, rounds=ROUNDS)
    M = export_model(b)
    print(f"classifier: {ROUNDS} rounds in {time.time() - t0:.1f}s, {len(M[1]):,} nodes")
    Xh = ph["X"][:3000]
    ref = b.predict(Xh, raw_score=True)
    mine = score_pairs(M, Xh)
    print(f"numba trees vs LightGBM predict(raw_score): max |diff| {np.abs(ref - mine).max():.2e}")
    assert np.abs(ref - mine).max() < 1e-9, "tree export does not reproduce LightGBM"
    print("held-out pair AUC by L: " + " ".join(f"{L}:{v:.3f}" for L, v in zip(LS, pair_auc_by_L(M, ph, np.arange(len(ph['y']))))))
    imp = b.feature_importance("gain")
    imp = imp / imp.sum()
    print("top inputs by gain: " + ", ".join(f"{INPUT_NAMES[j]} {100 * imp[j]:.1f}%" for j in np.argsort(-imp)[:12]))

    rng = np.random.default_rng(0)
    for i in rng.choice(ho, 20, replace=False):
        _, h, o = S[i]
        Bf = run_series_pc(h, o, M)
        assert np.array_equal(Bf, run_series_pc(h, o, M)), "two runs differ"
        st = PcStream(h, M)
        Sm = np.array([st.push(x).copy() for x in o])
        assert np.array_equal(Bf, Sm), f"streaming != batch (max diff {np.abs(Bf - Sm).max()})"
        assert np.array_equal(run_series_pc(h, o, M, 3), Sm[::3]), "sub-sampled batch != streaming"
        assert np.isfinite(Bf).all(), "non-finite feature"
        if len(o) > 80:
            o2 = o.copy()
            o2[70:] = rng.standard_normal(len(o) - 70) * 3 * h.std()
            B2 = run_series_pc(h, o2, M)
            assert np.array_equal(B2[:70], Bf[:70]), "future points changed earlier features"
            assert not np.array_equal(B2[70:], Bf[70:])
    print("streaming == batch (all rows and every 3rd; block boundaries off the kept grid), causal, finite, "
          "deterministic: OK")

    h = S[ho[0]][1]
    o = np.r_[rng.standard_normal(300) * h.std() + h.mean(), rng.standard_normal(200) * 2.0 * h.std() + h.mean()]
    Bs = run_series_pc(h, o, M)
    j = FEATURE_NAMES.index
    print(f"sanity, variance x4 after step 300: pc_logodds at 290 {Bs[290, j('pc_logodds')]:+.1f}, 330 "
          f"{Bs[330, j('pc_logodds')]:+.1f}, 499 {Bs[499, j('pc_logodds')]:+.1f}; pc_s64 at 290 {Bs[290, j('pc_s64')]:+.2f}"
          f", 400 {Bs[400, j('pc_s64')]:+.2f}; posterior time since break at 499 {np.expm1(Bs[499, j('pc_tsb')]):.0f} "
          f"(true 200)")

    items = [(S[i][1], S[i][2]) for i in ho[:200]]
    npts = sum(len(q[1]) for q in items)
    t0 = time.time()
    for hh, oo in items:
        make_pc_params(hh)
    t_setup = time.time() - t0
    t0 = time.time()
    for hh, oo in items:
        st = PcStream(hh, M)
        for x in oo:
            st.push(x)
    t_stream = time.time() - t0 - t_setup
    t0 = time.time()
    for hh, oo in items:
        run_series_pc(hh, oo, M, 3)
    t_b3 = time.time() - t0 - t_setup
    print(f"timing (single thread, {len(items)} series, {npts:,} points): history setup {1e3 * t_setup / len(items):.1f} "
          f"ms per series; streaming push {1e6 * t_stream / npts:.1f} us per point (incl. Python call overhead); "
          f"batch at every 3rd row {1e6 * t_b3 / npts:.1f} us per point")
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "test"
    if cmd != "test":
        sys.exit("usage: python -m sbrt.pairclf test")
    _selftest()
