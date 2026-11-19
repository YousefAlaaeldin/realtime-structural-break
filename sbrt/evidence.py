"""Likelihood-ratio evidence accumulation and Bayesian model averaging across break types (streaming, causal).

Separate from sbrt/features.py on purpose: the 270-feature kernel, its caches, the GRU input hash and main.py stay
unchanged. Columns are appended only when cv.py is run with --ev.

1. Whitening with a per-series null. Each point is turned into the history-fitted AR(10) residual e_t (same model
   as the main kernel), then mapped through that series' OWN history residual distribution (empirical CDF) and the
   standard normal quantile: w_t = Phi^-1(ECDF_hist(e_t)). Before a break, w_t is ~ N(0, 1) for every series, also
   heavy-tailed or skewed ones, so tail/variance alternatives do not fire just because a series is non-Gaussian.
   Points beyond the history's range continue linearly past the last quantile.
2. Alternatives (post-break densities of w_t), 14 in 4 groups:
       variance    N(0, s^2)                         s in 0.6, 0.8, 1.25, 1.6
       mean        N(m, 1)                           m in -0.6, -0.3, 0.3, 0.6
       dependence  N(r w_{t-1}, 1 - r^2)             r in -0.4, -0.2, 0.2, 0.4
       tails       Student-t, unit variance          nu in 3, 6
   l_k(t) = log f_k(w_t | w_{t-1}) - log phi(w_t)    (log-likelihood ratio of one point)
3. Evidence per alternative, all O(1) recursions per point:
       CUSUM              W_t = max(0, W_{t-1} + l_t)                       (Page; = max over start of the LR sum)
       Shiryaev-Roberts   log R_t = log(1 + R_{t-1}) + l_t                   (sum over start of the LR product)
       Shiryaev posterior log O_t = log(O_{t-1} + h_t) - log(1 - h_t) + l_t   O = P(break by t | data) / P(not)
   h_t is the prior hazard P(break at t | none before, series still running), computed from the generator's
   description only (50% of series break, break uniform over the online segment, online length uniform 10-999):
   no labels are used.
4. Bayesian model averaging. With prior weight w_k on alternative k the exact posterior odds of "break by t" are
   sum_k w_k O_t^k (the recursions are linear in the joint terms). With a uniform prior:
       ev_mix_logodds  = log mean_k O_t^k
       ev_logbf        = ev_mix_logodds - log prior odds_t      (Bayes factor: evidence beyond the prior)
       ev_post_<group> = posterior share of each break-type group
   These 6 are the COMPACT set (cv.py --ev compact). FULL adds the 3 x 14 per-alternative statistics (--ev full),
   from which the trees can learn their own weighting of break types (an empirical-Bayes step done per fold).

Self-test:  python -m sbrt.evidence
"""
import math

import numpy as np
from numba import njit

from sbrt import features as F

AR = F.AR
GROUPS = ("variance", "mean", "dependence", "tails")
ALTS = ([("variance", 0, s) for s in (0.6, 0.8, 1.25, 1.6)]
        + [("mean", 1, m) for m in (-0.6, -0.3, 0.3, 0.6)]
        + [("dependence", 2, r) for r in (-0.4, -0.2, 0.2, 0.4)]
        + [("tails", 3, nu) for nu in (3.0, 6.0)])
NA = len(ALTS)
KIND = np.array([a[1] for a in ALTS], dtype=np.int64)
PAR = np.array([a[2] for a in ALTS], dtype=np.float64)
GRP = np.array([GROUPS.index(a[0]) for a in ALTS], dtype=np.int64)


def _const(kind, p):
    if kind == 0:
        return -math.log(p)
    if kind == 2:
        return -0.5 * math.log(1.0 - p * p)
    if kind == 3:  # unit-variance Student-t: log normaliser minus the N(0,1) one
        return (math.lgamma((p + 1) / 2) - math.lgamma(p / 2) - 0.5 * math.log(math.pi * (p - 2))
                + 0.5 * math.log(2 * math.pi))
    return 0.0


CONST = np.array([_const(a[1], a[2]) for a in ALTS])


def _alt_name(a):
    return f"{a[0]}_{a[2]:g}".replace("-", "m")


COMPACT = ["ev_mix_logodds", "ev_logbf"] + [f"ev_post_{g}" for g in GROUPS]
FEATURE_NAMES = (COMPACT + [f"ev_cusum_{_alt_name(a)}" for a in ALTS] + [f"ev_logsr_{_alt_name(a)}" for a in ALTS]
                 + [f"ev_logodds_{_alt_name(a)}" for a in ALTS])
N_EV = len(FEATURE_NAMES)
N_COMPACT = len(COMPACT)
VERSION = 1   # bump when the kernel changes (part of the cache names)


def hazard_table(l_min=10, l_max=999, p_break=0.5):
    """h[t] = P(break at online step t | no break before t, series still running at t) under the generator."""
    Ls = np.arange(l_min, l_max + 1, dtype=np.float64)
    pL = np.full(len(Ls), 1.0 / len(Ls))
    h = np.zeros(l_max + 1)
    for t in range(l_max + 1):
        m = Ls > t
        if not m.any():
            h[t] = h[t - 1]
            continue
        num = p_break * np.sum(pL[m] / Ls[m])
        den = np.sum(pL[m] * ((1 - p_break) + p_break * (Ls[m] - t) / Ls[m]))
        h[t] = num / den
    return np.clip(h, 1e-9, 0.5)


HAZARD = hazard_table()

# ---------------------------------------------------------------------------------------------
NEG = -1e30


@njit(cache=True, nogil=True)
def _lae(a, b):
    m = a if a > b else b
    if m <= NEG:
        return NEG
    return m + math.log(math.exp(a - m) + math.exp(b - m))


@njit(cache=True, nogil=True)
def ndtri(p):
    """Standard normal quantile (Acklam's rational approximation, relative error < 1.2e-9)."""
    a0, a1, a2, a3, a4, a5 = -3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, \
        1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00
    b0, b1, b2, b3, b4 = -5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, \
        6.680131188771972e+01, -1.328068155288572e+01
    c0, c1, c2, c3, c4, c5 = -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, \
        -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00
    d0, d1, d2, d3 = 7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00
    plow = 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c0 * q + c1) * q + c2) * q + c3) * q + c4) * q + c5) / ((((d0 * q + d1) * q + d2) * q + d3) * q + 1.0)
    if p > 1.0 - plow:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c0 * q + c1) * q + c2) * q + c3) * q + c4) * q + c5) / ((((d0 * q + d1) * q + d2) * q + d3) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a0 * r + a1) * r + a2) * r + a3) * r + a4) * r + a5) * q / \
        (((((b0 * r + b1) * r + b2) * r + b3) * r + b4) * r + 1.0)


@njit(cache=True, nogil=True)
def _whiten(e, eref):
    n = len(eref)
    k = np.searchsorted(eref, e)
    w = ndtri((k + 0.5) / (n + 1.0))
    if e > eref[n - 1]:
        w = ndtri((n + 0.5) / (n + 1.0)) + (e - eref[n - 1])
    elif e < eref[0]:
        w = ndtri(0.5 / (n + 1.0)) - (eref[0] - e)
    return w


@njit(cache=True, nogil=True)
def _llr(kind, par, const, w, wprev):
    if kind == 0:
        return const - 0.5 * w * w * (1.0 / (par * par) - 1.0)
    if kind == 1:
        return par * w - 0.5 * par * par
    if kind == 2:
        d = w - par * wprev
        return const - 0.5 * d * d / (1.0 - par * par) + 0.5 * w * w
    # Student-t with unit variance, nu = par
    return const - 0.5 * (par + 1.0) * math.log(1.0 + w * w / (par - 2.0)) + 0.5 * w * w


# parameter vector Q: [mu, sd, se10, coef(AR)]; state ST: [t, wprev, logP, lag(AR), logO(NA), logR(NA), W(NA)]
Q_MU, Q_SD, Q_SE, Q_AR = 0, 1, 2, 3
S_T, S_WP, S_LP, S_LAG = 0, 1, 2, 3
S_LO = S_LAG + AR
S_LR = S_LO + NA
S_W = S_LR + NA
S_LEN = S_W + NA


@njit(cache=True, nogil=True)
def ev_step(x, Q, eref, hz, kind, par, const, grp, ST, out):
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
    w = _whiten(e, eref)
    wp = ST[S_WP]
    h = hz[t] if t < len(hz) else hz[len(hz) - 1]
    lh = math.log(h)
    l1h = math.log(1.0 - h)
    ST[S_LP] = _lae(ST[S_LP], lh) - l1h
    na = len(kind)
    tot = NEG
    glse = np.full(4, NEG)
    for k in range(na):
        l = _llr(kind[k], par[k], const[k], w, wp)
        wk = ST[S_W + k] + l
        ST[S_W + k] = wk if wk > 0.0 else 0.0
        ST[S_LR + k] = _lae(0.0, ST[S_LR + k]) + l
        lo = _lae(ST[S_LO + k], lh) - l1h + l
        ST[S_LO + k] = lo
        tot = _lae(tot, lo)
        glse[grp[k]] = _lae(glse[grp[k]], lo)
    mix = tot - math.log(na)
    out[0] = mix
    out[1] = mix - ST[S_LP]
    for g in range(4):
        out[2 + g] = math.exp(glse[g] - tot)
    o = 6
    for k in range(na):
        out[o + k] = ST[S_W + k]
        out[o + na + k] = ST[S_LR + k]
        out[o + 2 * na + k] = ST[S_LO + k]
    for i in range(AR - 1, 0, -1):
        ST[S_LAG + i] = ST[S_LAG + i - 1]
    ST[S_LAG] = z
    ST[S_WP] = w
    ST[S_T] = t + 1.0


@njit(cache=True, nogil=True)
def _ev_run(online, Q, eref, hz, kind, par, const, grp, ST, out):
    for i in range(len(online)):
        ev_step(online[i], Q, eref, hz, kind, par, const, grp, ST, out[i])


def make_ev_params(hist):
    """Per-series constants: the main kernel's AR(10) fit, the sorted history residuals, and the initial state."""
    P, _, z = F.make_params(hist)
    coef = P[F.P_AR:F.P_AR + AR]
    se = P[F.P_SE10]
    n = len(z)
    Xl = np.column_stack([z[AR - 1 - i: n - 1 - i] for i in range(AR)])
    eh = np.clip((z[AR:] - Xl @ coef) / se, -10.0, 10.0)
    eref = np.sort(eh)
    Q = np.r_[P[F.P_MU], P[F.P_SD], se, coef].astype(np.float64)
    ST = np.zeros(S_LEN)
    ST[S_LP] = NEG
    ST[S_LO:S_LO + NA] = NEG
    ST[S_LR:S_LR + NA] = NEG
    for i in range(AR):
        ST[S_LAG + i] = z[-1 - i]
    ST[S_WP] = _whiten(float(eh[-1]), eref)
    return Q, eref, ST


def run_series_ev(hist, online):
    """Batch evidence matrix (T, N_EV) for one series -- identical to streaming."""
    Q, eref, ST = make_ev_params(np.asarray(hist, dtype=np.float64))
    online = np.asarray(online, dtype=np.float64)
    out = np.zeros((len(online), N_EV))
    _ev_run(online, Q, eref, HAZARD, KIND, PAR, CONST, GRP, ST, out)
    return out


class EvStream:
    """One series' streaming evidence state for use inside infer()."""

    def __init__(self, hist):
        self.Q, self.eref, self.ST = make_ev_params(np.asarray(hist, dtype=np.float64))
        self.out = np.zeros(N_EV)

    def push(self, x):
        ev_step(float(x), self.Q, self.eref, HAZARD, KIND, PAR, CONST, GRP, self.ST, self.out)
        return self.out


def _selftest(n_series=2000):
    import os
    import sys
    import time
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from scipy.special import ndtri as sp_ndtri
    from scipy.stats import t as student_t

    from sbrt.data import load_train
    from sbrt.metric import ts_auc

    p = np.r_[1e-8, 1e-4, 0.01, 0.02425, 0.3, 0.5, 0.77, 0.99, 1 - 1e-6]
    assert np.allclose([ndtri(v) for v in p], sp_ndtri(p), rtol=1e-8, atol=1e-8)
    print("normal quantile approximation matches scipy: OK")
    for nu in (3.0, 6.0):  # Student-t log-likelihood ratio vs scipy
        k = [i for i, a in enumerate(ALTS) if a[1] == 3 and a[2] == nu][0]
        for w in (-3.0, -0.5, 0.0, 1.2, 4.0):
            sc = np.sqrt((nu - 2) / nu)
            ref = student_t.logpdf(w / sc, nu) - np.log(sc) - (-0.5 * w * w - 0.5 * np.log(2 * np.pi))
            assert abs(_llr(3, nu, CONST[k], w, 0.0) - ref) < 1e-9
    print("Student-t likelihood ratio matches scipy: OK")

    S, tau = load_train()
    S, tau = S[:n_series], tau[:n_series]
    rng = np.random.default_rng(0)
    t0 = time.time()
    for i in rng.choice(n_series, 30, replace=False):
        _, h, o = S[i]
        B = run_series_ev(h, o)
        st = EvStream(h)
        Sm = np.array([st.push(x).copy() for x in o])
        assert np.allclose(B, Sm, rtol=0, atol=0), "streaming != batch"
        assert np.isfinite(B).all()
        if len(o) > 20:
            o2 = o.copy()
            o2[15:] = rng.standard_normal(len(o) - 15) * 3
            assert np.array_equal(run_series_ev(h, o2)[:15], B[:15]), "future points changed earlier features"
    print("streaming == batch, causal, finite: OK")

    # null calibration: with no break, whitened points are ~N(0,1) per series
    W = []
    for i in np.flatnonzero(tau < 0)[:300]:
        _, h, o = S[i]
        Q, eref, ST = make_ev_params(h)
        z = (np.r_[h, o] - Q[0]) / Q[1]
        W.append([_whiten(float(e), eref) for e in
                  np.clip((z[len(h):] - np.column_stack([z[len(h) - 1 - k: len(z) - 1 - k] for k in range(AR)]) @ Q[3:]) / Q[2], -10, 10)])
    W = np.concatenate(W)
    print(f"whitened online points of no-break series: mean {W.mean():+.3f}, sd {W.std():.3f}, "
          f"share |w|>3 {np.mean(np.abs(W) > 3):.4f} (N(0,1): 0.0027)")

    # signal check on SUB-sampled rows, like cv.py
    t0 = time.time()
    mats = [run_series_ev(s[1], s[2])[::3] for s in S]
    print(f"evidence features for {n_series} series in {time.time() - t0:.1f}s")
    E = np.concatenate(mats)
    step = np.concatenate([np.arange(0, len(s[2]), 3) for s in S])
    sid = np.repeat(np.arange(n_series), [len(m) for m in mats])
    y = ((tau[sid] >= 0) & (step >= tau[sid])).astype(np.int8)
    print("single-feature TS-AUC on these series (0.5 = no signal):")
    for j, nm in enumerate(FEATURE_NAMES[:N_COMPACT]):
        print(f"  {nm:22s} {ts_auc(E[:, j], y, step):.4f}")
    best = sorted(((ts_auc(E[:, j], y, step), FEATURE_NAMES[j]) for j in range(N_COMPACT, N_EV)), reverse=True)[:5]
    print("  best per-alternative statistics:", ", ".join(f"{n} {v:.4f}" for v, n in best))
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    _selftest()
