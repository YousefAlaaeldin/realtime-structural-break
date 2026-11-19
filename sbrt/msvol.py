"""Markov-switching model of "normal" (Hamilton 1989): HANDOFF section 45, owner's request 2026-10-16.

Idea. Break-free series' variance wanders about as much as a typical break moves it (HANDOFF 15.2). GARCH (volstate)
treats that wandering as continuous persistence. A Markov-switching (MS) model treats it as visits to a few RECURRING
regimes the history has already shown, each with a known expected duration. Under that null, an online stretch in a
known high regime is cheap; staying there much longer than the history's switching allows, or landing on a variance
no regime has, is evidence of a break. Expectation (written before any run): LOW. Section 39 found AR(10) + GARCH-t
already describes the histories (one-step log score: extended GARCH +0.0002 nats/point). This is not closed by 39 /
42 because those tested one-step density and frozen variance FORECASTS; this tests a different NULL, whose evidence
comes from regime DURATION. Expected trees gain: +0.000 to +0.001.

Model (per series, fitted on the history only, then FROZEN):
  e_t   the kernel's history-fitted AR(10) residual (volstate's `eh`, clipped to +-10); online the same AR(10)
        continues from the last 10 history points (as volstate).
  S_t in {1..K} hidden regime, Markov with transition matrix P;  e_t | S_t = k ~ Student-t(nu, scale^2 s2_k).
  K in {1, 2, 3} and nu in NU_GRID = {4, 8, 30} chosen per series by BIC. EM (Baum-Welch with the t-weights,
  u = (nu + 1) / (nu + e^2 / s2_k)), deterministic start (block-variance quantiles, P_kk = 0.98), regimes ordered by
  s2. K = 1 is a constant-scale t (no regimes).
Online: the Hamilton filter runs on with the frozen parameters, starting from the filtered regime probabilities at the
end of the history. It may move between the KNOWN regimes, never learns a new one.

Features (N_MS = 13, prefix ms_):
  ms_K, ms_lspread (log s2_K / s2_1), ms_ldur (log expected duration of the top regime, -log(1 - P_KK)),
     ms_ptop (filtered probability of the top regime at the end of the history): per-series constants (0 if K = 1).
  ms_logodds, ms_tsb, ms_lvr: exact one-break posterior with bocpd.py's hazard prior. NO break = the MS filter's
     predictive density. Break at online step j = the points from j on are a NEW i.i.d. regime of the SAME t family
     (the series' nu, scale^2 s0^2 rho, s0^2 = the K = 1 fit with that nu, rho on a 15-point log grid 0.2-5 with weights
     from bocpd's empirical-Bayes prior IG(ALPHA0, BETA0) on the variance ratio):
         log BF_j = log sum_g w_g prod_{i>=j} t(e_i; s0^2 rho_g) - sum_{i>=j} log p_MS(e_i | e_<i)
     Same tail shape on both sides on purpose: otherwise every tail point of a heavy-tailed series would favour one side
     for reasons unrelated to a break, and the score's baseline would differ between series (bad for a cross-series
     ranking). logodds = log posterior odds of "break by t", tsb = posterior mean log(1 + time since the break),
     lvr = posterior mean log rho. Leak: the grid weights reuse bocpd's prior (3 numbers fitted on training breaks of
     series 0-4999), exactly the leak bo_ already has.
  ms_ls_full, ms_ls_win100, ms_ls_scanmax: standardised running mean of the surprise -log p_MS(e_t | past) minus its
     in-sample history mean (batch-means long-run variance from the history): all online / last 100 / scan over starts.
  ms_occ_full, ms_occ_scanmax, ms_occ_scanmin: the same for the filtered expected log regime variance ("has the series
     sat in a high / low regime longer than the history's switching allows?"). Exactly 0 when K = 1.

Self-test, instrument and mechanism report (CPU):  python -m sbrt.msvol [offset] [n_series]
  (default offset 5000, 2000 series; writes cv_out/msvol_selftest.txt, or _smoke.txt when n_series < 2000)

DECISION RULE (written 2026-10-16 before any run; do not move it after seeing numbers):
  INSTRUMENT (all must hold, else fix the module before any cv.py run; nothing counts until they do):
    I1 streaming == batch, causal, deterministic, finite.
    I2 positive control (simulated 2-regime series, variances 0.5 / 2.0, P_kk 0.995, t8 noise): K >= 2 chosen on at
       least 3 of 4, fitted variance ratio within 4.0 +- 35%; held-out log score MS - GARCH-t >= +0.002 nats/point;
       ms_logodds at online step 450 separates a break at step 200 into a new variance (3.5) from no break, AUC >= 0.8;
       and on the no-break simulated series the MS null explains regime visits: mean ms_logodds < the constant-t
       twin's.
    I3 negative control (simulated GARCH(1,1)-t6 series, no regimes): held-out MS - GARCH-t <= +0.002 nats/point.
    I4 comparability: on no-break pseudo-online segments (fit on the first 80% of each history, run the last 20%),
       the spread of the mean ms_logodds at pseudo-step 199 across history-kurtosis quintiles, divided by its overall
       sd, is at most production bo_w_logodds's spread + 0.10.
  MECHANISM (reported, NOT a gate; cheap gates have misled both ways here, HANDOFF 35.1 / 38): held-out history log
    score MS vs GARCH-t (comparable with section 39's R2 rows); shares of K and nu; single-feature TS-AUC of every ms_
    column and of three twins with the same break model but a different null: (a) online-updated GARCH-t,
    (b) volstate's frozen GARCH path (what production has), (c) constant-scale t; rank correlations.
  DECISION: the paired cv.py run (standing rule, HANDOFF section 11): trees `--ms` seed 0 vs
    oof_aug2_df_bo_nbm_bs24_ep25 with the early stop (folds 0 and 1 both <= 0 -> CLOSED); then seed 1 and the
    seed-average comparison; then blend.py vs 0.6489 (adopt at >= +0.0005, >= 4/5 folds, > 2 SE).
"""
import math

import numpy as np
from numba import njit

from sbrt import bocpd as BO
from sbrt import features as F
from sbrt import volstate as VS

AR = F.AR
CAP = F.CAP
QCAP = VS.QCAP
KMAX = 3
NU_GRID = np.array([4.0, 8.0, 30.0])
P_STAY0 = 0.98
EM_ITERS = 200
EM_TOL = 1e-7          # stop when the log likelihood gains less than this per point
BLOCK = 50             # block length for the deterministic EM start
BURN = 50              # first history points left out of the in-sample reference statistics
MINSEG = 8
G = 15
RHO = np.exp(np.linspace(math.log(0.2), math.log(5.0), G))
LRHO = np.log(RHO)
_lw = -BO.ALPHA0 * LRHO - BO.BETA0 / RHO               # IG(ALPHA0, BETA0) density of rho, times rho (log grid)
LOGW = _lw - (_lw.max() + math.log(float(np.exp(_lw - _lw.max()).sum())))
LOGH, LP = BO.LOGH, BO.LP
VERSION = 1            # bump when the model or the features change (part of the cache names)

FEATURE_NAMES = ["ms_K", "ms_lspread", "ms_ldur", "ms_ptop",
                 "ms_logodds", "ms_tsb", "ms_lvr",
                 "ms_ls_full", "ms_ls_win100", "ms_ls_scanmax",
                 "ms_occ_full", "ms_occ_scanmax", "ms_occ_scanmin"]
N_MS = len(FEATURE_NAMES)


# ---------------------------------------------------------------------------------------------
@njit(cache=True)
def _tc(nu):
    return math.lgamma(0.5 * (nu + 1.0)) - math.lgamma(0.5 * nu) - 0.5 * math.log(nu * math.pi)


@njit(cache=True, nogil=True)
def _lt(e, s2, nu, c):
    """log Student-t(nu) density of e with scale^2 s2 (c = _tc(nu))."""
    return c - 0.5 * math.log(s2) - 0.5 * (nu + 1.0) * math.log1p(e * e / (nu * s2))


@njit(cache=True, nogil=True)
def _em(e, K, nu, s2init, iters, tol, floor):
    """Baum-Welch for a K-regime Student-t variance HMM (EM with the t latent weights). Returns s2, P, pi0, loglik;
    the log likelihood is that of the returned parameters."""
    n = len(e)
    c = _tc(nu)
    s2 = s2init.copy()
    P = np.zeros((K, K))
    for i in range(K):
        for k in range(K):
            if K == 1:
                P[i, k] = 1.0
            elif i == k:
                P[i, k] = P_STAY0
            else:
                P[i, k] = (1.0 - P_STAY0) / (K - 1)
    pi0 = np.full(K, 1.0 / K)
    B = np.empty((n, K))
    lsc = np.empty(n)
    al = np.empty((n, K))
    be = np.empty((n, K))
    cs = np.empty(n)
    ll = -1e300
    ll_old = -1e300
    for it in range(iters + 1):
        for t in range(n):
            mx = -1e300
            for k in range(K):
                v = _lt(e[t], s2[k], nu, c)
                B[t, k] = v
                if v > mx:
                    mx = v
            lsc[t] = mx
            for k in range(K):
                B[t, k] = math.exp(B[t, k] - mx)
        ll = 0.0
        for t in range(n):
            tot = 0.0
            for k in range(K):
                if t == 0:
                    pr = pi0[k]
                else:
                    pr = 0.0
                    for i in range(K):
                        pr += al[t - 1, i] * P[i, k]
                al[t, k] = pr * B[t, k]
                tot += al[t, k]
            if tot < 1e-300:
                tot = 1e-300
            cs[t] = tot
            for k in range(K):
                al[t, k] /= tot
            ll += math.log(tot) + lsc[t]
        if it == iters or (it > 0 and ll - ll_old < tol * n):
            break
        ll_old = ll
        for k in range(K):
            be[n - 1, k] = 1.0
        for t in range(n - 2, -1, -1):
            for i in range(K):
                acc = 0.0
                for k in range(K):
                    acc += P[i, k] * B[t + 1, k] * be[t + 1, k]
                be[t, i] = acc / cs[t + 1]
        num = np.zeros(K)
        den = np.zeros(K)
        xs = np.zeros((K, K))
        gs = np.zeros(K)
        for t in range(n):
            e2 = e[t] * e[t]
            for k in range(K):
                g = al[t, k] * be[t, k]
                num[k] += g * (nu + 1.0) / (nu + e2 / s2[k]) * e2
                den[k] += g
                if t < n - 1:
                    gs[k] += g
            if t < n - 1:
                for i in range(K):
                    for k in range(K):
                        xs[i, k] += al[t, i] * P[i, k] * B[t + 1, k] * be[t + 1, k] / cs[t + 1]
        for k in range(K):
            s2[k] = max(num[k] / max(den[k], 1e-12), floor)
            pi0[k] = al[0, k] * be[0, k]
        if K > 1:
            for i in range(K):
                rs = 0.0
                for k in range(K):
                    v = max(xs[i, k] / max(gs[i], 1e-12), 1e-6)
                    P[i, k] = v
                    rs += v
                for k in range(K):
                    P[i, k] /= rs
    return s2, P, pi0, ll


@njit(cache=True, nogil=True)
def _fstep(e, K, nu, c, s2, lv, P, al, pr, first):
    """One Hamilton-filter step. `al` holds the previous filtered probabilities (pi0 on the first call) and is updated
    in place. Returns the one-step predictive log density of e and the filtered expected log regime variance."""
    if first:
        for k in range(K):
            pr[k] = al[k]
    else:
        for k in range(K):
            v = 0.0
            for i in range(K):
                v += al[i] * P[i, k]
            pr[k] = v
    mx = -1e300
    for k in range(K):
        d = _lt(e, s2[k], nu, c)
        al[k] = d
        if d > mx:
            mx = d
    tot = 0.0
    for k in range(K):
        al[k] = pr[k] * math.exp(al[k] - mx)
        tot += al[k]
    if tot < 1e-300:
        tot = 1e-300
    occ = 0.0
    for k in range(K):
        al[k] /= tot
        occ += al[k] * lv[k]
    return mx + math.log(tot), occ


@njit(cache=True, nogil=True)
def _hfilter(e, K, nu, s2, lv, P, pi0):
    """Forward filter over a residual sequence: predictive log densities, filtered occupancy, final filtered probs."""
    n = len(e)
    c = _tc(nu)
    al = pi0.copy()
    pr = np.zeros(len(pi0))
    lp = np.empty(n)
    occ = np.empty(n)
    for t in range(n):
        a, b = _fstep(e[t], K, nu, c, s2, lv, P, al, pr, t == 0)
        lp[t] = a
        occ[t] = b
    return lp, occ, al


@njit(cache=True, nogil=True)
def _post(C, n, RHO_, LOGW_, LRHO_, LOGH_, LP_, buf, buf2, out, o):
    """One-break posterior at n points. C[0] = cumulative null log predictive, C[3 + g] = cumulative log density of
    the post-break regime with variance ratio rho_g."""
    Gn = len(LOGW_)
    mx = -1e300
    for j in range(n):
        vm = -1e300
        for g in range(Gn):
            v = LOGW_[g] + C[3 + g, n] - C[3 + g, j]
            if v > vm:
                vm = v
        sg = 0.0
        sl = 0.0
        for g in range(Gn):
            wv = math.exp(LOGW_[g] + C[3 + g, n] - C[3 + g, j] - vm)
            sg += wv
            sl += wv * LRHO_[g]
        lr = vm + math.log(sg) - (C[0, n] - C[0, j])
        a = LP_[j] + LOGH_[j] + lr
        buf[j] = a
        buf2[j] = sl / sg
        if a > mx:
            mx = a
    s = 0.0
    etsb = 0.0
    elvr = 0.0
    for j in range(n):
        wgt = math.exp(buf[j] - mx)
        s += wgt
        etsb += wgt * math.log(1.0 + n - j)
        elvr += wgt * buf2[j]
    out[o] = mx + math.log(s) - LP_[n]
    out[o + 1] = etsb / s
    out[o + 2] = elvr / s


# ---------------------------------------------------------------------------------------------
def fit_ms(eh):
    """Grid over K in {1, 2, 3} and nu in NU_GRID, EM for each, BIC choice. Deterministic."""
    e = np.ascontiguousarray(eh, dtype=np.float64)
    n = len(e)
    m2 = float(np.mean(e * e))
    floor = 1e-3 * m2
    nb = n // BLOCK
    bv = np.sort((e[: nb * BLOCK] ** 2).reshape(nb, BLOCK).mean(axis=1))
    best = None
    k1 = {}
    for K in (1, 2, 3):
        for nu in NU_GRID:
            f = (nu - 2.0) / nu
            if K == 1:
                s2i = np.array([m2 * f])
            else:
                s2i = np.array([max(float(bv[min(nb - 1, int((k + 0.5) / K * nb))]) * f, floor) for k in range(K)])
            s2, P, pi0, ll = _em(e, K, float(nu), s2i, EM_ITERS, EM_TOL, floor)
            bic = -2.0 * ll + (K + K * (K - 1) + 1) * math.log(n)
            if K == 1:
                k1[float(nu)] = (float(s2[0]), float(ll))
            if best is None or bic < best[0] - 1e-9:
                best = (bic, K, float(nu), s2, P, pi0, float(ll))
    bic, K, nu, s2, P, pi0, ll = best
    idx = np.argsort(s2, kind="stable")
    s2, P, pi0 = s2[idx], P[idx][:, idx], pi0[idx]
    nu_c = max(k1, key=lambda v: k1[v][1])          # constant-scale t: the best nu by likelihood (twin c)
    return dict(K=K, nu=nu, s2=s2, P=P, pi0=pi0, ll=ll, bic=bic, s0sq=k1[nu][0], k1=k1, nu_c=nu_c)


def _full(M):
    """Model arrays padded to KMAX (numba-friendly)."""
    K = M["K"]
    s2 = np.ones(KMAX)
    s2[:K] = M["s2"]
    P = np.eye(KMAX)
    P[:K, :K] = M["P"]
    pi0 = np.zeros(KMAX)
    pi0[:K] = M["pi0"]
    return s2, np.log(s2), P, pi0


def online_resid(H, x):
    """Clipped AR(10) residuals of points x that follow a fitted history H (volstate.fit_history output)."""
    Pp = H["P"]
    zx = (np.asarray(x, dtype=np.float64) - Pp[F.P_MU]) / Pp[F.P_SD]
    zz = np.r_[H["z"], zx]
    nh = len(H["z"])
    Xo = np.column_stack([zz[nh - 1 - i: len(zz) - 1 - i] for i in range(AR)])
    return np.clip((zx - Xo @ H["coef"]) / H["se"], -10.0, 10.0)


# parameter vector Q and state ST layouts
(Q_MU, Q_SD, Q_SE, Q_K, Q_NU, Q_C, Q_S0, Q_CLS, Q_ILS, Q_COC, Q_IOC, Q_LSPR, Q_LDUR, Q_PTOP, Q_AR) = range(15)
Q_LEN = Q_AR + AR
S_T, S_LAG = 0, 1
S_LEN = S_LAG + AR


def make_ms_params(hist):
    H = VS.fit_history(np.asarray(hist, dtype=np.float64))
    M = fit_ms(H["eh"])
    K, nu = M["K"], M["nu"]
    s2, lv, P, pi0 = _full(M)
    lp, occ, al = _hfilter(H["eh"], K, nu, s2, lv, P, pi0)
    x = -lp[BURN:]
    cls = float(x.mean())
    lrv_ls = F._lrv(x - cls, float(x.var()), 1e-6)
    oc = occ[BURN:]
    coc = float(oc.mean())
    Q = np.zeros(Q_LEN)
    Q[Q_MU], Q[Q_SD], Q[Q_SE] = H["P"][F.P_MU], H["P"][F.P_SD], H["se"]
    Q[Q_K], Q[Q_NU], Q[Q_C], Q[Q_S0] = K, nu, _tc(nu), M["s0sq"]
    Q[Q_CLS], Q[Q_ILS] = cls, 1.0 / math.sqrt(lrv_ls)
    Q[Q_COC] = coc
    if K > 1:
        Q[Q_IOC] = 1.0 / math.sqrt(F._lrv(oc - coc, float(oc.var()), 1e-8))
        Q[Q_LSPR] = math.log(s2[K - 1] / s2[0])
        Q[Q_LDUR] = -math.log(max(1.0 - P[K - 1, K - 1], 1e-6))
        Q[Q_PTOP] = al[K - 1]
    Q[Q_AR:] = H["coef"]
    ST = np.zeros(S_LEN)
    z = H["z"]
    for i in range(AR):
        ST[S_LAG + i] = z[-1 - i]
    C = np.zeros((3 + G, CAP + 1))
    return Q, s2, lv, P, al.copy(), ST, C, H, M


@njit(cache=True, nogil=True)
def ms_step(x, Q, s2, lv, P, al, pr, C, ST, RHO_, LOGW_, LRHO_, LOGH_, LP_, buf, buf2, out, do_post):
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
    K = int(Q[Q_K])
    nu = Q[Q_NU]
    c = Q[Q_C]
    lp, occ = _fstep(e, K, nu, c, s2, lv, P, al, pr, False)
    n = t + 1
    C[0, n] = C[0, t] + lp
    C[1, n] = C[1, t] + (-lp - Q[Q_CLS])
    C[2, n] = C[2, t] + (occ - Q[Q_COC])
    s0 = Q[Q_S0]
    for g in range(len(RHO_)):
        C[3 + g, n] = C[3 + g, t] + _lt(e, s0 * RHO_[g], nu, c)
    if do_post:
        out[0] = K * 1.0
        out[1] = Q[Q_LSPR]
        out[2] = Q[Q_LDUR]
        out[3] = Q[Q_PTOP]
        _post(C, n, RHO_, LOGW_, LRHO_, LOGH_, LP_, buf, buf2, out, 4)
        ils = Q[Q_ILS]
        out[7] = C[1, n] * ils / math.sqrt(n * 1.0)
        w = 100 if n > 100 else n
        out[8] = (C[1, n] - C[1, n - w]) * ils / math.sqrt(w * 1.0)
        ioc = Q[Q_IOC]
        out[10] = C[2, n] * ioc / math.sqrt(n * 1.0)
        gmax = -1e9
        omax = -1e9
        omin = 1e9
        for j in range(0, n - MINSEG + 1):
            r = 1.0 / math.sqrt((n - j) * 1.0)
            s = (C[1, n] - C[1, j]) * ils * r
            if s > gmax:
                gmax = s
            s = (C[2, n] - C[2, j]) * ioc * r
            if s > omax:
                omax = s
            if s < omin:
                omin = s
        out[9] = gmax if gmax > -1e8 else 0.0
        out[11] = omax if omax > -1e8 else 0.0
        out[12] = omin if omin < 1e8 else 0.0
    for i in range(AR - 1, 0, -1):
        ST[S_LAG + i] = ST[S_LAG + i - 1]
    ST[S_LAG] = z
    ST[S_T] = n * 1.0


@njit(cache=True, nogil=True)
def _ms_run(online, Q, s2, lv, P, al, C, ST, RHO_, LOGW_, LRHO_, LOGH_, LP_, out, sub):
    pr = np.zeros(len(al))
    buf = np.zeros(CAP + 1)
    buf2 = np.zeros(CAP + 1)
    row = np.zeros(out.shape[1])
    r = 0
    for i in range(len(online)):
        keep = i % sub == 0
        ms_step(online[i], Q, s2, lv, P, al, pr, C, ST, RHO_, LOGW_, LRHO_, LOGH_, LP_, buf, buf2, row, keep)
        if keep:
            for k in range(out.shape[1]):
                out[r, k] = row[k]
            r += 1


def run_series_ms(hist, online, sub=1):
    """Batch matrix (rows for online steps 0, sub, 2 sub, ..., N_MS) for one series -- identical to streaming."""
    Q, s2, lv, P, al, ST, C, _, _ = make_ms_params(hist)
    online = np.asarray(online, dtype=np.float64)
    if len(online) >= CAP:
        raise ValueError(f"online segment longer than the supported {CAP - 1} points")
    out = np.zeros(((len(online) + sub - 1) // sub, N_MS))
    _ms_run(online, Q, s2, lv, P, al, C, ST, RHO, LOGW, LRHO, LOGH, LP, out, sub)
    return out


class MsStream:
    """One series' streaming state for use inside infer()."""

    def __init__(self, hist):
        self.Q, self.s2, self.lv, self.P, self.al, self.ST, self.C, _, _ = make_ms_params(hist)
        self.pr = np.zeros(KMAX)
        self.buf = np.zeros(CAP + 1)
        self.buf2 = np.zeros(CAP + 1)
        self.out = np.zeros(N_MS)

    def push(self, x):
        if self.ST[S_T] >= CAP - 1:
            return self.out
        ms_step(float(x), self.Q, self.s2, self.lv, self.P, self.al, self.pr, self.C, self.ST, RHO, LOGW, LRHO,
                LOGH, LP, self.buf, self.buf2, self.out, True)
        return self.out


# ---------------------------------------------------------------------------------------------
# self-test helpers: twins with the same break model and a different null
@njit(cache=True, nogil=True)
def _post_series(e, lnull, nu, s0sq, RHO_, LOGW_, LRHO_, LOGH_, LP_, sub, out):
    """(logodds, tsb, lvr) on rows 0, sub, ... for any null predictive log density lnull[i] of e[i]."""
    n_pts = len(e)
    c = _tc(nu)
    C = np.zeros((3 + len(RHO_), n_pts + 1))
    buf = np.zeros(n_pts + 1)
    buf2 = np.zeros(n_pts + 1)
    row = np.zeros(3)
    r = 0
    for i in range(n_pts):
        C[0, i + 1] = C[0, i] + lnull[i]
        for g in range(len(RHO_)):
            C[3 + g, i + 1] = C[3 + g, i] + _lt(e[i], s0sq * RHO_[g], nu, c)
        if i % sub == 0:
            _post(C, i + 1, RHO_, LOGW_, LRHO_, LOGH_, LP_, buf, buf2, row, 0)
            out[r, 0], out[r, 1], out[r, 2] = row[0], row[1], row[2]
            r += 1


@njit(cache=True, nogil=True)
def _fit_t1(u, nu, iters):
    """Student-t scale^2 with a fixed df by EM; returns (scale^2, loglik)."""
    n = len(u)
    s2 = 0.0
    for x in u:
        s2 += x * x
    s2 = max(s2 / n * (nu - 2.0) / nu if nu > 2.0 else s2 / n, 1e-12)
    for it in range(iters):
        acc = 0.0
        for x in u:
            acc += (nu + 1.0) / (nu + x * x / s2) * x * x
        s2 = max(acc / n, 1e-12)
    c = _tc(nu)
    ll = 0.0
    for x in u:
        ll += _lt(x, s2, nu, c)
    return s2, ll


def fit_t_grid(u, nus):
    best = None
    for nu in nus:
        s2, ll = _fit_t1(np.ascontiguousarray(u, dtype=np.float64), float(nu), 30)
        if best is None or ll > best[2]:
            best = (s2, float(nu), ll)
    return best[0], best[1]


EQ_NU = np.array([2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 12.0, 20.0, 50.0, 200.0])   # equation_check's grid (M2 comparable)


def lt_vec(e, s2, nu):
    c = _tc(nu)
    return c - 0.5 * np.log(s2) - 0.5 * (nu + 1.0) * np.log1p(e * e / (nu * s2))


def twins(H, eo, sub):
    """Twin posteriors (rows like run_series_ms): (a) online-updated GARCH-t, (b) volstate's frozen GARCH path,
    (c) constant-scale t. Each with its own df on both sides of the break."""
    eh = H["eh"]
    q = np.minimum(np.r_[eh, eo] ** 2, QCAP)
    s2all, _ = VS._filter(q, H["vbar"], H["a"], H["p"] - H["a"])
    s2h, s2o = s2all[: len(eh)], s2all[len(eh):]
    sc2, nu2 = fit_t_grid(eh[BURN:] / np.sqrt(s2h[BURN:]), NU_GRID)
    s0g, _ = _fit_t1(np.ascontiguousarray(eh), nu2, 30)
    k = np.arange(len(eo))
    s2f = H["L"] + (H["s2T1"] - H["L"]) * H["p"] ** k
    out = {}
    for nm, ln, nu, s0 in (("a", lt_vec(eo, sc2 * s2o, nu2), nu2, s0g), ("b", lt_vec(eo, sc2 * s2f, nu2), nu2, s0g)):
        o = np.zeros(((len(eo) + sub - 1) // sub, 3))
        _post_series(np.ascontiguousarray(eo), np.ascontiguousarray(ln), nu, s0, RHO, LOGW, LRHO, LOGH, LP, sub, o)
        out[nm] = o
    M = fit_ms(eh)
    nu_c = M["nu_c"]
    s0c = M["k1"][nu_c][0]
    o = np.zeros(((len(eo) + sub - 1) // sub, 3))
    _post_series(np.ascontiguousarray(eo), np.ascontiguousarray(lt_vec(eo, s0c, nu_c)), nu_c, s0c, RHO, LOGW, LRHO,
                 LOGH, LP, sub, o)
    out["c"] = o
    return out


def heldout(h, split=0.8):
    """Held-out one-step log score on the last 20% of a history: MS vs GARCH-t (section 39's M2, same recipe)."""
    h = np.asarray(h, dtype=np.float64)
    s = int(split * len(h))
    H = VS.fit_history(h[:s])
    eo = online_resid(H, h[s:])
    et = np.r_[H["eh"], eo]
    m = len(H["eh"])
    M = fit_ms(H["eh"])
    s2, lv, P, pi0 = _full(M)
    lp, _, _ = _hfilter(et, M["K"], M["nu"], s2, lv, P, pi0)
    q = np.minimum(et * et, QCAP)
    s2g = VS._filter(q, H["vbar"], H["a"], H["p"] - H["a"])[0]
    sdg = np.sqrt(s2g)
    sc2, nu2 = fit_t_grid(H["eh"][BURN:] / sdg[BURN:m], EQ_NU)
    m2 = lt_vec(et[m:], sc2 * s2g[m:], nu2)
    return float(lp[m:].mean() - m2.mean()), M["K"], M["nu"]


# ---------------------------------------------------------------------------------------------
def _sim_ms(rng, n, s2=(0.5, 2.0), pstay=0.995, nu=8.0):
    k = 0
    x = np.empty(n)
    sc = math.sqrt((nu - 2.0) / nu)
    for t in range(n):
        if rng.random() > pstay:
            k = 1 - k
        x[t] = math.sqrt(s2[k]) * sc * rng.standard_t(nu)
    return x


def _sim_garch(rng, n, a=0.08, b=0.90, nu=6.0):
    x = np.empty(n)
    s2 = 1.0
    sc = math.sqrt((nu - 2.0) / nu)
    for t in range(n):
        x[t] = math.sqrt(s2) * sc * rng.standard_t(nu)
        s2 = (1 - a - b) + a * x[t] ** 2 + b * s2
    return x


def _selftest(offset=5000, n_series=2000, threads=8):
    import os
    import sys
    import time
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from joblib import Parallel, delayed
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score

    from sbrt.data import load_train
    from sbrt.metric import ts_auc

    smoke = n_series < 2000
    os.makedirs("cv_out", exist_ok=True)
    log = open(os.path.join("cv_out", "msvol_selftest_smoke.txt" if smoke else "msvol_selftest.txt"), "w",
               encoding="utf-8")

    def say(s=""):
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    par = Parallel(n_jobs=threads, backend="threading")
    T0 = time.time()
    say(f"msvol self-test: series {offset}..{offset + n_series - 1}, threads {threads}"
        + ("  [SMOKE: numbers meaningless]" if smoke else ""))
    fails = []
    rng = np.random.default_rng(0)

    # I2 positive control: recovery, held-out score, break separation, regime visits explained
    rec = []
    for rep in range(4):
        x = _sim_ms(rng, 4000)
        H = VS.fit_history(x)
        M = fit_ms(H["eh"])
        ratio = M["s2"][-1] / M["s2"][0]
        rec.append((M["K"], ratio, M["P"][-1, -1] if M["K"] > 1 else 0.0, M["nu"]))
    say("[I2] simulated 2-regime t8 series (variances 0.5 / 2.0, P_kk 0.995): fitted (K, s2_max/s2_min, P_KK, nu) = "
        + "; ".join(f"({k}, {r:.2f}, {p:.4f}, {nu:.0f})" for k, r, p, nu in rec))
    ok = sum(k >= 2 and abs(r / 4.0 - 1.0) <= 0.35 for k, r, _, _ in rec) >= 3
    say(f"     recovery (K >= 2 and ratio within 4.0 +- 35% on >= 3 of 4): {'OK' if ok else 'FAILED'}")
    if not ok:
        fails.append("I2 recovery")
    d_ms = [heldout(_sim_ms(rng, 4000))[0] for _ in range(6)]
    say(f"[I2] held-out MS - GARCH-t on simulated 2-regime series: {np.mean(d_ms):+.4f} nats/point "
        f"(per series {np.round(d_ms, 4)}) -> {'OK' if np.mean(d_ms) >= 0.002 else 'FAILED'}")
    if np.mean(d_ms) < 0.002:
        fails.append("I2 held-out score")
    lo_b, lo_n, tw_n = [], [], []
    for rep in range(80):
        x = _sim_ms(rng, 3600)
        if rep % 2 == 0:   # break at online step 200: a new i.i.d. regime with variance 3.5
            x[3200:] = math.sqrt(3.5 * 6.0 / 8.0) * rng.standard_t(8.0, 400)
        h, o = x[:3000], x[3000:]
        R = run_series_ms(h, o)
        (lo_b if rep % 2 == 0 else lo_n).append(R[450, FEATURE_NAMES.index("ms_logodds")])
        if rep % 2 == 1:
            H = VS.fit_history(h)
            tw_n.append(twins(H, online_resid(H, o), 1)["c"][450, 0])
    auc = roc_auc_score(np.r_[np.ones(len(lo_b)), np.zeros(len(lo_n))], np.r_[lo_b, lo_n])
    say(f"[I2] break separation at online step 450 (40 break / 40 no-break simulated series): ms_logodds AUC {auc:.3f}"
        f" -> {'OK' if auc >= 0.8 else 'FAILED'}")
    if auc < 0.8:
        fails.append("I2 break separation")
    say(f"[I2] no-break simulated series, mean logodds at step 450: MS null {np.mean(lo_n):+.3f} vs constant-t twin "
        f"{np.mean(tw_n):+.3f} -> {'OK' if np.mean(lo_n) < np.mean(tw_n) else 'FAILED'} (the MS null explains regime visits)")
    if not np.mean(lo_n) < np.mean(tw_n):
        fails.append("I2 regime visits")
    # I3 negative control
    d_g = [heldout(_sim_garch(rng, 4000))[0] for _ in range(6)]
    say(f"[I3] held-out MS - GARCH-t on simulated GARCH(1,1)-t6 series: {np.mean(d_g):+.4f} nats/point "
        f"(per series {np.round(d_g, 4)}) -> {'OK' if np.mean(d_g) <= 0.002 else 'FAILED'}")
    if np.mean(d_g) > 0.002:
        fails.append("I3 negative control")

    S, tau = load_train()
    S, tau = S[offset:offset + n_series], tau[offset:offset + n_series]

    # I1 streaming == batch, causal, deterministic, finite
    for i in rng.choice(n_series, 20, replace=False):
        _, h, o = S[i]
        B = run_series_ms(h, o)
        assert np.array_equal(B, run_series_ms(h, o)), "two runs differ"
        st = MsStream(h)
        Sm = np.array([st.push(x).copy() for x in o])
        assert np.array_equal(B, Sm), "streaming != batch"
        assert np.array_equal(run_series_ms(h, o, 3), B[::3]), "sub-sampled rows != every 3rd streaming row"
        assert np.isfinite(B).all(), "non-finite feature"
        if len(o) > 20:
            o2 = o.copy()
            o2[15:] = rng.standard_normal(len(o) - 15) * 3
            assert np.array_equal(run_series_ms(h, o2)[:15], B[:15]), "future points changed earlier features"
    say("[I1] streaming == batch (also with sub = 3), causal, deterministic, finite: OK")

    # fits on the real histories
    t0 = time.time()
    FITS = par(delayed(lambda s: fit_ms(VS.fit_history(s[1])["eh"]))(s) for s in S)
    say(f"\nMS fits on {n_series} real histories in {time.time() - t0:.0f}s ({(time.time() - t0) * threads / n_series * 1000:.0f} "
        f"ms per series per thread)")
    Ks = np.array([m["K"] for m in FITS])
    nus = np.array([m["nu"] for m in FITS])
    say(f"  K chosen by BIC: 1 {np.mean(Ks == 1):.1%}, 2 {np.mean(Ks == 2):.1%}, 3 {np.mean(Ks == 3):.1%};  nu: "
        + ", ".join(f"{v:.0f} {np.mean(nus == v):.1%}" for v in NU_GRID))
    multi = [m for m in FITS if m["K"] > 1]
    if multi:
        spr = np.array([m["s2"][-1] / m["s2"][0] for m in multi])
        dur = np.array([1.0 / max(1.0 - m["P"][-1, -1], 1e-6) for m in multi])
        say(f"  K > 1: variance ratio top/bottom quantiles 10/50/90% {np.round(np.quantile(spr, [.1, .5, .9]), 2)}; "
            f"expected top-regime duration (points) 10/50/90% {np.round(np.quantile(dur, [.1, .5, .9]), 0)}")

    # mechanism: held-out log score vs GARCH-t (comparable with section 39 R2)
    t0 = time.time()
    HO = par(delayed(heldout)(s[1]) for s in S)
    d = np.array([v[0] for v in HO])
    hk = np.array([v[1] for v in HO])
    say(f"\n[mechanism] held-out one-step log score on the last 20% of each history, MS - GARCH-t (section 39's M2): "
        f"{d.mean():+.4f} +- {d.std(ddof=1) / math.sqrt(len(d)):.4f} nats/point, median {np.median(d):+.4f}; "
        f"share > +0.002: {np.mean(d > 0.002):.1%}, < -0.002: {np.mean(d < -0.002):.1%}  ({time.time() - t0:.0f}s)")
    for K in (1, 2, 3):
        if np.any(hk == K):
            say(f"  K = {K} (on the 80% fit, {np.mean(hk == K):.1%}): {d[hk == K].mean():+.4f}")
    say("  (section 39 R2 reference: extended GARCH +0.0002, NBM-scale* +0.0001; the R2 bar was +0.002 and > 2 SE)")

    # I4 comparability on no-break pseudo-online segments
    def pseudo(s):
        h = np.asarray(s[1], dtype=np.float64)
        sp = int(0.8 * len(h))
        hh, oo = h[:sp], h[sp:]
        R = run_series_ms(hh, oo)
        Bo = BO.run_series_bocpd(hh, oo)
        H = VS.fit_history(hh)
        Tw = twins(H, online_resid(H, oo), 1)
        eh = H["eh"]
        kurt = float(np.mean(eh ** 4) / np.mean(eh ** 2) ** 2 - 3.0)
        j = 199
        return (R[j, FEATURE_NAMES.index("ms_logodds")], Bo[j, BO.FEATURE_NAMES.index("bo_w_logodds")],
                Bo[j, BO.FEATURE_NAMES.index("bo_wu_logodds")], Tw["a"][j, 0], Tw["b"][j, 0], Tw["c"][j, 0], kurt,
                R[j, 0])

    t0 = time.time()
    PS = np.array(par(delayed(pseudo)(s) for s in S))
    say(f"\n[I4] no-break pseudo-online (fit on the first 80% of each history, run the last 20%), logodds at "
        f"pseudo-step 199, by history-kurtosis quintile ({time.time() - t0:.0f}s):")
    qs = np.quantile(PS[:, 6], [0.2, 0.4, 0.6, 0.8])
    bk = np.searchsorted(qs, PS[:, 6])
    flat = {}
    for c, nm in enumerate(["ms_logodds", "bo_w_logodds", "bo_wu_logodds", "twin a (updated GARCH)",
                            "twin b (frozen GARCH)", "twin c (constant t)"]):
        mns = np.array([PS[bk == b, c].mean() for b in range(5)])
        flat[nm] = (mns.max() - mns.min()) / PS[:, c].std()
        say(f"  {nm:24s} quintile means {np.round(mns, 2)}  spread/sd {flat[nm]:.3f}")
    ok = flat["ms_logodds"] <= flat["bo_w_logodds"] + 0.10
    say(f"  I4 (ms spread <= bo_w spread + 0.10): {'OK' if ok else 'FAILED'}")
    if not ok:
        fails.append("I4 comparability")
    kk = PS[:, 7]
    say("  ms_logodds mean by K chosen: " + ", ".join(f"K={K}: {PS[kk == K, 0].mean():+.2f} ({np.mean(kk == K):.0%})"
                                                       for K in (1, 2, 3) if np.any(kk == K)))

    # mechanism: single-feature TS-AUC on the real online segments (every 3rd step)
    t0 = time.time()

    def feats(s):
        h, o = s[1], s[2]
        R = run_series_ms(h, o, 3)
        H = VS.fit_history(np.asarray(h, dtype=np.float64))
        Tw = twins(H, online_resid(H, o), 3)
        Bo = BO.run_series_bocpd(h, o, 3)
        V = VS.run_series_vol(h, o)[::3]
        Kf = F.run_series(h, o)[::3]
        return np.column_stack([R, Tw["a"][:, 0], Tw["b"][:, 0], Tw["c"][:, 0],
                                Bo[:, BO.FEATURE_NAMES.index("bo_w_logodds")],
                                Bo[:, BO.FEATURE_NAMES.index("bo_wu_logodds")],
                                V[:, VS.FEATURE_NAMES.index("vol_lr_full")],
                                Kf[:, F.FEATURE_NAMES.index("glrmax_e10sq")]])

    MATS = par(delayed(feats)(s) for s in S)
    E = np.concatenate(MATS)
    step = np.concatenate([np.arange(0, len(s[2]), 3) for s in S])
    sid = np.repeat(np.arange(n_series), [len(m) for m in MATS])
    y = ((tau[sid] >= 0) & (step >= tau[sid])).astype(np.int8)
    names = FEATURE_NAMES + ["twin_a_logodds (updated GARCH)", "twin_b_logodds (frozen GARCH)",
                             "twin_c_logodds (constant t)", "bo_w_logodds", "bo_wu_logodds", "vol_lr_full",
                             "glrmax_e10sq"]
    buckets = ((0, 100), (100, 300), (300, 700), (700, 10000))
    say(f"\n[mechanism] single-feature TS-AUC on the real online segments ({n_series} series, every 3rd step, "
        f"{time.time() - t0:.0f}s); 0.5 = no signal")
    say(f"  {'feature':32s} {'all':>7s}" + "".join(f"  {lo:>4d}-{hi if hi < 10000 else '':<4}" for lo, hi in buckets))
    for c, nm in enumerate(names):
        if c < 4:
            continue   # per-series constants carry no per-step ranking on their own
        x = E[:, c]
        r = [ts_auc(x, y, step)] + [ts_auc(x[(step >= lo) & (step < hi)], y[(step >= lo) & (step < hi)],
                                           step[(step >= lo) & (step < hi)]) for lo, hi in buckets]
        say(f"  {nm:32s} " + "  ".join(f"{v:7.4f}" for v in r))
    sub = np.random.default_rng(1).choice(len(E), min(200000, len(E)), replace=False)
    j = FEATURE_NAMES.index("ms_logodds")
    say("  rank correlation of ms_logodds with: " + ", ".join(
        f"{names[c].split(' ')[0]} {spearmanr(E[sub, j], E[sub, c])[0]:.2f}" for c in range(N_MS, len(names))))
    say(f"\nINSTRUMENT: {'OK' if not fails else 'FAILED: ' + ', '.join(fails)}   (total {time.time() - T0:.0f}s)")
    say("Decision: the paired cv.py run decides (docstring / HANDOFF section 45); this report is the mechanism check.")
    log.close()


if __name__ == "__main__":
    import sys
    _selftest(offset=int(sys.argv[1]) if len(sys.argv) > 1 else 5000,
              n_series=int(sys.argv[2]) if len(sys.argv) > 2 else 2000)
