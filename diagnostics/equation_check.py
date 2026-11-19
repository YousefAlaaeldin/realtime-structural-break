"""Equation check (HANDOFF section 39, owner's idea 2026-10-15): is there a data-generating EQUATION left to learn
from the history that the models do not already have?

For a noisy series the generating equation is  x_t = f(past) + g(past) * eps_t.  The models already carry
f = AR(10) (mean), g = GARCH with a frozen path to a slow level (sbrt/volstate.py, +0.0070 in the trees) and a
learned f + g + eps (the NBM, sbrt/nbm.py, +0.0049). This script asks, without building anything into cv.py,
whether a RICHER explicit equation, fitted per series or learned across series, would add something. Three parts:

Part 1  Mean equation, per series (no labels). Fit on the first 80% of each history: AR(10) vs AR(10) + a library
        of 11 nonlinear terms of the last two points (squares, |.|, signed squares, a clipped cube, thresholds,
        cross products; ridge, penalty picked on the last quarter of the fit part). Out-of-sample MSE of the AR
        residual on the last 20%. Positive control: a simulated threshold-AR series.
Part 2  One-step conditional density (no labels). Mean log score (nats per point) on the last 20% of each history,
        every model fitted on the first 80%:
          M0 Gaussian, constant variance           M1 Student-t, constant scale (df per series)
          M2 GARCH(1,1)-t (volstate's fit)          M3 extended GARCH-t, nesting M2:
             s2_{t+1} = om + a q_t + g q_t 1{e_t < 0} + b s2_t + d min(z_t^2, 25) + k EWMA200(q)_t
             (leverage, level-dependent variance, slow component), Gaussian QMLE (L-BFGS-B) started at M2's fit
             and kept only if it improves the in-sample fit; Student-t df per series
          M4 = M3 + the Part-1 nonlinear mean
          NBM-full (data/nbm_v1.pt, trained on series 0-4499, so held out here), NBM-scale (the NBM's scale and
          its one global df, location removed), NBM-scale* (NBM scale with df and scale multiplier refitted per
          series: the fair comparison with M2), NBM-loc (the NBM's location on top of M2): splits the NBM's gain
          into location vs scale vs tail shape.
        Positive controls: a simulated threshold-AR series (Part 1) and simulated leverage-GARCH series (M3 must
        beat M2; first smoke 2026-10-15: +0.007 to +0.016 nats/point, leverage coefficient recovered).
Part 3  Frozen forecast of the online variance: the main test. The obstacle (HANDOFF 15.2) is that no-break series
        wander about as much as a break moves them; an equation helps only if it PREDICTS that wandering from the
        history, without looking at online data. For each forecast F the statistic is
            r_F(n) = log((sum_{t<n} q_t + 1e-3 n) / n) - log vbar - yhat_F(n),   q = min(e^2, 25)
        (with F1 this is exactly volstate's vol_lr_full; checked below). Forecasts:
          F0 whole-history level (yhat = 0)         F1 volstate's frozen path (production)
          F2 per-series multi-scale variance equation  q_{t+1} = c + sum_h b_h EWMA_h(q)_t, h = 5 / 50 / 500,
             non-negative least squares on the history, iterated forward on its own expectations (frozen)
          F3g / F3l  an equation LEARNED ACROSS SERIES (LightGBM / ridge): 24 history-end summaries (EWMAs at
             half-lives 5-1000, last-window levels, GARCH a and p, q autocorrelations, kurtosis, rho1, length, F2's
             coefficients) plus F1's and F2's own forecasts -> log mean q over the next k points, one model per
             k in 1 / 3 / 10 / 30 / 100 / 300 / 1000, trained on pseudo-splits INSIDE the histories of the training
             series (no online data, no labels), interpolated in log k.
        Reported: the spread (sd) of r_F over NO-BREAK eval series at k = 30 / 100 / 300 (bootstrap SE of the
        difference to F1), and the same-form single-feature TS-AUC on all eval series (every 3rd step), with
        5 paired series groups.

Series: F3 trains on the histories of series 0-4999; everything is evaluated on series 5000-9999 (held out from
the NBM gate model and from the choice of volstate's half-life).

STOP RULE (written 2026-10-15 before any full run; do not move it after seeing numbers):
  R1 mean equation OPEN only if the per-series library cuts out-of-sample MSE by >= 1% on average (mean
     per-series change <= -1.0%); the pooled GBM correction in the 2026-10-13 bake-off gave -0.05%.
  R2 one-step equation OPEN only if M3 or M4 beats the better NBM row (NBM-full or NBM-scale*) by >= +0.002
     nats/point AND > 2 SE (otherwise it is dominated by streams the trees already have; 0.002 is about a tenth of
     what GARCH itself adds over a constant variance).
     (Amended after the first smoke, before any full run: the first M3, a log-linear ridge library, FAILED its
     positive control and was replaced by the nested extended GARCH; the minimum effect size was added because at
     5,000 series an SE of about 0.0001 would let a useless +0.0003 pass.)
  R3 frozen forecast (the main one) PASSES for a candidate F2 / F3g / F3l if
     (b) its same-form TS-AUC beats F1's by >= +0.003 with >= 4/5 groups better, and the INSTRUMENT line is OK.
     (a) its no-break spread at k = 100 vs F1 (sd and robust sd, with bootstrap SE) is REPORTED as the mechanism
         check, not required.
     (Amended after the smokes, before any full run: the first version also required (a) a spread cut of >= 0.02
     and > 2 SE. Dropped because (1) volstate showed that a pre-combined history-conditioned variance ratio adds to
     the trees (+0.0070) although they already had the raw ratio and the history constants, so a rank gain without
     a spread cut is not "already in the model"; (2) the plain sd is driven by a few outlier series (1,000-series
     smoke: F3g sd +0.023 worse, robust sd +0.003), so it should not veto a rank-based gain; (3) cheap gates
     mislead both ways (HANDOFF 35.1), the paired cv.py run is the real test.)
  If R1, R2 and R3 all fail, the equation idea is CLOSED. Otherwise only the passing part gets built (R3: new
  vol_-style columns from the better forecast; R2: a new whitened stream into the bocpd posterior; R1: the library
  mean before the residual) and goes to a paired cv.py run with the usual rule (>= +0.003, > 2 SE, >= 4/5 folds).
  The run counts only if the instrument checks hold: F1's spread clearly below F0's (earlier 0.335 -> 0.251 at
  k = 100 on other series), F1's TS-AUC above F0's, F1's r equal to vol_lr_full, the history-only controls
  (-yhat alone) within 0.5 +- 0.015 (about 2 SE at 5,000 series), both positive controls OK (library >= 5% MSE cut; M3 - M2 >= +0.002 on leverage data),
  and the numba NBM loop equal to NbmStream.

Run:  python diagnostics/equation_check.py                                   (full; CPU only, a few minutes)
      python diagnostics/equation_check.py --n-train 300 --n-eval 300        (smoke; numbers meaningless)
-> cv_out/equation_check.txt (smoke runs: cv_out/equation_check_smoke.txt)
"""
import argparse
import math
import os
import sys
import time

import numpy as np
from numba import njit
from scipy.optimize import nnls
from scipy.special import gammaln

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
os.chdir(HERE)
from sbrt.data import load_train     # noqa: E402
from sbrt import features as F       # noqa: E402
from sbrt import nbm as NBM          # noqa: E402
from sbrt import volstate as VS      # noqa: E402
from sbrt.metric import ts_auc       # noqa: E402

AR = VS.AR
QCAP = VS.QCAP
SPLIT = 0.8                                  # Parts 1-2: fit on the first 80% of the history, score the rest
BURN = 50                                    # first rows of a fit part not used as regression targets
HAR_HL = np.array([5.0, 50.0, 500.0])        # F2
EW_HL = (5.0, 20.0, 50.0, 200.0, 1000.0)     # F3 summaries
HZ = np.array([1, 3, 10, 30, 100, 300, 1000])
PSPLITS = (0.5, 0.65, 0.8)                   # F3 pseudo-split points inside each training history
MIN_FIT = 400                                # shortest history part that gets fitted
PATH_MIN = 1000                              # forecast paths are at least this long (F3 horizon features)
LAMS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
NU_GRID = np.array([2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 12.0, 20.0, 50.0, 200.0])
KS = (30, 100, 300)
BUCKETS = ((0, 100), (100, 300), (300, 700), (700, 10000))
BASE_NAMES = ["lew5", "lew20", "lew50", "lew200", "lew1000", "lw50", "lw200", "lw500", "garch_a", "garch_p",
              "ls2T", "lL", "logn", "kurt", "rho1", "coef0", "logse", "acf1", "acf10", "acf50",
              "har_c", "har_b5", "har_b50", "har_b500"]
MEAN_LIB = ["z1^2", "z1|z1|", "z1^3", "|z1|", "e1^2", "|e1|", "e1e2", "z1z2", "z1+", "e1+", "e1|e1|"]

_row = NBM._row
_gru_step = NBM._gru_step
CIN = NBM.CIN


# ---------------------------------------------------------------------------------------------
# numba helpers
@njit(cache=True, nogil=True)
def _ewma_all(q, hl, init):
    """EWMA state after each point (half-life hl, started at init)."""
    lam = 0.5 ** (1.0 / hl)
    out = np.empty(len(q))
    s = init
    for t in range(len(q)):
        s = lam * s + (1.0 - lam) * q[t]
        out[t] = s
    return out


@njit(cache=True, nogil=True)
def _har_path(c, b, E0, hl, K):
    """Frozen K-step forecast of q from q_{t+1} = c + sum_i b_i EWMA_i: the EWMAs are fed their own forecasts."""
    lam = np.empty(len(hl))
    for i in range(len(hl)):
        lam[i] = 0.5 ** (1.0 / hl[i])
    E = E0.copy()
    f = np.empty(K)
    for k in range(K):
        v = c
        for i in range(len(b)):
            v += b[i] * E[i]
        f[k] = v
        for i in range(len(b)):
            E[i] = lam[i] * E[i] + (1.0 - lam[i]) * v
    return f


@njit(cache=True, nogil=True)
def _t_ll(u, s, nu):
    c = math.lgamma(0.5 * (nu + 1.0)) - math.lgamma(0.5 * nu) - 0.5 * math.log(nu * math.pi) - math.log(s)
    tot = 0.0
    for x in u:
        r = x / s
        tot += c - 0.5 * (nu + 1.0) * math.log1p(r * r / nu)
    return tot


@njit(cache=True, nogil=True)
def _fit_t(u, nus):
    """Student-t scale and df by maximum likelihood (EM for the scale on a df grid). Returns (scale, df)."""
    n = len(u)
    m2 = 0.0
    for x in u:
        m2 += x * x
    m2 = max(m2 / n, 1e-12)
    best, bs, bnu = -1e300, math.sqrt(m2), nus[-1]
    for nu in nus:
        s2 = m2 * (nu - 2.0) / nu if nu > 2.0 else m2
        for it in range(30):
            acc = 0.0
            for x in u:
                acc += (nu + 1.0) / (nu + x * x / s2) * x * x
            s2 = max(acc / n, 1e-12)
        ll = _t_ll(u, math.sqrt(s2), nu)
        if ll > best:
            best, bs, bnu = ll, math.sqrt(s2), nu
    return bs, bnu


@njit(cache=True, nogil=True)
def _nbm_preds(zt, et, ctx, Wi, bi, Wih, bih, Whh, bhh, Wo1, bo1, Wo2, bo2):
    """The NBM GRU over a residual sequence (same calls as nbm._init_core): pm[p], pls[p] = location and log scale
    of the prediction for et[p], made from points before p."""
    Hn = Wi.shape[0]
    h = np.zeros(Hn)
    a = np.zeros(Hn)
    gi = np.zeros(3 * Hn)
    gh = np.zeros(3 * Hn)
    hid = np.zeros(Hn)
    mo = np.zeros(2)
    xin = np.zeros(CIN)
    n = len(et)
    pm = np.zeros(n)
    pls = np.zeros(n)
    mp, lp, ep = 0.0, 0.0, 0.0
    for p in range(n):
        pm[p] = mp
        pls[p] = lp
        _row(xin, zt[p], et[p], ep, ctx)
        _gru_step(xin, h, Wi, bi, Wih, bih, Whh, bhh, Wo1, bo1, Wo2, bo2, a, gi, gh, hid, mo)
        mp, lp, ep = mo[0], mo[1], et[p]
    return pm, pls


# ---------------------------------------------------------------------------------------------
# per-series pieces
def fit_part(h):
    """AR(10) + GARCH + slow level on a history (part), exactly volstate's fit."""
    return VS.fit_history(np.asarray(h, dtype=np.float64))


def continuation(H, x):
    """z and the clipped AR residual e of the points x that follow the fitted part (same units as the part)."""
    P = H["P"]
    zx = (np.asarray(x, dtype=np.float64) - P[F.P_MU]) / P[F.P_SD]
    zz = np.r_[H["z"], zx]
    nh = len(H["z"])
    Xo = np.column_stack([zz[nh - 1 - i: len(zz) - 1 - i] for i in range(AR)])
    e = np.clip((zx - Xo @ H["coef"]) / H["se"], -10.0, 10.0)
    return zx, e


def ridge_fit(X, y):
    """Ridge on standardised columns; the penalty is picked on the last quarter of the rows, then refitted on all.
    Returns a predict function."""
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    c = int(0.75 * len(y))

    def solve(Zs, ys, lam):
        A = Zs.T @ Zs + lam * len(ys) * np.eye(Zs.shape[1])
        return np.linalg.solve(A, Zs.T @ (ys - ys.mean())), ys.mean()

    best, bl = np.inf, LAMS[-1]
    for lam in LAMS:
        b, m0 = solve(Z[:c], y[:c], lam)
        err = float(np.mean((y[c:] - m0 - Z[c:] @ b) ** 2))
        if err < best:
            best, bl = err, lam
    b, m0 = solve(Z, y, bl)
    return lambda Xn: m0 + ((Xn - mu) / sd) @ b


@njit(cache=True, nogil=True)
def _xgarch_path(th, q, neg, lev2, slow, vbar):
    """Extended GARCH one-step variances: s2_{t+1} = om + a q_t + g q_t 1{e_t < 0} + b s2_t + d lev2_t + k slow_t
    (lev2 = min(z_t^2, 25): level-dependent variance; slow = EWMA of q with half-life 200: slow component)."""
    om, a, g, b, d, k = th[0], th[1], th[2], th[3], th[4], th[5]
    s2 = np.empty(len(q))
    s = vbar
    for t in range(len(q)):
        s2[t] = s
        s = om + a * q[t] + g * q[t] * neg[t] + b * s + d * lev2[t] + k * slow[t]
        if s < 1e-4:
            s = 1e-4
    return s2


@njit(cache=True, nogil=True)
def _xgarch_nll(th, q, neg, lev2, slow, vbar, n_fit):
    s2 = _xgarch_path(th, q, neg, lev2, slow, vbar)
    tot = 0.0
    for t in range(n_fit):
        tot += math.log(s2[t]) + q[t] / s2[t]
    if th[1] + 0.5 * th[2] + th[3] + th[5] > 0.999:      # outside the stationary region
        tot += 1e6 * (th[1] + 0.5 * th[2] + th[3] + th[5] - 0.999)
    return tot


def xgarch_fit(q, neg, lev2, slow, vbar, n_fit, a0, p0):
    """Gaussian QMLE of the extended GARCH on the first n_fit points, started at the production GARCH fit (so the
    in-sample fit is never worse than M2's); L-BFGS-B with numerical gradients. Returns the parameter vector."""
    from scipy.optimize import minimize
    th0 = np.array([vbar * (1.0 - p0), a0, 0.0, p0 - a0, 0.0, 0.0])   # = M2 exactly (flat when p0 = 0)
    f0 = _xgarch_nll(th0, q, neg, lev2, slow, vbar, n_fit)
    bounds = [(1e-6, 10.0), (0.0, 0.5), (0.0, 0.5), (0.0, 0.999), (0.0, 0.5), (0.0, 0.5)]
    r = minimize(_xgarch_nll, th0, args=(q, neg, lev2, slow, vbar, n_fit), method="L-BFGS-B", bounds=bounds,
                 options=dict(maxiter=200))
    return r.x if r.fun <= f0 else th0


def t_lp(x, m, s, nu):
    r = (x - m) / s
    return gammaln(0.5 * (nu + 1)) - gammaln(0.5 * nu) - 0.5 * np.log(nu * np.pi) - np.log(s) \
        - 0.5 * (nu + 1) * np.log1p(r * r / nu)


def mean_library(zt, et):
    """Nonlinear mean-library rows for predicting et[j] from points before j (rows j >= 2 are valid)."""
    z1, z2 = np.r_[0.0, zt[:-1]], np.r_[0.0, 0.0, zt[:-2]]
    e1, e2 = np.r_[0.0, et[:-1]], np.r_[0.0, 0.0, et[:-2]]
    return np.column_stack([z1 * z1, z1 * np.abs(z1), np.clip(z1, -4, 4) ** 3, np.abs(z1), np.minimum(e1 * e1, QCAP),
                            np.abs(e1), np.clip(e1 * e2, -25, 25), z1 * z2, np.maximum(z1, 0), np.maximum(e1, 0),
                            np.clip(e1 * np.abs(e1), -25, 25)])


def onestep(h, M=None, nu_nbm=None, nus=NU_GRID):
    """Parts 1-2 for one history: fit on the first 80%, score the last 20%. Returns a dict of per-series numbers."""
    s = int(SPLIT * len(h))
    H = fit_part(h[:s])
    zx, ec = continuation(H, h[s:])
    zt = np.r_[H["z"][AR:], zx]
    et = np.r_[H["eh"], ec]
    m = len(H["eh"])
    tr = np.arange(max(2, BURN), m)
    te = np.arange(m, len(et))
    Xm = mean_library(zt, et)
    q = np.minimum(et * et, QCAP)
    out = {}
    # Part 1: mean equation
    base = float(et[tr].mean())
    mse_ar = float(np.mean((et[te] - base) ** 2))
    fm = ridge_fit(Xm[tr], et[tr])
    loc = fm(Xm)
    out["mse_pct"] = 100.0 * (float(np.mean((et[te] - loc[te]) ** 2)) / max(mse_ar, 1e-12) - 1.0)
    # Part 2: one-step densities
    s2c = float(np.mean(et[tr] ** 2))
    out["M0"] = float(np.mean(-0.5 * np.log(2 * np.pi * s2c) - 0.5 * et[te] ** 2 / s2c))
    sc1, nu1 = _fit_t(et[tr], nus)
    out["M1"] = float(np.mean(t_lp(et[te], 0.0, sc1, nu1)))
    s2g = VS._filter(q, H["vbar"], H["a"], H["p"] - H["a"])[0]
    sdg = np.sqrt(s2g)
    sc2, nu2 = _fit_t(et[tr] / sdg[tr], nus)
    out["M2"] = float(np.mean(t_lp(et[te], 0.0, sc2 * sdg[te], nu2)))
    neg = (et < 0).astype(np.float64)
    lev2 = np.minimum(zt * zt, QCAP)
    slow = _ewma_all(q, VS.HL_SLOW, H["vbar"])
    th = xgarch_fit(q, neg, lev2, slow, H["vbar"], m, H["a"], H["p"])
    sdv = np.sqrt(_xgarch_path(th, q, neg, lev2, slow, H["vbar"]))
    out["xg"] = th
    sc3, nu3 = _fit_t(et[tr] / sdv[tr], nus)
    out["M3"] = float(np.mean(t_lp(et[te], 0.0, sc3 * sdv[te], nu3)))
    sc4, nu4 = _fit_t((et[tr] - loc[tr]) / sdv[tr], nus)
    out["M4"] = float(np.mean(t_lp(et[te], loc[te], sc4 * sdv[te], nu4)))
    if M is not None:
        ctx = NBM._ctx(H["z"], H["coef"], H["se"])
        pm, pls = _nbm_preds(np.ascontiguousarray(zt), np.ascontiguousarray(et), ctx, *M)
        out["NBM-full"] = float(np.mean(t_lp(et[te], pm[te], np.exp(pls[te]), nu_nbm)))
        out["NBM-scale"] = float(np.mean(t_lp(et[te], 0.0, np.exp(pls[te]), nu_nbm)))
        sn = np.exp(pls)
        scn, nun = _fit_t(et[tr] / sn[tr], nus)
        out["NBM-scale*"] = float(np.mean(t_lp(et[te], 0.0, scn * sn[te], nun)))
        out["NBM-loc"] = float(np.mean(t_lp(et[te], pm[te], sc2 * sdg[te], nu2)))
    out["n_test"] = len(te)
    return out


def har_fit(qh, vbar):
    """F2: q_{t+1} = c + sum_h b_h EWMA_h(q)_t by non-negative least squares; persistence capped at 0.995."""
    m = len(qh)
    E = np.column_stack([_ewma_all(qh, hl, vbar) for hl in HAR_HL])
    t0 = min(100, m // 4)
    X = np.column_stack([np.ones(m - 1 - t0), E[t0:m - 1]])
    y = qh[t0 + 1:]
    coef, _ = nnls(X, y)
    c, b = float(coef[0]), coef[1:].copy()
    sb = float(b.sum())
    if sb > 0.995:
        b *= 0.995 / sb
        c = float(y.mean()) * (1.0 - 0.995)    # keep the long-run level at the sample mean
    return c, b, np.ascontiguousarray(E[-1])


def frozen(h, x):
    """Part 3 for one (history part, continuation): q of the continuation, the F1 / F2 frozen paths (at least
    PATH_MIN long) and the history-end summaries."""
    H = fit_part(h)
    _, e = continuation(H, x)
    q = np.minimum(e * e, QCAP)
    vbar, qh, eh = H["vbar"], H["qh"], H["eh"]
    K = max(len(x), PATH_MIN)
    F1 = H["L"] + H["p"] ** np.arange(K) * (H["s2T1"] - H["L"])
    c, b, Eend = har_fit(qh, vbar)
    F2 = np.maximum(_har_path(c, b, Eend, HAR_HL, K), 1e-6)
    qc = qh - qh.mean()
    den = max(float(qc @ qc), 1e-12)
    z = H["z"]
    zc = z - z.mean()
    f = [math.log(_ewma_all(qh, hl, vbar)[-1] / vbar + 1e-6) for hl in EW_HL]
    f += [math.log(float(qh[-w:].mean()) / vbar + 1e-6) for w in (50, 200, 500)]
    f += [H["a"], H["p"], math.log(H["s2T1"] / H["L"]), math.log(H["L"] / vbar), math.log(len(h)),
          float(np.mean(eh ** 4) / max(np.mean(eh ** 2) ** 2, 1e-12)),
          float(zc[:-1] @ zc[1:]) / max(float(zc @ zc), 1e-12), float(H["coef"][0]), math.log(H["se"])]
    f += [float(qc[:-lag] @ qc[lag:]) / den for lag in (1, 10, 50)]
    f += [c, float(b[0]), float(b[1]), float(b[2])]
    return dict(q=q, vbar=vbar, F1=F1, F2=F2, base=np.array(f))


def horizon_rows(R, k):
    """F3 feature row for horizon k: base summaries + F1's and F2's forecasts of log mean q over k points."""
    return np.r_[R["base"], math.log(R["F1"][:k].mean() / R["vbar"]), math.log(R["F2"][:k].mean() / R["vbar"])]


def log_mean_q(R, n):
    """log((sum of the first n q + 1e-3 n) / n) - log vbar, for n = 1..len(q)."""
    cq = np.cumsum(R["q"][:n])
    k = np.arange(1, len(cq) + 1)
    return np.log((cq + 1e-3 * k) / k) - math.log(R["vbar"])


# ---------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-from", type=int, default=0)
    ap.add_argument("--n-train", type=int, default=5000)
    ap.add_argument("--eval-from", type=int, default=5000)
    ap.add_argument("--n-eval", type=int, default=5000)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--boot", type=int, default=500)
    args = ap.parse_args()
    from joblib import Parallel, delayed
    import lightgbm as lgb
    import torch

    smoke = args.n_train < 5000 or args.n_eval < 5000
    os.makedirs("cv_out", exist_ok=True)
    path = os.path.join("cv_out", "equation_check_smoke.txt" if smoke else "equation_check.txt")
    log = open(path, "w", encoding="utf-8")

    def say(s=""):
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    par = Parallel(n_jobs=args.threads, backend="threading")
    T0 = time.time()
    say(f"equation check: settings {vars(args)}" + ("  [SMOKE: numbers meaningless]" if smoke else ""))
    S, tau = load_train()
    TR = S[args.train_from:args.train_from + args.n_train]
    EV_ = S[args.eval_from:args.eval_from + args.n_eval]
    tau_e = tau[args.eval_from:args.eval_from + args.n_eval]
    assert args.eval_from >= args.train_from + args.n_train or args.train_from >= args.eval_from + args.n_eval, \
        "train and eval series overlap"
    state = torch.load(NBM.WEIGHTS, map_location="cpu")
    M = NBM.model_arrays(state)
    nu_nbm = 2.0 + float(torch.exp(state["lnu"]).clamp(0.1, 60.0))
    if args.eval_from < 5000:
        say("WARNING: eval series overlap the NBM gate model's training series 0-4999; NBM rows are optimistic")

    # ---- instrument checks -----------------------------------------------------------------
    rng = np.random.default_rng(0)
    n = 3000
    zs = np.zeros(n)
    for t in range(1, n):
        zs[t] = (0.6 if zs[t - 1] > 0 else -0.4) * zs[t - 1] + rng.standard_normal()
    zs = (zs - zs.mean()) / zs.std()
    inst = []   # failed instrument checks
    ctrl = onestep(zs)["mse_pct"]
    if ctrl > -5:
        inst.append("library positive control")
    say(f"[check] library positive control (threshold AR, the |z| term carries ~9% of the variance): "
        f"out-of-sample MSE change {ctrl:+.1f}%  -> {'OK' if ctrl <= -5 else 'FAILED'}")
    gains = []
    for rep in range(4):   # leverage (GJR) variance: M3 has the leverage term, M2's GARCH is symmetric
        x = np.zeros(n)
        s2 = 1.0
        for t in range(n):
            x[t] = math.sqrt(s2) * rng.standard_normal()
            s2 = 0.05 + 0.02 * x[t] ** 2 + 0.15 * x[t] ** 2 * (x[t] < 0) + 0.85 * s2
        o = onestep(x)
        gains.append(o["M3"] - o["M2"])
    say(f"[check] variance-equation positive control (simulated leverage GARCH, 4 series): M3 - M2 log score "
        f"{np.mean(gains):+.4f} nats/point (per series {np.round(gains, 4)})  -> "
        f"{'OK' if np.mean(gains) >= 0.002 else 'WEAK: R2 cannot be trusted'}")
    if np.mean(gains) < 0.002:
        inst.append("variance-equation positive control")
    for i in range(3):
        _, h, o = EV_[i]
        H = fit_part(h)
        pm, pls = _nbm_preds(np.ascontiguousarray(H["z"][AR:]), np.ascontiguousarray(H["eh"]),
                             NBM._ctx(H["z"], H["coef"], H["se"]), *M)
        v = np.sort(((H["eh"] - pm) * np.exp(-pls))[NBM.LOSS_FROM + 1:])
        ref = NBM.NbmStream(h, M, (1.0, 2.0, 1.0)).ref
        assert len(v) == len(ref) and np.allclose(v, ref, atol=1e-8), "numba NBM loop != NbmStream"
        R = frozen(h, o)
        mine = log_mean_q(R, len(o)) - np.log(np.cumsum(R["F1"][:len(o)]) / np.arange(1, len(o) + 1) / R["vbar"])
        vol = VS.run_series_vol(h, o)[:, VS.FEATURE_NAMES.index("vol_lr_full")]
        assert np.allclose(mine, vol, atol=1e-8), "F1 statistic != vol_lr_full"
    say("[check] numba NBM loop == NbmStream history stream; F1 statistic == vol_lr_full: OK")

    # ---- Parts 1-2 -------------------------------------------------------------------------------
    t0 = time.time()
    OS = par(delayed(onestep)(s[1], M, nu_nbm) for s in EV_)
    say(f"\nParts 1-2 on {len(OS):,} eval histories (fit first {SPLIT:.0%}, score the rest; "
        f"{sum(o['n_test'] for o in OS):,} test points) in {time.time() - t0:.0f}s")
    pct = np.array([o["mse_pct"] for o in OS])
    say("\nPART 1  mean equation: AR(10) + 11-term nonlinear library (per-series ridge) vs AR(10), out of sample")
    say(f"  MSE change per series: mean {pct.mean():+.2f}% (SE {pct.std(ddof=1) / math.sqrt(len(pct)):.2f}), median "
        f"{np.median(pct):+.2f}%; better by >1%: {np.mean(pct < -1):.1%}, by >3%: {np.mean(pct < -3):.1%}; "
        f"worse by >1%: {np.mean(pct > 1):.1%}")
    r1 = pct.mean() <= -1.0
    say(f"  R1 (mean change <= -1.0%): {'OPEN' if r1 else 'CLOSED'}")

    names = ["M0", "M1", "M2", "M3", "M4", "NBM-full", "NBM-scale", "NBM-scale*", "NBM-loc"]
    desc = {"M0": "Gaussian, constant variance", "M1": "Student-t, constant scale (df per series)",
            "M2": "GARCH(1,1)-t (production variance equation)",
            "M3": "extended GARCH-t: + leverage, level, slow component (nests M2)",
            "M4": "M3 + nonlinear mean", "NBM-full": "NBM (location + scale + its one global df)",
            "NBM-scale": "NBM scale + global df, location removed",
            "NBM-scale*": "NBM scale, df and scale multiplier refitted per series (fair vs M2)",
            "NBM-loc": "NBM location on top of M2"}
    L = {nm: np.array([o[nm] for o in OS]) for nm in names}
    say("\nPART 2  one-step log score on the held-out 20% of each history (nats per point, mean over series; higher "
        "is better)")
    say(f"  {'model':10s} {'log score':>10s} {'vs M2':>18s} {'vs NBM-full':>18s}   what")

    def pd(a, b):
        d = L[a] - L[b]
        return d.mean(), d.std(ddof=1) / math.sqrt(len(d))

    for nm in names:
        a, sa = pd(nm, "M2")
        b, sb = pd(nm, "NBM-full")
        say(f"  {nm:10s} {L[nm].mean():10.4f} {a:+9.4f} +- {sa:.4f} {b:+9.4f} +- {sb:.4f}   {desc[nm]}")
    for a_, b_, what in (("NBM-full", "NBM-scale", "NBM location"), ("NBM-scale*", "NBM-scale", "per-series df"),
                         ("NBM-scale*", "M2", "NBM scale vs GARCH scale, both with per-series df"),
                         ("NBM-loc", "M2", "NBM location added to GARCH-t")):
        d, se = pd(a_, b_)
        say(f"  decomposition: {what:50s} {a_} - {b_} = {d:+.4f} +- {se:.4f}")
    best_nbm = max(("NBM-full", "NBM-scale*"), key=lambda nm: L[nm].mean())
    r2 = any(pd(nm, best_nbm)[0] >= 0.002 and pd(nm, best_nbm)[0] > 2 * pd(nm, best_nbm)[1] for nm in ("M3", "M4"))
    xg = np.array([o["xg"] for o in OS])
    say(f"  M3 fitted terms, share of series with the term > 0.01: leverage g {np.mean(xg[:, 2] > 0.01):.1%}, level d "
        f"{np.mean(xg[:, 4] > 0.01):.1%}, slow k {np.mean(xg[:, 5] > 0.01):.1%}; median g {np.median(xg[:, 2]):.3f}")
    say(f"  R2 (M3 or M4 beats the better NBM row, {best_nbm}, by >= +0.002 and > 2 SE): {'OPEN' if r2 else 'CLOSED'}")

    # ---- Part 3 ----------------------------------------------------------------------------------
    t0 = time.time()
    jobs = []
    for s in TR:
        nh = len(s[1])
        for fr in PSPLITS:
            sp = int(fr * nh)
            if sp >= MIN_FIT and nh - sp >= 1:
                jobs.append((s[1][:sp], s[1][sp:]))
    PR = par(delayed(frozen)(a, b) for a, b in jobs)
    say(f"\nPART 3  frozen forecast of the online variance. F3 training: {len(PR):,} pseudo-splits inside "
        f"{len(TR):,} training histories ({time.time() - t0:.0f}s)")
    t0 = time.time()
    ER = par(delayed(frozen)(s[1], s[2]) for s in EV_)
    say(f"  eval: {len(ER):,} series with their real online segments ({time.time() - t0:.0f}s)")
    lgb_p = dict(objective="regression", learning_rate=0.03, num_leaves=15, min_data_in_leaf=50, feature_fraction=0.8,
                 bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, seed=0, deterministic=True,
                 force_row_wise=True, num_threads=args.threads, verbose=-1)
    P3 = {"F3g": np.zeros((len(ER), len(HZ))), "F3l": np.zeros((len(ER), len(HZ)))}
    Xe_all = {}
    imp = np.zeros(len(BASE_NAMES) + 2)
    for j, k in enumerate(HZ):
        rows = [(horizon_rows(R, k), log_mean_q(R, k)[-1]) for R in PR if len(R["q"]) >= k]
        X = np.array([r[0] for r in rows])
        y = np.array([r[1] for r in rows])
        Xe = np.array([horizon_rows(R, k) for R in ER])
        Xe_all[k] = Xe
        m = lgb.train(lgb_p, lgb.Dataset(X, y), 400)
        P3["F3g"][:, j] = m.predict(Xe)
        g = m.feature_importance("gain")
        imp += g / max(g.sum(), 1e-12)
        P3["F3l"][:, j] = ridge_fit(X, y)(Xe)
        say(f"  horizon k = {k:4d}: {len(y):6,} training rows; target sd {y.std():.3f}; "
            f"in-sample R2 LightGBM {1 - np.var(y - m.predict(X)) / np.var(y):.3f}")
    nm_imp = BASE_NAMES + ["F1_k", "F2_k"]
    top = np.argsort(-imp)[:10]
    say("  F3g mean gain share, top 10: " + ", ".join(f"{nm_imp[i]} {100 * imp[i] / len(HZ):.0f}%" for i in top))

    models = ["F0", "F1", "F2", "F3g", "F3l"]

    def yhat(i, mdl, n):
        R = ER[i]
        k = np.arange(1, n + 1)
        if mdl == "F0":
            return np.zeros(n)
        if mdl in ("F1", "F2"):
            return np.log(np.cumsum(R[mdl][:n]) / k / R["vbar"])
        return np.interp(np.log(k), np.log(HZ), P3[mdl][i])

    # spread over no-break eval series
    say("\n  spread (sd over NO-BREAK eval series) of r_F(k) = log(mean online q over the first k points / forecast); "
        "lower = the forecast explains more of the natural wandering")
    say(f"  {'k':>5s} {'n':>5s} " + " ".join(f"{m_:>8s}" for m_ in models) + "   (second line: robust sd = IQR/1.349)")
    boot = np.random.default_rng(1)
    spread_ok = {}
    rsd = lambda v: np.subtract(*np.percentile(v, [75, 25])) / 1.349
    for k in KS:
        idx = [i for i in range(len(ER)) if tau_e[i] < 0 and len(ER[i]["q"]) >= k]
        if len(idx) < 10:
            say(f"  {k:5d}: too few no-break series")
            continue
        r = {m_: np.array([log_mean_q(ER[i], k)[-1] - yhat(i, m_, k)[-1] for i in idx]) for m_ in models}
        sd = {m_: r[m_].std() for m_ in models}
        say(f"  {k:5d} {len(idx):5d} " + " ".join(f"{sd[m_]:8.3f}" for m_ in models))
        say(f"  {'':5s} {'':5s} " + " ".join(f"{rsd(r[m_]):8.3f}" for m_ in models))
        bs = boot.integers(0, len(idx), (args.boot, len(idx)))
        for m_ in ("F0", "F2", "F3g", "F3l"):
            d = np.array([r[m_][b].std() - r["F1"][b].std() for b in bs])
            dr = np.array([rsd(r[m_][b]) - rsd(r["F1"][b]) for b in bs])
            say(f"         {m_} - F1: sd {sd[m_] - sd['F1']:+.3f} +- {d.std():.3f}; robust sd "
                f"{rsd(r[m_]) - rsd(r['F1']):+.3f} +- {dr.std():.3f}   (mean of r_{m_} {r[m_].mean():+.3f})")
            if k == 100:
                spread_ok[m_] = (sd[m_] - sd["F1"] <= -0.02) and (sd[m_] - sd["F1"] < -2 * d.std())
        if k == 100 and not sd["F1"] < sd["F0"] - 0.03:
            inst.append("F1 spread not clearly below F0 at k = 100")

    # same-form single-feature TS-AUC on all eval series (every 3rd step)
    stat = {m_: [] for m_ in models}
    steps, ys, sids = [], [], []
    for i, R in enumerate(ER):
        T = len(R["q"])
        st = np.arange(0, T, 3)
        lq = log_mean_q(R, T)
        for m_ in models:
            stat[m_].append((lq - yhat(i, m_, T))[st])
        steps.append(st)
        ys.append(((tau_e[i] >= 0) & (st >= tau_e[i])).astype(np.int8))
        sids.append(np.full(len(st), i))
    stat = {m_: np.concatenate(v) for m_, v in stat.items()}
    step, y, sid = np.concatenate(steps), np.concatenate(ys), np.concatenate(sids)
    grp = np.random.default_rng(0).integers(0, 5, len(ER))[sid]
    say("\n  same-form single-feature TS-AUC (all eval series, every 3rd step; F1 = vol_lr_full)")
    say(f"  {'model':6s} {'all':>7s}" + "".join(f"  {lo:>4d}-{hi if hi < 10000 else '':<4}" for lo, hi in BUCKETS)
        + "   paired vs F1 over 5 series groups")
    base_g = [ts_auc(stat["F1"][grp == g], y[grp == g], step[grp == g]) for g in range(5)]
    auc_ok, auc_all = {}, {}
    for m_ in models:
        x = stat[m_]
        row = [ts_auc(x, y, step)]
        for lo, hi in BUCKETS:
            mk = (step >= lo) & (step < hi)
            row.append(ts_auc(x[mk], y[mk], step[mk]) if mk.any() else float("nan"))
        d = np.array([ts_auc(x[grp == g], y[grp == g], step[grp == g]) - base_g[g] for g in range(5)])
        say(f"  {m_:6s} " + "  ".join(f"{v:7.4f}" for v in row)
            + f"   {d.mean():+.4f} +- {d.std(ddof=1) / math.sqrt(5):.4f}, {int((d > 0).sum())}/5")
        auc_ok[m_] = d.mean() >= 0.003 and (d > 0).sum() >= 4
        auc_all[m_] = row[0]
    say("  history-only control: -yhat_F alone (no online data; the history carries no label signal, HANDOFF 3, so "
        "these must sit near 0.5):")
    for m_ in models[1:]:
        v = ts_auc(-np.concatenate([yhat(i, m_, len(R["q"]))[np.arange(0, len(R["q"]), 3)] for i, R in enumerate(ER)]),
                   y, step)
        say(f"  -yhat {m_:4s} {v:.4f}")
        if abs(v - 0.5) > 0.015:
            inst.append(f"history-only control {m_} = {v:.3f} (outside 0.5 +- 0.015)")
    if not auc_all["F1"] > auc_all["F0"]:
        inst.append("F1 TS-AUC not above F0")

    say("\n  instrument: F1 spread below F0 at k = 100 and F1 TS-AUC above F0 are expected (volstate, HANDOFF 16)")
    r3 = {m_: auc_ok[m_] for m_ in ("F2", "F3g", "F3l")}
    for m_ in ("F2", "F3g", "F3l"):
        say(f"  R3 {m_}: TS-AUC (b) {'pass' if auc_ok[m_] else 'fail'} -> {'PASS' if r3[m_] else 'fail'}   "
            f"[mechanism, reported only: sd cut >= 0.02 and > 2 SE at k = 100: "
            f"{'yes' if spread_ok.get(m_, False) else 'no'}]")

    say("\nVERDICT (rule in the docstring, written before the run):")
    say("  INSTRUMENT " + ("OK" if not inst else "FAILED (" + "; ".join(inst) + "): the verdict below does not count"))
    say(f"  R1 mean equation {'OPEN' if r1 else 'closed'}; R2 one-step equation {'OPEN' if r2 else 'closed'}; "
        f"R3 frozen forecast {'PASS: ' + ', '.join(m_ for m_ in r3 if r3[m_]) if any(r3.values()) else 'fail'}")
    if not (r1 or r2 or any(r3.values())):
        say("  -> the equation idea is CLOSED (nothing an explicit equation adds that the models lack)")
    else:
        say("  -> build only the passing part, then a paired cv.py run with the usual rule")
    if smoke:
        say("  [SMOKE run: the verdict above means nothing at this size]")
    say(f"\ntotal {time.time() - T0:.0f}s -> {path}")


if __name__ == "__main__":
    main()
