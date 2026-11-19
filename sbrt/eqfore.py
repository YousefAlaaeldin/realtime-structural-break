"""Frozen variance forecasts from a history EQUATION (HANDOFF 39 R3 and 42): the F2 and F3l columns for cv.py --eqf.

Same maths as diagnostics/equation_check.py Part 3 (the gate that passed R3 for F2 and F3l); checked equal by
`python -m sbrt.eqfore test`. Both columns have the form of volstate's vol_lr_full, with a different forecast:

    r_F(n) = log((sum_{t<n} q_t + 1e-3 n) / n) - log vbar - yhat_F(n),   q = min(e^2, 25), n = online points so far

  F1 (= vol_lr_full, already in the models): volstate's frozen GARCH path to the slow level.
  eq_f2_lr   F2: per-series multi-scale variance equation  q_{t+1} = c + sum_h b_h EWMA_h(q)_t, h = 5 / 50 / 500,
             non-negative least squares on the history, iterated forward on its own expectations (never updated with
             online data); yhat_F2(n) = log(mean of the first n path values / vbar).
  eq_f3l_lr  F3l: an equation LEARNED ACROSS SERIES: ridge from 24 history-end summaries + F1's and F2's own
             forecasts to log(mean q over the next k points / vbar), one ridge per k in HZ, trained on pseudo-splits
             INSIDE training histories (split at 50 / 65 / 80% of each history: no online data, no labels);
             yhat_F3l(n) interpolated in log n between the HZ horizons (constant beyond 1000).

Everything that depends only on a series is cached once (`series_parts`, `pseudo_rows`); only the ridge depends on
the CV fold (cv.py trains it on the fold's ORIGINAL training histories). Causal: the forecasts use the history only,
and the running mean at row n uses the first n online points only.

    python -m sbrt.eqfore test        # parity with diagnostics/equation_check.py, causality, timing (CPU, ~1 min)
"""
import math
import os
import sys
import time

import numpy as np
from numba import njit
from scipy.optimize import nnls

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sbrt import features as F    # noqa: E402
from sbrt import volstate as VS   # noqa: E402

VERSION = 1
AR = VS.AR
QCAP = VS.QCAP
HAR_HL = np.array([5.0, 50.0, 500.0])        # F2
EW_HL = (5.0, 20.0, 50.0, 200.0, 1000.0)     # F3 summaries
HZ = np.array([1, 3, 10, 30, 100, 300, 1000])
LOG_HZ = np.log(HZ)
PSPLITS = (0.5, 0.65, 0.8)                   # F3 pseudo-split points inside each training history
MIN_FIT = 400                                # shortest history part that gets a pseudo-split
PATH_MIN = 1000                              # forecast paths are at least this long (F3 horizon features)
LAMS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
N_X = 26                                     # 24 summaries + F1's and F2's forecasts at the horizon
FEATURE_NAMES = ["eq_f2_lr", "eq_f3l_lr"]
N_EQ = len(FEATURE_NAMES)


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


def continuation(H, x):
    """z and the clipped AR residual e of the points x that follow the fitted history (same units)."""
    P = H["P"]
    zx = (np.asarray(x, dtype=np.float64) - P[F.P_MU]) / P[F.P_SD]
    zz = np.r_[H["z"], zx]
    nh = len(H["z"])
    Xo = np.column_stack([zz[nh - 1 - i: len(zz) - 1 - i] for i in range(AR)])
    e = np.clip((zx - Xo @ H["coef"]) / H["se"], -10.0, 10.0)
    return zx, e


def frozen(h, x):
    """One (history, continuation): q of the continuation, the F1 / F2 frozen paths (at least PATH_MIN long) and
    the 24 history-end summaries."""
    H = VS.fit_history(np.asarray(h, dtype=np.float64))
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
    """F3 input row for horizon k: the summaries + F1's and F2's forecasts of log mean q over k points."""
    return np.r_[R["base"], math.log(R["F1"][:k].mean() / R["vbar"]), math.log(R["F2"][:k].mean() / R["vbar"])]


def log_mean_q(R, n):
    """log((sum of the first n q + 1e-3 n) / n) - log vbar, for n = 1..len(q)."""
    cq = np.cumsum(R["q"][:n])
    k = np.arange(1, len(cq) + 1)
    return np.log((cq + 1e-3 * k) / k) - math.log(R["vbar"])


def series_parts(h, o, sub=1):
    """Fold-independent pieces of one series at the kept rows (every `sub`-th online step):
    lq (running log mean q vs vbar), the eq_f2_lr column, and XH (len(HZ), N_X) = the F3 inputs per horizon."""
    R = frozen(h, o)
    T = len(o)
    lq = log_mean_q(R, T)
    y2 = np.log(np.cumsum(R["F2"][:T]) / np.arange(1, T + 1) / R["vbar"])
    st = np.arange(0, T, sub)
    return lq[st], (lq - y2)[st], np.array([horizon_rows(R, k) for k in HZ])


def pseudo_rows(h):
    """F3 training rows from one history: split it at PSPLITS, treat the rest as the 'online' continuation.
    Returns XP (m, len(HZ), N_X) and YP (m, len(HZ)) with NaN where the continuation is shorter than the horizon."""
    nh = len(h)
    XP, YP = [], []
    for fr in PSPLITS:
        sp = int(fr * nh)
        if sp >= MIN_FIT and nh - sp >= 1:
            R = frozen(h[:sp], h[sp:])
            XP.append([horizon_rows(R, k) for k in HZ])
            YP.append([log_mean_q(R, k)[-1] if len(R["q"]) >= k else np.nan for k in HZ])
    if not XP:
        return np.zeros((0, len(HZ), N_X)), np.zeros((0, len(HZ)))
    return np.array(XP), np.array(YP)


def ridge_fit(X, y):
    """Ridge on standardised columns; the penalty is picked on the last quarter of the rows, then refitted on all.
    Returns the parameters (m0, mu, sd, b) of  m0 + ((x - mu) / sd) @ b  (equation_check.ridge_fit's predictor)."""
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
    return m0, mu, sd, b


def fit_f3l(XP, YP):
    """One ridge per horizon on the pseudo-split rows (rows whose continuation reaches that horizon)."""
    out = []
    for j in range(len(HZ)):
        mk = np.isfinite(YP[:, j])
        out.append(ridge_fit(XP[mk, j], YP[mk, j]))
    return out


def predict_f3l(models, XH):
    """(n_series, len(HZ)) forecasts of log(mean q over the first k online points / vbar)."""
    return np.column_stack([m0 + ((XH[:, j] - mu) / sd) @ b for j, (m0, mu, sd, b) in enumerate(models)])


def f3l_column(lq, counts, Pred, sub=1):
    """eq_f3l_lr at the kept rows: lq minus the F3l forecast interpolated in log n (n = row step + 1)."""
    out = np.empty(len(lq))
    o = 0
    for i, c in enumerate(counts):
        n = np.arange(c) * sub + 1.0
        out[o:o + c] = lq[o:o + c] - np.interp(np.log(n), LOG_HZ, Pred[i])
        o += c
    return out


# ---------------------------------------------------------------------------------------------
def selftest(n_fit=600, n_eval=300, offset=5000, threads=8):
    from joblib import Parallel, delayed
    from sbrt.data import load_train
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "diagnostics"))
    import equation_check as EC   # the gate's own code (importing it changes the working directory to the project)
    S, tau = load_train()
    par = Parallel(n_jobs=threads, backend="threading")

    # 1. the gate's F3 training rows and ridge vs ours
    TR = S[:n_fit]
    jobs = [(s[1][:int(fr * len(s[1]))], s[1][int(fr * len(s[1])):]) for s in TR for fr in PSPLITS
            if int(fr * len(s[1])) >= MIN_FIT and len(s[1]) - int(fr * len(s[1])) >= 1]
    PR = par(delayed(EC.frozen)(a, b) for a, b in jobs)
    PS_ = par(delayed(pseudo_rows)(s[1]) for s in TR)
    XP = np.concatenate([p[0] for p in PS_])
    YP = np.concatenate([p[1] for p in PS_])
    assert len(XP) == len(PR)
    models = fit_f3l(XP, YP)
    EV_ = S[offset:offset + n_eval]
    ER = par(delayed(EC.frozen)(s[1], s[2]) for s in EV_)
    mine = par(delayed(series_parts)(s[1], s[2], 3) for s in EV_)
    worst = 0.0
    for j, k in enumerate(HZ):
        rows = [(EC.horizon_rows(R, k), EC.log_mean_q(R, k)[-1]) for R in PR if len(R["q"]) >= k]
        Xg = np.array([r[0] for r in rows])
        yg = np.array([r[1] for r in rows])
        mk = np.isfinite(YP[:, j])
        assert np.array_equal(Xg, XP[mk, j]) and np.array_equal(yg, YP[mk, j]), f"training rows differ at k={k}"
        Xe = np.array([EC.horizon_rows(R, k) for R in ER])
        assert np.array_equal(Xe, np.array([m[2][j] for m in mine])), f"eval rows differ at k={k}"
        pg = EC.ridge_fit(Xg, yg)(Xe)
        pm = predict_f3l([models[j]], Xe[:, None, :])[:, 0]
        worst = max(worst, float(np.abs(pg - pm).max()))
    assert worst == 0.0, worst
    print(f"F3 training rows ({len(XP):,} pseudo-splits from {n_fit} histories), eval rows and ridge predictions "
          f"== equation_check: OK")

    # 2. the two columns vs the gate's statistic (every 3rd row) and F1 vs vol_lr_full
    Pred = predict_f3l(models, np.array([m[2] for m in mine]))
    counts = [len(m[0]) for m in mine]
    lq = np.concatenate([m[0] for m in mine])
    f3 = f3l_column(lq, counts, Pred, 3)
    w2, w3, w1 = 0.0, 0.0, 0.0
    o = 0
    for i, (R, s) in enumerate(zip(ER, EV_)):
        T = len(R["q"])
        st = np.arange(0, T, 3)
        k = np.arange(1, T + 1)
        lqg = EC.log_mean_q(R, T)
        g2 = (lqg - np.log(np.cumsum(R["F2"][:T]) / k / R["vbar"]))[st]
        g3 = (lqg - np.interp(np.log(k), np.log(HZ), Pred[i]))[st]
        g1 = (lqg - np.log(np.cumsum(R["F1"][:T]) / k / R["vbar"]))[st]
        vol = VS.run_series_vol(s[1], s[2])[::3, VS.FEATURE_NAMES.index("vol_lr_full")]
        w2 = max(w2, float(np.abs(g2 - mine[i][1]).max()))
        w3 = max(w3, float(np.abs(g3 - f3[o:o + len(st)]).max()))
        w1 = max(w1, float(np.abs(g1 - vol).max()))
        o += len(st)
    assert w2 == 0.0 and w3 < 1e-12 and w1 < 1e-8, (w2, w3, w1)
    print(f"eq_f2_lr == gate statistic (max diff {w2:.1e}); eq_f3l_lr == gate statistic ({w3:.1e}); "
          f"F1 statistic == vol_lr_full ({w1:.1e}): OK")

    # 3. causal: perturbing online points from 60 on leaves rows 0..59 unchanged
    rng = np.random.default_rng(0)
    for s in EV_[:20]:
        h, o_ = s[1], s[2]
        if len(o_) < 90:
            continue
        o2 = o_.copy()
        o2[60:] = rng.standard_normal(len(o_) - 60) * 3 + 5
        a, b = series_parts(h, o_, 1), series_parts(h, o2, 1)
        assert np.array_equal(a[0][:60], b[0][:60]) and np.array_equal(a[1][:60], b[1][:60]), "future changed past"
        assert np.array_equal(a[2], b[2]), "online points changed the history forecasts"
    print("causal (online points from 60 on leave rows 0-59 and the forecasts unchanged): OK")

    # 4. timing
    t0 = time.time()
    for s in EV_[:100]:
        series_parts(s[1], s[2], 3)
        pseudo_rows(s[1])
    print(f"timing: {(time.time() - t0) * 10:.0f} ms per series single-threaded (series_parts + pseudo_rows)")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        selftest()
    else:
        print(__doc__)
