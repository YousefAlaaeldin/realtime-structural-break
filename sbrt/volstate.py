"""Volatility state at the end of the history: GARCH(1,1) fit + FROZEN variance forecast path (streaming, causal).

Why (HANDOFF section 15.2): no-break series' online variance wanders by about as much as a typical break, and that
wandering is partly predictable from the END of the history (corr 0.49-0.55 with the log variance of the last
50-500 history points). The 270-feature kernel compares every online point with WHOLE-history constants, so a
series that ended its history in a calm (or wild) patch looks broken (or hides a break). Here the online residuals
are compared with what the history's volatility state predicts for each online step instead.

Separate from sbrt/features.py on purpose (same as sbrt/evidence.py): the 270-feature kernel, its caches, the GRU
input hash and main.py stay unchanged. Columns are appended only when cv.py / gru_cv.py run with --vol.

1. Residuals: the kernel's own history-fitted AR(10) residual e_t (= feature `e10`, clipped to +-10).
   q_t = min(e_t^2, 25) (squares capped so one outlier cannot dominate the fit or the statistics).
2. GARCH(1,1) on the history, variance-targeted:  s2_{t+1} = vbar (1 - a - b) + a q_t + b s2_t,   vbar = mean(q_hist).
   Gaussian quasi-likelihood, fitted by a fixed grid over persistence p = a + b and a, then one finer grid around the
   best point. Deterministic (no optimiser, no random restarts). a = b = 0 (flat variance) is one of the candidates.
3. Slow level: L = EWMA of q over the history with half-life 200 points, read at the history's end. Measured on 3,000
   series (single-feature TS-AUC of log(realized/expected online variance)): whole-history level 0.5719, GARCH path
   decaying to vbar 0.5718 (null spread at step 99 drops 0.360 -> 0.276, but the ranking does not improve: the fitted
   half-lives are short, median 8 steps), slow level L 0.5805, GARCH path decaying to L 0.5791. The persistent part of
   the volatility state is what carries the signal; the fast GARCH part is kept only as the start of the path.
4. Frozen path: s2f(k) = L + p^(k-1) (s2_{T+1} - L) for online step k = 1, 2, ... It starts at the end-of-history
   GARCH forecast and decays to the slow level; it is NEVER updated with online data, so it cannot absorb a break.
   u_t = e_t / sqrt(s2f(t+1)) is the online residual measured against that forecast.
5. Features (N_VOL = 13), all prefixed vol_:
     lvh200, lvh1000             per-series constants: log(L / vbar), log(mean q over the last 1000 history points / vbar)
     logs2T, persist             per-series constants: log(s2_{T+1} / L), p
     full, win30, win100         standardised running mean of r_t - c over all online points / the last 30 / 100,
                                 r_t = min(u_t^2, 25), c and the scale from the history's in-sample GARCH-standardised
                                 squares (batch-means long-run variance, like the kernel's _lrv)
     scanmax, scanmin            max / min over start j of the standardised segment sum (min segment 8), like glrmax/min
     lr_full, lr100              log(sum q / sum s2f) over all online points / the last 100: realized vs frozen forecast
     lr1000                      log(sum q / (n * mean of the last 1000 history q)): a second, longer horizon
     gap                         log(online-UPDATED GARCH forecast / frozen forecast) for the next point (the updated
                                 filter uses the same a, p and slow level)
   Two more outputs per step are not features but GRU channels: u_t (clipped +-8) and log(s2f(t+1)/vbar).

Self-test (includes the go/no-go oracle check):  python -m sbrt.volstate [offset]
(offset 5000 = series 5000-6999, held out from the 3,000 series the slow level's half-life was chosen on)
"""
import math

import numpy as np
from numba import njit

from sbrt import features as F

AR = F.AR
CAP = F.CAP
QCAP = 25.0
MINSEG = 8
HL_SLOW = 200.0
VERSION = 2   # bump when the kernel changes (part of the cache names)

FEATURE_NAMES = ["vol_lvh200", "vol_lvh1000", "vol_logs2T", "vol_persist",
                 "vol_full", "vol_win30", "vol_win100", "vol_scanmax", "vol_scanmin",
                 "vol_lr_full", "vol_lr100", "vol_lr1000", "vol_gap"]
N_VOL = len(FEATURE_NAMES)
N_OUT = N_VOL + 2           # + GRU channels: u, log s2f
GRU_NAMES = ["vol_u", "vol_u2m1", "vol_logs2f"]
N_GRU = len(GRU_NAMES)

P_GRID = np.array([0.5, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995, 0.998])
A_GRID = np.array([0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3])
REFINE = np.array([-0.3, -0.15, 0.0, 0.15, 0.3])
P_MAX = 0.9995


# ---------------------------------------------------------------------------------------------
@njit(cache=True, nogil=True)
def _nll(q, vbar, a, b):
    om = vbar * (1.0 - a - b)
    s2 = vbar
    nll = 0.0
    for t in range(len(q)):
        nll += math.log(s2) + q[t] / s2
        s2 = om + a * q[t] + b * s2
    return nll


@njit(cache=True, nogil=True)
def _fit(q, vbar, pg, ag, rf):
    """Grid QMLE of (a, p). Returns a, p (a = p = 0 means flat variance)."""
    best = _nll(q, vbar, 0.0, 0.0)
    ba, bp = 0.0, 0.0
    for p in pg:
        for a in ag:
            if a >= p:
                continue
            v = _nll(q, vbar, a, p - a)
            if v < best:
                best, ba, bp = v, a, p
    if bp > 0.0:
        a0, p0 = ba, bp
        for dx in rf:
            p = 1.0 - (1.0 - p0) * math.exp(dx)
            if p > P_MAX:
                p = P_MAX
            for dy in rf:
                a = a0 * math.exp(dy)
                if a >= p - 1e-3:
                    continue
                v = _nll(q, vbar, a, p - a)
                if v < best:
                    best, ba, bp = v, a, p
    return ba, bp


@njit(cache=True, nogil=True)
def _filter(q, vbar, a, b):
    """One-step in-sample variance forecasts s2[t] for q[t], and the forecast for the point after the last one."""
    om = vbar * (1.0 - a - b)
    s2 = np.empty(len(q))
    s = vbar
    for t in range(len(q)):
        s2[t] = s
        s = om + a * q[t] + b * s
    return s2, s


def fit_history(hist):
    """Everything the online statistics need from one history (also used for the GRU's burn-in channels)."""
    P, _, z = F.make_params(np.asarray(hist, dtype=np.float64))
    coef = P[F.P_AR:F.P_AR + AR]
    se = P[F.P_SE10]
    n = len(z)
    Xl = np.column_stack([z[AR - 1 - i: n - 1 - i] for i in range(AR)])
    eh = np.clip((z[AR:] - Xl @ coef) / se, -10.0, 10.0)
    qh = np.minimum(eh * eh, QCAP)
    vbar = max(float(qh.mean()), 1e-6)
    a, p = _fit(qh, vbar, P_GRID, A_GRID, REFINE)
    s2h, s2T1 = _filter(qh, vbar, a, p - a)
    rh = np.minimum(eh * eh / s2h, QCAP)
    cr = float(rh.mean())
    lrv = F._lrv(rh - cr, float(rh.var()), 1e-6)
    wl = 0.5 ** (np.arange(len(qh))[::-1] / HL_SLOW)
    L = max(float((wl * qh).sum() / wl.sum()), 1e-6)
    L1000 = max(float(qh[-1000:].mean()), 1e-6)
    return dict(P=P, z=z, coef=coef, se=se, eh=eh, qh=qh, s2h=s2h, s2T1=float(s2T1), vbar=vbar, a=float(a),
                p=float(p), cr=cr, lrv=lrv, L=L, L1000=L1000)


# parameter vector Q and state ST layouts
Q_MU, Q_SD, Q_SE, Q_A, Q_P, Q_VBAR, Q_L, Q_L1000, Q_CR, Q_INV, Q_LV200, Q_LV1000, Q_LS2T, Q_AR = range(14)
Q_LEN = Q_AR + AR
S_T, S_S2O, S_D, S_LAG = 0, 1, 2, 3
S_LEN = S_LAG + AR


def make_vol_params(hist):
    H = fit_history(hist)
    Q = np.zeros(Q_LEN)
    Q[Q_MU], Q[Q_SD], Q[Q_SE] = H["P"][F.P_MU], H["P"][F.P_SD], H["se"]
    Q[Q_A], Q[Q_P], Q[Q_VBAR], Q[Q_L], Q[Q_L1000] = H["a"], H["p"], H["vbar"], H["L"], H["L1000"]
    Q[Q_CR], Q[Q_INV] = H["cr"], 1.0 / math.sqrt(H["lrv"])
    Q[Q_LV200], Q[Q_LV1000] = math.log(H["L"] / H["vbar"]), math.log(H["L1000"] / H["vbar"])
    Q[Q_LS2T] = math.log(H["s2T1"] / H["L"])
    Q[Q_AR:Q_AR + AR] = H["coef"]
    ST = np.zeros(S_LEN)
    ST[S_S2O] = H["s2T1"]
    ST[S_D] = H["s2T1"] - H["L"]
    z = H["z"]
    for i in range(AR):
        ST[S_LAG + i] = z[-1 - i]
    return Q, ST, np.zeros((3, CAP + 1)), H


@njit(cache=True, nogil=True)
def vol_step(x, Q, C, ST, out):
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
    q = min(e * e, QCAP)
    lev = Q[Q_L]
    a = Q[Q_A]
    p = Q[Q_P]
    s2f = lev + ST[S_D]                        # frozen forecast for this point (online step t+1)
    u = e / math.sqrt(s2f)
    r = min(u * u, QCAP)
    n = t + 1
    C[0, n] = C[0, t] + r - Q[Q_CR]
    C[1, n] = C[1, t] + q
    C[2, n] = C[2, t] + s2f
    inv = Q[Q_INV]

    out[0] = Q[Q_LV200]
    out[1] = Q[Q_LV1000]
    out[2] = Q[Q_LS2T]
    out[3] = p
    out[4] = C[0, n] * inv / math.sqrt(n * 1.0)
    w = 30 if n > 30 else n
    out[5] = (C[0, n] - C[0, n - w]) * inv / math.sqrt(w * 1.0)
    w = 100 if n > 100 else n
    out[6] = (C[0, n] - C[0, n - w]) * inv / math.sqrt(w * 1.0)
    gmax = -1e9
    gmin = 1e9
    for j in range(0, n - MINSEG + 1):
        s = (C[0, n] - C[0, j]) * inv / math.sqrt((n - j) * 1.0)
        if s > gmax:
            gmax = s
        if s < gmin:
            gmin = s
    out[7] = gmax if gmax > -1e8 else 0.0
    out[8] = gmin if gmin < 1e8 else 0.0
    out[9] = math.log((C[1, n] + 1e-3 * n) / C[2, n])
    out[10] = math.log((C[1, n] - C[1, n - w] + 1e-3 * w) / (C[2, n] - C[2, n - w]))
    out[11] = math.log((C[1, n] + 1e-3 * n) / (n * Q[Q_L1000]))
    # online-updated GARCH forecast vs frozen forecast, both for the NEXT point
    s2o = lev * (1.0 - p) + a * q + (p - a) * ST[S_S2O]
    ST[S_S2O] = s2o
    ST[S_D] = ST[S_D] * p
    out[12] = math.log(s2o / (lev + ST[S_D]))
    # GRU channels (not features)
    out[13] = min(max(u, -8.0), 8.0)
    out[14] = math.log(s2f / Q[Q_VBAR])

    for i in range(AR - 1, 0, -1):
        ST[S_LAG + i] = ST[S_LAG + i - 1]
    ST[S_LAG] = z
    ST[S_T] = n * 1.0


@njit(cache=True, nogil=True)
def _vol_run(online, Q, C, ST, out):
    for i in range(len(online)):
        vol_step(online[i], Q, C, ST, out[i])


def run_series_vol(hist, online, full=False):
    """Batch matrix (T, N_VOL) for one series -- identical to streaming. full=True also returns the GRU outputs."""
    Q, ST, C, _ = make_vol_params(hist)
    online = np.asarray(online, dtype=np.float64)
    if len(online) >= CAP:
        raise ValueError(f"online segment longer than the supported {CAP - 1} points")
    out = np.zeros((len(online), N_OUT))
    _vol_run(online, Q, C, ST, out)
    return out if full else out[:, :N_VOL]


def gru_channels(hist, online, burn):
    """(burn + T, 3) channels for gru_cv.py: u, min(u^2, 25) - 1, log(s2/vbar). In the burn-in (last `burn` history
    points) s2 is the in-sample one-step GARCH forecast; online it is the frozen path (they agree at the boundary)."""
    Q, ST, C, H = make_vol_params(hist)
    online = np.asarray(online, dtype=np.float64)
    out = np.zeros((len(online), N_OUT))
    _vol_run(online, Q, C, ST, out)
    eh, s2h, vbar = H["eh"][-burn:], H["s2h"][-burn:], H["vbar"]
    uh = eh / np.sqrt(s2h)
    uo = out[:, N_VOL]   # clipped at +-8, harmless for u^2 since the cap is 25
    u = np.r_[np.clip(uh, -8.0, 8.0), uo]
    u2 = np.r_[np.minimum(uh * uh, QCAP), np.minimum(uo * uo, QCAP)] - 1.0
    ls = np.r_[np.log(s2h / vbar), out[:, N_VOL + 1]]
    return np.nan_to_num(np.column_stack([u, u2, ls])).astype(np.float32)


class VolStream:
    """One series' streaming state for use inside infer()."""

    def __init__(self, hist):
        self.Q, self.ST, self.C, _ = make_vol_params(hist)
        self.out = np.zeros(N_OUT)

    def push(self, x):
        if self.ST[S_T] >= CAP - 1:
            return self.out[:N_VOL]
        vol_step(float(x), self.Q, self.C, self.ST, self.out)
        return self.out[:N_VOL]


# ---------------------------------------------------------------------------------------------
def _selftest(n_series=2000, offset=0, threads=8):
    import os
    import sys
    import time
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from joblib import Parallel, delayed

    from sbrt.data import load_train
    from sbrt.metric import ts_auc

    # 1. the fit recovers known GARCH parameters
    rng = np.random.default_rng(0)
    for a0, b0 in ((0.08, 0.90), (0.03, 0.96), (0.0, 0.0)):
        est = []
        for rep in range(5):
            n = 4000
            e = np.empty(n)
            s2 = 1.0
            for t in range(n):
                e[t] = math.sqrt(s2) * rng.standard_normal()
                s2 = (1 - a0 - b0) + a0 * e[t] ** 2 + b0 * s2
            q = np.minimum(e * e, QCAP)
            est.append(_fit(q, q.mean(), P_GRID, A_GRID, REFINE))
        est = np.array(est)
        print(f"simulated GARCH a={a0:.2f} p={a0 + b0:.2f}: fitted a {np.round(est[:, 0], 3)}  p {np.round(est[:, 1], 3)}")

    S, tau = load_train()
    S, tau = S[offset:offset + n_series], tau[offset:offset + n_series]
    print(f"series {offset}..{offset + n_series - 1}")

    # 2. streaming == batch, causal, finite, deterministic
    for i in rng.choice(n_series, 30, replace=False):
        _, h, o = S[i]
        B = run_series_vol(h, o)
        assert np.array_equal(B, run_series_vol(h, o)), "two runs differ"
        st = VolStream(h)
        Sm = np.array([st.push(x).copy() for x in o])
        assert np.array_equal(B, Sm), "streaming != batch"
        assert np.isfinite(B).all()
        if len(o) > 20:
            o2 = o.copy()
            o2[15:] = rng.standard_normal(len(o) - 15) * 3
            assert np.array_equal(run_series_vol(h, o2)[:15], B[:15]), "future points changed earlier features"
        G = gru_channels(h, o, 100)
        assert G.shape == (100 + len(o), N_GRU) and np.isfinite(G).all()
    print("streaming == batch, causal, finite, deterministic, GRU channels OK")

    # 3. fitted parameters on real histories
    t0 = time.time()
    fits = Parallel(n_jobs=threads, backend="threading")(delayed(fit_history)(s[1]) for s in S)
    print(f"history fits: {n_series} series in {time.time() - t0:.1f}s")
    pp = np.array([f["p"] for f in fits])
    aa = np.array([f["a"] for f in fits])
    print(f"persistence p = a+b: share flat (a=0) {np.mean(pp == 0):.3f}; quantiles 10/25/50/75/90% "
          f"{np.round(np.quantile(pp, [.1, .25, .5, .75, .9]), 3)}")
    print(f"  half-life of the frozen path (steps), median over non-flat: "
          f"{np.median(np.log(0.5) / np.log(pp[pp > 0])):.0f};  share with p >= 0.99: {np.mean(pp >= 0.99):.3f}")
    print(f"a: quantiles 10/50/90% {np.round(np.quantile(aa, [.1, .5, .9]), 3)}")
    ls = np.array([math.log(f["s2T1"] / f["vbar"]) for f in fits])
    print(f"log(s2_T+1 / vbar): sd {ls.std():.3f}, quantiles 10/50/90% {np.round(np.quantile(ls, [.1, .5, .9]), 3)}")

    # 4. features + kernel references on every 3rd step (like cv.py)
    t0 = time.time()
    V = Parallel(n_jobs=threads, backend="threading")(delayed(lambda s: run_series_vol(s[1], s[2])[::3])(s) for s in S)
    ref_names = ["full_e10sq", "win100_e10sq", "glrmax_e10sq", "full_abse10", "glrmax_tail3"]
    cols = [F.FEATURE_NAMES.index(nm) for nm in ref_names]
    K = Parallel(n_jobs=threads, backend="threading")(delayed(lambda s: F.run_series(s[1], s[2])[::3][:, cols])(s) for s in S)
    print(f"features for {n_series} series in {time.time() - t0:.0f}s")
    E, KR = np.concatenate(V), np.concatenate(K)
    step = np.concatenate([np.arange(0, len(s[2]), 3) for s in S])
    sid = np.repeat(np.arange(n_series), [len(m) for m in V])
    y = ((tau[sid] >= 0) & (step >= tau[sid])).astype(np.int8)
    # same-form UNCONDITIONED log ratio (realized vs whole-history level vbar): isolates the effect of conditioning
    def unconditioned(s, f):
        nh = len(f["z"])
        zz = np.r_[f["z"], (s[2] - f["P"][F.P_MU]) / f["P"][F.P_SD]]
        Xo = np.column_stack([zz[nh - 1 - i: len(zz) - 1 - i] for i in range(AR)])
        q = np.minimum(np.clip((zz[nh:] - Xo @ f["coef"]) / f["se"], -10, 10) ** 2, QCAP)
        k = np.arange(1, len(q) + 1)
        return np.log((np.cumsum(q) + 1e-3 * k) / (f["vbar"] * k))[::3]

    lr_unc = np.concatenate([unconditioned(s, f) for s, f in zip(S, fits)])
    assert len(lr_unc) == len(y)
    # null spread: no-break series around online step 99 (row step 99), raw vs conditioned
    m99 = (step == 99) & (tau[sid] < 0)
    j = FEATURE_NAMES.index("vol_lr_full")
    print(f"\nno-break series at step 99 ({int(m99.sum())} series): sd of log(realized/expected variance) "
          f"vs whole-history level {lr_unc[m99].std():.3f}  vs frozen GARCH path {E[m99, j].std():.3f}")
    buckets = ((0, 100), (100, 300), (300, 700), (700, 10000))
    print("\nsingle-feature TS-AUC (0.5 = no signal; higher = better)   by online-step bucket")
    print(f"  {'feature':22s} {'all':>7s}" + "".join(f"  {lo:>4d}-{hi if hi < 10000 else '':<4}" for lo, hi in buckets))

    def row(nm, x):
        r = [ts_auc(x, y, step)] + [ts_auc(x[(step >= lo) & (step < hi)], y[(step >= lo) & (step < hi)],
                                          step[(step >= lo) & (step < hi)]) for lo, hi in buckets]
        print(f"  {nm:22s} " + "  ".join(f"{v:7.4f}" for v in r))
    print("  -- same-form comparison (conditioning is the only difference):")
    row("lr_full UNconditioned", lr_unc)
    row("vol_lr_full", E[:, j])
    print("  -- kernel references:")
    for c, nm in enumerate(ref_names):
        row(nm, KR[:, c])
    print("  -- new features:")
    for c, nm in enumerate(FEATURE_NAMES):
        row(nm, E[:, c])
    print("\nGo/no-go: the new statistics should beat their kernel counterparts (vol_full vs full_e10sq, vol_win100 vs "
          "win100_e10sq, vol_scanmax vs glrmax_e10sq) and lr_full should beat its unconditioned twin, especially "
          "in 100-300.")
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    import sys
    _selftest(offset=int(sys.argv[1]) if len(sys.argv) > 1 else 0)
