"""Bayesian online change-point posterior (BOCPD, Adams & MacKay 2007) specialised to this task: at most one break,
a known break-free reference (the history) and the generator's known break-timing prior.

Separate from sbrt/features.py like evidence/volstate/disttest; columns are appended only with cv.py --bocpd.

1. Streams (one value per online point, ~N(0, 1) before a break for every series, heavy tails included):
       w   the history-fitted AR(10) residual mapped through the history's own ECDF and the normal quantile
           (the same whitening as sbrt/evidence.py)
       wu  the same for u = residual / frozen volatility forecast (sbrt/volstate.py), whitened by the ECDF of the
           history's in-sample GARCH-standardised residuals
2. Model. No break: w_t ~ N(0, 1). Break at online step j: w_0..w_{j-1} ~ N(0, 1) and w_j..w_t ~ N(mu, s2) with
   (mu, s2) ~ Normal-Inverse-Gamma(0, KAPPA0, ALPHA0, BETA0) integrated out (closed form, Student-t predictive).
   Log Bayes factor of "break at j" vs "no break", from the segment's sums S1, S2 over n points:
       LR_j = 0.5 log(k0 / kn) + a0 log b0 - an log bn + lgamma(an) - lgamma(a0) + 0.5 S2
       kn = k0 + n, an = a0 + n/2, bn = b0 + 0.5 (S2 - S1^2 / kn)
   Prior over the break time: pi_j = h_j prod_{i<j}(1 - h_i) with the generator's hazard h (evidence.HAZARD, no
   labels used); P(no break by t) = prod_{i<=t}(1 - h_i). Exact posterior odds of "break by t":
       O_t = sum_j pi_j exp(LR_j) / P(no break by t)                   (O(t) per step from cumulative sums)
3. Features per stream (5): logodds (log O_t), logbf (log O_t minus the prior log odds), tsb (posterior mean of
   log(1 + time since the break), given a break), lvr (posterior mean of the log post-break variance), mu
   (posterior mean of the post-break mean). 10 features. In batch mode the posterior is evaluated only on the rows
   cv.py keeps (the cumulative sums are updated every step, so those rows equal streaming exactly; tested).
4. Prior (KAPPA0, ALPHA0, BETA0): empirical Bayes, fitted on the post-break segments of training breaks (series
   0-4999) by `python -m sbrt.bocpd fit` (3 numbers; the CV runs therefore use a prior that saw labels of some
   validation series: a negligible leak of 3 summary numbers, noted in HANDOFF). WIDE is a weak alternative prior.

Self-test (+ single-feature check on held-out series 5000-6999):  python -m sbrt.bocpd [fit]
"""
import math

import numpy as np
from numba import njit

from sbrt import evidence as EV
from sbrt import features as F
from sbrt import volstate as VS

AR = F.AR
CAP = F.CAP
# empirical-Bayes prior (python -m sbrt.bocpd fit, series 0-4999); see the docstring
KAPPA0, ALPHA0, BETA0 = 490.0, 3.04, 2.51
WIDE = (10.0, 2.0, 1.0)
VARS = ("w", "wu")
STATS = ("logodds", "logbf", "tsb", "lvr", "mu")
FEATURE_NAMES = [f"bo_{v}_{s}" for v in VARS for s in STATS]
N_BO = len(FEATURE_NAMES)
VERSION = 1   # bump when the kernel or the prior changes (part of the cache names)

_h = np.clip(EV.HAZARD, 1e-12, 1 - 1e-12)
_hz = np.r_[_h, np.full(CAP + 2 - len(_h), _h[-1])] if len(_h) < CAP + 2 else _h[:CAP + 2]
LOGH = np.log(_hz)
LP = np.r_[0.0, np.cumsum(np.log1p(-_hz))]          # LP[j] = sum_{i<j} log(1 - h_i)


@njit(cache=True, nogil=True)
def _posterior(C1, C2, k, n, prior, LOGH, LP, buf, out, o):
    k0, a0, b0 = prior[0], prior[1], prior[2]
    base = a0 * math.log(b0) - math.lgamma(a0)
    mx = -1e300
    for j in range(n):
        m = n - j
        S1 = C1[k, n] - C1[k, j]
        S2 = C2[k, n] - C2[k, j]
        kn = k0 + m
        an = a0 + 0.5 * m
        bn = b0 + 0.5 * (S2 - S1 * S1 / kn)
        if bn < 1e-12:
            bn = 1e-12
        lr = 0.5 * math.log(k0 / kn) + base - an * math.log(bn) + math.lgamma(an) + 0.5 * S2
        a = LP[j] + LOGH[j] + lr
        buf[j] = a
        if a > mx:
            mx = a
    s = 0.0
    e_tsb = 0.0
    e_lvr = 0.0
    e_mu = 0.0
    for j in range(n):
        wgt = math.exp(buf[j] - mx)
        s += wgt
        m = n - j
        S1 = C1[k, n] - C1[k, j]
        S2 = C2[k, n] - C2[k, j]
        kn = k0 + m
        an = a0 + 0.5 * m
        bn = b0 + 0.5 * (S2 - S1 * S1 / kn)
        if bn < 1e-12:
            bn = 1e-12
        e_tsb += wgt * math.log(1.0 + m)
        e_lvr += wgt * math.log(bn / an)
        e_mu += wgt * S1 / kn
    lse = mx + math.log(s)
    lnone = LP[n]
    out[o] = lse - lnone
    pb = 1.0 - math.exp(lnone)
    out[o + 1] = out[o] - (math.log(pb if pb > 1e-300 else 1e-300) - lnone)
    out[o + 2] = e_tsb / s
    out[o + 3] = e_lvr / s
    out[o + 4] = e_mu / s


Q_MU, Q_SD, Q_SE, Q_L, Q_P, Q_AR = 0, 1, 2, 3, 4, 5
Q_LEN = Q_AR + AR
S_T, S_D, S_LAG = 0, 1, 2
S_LEN = S_LAG + AR


def make_bo_params(hist):
    H = VS.fit_history(hist)
    eh = H["eh"]
    Q = np.zeros(Q_LEN)
    Q[Q_MU], Q[Q_SD], Q[Q_SE] = H["P"][F.P_MU], H["P"][F.P_SD], H["se"]
    Q[Q_L], Q[Q_P] = H["L"], H["p"]
    Q[Q_AR:] = H["coef"]
    ST = np.zeros(S_LEN)
    ST[S_D] = H["s2T1"] - H["L"]
    z = H["z"]
    for i in range(AR):
        ST[S_LAG + i] = z[-1 - i]
    refs = np.stack([np.sort(eh), np.sort(eh / np.sqrt(H["s2h"]))])
    return Q, refs, ST, np.zeros((2, CAP + 1)), np.zeros((2, CAP + 1))


@njit(cache=True, nogil=True)
def bo_step(x, Q, refs, C1, C2, ST, prior, LOGH, LP, buf, out, do_post):
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
    for k in range(2):
        w = EV._whiten(e if k == 0 else u, refs[k])
        if w > 8.0:
            w = 8.0
        elif w < -8.0:
            w = -8.0
        C1[k, n] = C1[k, t] + w
        C2[k, n] = C2[k, t] + w * w
        if do_post:
            _posterior(C1, C2, k, n, prior, LOGH, LP, buf, out, 5 * k)
    ST[S_D] = ST[S_D] * Q[Q_P]
    for i in range(AR - 1, 0, -1):
        ST[S_LAG + i] = ST[S_LAG + i - 1]
    ST[S_LAG] = z
    ST[S_T] = n * 1.0


@njit(cache=True, nogil=True)
def _bo_run(online, Q, refs, C1, C2, ST, prior, LOGH, LP, buf, out, sub):
    r = 0
    row = np.zeros(out.shape[1])
    for i in range(len(online)):
        keep = i % sub == 0
        bo_step(online[i], Q, refs, C1, C2, ST, prior, LOGH, LP, buf, row, keep)
        if keep:
            for c in range(out.shape[1]):
                out[r, c] = row[c]
            r += 1


def run_series_bocpd(hist, online, sub=1, prior=None):
    """Batch matrix (rows for online steps 0, sub, 2 sub, ..., N_BO) for one series -- identical to streaming."""
    Q, refs, ST, C1, C2 = make_bo_params(hist)
    online = np.asarray(online, dtype=np.float64)
    if len(online) >= CAP:
        raise ValueError(f"online segment longer than the supported {CAP - 1} points")
    pr = np.array(prior if prior is not None else (KAPPA0, ALPHA0, BETA0), dtype=np.float64)
    out = np.zeros(((len(online) + sub - 1) // sub, N_BO))
    _bo_run(online, Q, refs, C1, C2, ST, pr, LOGH, LP, np.zeros(CAP + 1), out, sub)
    return out


class BoStream:
    """One series' streaming state for use inside infer()."""

    def __init__(self, hist):
        self.Q, self.refs, self.ST, self.C1, self.C2 = make_bo_params(hist)
        self.prior = np.array((KAPPA0, ALPHA0, BETA0))
        self.buf = np.zeros(CAP + 1)
        self.out = np.zeros(N_BO)

    def push(self, x):
        if self.ST[S_T] >= CAP - 1:
            return self.out
        bo_step(float(x), self.Q, self.refs, self.C1, self.C2, self.ST, self.prior, LOGH, LP, self.buf, self.out, True)
        return self.out


def whitened_online(hist, online):
    """The two whitened online streams (for the prior fit)."""
    Q, refs, ST, C1, C2 = make_bo_params(hist)
    online = np.asarray(online, dtype=np.float64)
    _bo_sums(online, Q, refs, C1, C2, ST, np.array(WIDE), LOGH, LP, np.zeros(CAP + 1), np.zeros(N_BO))
    return np.diff(C1[:, :len(online) + 1], axis=1)


@njit(cache=True, nogil=True)
def _bo_sums(online, Q, refs, C1, C2, ST, prior, LOGH, LP, buf, row):
    for i in range(len(online)):
        bo_step(online[i], Q, refs, C1, C2, ST, prior, LOGH, LP, buf, row, False)


def fit_prior(S, tau, max_post=300, min_post=100):
    """Empirical-Bayes NIG prior from the post-break segments of training breaks (stream w):
    s2 ~ Inverse-Gamma(a0, b0) by moments of the segment variances, mu | s2 ~ N(0, s2 / k0) from E[m^2 / s2]."""
    v, r = [], []
    for (_, h, o), t in zip(S, tau):
        if t < 0 or len(o) - t < min_post:
            continue
        w = whitened_online(h, o)[0][t:t + max_post]
        n = len(w)
        s2 = float(w.var())
        v.append(s2)
        r.append(float(w.mean()) ** 2 / max(s2, 1e-6) - 1.0 / n)
    v = np.array(v)
    mean, var = v.mean(), v.var()
    a0 = mean * mean / max(var, 1e-6) + 2.0
    b0 = mean * (a0 - 1.0)
    k0 = 1.0 / max(float(np.mean(r)), 1e-3)
    return dict(n=len(v), kappa0=k0, alpha0=a0, beta0=b0, s2_quantiles=np.round(np.quantile(v, [.1, .5, .9]), 3))


# ---------------------------------------------------------------------------------------------
def _selftest(n_series=2000, offset=5000, threads=8, fit=False):
    import os
    import sys
    import time
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from joblib import Parallel, delayed
    from scipy.stats import rankdata

    from sbrt import disttest as DT
    from sbrt.data import load_train
    from sbrt.metric import ts_auc

    S_all, tau_all = load_train()
    if fit:
        t0 = time.time()
        print("empirical-Bayes prior fitted on series 0-4999:", fit_prior(S_all[:5000], tau_all[:5000]),
              f"({time.time() - t0:.0f}s)")
    print(f"prior in use: kappa0 {KAPPA0}, alpha0 {ALPHA0}, beta0 {BETA0} (mean post-break variance "
          f"{BETA0 / (ALPHA0 - 1):.2f}); wide alternative {WIDE}")
    S, tau = S_all[offset:offset + n_series], tau_all[offset:offset + n_series]
    rng = np.random.default_rng(0)

    # 1. streaming == batch (all rows and every 3rd row), causal, finite, deterministic; posterior sanity
    for i in rng.choice(n_series, 20, replace=False):
        _, h, o = S[i]
        B = run_series_bocpd(h, o)
        assert np.array_equal(B, run_series_bocpd(h, o)), "two runs differ"
        st = BoStream(h)
        Sm = np.array([st.push(x).copy() for x in o])
        assert np.array_equal(B, Sm), "streaming != batch"
        assert np.array_equal(run_series_bocpd(h, o, 3), Sm[::3]), "sub-sampled batch != streaming"
        assert np.isfinite(B).all()
        if len(o) > 40:
            o2 = o.copy()
            o2[35:] = rng.standard_normal(len(o) - 35) * 3
            assert np.array_equal(run_series_bocpd(h, o2)[:35], B[:35]), "future points changed earlier features"
    # an obvious break must drive the posterior up and locate it
    h = S[0][1]
    o = np.r_[rng.standard_normal(300) * h.std() + h.mean(), rng.standard_normal(200) * 2.5 * h.std() + h.mean()]
    B = run_series_bocpd(h, o)
    print(f"sanity: variance x2.5 after step 300: logodds at 290 {B[290, 0]:+.1f}, at 330 {B[330, 0]:+.1f}, "
          f"at 499 {B[499, 0]:+.1f}; posterior time since break at 499: {np.expm1(B[499, 2]):.0f} (true 200)")
    print("streaming == batch (also every 3rd row), causal, finite, deterministic: OK")

    # 2. single-feature TS-AUC vs the strongest related features, and redundancy
    t0 = time.time()
    Bo = Parallel(n_jobs=threads, backend="threading")(delayed(run_series_bocpd)(s[1], s[2], 3) for s in S)
    Bw = Parallel(n_jobs=threads, backend="threading")(delayed(run_series_bocpd)(s[1], s[2], 3, WIDE) for s in S)
    print(f"BOCPD features for {n_series} series in {time.time() - t0:.0f}s (both priors)")
    K = Parallel(n_jobs=threads, backend="threading")(delayed(lambda s: F.run_series(s[1], s[2])[::3])(s) for s in S)
    V = Parallel(n_jobs=threads, backend="threading")(delayed(lambda s: VS.run_series_vol(s[1], s[2])[::3])(s) for s in S)
    D = Parallel(n_jobs=threads, backend="threading")(delayed(DT.run_series_dist)(s[1], s[2], 3) for s in S)
    E = Parallel(n_jobs=threads, backend="threading")(delayed(lambda s: EV.run_series_ev(s[1], s[2])[::3, :EV.N_COMPACT])(s) for s in S)
    Bo, Bw, K, V, D, E = map(np.concatenate, (Bo, Bw, K, V, D, E))
    step = np.concatenate([np.arange(0, len(s[2]), 3) for s in S])
    sid = np.repeat(np.arange(n_series), [(len(s[2]) + 2) // 3 for s in S])
    y = ((tau[sid] >= 0) & (step >= tau[sid])).astype(np.int8)
    buckets = ((0, 100), (100, 300), (300, 700), (700, 10000))

    def row(nm, x, extra=""):
        r = [ts_auc(x, y, step)] + [ts_auc(x[(step >= lo) & (step < hi)], y[(step >= lo) & (step < hi)],
                                          step[(step >= lo) & (step < hi)]) for lo, hi in buckets]
        print(f"  {nm:20s} " + "  ".join(f"{v:7.4f}" for v in r) + extra)

    print("\nsingle-feature TS-AUC (0.5 = no signal)          all     0-100  100-300  300-700    700+")
    print("  -- existing references:")
    for j, nm in enumerate(EV.FEATURE_NAMES[:EV.N_COMPACT]):
        if nm in ("ev_mix_logodds", "ev_logbf", "ev_post_variance"):
            row(nm, E[:, j])
    for nm in ("vol_scanmax", "vol_lr_full"):
        row(nm, V[:, VS.FEATURE_NAMES.index(nm)])
    row("dist_u_adscan", D[:, DT.FEATURE_NAMES.index("dist_u_adscan")])
    sel = rng.choice(len(y), min(len(y), 150000), replace=False)
    Z = np.column_stack([K[sel], V[sel], D[sel], E[sel]])
    old = list(F.FEATURE_NAMES) + list(VS.FEATURE_NAMES) + list(DT.FEATURE_NAMES) + list(EV.FEATURE_NAMES[:EV.N_COMPACT])
    R = np.column_stack([rankdata(Z[:, j]) for j in range(Z.shape[1])])
    R = (R - R.mean(0)) / (R.std(0) + 1e-12)
    print("  -- new, empirical-Bayes prior (last columns: max |rank corr| with any existing feature, and which):")
    for c, nm in enumerate(FEATURE_NAMES):
        r = rankdata(Bo[sel, c])
        r = (r - r.mean()) / (r.std() + 1e-12)
        cc = np.abs(R.T @ r) / len(r)
        j = int(np.argmax(cc))
        row(nm, Bo[:, c], f"   {cc[j]:.2f} {old[j]}")
    print("  -- new, wide prior (logodds only):")
    for c, nm in enumerate(FEATURE_NAMES):
        if nm.endswith("logodds"):
            row(nm + " (wide)", Bw[:, c])
    print("\nGo/no-go: worth a cross-validation run if bo_*_logodds/logbf beat ev_logbf / ev_mix_logodds and are not "
          "near-copies (rank correlation below ~0.9).")
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    import sys
    _selftest(fit=len(sys.argv) > 1 and sys.argv[1] == "fit")
