"""Richer one-break Bayesian posterior (extends sbrt/bocpd.py; that module stays unchanged for production).

Same frame as bocpd.py: the history is the known pre-break regime, at most one break, the generator's break-timing
prior (evidence.HAZARD), exact posterior over the break time j, O(t) per step from cumulative sums. Three additions:

1. Dependence as well as mean and variance. After a break at j the whitened stream follows
       w_t = mu + phi * w_{t-1} + eps,  eps ~ N(0, s2)
   with a Normal-Inverse-Gamma prior on (mu, phi, s2): (mu, phi) | s2 ~ N(0, s2 diag(1/k_mu, 1/k_phi)),
   s2 ~ IG(a0, b0). Before the break w_t ~ N(0, 1) (phi = 0). Closed-form log Bayes factor of "break at j" vs none,
   from the segment sums (m points; x_t = (1, w_{t-1})):
       L = diag(k_mu, k_phi) + sum x x',  r = sum x w,  bn = b0 + 0.5 (sum w^2 - r' L^-1 r),  an = a0 + m / 2
       LR_j = 0.5 log(k_mu k_phi / det L) + a0 log b0 - an log bn + lgamma(an) - lgamma(a0) + 0.5 sum w^2
   This contains bocpd.py's mean/variance model (phi = 0) and adds dependence breaks.
2. Tail frequency. Exceedances b_t = 1{|w_t| > 2}: before the break b_t ~ Bernoulli(p0 = P(|N(0,1)| > 2)); after,
   rate ~ Beta(ta, tb) integrated out. Catches tail changes without assuming a normal shape after the break.
   (A Student-t observation model is not needed for heavy-tailed SERIES: the ECDF whitening already makes w normal
   before the break for every series; heavier tails AFTER a break are what this stream measures.)
3. Type-conditioned empirical-Bayes priors (TESTED, NOT USED: USE_BINS = False; the global prior was as good). The prior describes how breaks change things, and that may differ by
   kind of series. Series are put into 8 bins by their HISTORY only: |lag-1 autocorrelation| > 0.3, excess kurtosis
   > 3, volatility persistence above the training median. Per-bin priors are fitted on post-break segments of
   training breaks (series 0-4999) and shrunk toward the global prior (weight n / (n + 100)). Prior table:
   sbrt/bocpd2_prior.json, written by `python -m sbrt.bocpd2 fit` (uses labels of some CV validation series: a few
   shrunk summary numbers per bin, noted in HANDOFF; the self-test scores held-out series 5000-6999 only).

Streams: w and wu, as in bocpd.py (history-ECDF-whitened AR residual, and the same for the volatility-adjusted one).
Features (12): per stream bo2_{w,wu}_{logodds, tsb, lvr, phi, mu} and bo2_tail_{logodds, lrate} on w.
In batch mode the posteriors are evaluated only on the rows cv.py keeps (identical to streaming; tested).

Self-test (+ single-feature check on held-out series 5000-6999):  python -m sbrt.bocpd2 [fit]
"""
import json
import math
import os

import numpy as np
from numba import njit

from sbrt import bocpd as BO
from sbrt import evidence as EV
from sbrt import features as F
from sbrt import volstate as VS

AR = F.AR
CAP = F.CAP
VERSION = 1   # bump when the kernel or the prior changes (part of the cache names)
TAIL = 2.0
P0 = math.erfc(TAIL / math.sqrt(2.0))   # P(|N(0,1)| > 2) = 0.0455
REG = ("logodds", "tsb", "lvr", "phi", "mu")
FEATURE_NAMES = [f"bo2_{v}_{s}" for v in ("w", "wu") for s in REG] + ["bo2_tail_logodds", "bo2_tail_lrate"]
N_BO2 = len(FEATURE_NAMES)
PRIOR_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bocpd2_prior.json")
NPAR = 6   # k_mu, k_phi, a0, b0, ta, tb
SHRINK_N = 100.0
LOGH, LP = BO.LOGH, BO.LP


def _load_prior():
    if not os.path.exists(PRIOR_FILE):
        return None
    with open(PRIOR_FILE, encoding="utf-8") as fh:
        d = json.load(fh)
    return dict(p_med=d["p_med"], table=np.array(d["table"], dtype=np.float64),   # table: (8 bins, 2 streams, 6)
                glob=np.array(d["glob"], dtype=np.float64))


PRIOR = _load_prior()
# Self-test (held-out 5000-6999): the per-bin priors did NOT beat the global prior (bo2_w_logodds 0.6061 vs 0.6074,
# bo2_wu 0.6051 vs 0.6060, tail 0.5592 vs 0.5617), so the features use the global prior; the bins stay for reference.
USE_BINS = False


def series_bin(hist, H=None):
    """Bin 0-7 from the history only: bit 0 |rho1| > 0.3, bit 1 excess kurtosis > 3, bit 2 vol persistence > median."""
    H = H if H is not None else VS.fit_history(hist)
    z = H["z"]
    zc = z - z.mean()
    rho1 = float(np.dot(zc[:-1], zc[1:]) / max(np.dot(zc, zc), 1e-12))
    kurt = float(np.mean(zc ** 4) / max(np.mean(zc ** 2) ** 2, 1e-12) - 3.0)
    p_med = PRIOR["p_med"] if PRIOR is not None else 0.9
    return int(abs(rho1) > 0.3) + 2 * int(kurt > 3.0) + 4 * int(H["p"] > p_med)


# ---- cumulative sums: index k = stream (0 w, 1 wu); C[k, q, n] for q in 0..5 ------------------
# q: 0 sum w, 1 sum w^2, 2 sum lag, 3 sum lag^2, 4 sum lag*w, 5 sum 1{|w| > TAIL}
@njit(cache=True, nogil=True)
def _post_reg(C, k, n, pr, LOGH, LP, buf, out, o):
    kmu, kphi, a0, b0 = pr[0], pr[1], pr[2], pr[3]
    base = a0 * math.log(b0) - math.lgamma(a0) + 0.5 * math.log(kmu * kphi)
    mx = -1e300
    for j in range(n):
        m = n - j
        s1 = C[k, 0, n] - C[k, 0, j]
        syy = C[k, 1, n] - C[k, 1, j]
        sl = C[k, 2, n] - C[k, 2, j]
        sll = C[k, 3, n] - C[k, 3, j]
        sly = C[k, 4, n] - C[k, 4, j]
        l00 = kmu + m
        l11 = kphi + sll
        det = l00 * l11 - sl * sl
        quad = (l11 * s1 * s1 - 2.0 * sl * s1 * sly + l00 * sly * sly) / det
        an = a0 + 0.5 * m
        bn = b0 + 0.5 * (syy - quad)
        if bn < 1e-12:
            bn = 1e-12
        a = LP[j] + LOGH[j] + base - 0.5 * math.log(det) - an * math.log(bn) + math.lgamma(an) + 0.5 * syy
        buf[j] = a
        if a > mx:
            mx = a
    s = 0.0
    e_tsb = 0.0
    e_lvr = 0.0
    e_phi = 0.0
    e_mu = 0.0
    for j in range(n):
        wgt = math.exp(buf[j] - mx)
        s += wgt
        m = n - j
        s1 = C[k, 0, n] - C[k, 0, j]
        syy = C[k, 1, n] - C[k, 1, j]
        sl = C[k, 2, n] - C[k, 2, j]
        sll = C[k, 3, n] - C[k, 3, j]
        sly = C[k, 4, n] - C[k, 4, j]
        l00 = kmu + m
        l11 = kphi + sll
        det = l00 * l11 - sl * sl
        mu = (l11 * s1 - sl * sly) / det
        phi = (l00 * sly - sl * s1) / det
        an = a0 + 0.5 * m
        bn = b0 + 0.5 * (syy - (s1 * mu + sly * phi))
        if bn < 1e-12:
            bn = 1e-12
        e_tsb += wgt * math.log(1.0 + m)
        e_lvr += wgt * math.log(bn / an)
        e_phi += wgt * phi
        e_mu += wgt * mu
    out[o] = mx + math.log(s) - LP[n]
    out[o + 1] = e_tsb / s
    out[o + 2] = e_lvr / s
    out[o + 3] = e_phi / s
    out[o + 4] = e_mu / s


@njit(cache=True, nogil=True)
def _lbeta(a, b):
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


@njit(cache=True, nogil=True)
def _post_tail(C, n, ta, tb, LOGH, LP, buf, out, o):
    lp0, lq0 = math.log(P0), math.log(1.0 - P0)
    b0 = _lbeta(ta, tb)
    mx = -1e300
    for j in range(n):
        m = n - j
        c = C[0, 5, n] - C[0, 5, j]
        a = LP[j] + LOGH[j] + _lbeta(ta + c, tb + m - c) - b0 - c * lp0 - (m - c) * lq0
        buf[j] = a
        if a > mx:
            mx = a
    s = 0.0
    e_rate = 0.0
    for j in range(n):
        wgt = math.exp(buf[j] - mx)
        s += wgt
        m = n - j
        c = C[0, 5, n] - C[0, 5, j]
        e_rate += wgt * math.log((ta + c) / (ta + tb + m) / P0)
    out[o] = mx + math.log(s) - LP[n]
    out[o + 1] = e_rate / s


def make_params(hist, prior=None):
    """Per-series state: BO's whitening params, the prior row (2 streams x 6) for the series' bin, cumulative sums."""
    H = VS.fit_history(hist)
    Q, refs, ST, _, _ = BO.make_bo_params(hist)
    if prior is None:
        if PRIOR is None:
            raise RuntimeError(f"{PRIOR_FILE} missing: run  python -m sbrt.bocpd2 fit")
        prior = PRIOR["table"][series_bin(hist, H)] if USE_BINS else PRIOR["glob"]
    lag = np.zeros(3)   # running lag per stream (w, wu) + spare
    return Q, refs, ST, np.ascontiguousarray(prior, dtype=np.float64), np.zeros((2, 6, CAP + 1)), lag


@njit(cache=True, nogil=True)
def bo2_step(x, Q, refs, ST, pr, C, lag, LOGH, LP, buf, out, do_post):
    t = int(ST[BO.S_T])
    z = (x - Q[BO.Q_MU]) / Q[BO.Q_SD]
    pred = 0.0
    for i in range(AR):
        pred += Q[BO.Q_AR + i] * ST[BO.S_LAG + i]
    e = (z - pred) / Q[BO.Q_SE]
    if e > 10.0:
        e = 10.0
    elif e < -10.0:
        e = -10.0
    u = e / math.sqrt(Q[BO.Q_L] + ST[BO.S_D])
    n = t + 1
    for k in range(2):
        w = EV._whiten(e if k == 0 else u, refs[k])
        if w > 8.0:
            w = 8.0
        elif w < -8.0:
            w = -8.0
        l = lag[k]
        C[k, 0, n] = C[k, 0, t] + w
        C[k, 1, n] = C[k, 1, t] + w * w
        C[k, 2, n] = C[k, 2, t] + l
        C[k, 3, n] = C[k, 3, t] + l * l
        C[k, 4, n] = C[k, 4, t] + l * w
        C[k, 5, n] = C[k, 5, t] + (1.0 if abs(w) > 2.0 else 0.0)
        lag[k] = w
        if do_post:
            _post_reg(C, k, n, pr[k], LOGH, LP, buf, out, 5 * k)
    if do_post:
        _post_tail(C, n, pr[0, 4], pr[0, 5], LOGH, LP, buf, out, 10)
    ST[BO.S_D] = ST[BO.S_D] * Q[BO.Q_P]
    for i in range(AR - 1, 0, -1):
        ST[BO.S_LAG + i] = ST[BO.S_LAG + i - 1]
    ST[BO.S_LAG] = z
    ST[BO.S_T] = n * 1.0


@njit(cache=True, nogil=True)
def _run(online, Q, refs, ST, pr, C, lag, LOGH, LP, buf, out, sub, do_post):
    r = 0
    row = np.zeros(12)
    for i in range(len(online)):
        keep = do_post and i % sub == 0
        bo2_step(online[i], Q, refs, ST, pr, C, lag, LOGH, LP, buf, row, keep)
        if keep:
            for c in range(12):
                out[r, c] = row[c]
            r += 1


def run_series_bocpd2(hist, online, sub=1, prior=None):
    """Batch matrix (rows for online steps 0, sub, 2 sub, ...; N_BO2 columns) for one series -- equals streaming."""
    Q, refs, ST, pr, C, lag = make_params(hist, prior)
    online = np.asarray(online, dtype=np.float64)
    if len(online) >= CAP:
        raise ValueError(f"online segment longer than the supported {CAP - 1} points")
    out = np.zeros(((len(online) + sub - 1) // sub, N_BO2))
    _run(online, Q, refs, ST, pr, C, lag, LOGH, LP, np.zeros(CAP + 1), out, sub, True)
    return out


class Bo2Stream:
    """One series' streaming state for use inside infer()."""

    def __init__(self, hist):
        self.Q, self.refs, self.ST, self.pr, self.C, self.lag = make_params(hist)
        self.buf = np.zeros(CAP + 1)
        self.out = np.zeros(N_BO2)

    def push(self, x):
        if self.ST[BO.S_T] >= CAP - 1:
            return self.out
        bo2_step(float(x), self.Q, self.refs, self.ST, self.pr, self.C, self.lag, LOGH, LP, self.buf, self.out, True)
        return self.out


def streams(hist, online):
    """(2, n) whitened streams w, wu of the online segment (for the prior fit and sbrt/lratio.py)."""
    Q, refs, ST, pr, C, lag = make_params(hist, np.ones((2, NPAR)))
    online = np.asarray(online, dtype=np.float64)
    _run(online, Q, refs, ST, pr, C, lag, LOGH, LP, np.zeros(CAP + 1), np.zeros((1, N_BO2)), 1, False)
    return np.diff(C[:, 0, :len(online) + 1], axis=1)


# ---- empirical-Bayes prior, per bin, shrunk to the global fit ---------------------------------
def _fit_group(seg):
    """seg: list of (w segment, lag segment). Moments -> (k_mu, k_phi, a0, b0, ta, tb)."""
    s2s, rmu, rphi, rate, nn = [], [], [], [], []
    for w, l in seg:
        n = len(w)
        Xd = np.column_stack([np.ones(n), l])
        beta, *_ = np.linalg.lstsq(Xd, w, rcond=None)
        res = w - Xd @ beta
        s2 = max(float(res.var()), 1e-6)
        s2s.append(s2)
        rmu.append(beta[0] ** 2 / s2 - 1.0 / n)
        rphi.append((beta[1] ** 2 - 1.0 / n) / s2)
        rate.append(float(np.mean(np.abs(w) > TAIL)))
        nn.append(n)
    s2s, rate, nn = np.array(s2s), np.array(rate), np.array(nn, float)
    mean, var = s2s.mean(), s2s.var()
    a0 = mean * mean / max(var, 1e-6) + 2.0
    b0 = mean * (a0 - 1.0)
    kmu = 1.0 / max(float(np.mean(rmu)), 1e-3)
    kphi = 1.0 / max(float(np.mean(rphi)), 1e-3)
    m = float(rate.mean())
    v = float(rate.var() - np.mean(m * (1 - m) / nn))
    ab = float(np.clip(m * (1 - m) / v - 1.0, 2.0, 1e4)) if v > 1e-9 else 1e4
    return np.array([kmu, kphi, a0, b0, m * ab, (1 - m) * ab])


def fit_prior(S, tau, max_post=300, min_post=100):
    ps = np.array([VS.fit_history(h)["p"] for (_, h, _) in S])
    p_med = float(np.median(ps))
    global PRIOR
    PRIOR = dict(p_med=p_med, table=np.ones((8, 2, NPAR)), glob=np.ones((2, NPAR)))   # bins need p_med
    segs = {b: ([], []) for b in range(8)}
    allseg = ([], [])
    for (_, h, o), t in zip(S, tau):
        if t < 0 or len(o) - t < min_post:
            continue
        W = streams(h, o)
        b = series_bin(h)
        for k in range(2):
            w = W[k][t:t + max_post]
            l = W[k][t - 1:t - 1 + len(w)] if t >= 1 else np.r_[0.0, W[k][:len(w) - 1]]
            segs[b][k].append((w, l))
            allseg[k].append((w, l))
    glob = np.stack([_fit_group(allseg[k]) for k in range(2)])
    table = np.zeros((8, 2, NPAR))
    counts = []
    for b in range(8):
        nb = len(segs[b][0])
        counts.append(nb)
        lam = nb / (nb + SHRINK_N)
        for k in range(2):
            pb = _fit_group(segs[b][k]) if nb >= 10 else glob[k]
            table[b, k] = np.exp(lam * np.log(pb) + (1 - lam) * np.log(glob[k]))
    PRIOR = dict(p_med=p_med, table=table, glob=glob)
    with open(PRIOR_FILE, "w", encoding="utf-8") as fh:
        json.dump(dict(p_med=p_med, table=table.tolist(), counts=counts, glob=glob.tolist(),
                       fitted_on="series 0-4999, post-break segments of 100-300 points"), fh, indent=1)
    return dict(p_med=p_med, counts=counts, glob=np.round(glob, 3), table=np.round(table, 3))


# ---------------------------------------------------------------------------------------------
def _selftest(n_series=2000, offset=5000, threads=8, fit=False):
    import sys
    import time
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from joblib import Parallel, delayed
    from scipy.stats import rankdata

    from sbrt.data import load_train
    from sbrt.metric import ts_auc

    S_all, tau_all = load_train()
    if fit or PRIOR is None:
        t0 = time.time()
        r = fit_prior(S_all[:5000], tau_all[:5000])
        print(f"prior fitted on series 0-4999 ({time.time() - t0:.0f}s): p_med {r['p_med']:.4f}, breaks per bin "
              f"{r['counts']}")
        print("  global (k_mu, k_phi, a0, b0, ta, tb), stream w :", r["glob"][0])
        print("  global                                stream wu:", r["glob"][1])
        for b in range(8):
            print(f"  bin {b} (|rho1|>0.3 {b & 1}, kurt>3 {(b >> 1) & 1}, persist>med {(b >> 2) & 1}): w {r['table'][b, 0]}")
    S, tau = S_all[offset:offset + n_series], tau_all[offset:offset + n_series]
    rng = np.random.default_rng(0)

    # 1. streaming == batch, sub-sampled == streaming, causal, finite, deterministic; the AR model nests bocpd.py
    for i in rng.choice(n_series, 20, replace=False):
        _, h, o = S[i]
        B = run_series_bocpd2(h, o)
        assert np.array_equal(B, run_series_bocpd2(h, o)), "two runs differ"
        st = Bo2Stream(h)
        Sm = np.array([st.push(x).copy() for x in o])
        assert np.array_equal(B, Sm), "streaming != batch"
        assert np.array_equal(run_series_bocpd2(h, o, 3), Sm[::3]), "sub-sampled batch != streaming"
        assert np.isfinite(B).all()
        if len(o) > 40:
            o2 = o.copy()
            o2[35:] = rng.standard_normal(len(o) - 35) * 3
            assert np.array_equal(run_series_bocpd2(h, o2)[:35], B[:35]), "future points changed earlier features"
    # with k_phi huge (phi pinned at 0) the regression model must equal bocpd.py's mean/variance model
    _, h, o = S[0]
    pr = np.tile([BO.KAPPA0, 1e12, BO.ALPHA0, BO.BETA0, 1.0, 20.0], (2, 1))
    A2 = run_series_bocpd2(h, o, 1, pr)
    A1 = BO.run_series_bocpd(h, o)
    d = max(np.abs(A2[:, 0] - A1[:, 0]).max(), np.abs(A2[:, 1] - A1[:, 2]).max(), np.abs(A2[:, 2] - A1[:, 3]).max())
    assert d < 1e-5, f"phi-pinned model differs from bocpd.py by {d}"
    # an obvious dependence break (variance unchanged) must show up in bo2_w_logodds and phi, much less in bocpd.py
    h = S[1][1]
    a = np.zeros(500)
    eps = rng.standard_normal(500)
    for t in range(500):
        a[t] = eps[t] if t < 300 else 0.6 * a[t - 1] + 0.8 * eps[t]
    o = a * h.std() + h.mean()
    B2, B1 = run_series_bocpd2(h, o), BO.run_series_bocpd(h, o)
    print(f"sanity: AR(1) 0.6 break at step 300 (same variance): bo2 logodds at 290 {B2[290, 0]:+.1f}, 400 "
          f"{B2[400, 0]:+.1f}, 499 {B2[499, 0]:+.1f}, phi at 499 {B2[499, 3]:+.2f}; bocpd.py logodds at 499 {B1[499, 0]:+.1f}")
    print("streaming == batch (also every 3rd row), causal, finite, deterministic, nests bocpd.py: OK")

    # 2. single-feature TS-AUC, type-conditioned vs global prior, redundancy with bocpd.py
    t0 = time.time()
    run = lambda s, pr=None: run_series_bocpd2(s[1], s[2], 3, pr)
    B2 = np.concatenate(Parallel(n_jobs=threads, backend="threading")(delayed(run)(s) for s in S))
    alt = lambda s: run(s, PRIOR["glob"] if USE_BINS else PRIOR["table"][series_bin(s[1])])
    B2g = np.concatenate(Parallel(n_jobs=threads, backend="threading")(delayed(alt)(s) for s in S))
    B1 = np.concatenate(Parallel(n_jobs=threads, backend="threading")(delayed(BO.run_series_bocpd)(s[1], s[2], 3) for s in S))
    print(f"features for {n_series} series in {time.time() - t0:.0f}s (bocpd2 x2 priors + bocpd)")
    step = np.concatenate([np.arange(0, len(s[2]), 3) for s in S])
    sid = np.repeat(np.arange(n_series), [(len(s[2]) + 2) // 3 for s in S])
    y = ((tau[sid] >= 0) & (step >= tau[sid])).astype(np.int8)
    buckets = ((0, 100), (100, 300), (300, 700), (700, 10000))

    def row(nm, x, extra=""):
        r = [ts_auc(x, y, step)] + [ts_auc(x[(step >= lo) & (step < hi)], y[(step >= lo) & (step < hi)],
                                          step[(step >= lo) & (step < hi)]) for lo, hi in buckets]
        print(f"  {nm:22s} " + "  ".join(f"{v:7.4f}" for v in r) + extra)

    print("\nsingle-feature TS-AUC (0.5 = no signal)            all     0-100  100-300  300-700    700+")
    print("  -- bocpd.py (current):")
    for c, nm in enumerate(BO.FEATURE_NAMES):
        if nm.endswith(("logodds", "lvr")):
            row(nm, B1[:, c])
    sel = rng.choice(len(y), min(len(y), 150000), replace=False)
    R = np.column_stack([rankdata(B1[sel, j]) for j in range(B1.shape[1])])
    R = (R - R.mean(0)) / (R.std(0) + 1e-12)
    print(f"  -- bocpd2, {'type-conditioned' if USE_BINS else 'global'} prior (last: max |rank corr| with a bocpd.py feature):")
    for c, nm in enumerate(FEATURE_NAMES):
        r = rankdata(B2[sel, c])
        r = (r - r.mean()) / (r.std() + 1e-12)
        cc = np.abs(R.T @ r) / len(r)
        j = int(np.argmax(cc))
        row(nm, B2[:, c], f"   {cc[j]:.2f} {BO.FEATURE_NAMES[j]}")
    print(f"  -- bocpd2, {'global' if USE_BINS else 'type-conditioned'} prior (the alternative):")
    for c, nm in enumerate(FEATURE_NAMES):
        if nm.endswith("logodds"):
            row(nm + (" (glob)" if USE_BINS else " (bins)"), B2g[:, c])
    print("\nGo/no-go: worth a CV run if bo2 logodds beat bo_*_logodds or the tail/phi features carry signal with "
          "rank correlation below ~0.9.")
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    import sys
    _selftest(fit=len(sys.argv) > 1 and sys.argv[1] == "fit")
