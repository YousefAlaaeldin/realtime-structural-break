"""Break transplant: synthetic breaks LEARNED from real ones, applied to other series' real break-free data.

Why (HANDOFF section 3: data-limited, doubling the data ~+0.015; section 14 of the full history: the hand-made
break types of sbrt/synth.py failed, their detectability grew with L in the wrong shape). Here nothing about the
break is hand-designed: every synthetic break copies the change of ONE real break (the donor), expressed in a
series-free, whitened form, and re-applies it to the real break-free continuation of another series (the target).
Histories are never generated (plenty of real break-free data exists; the NBM data curve is flat).

1. Normal model of a break-free stream `pre` (the data before the break point; volstate.fit_history):
   z = (x - mu) / sd, AR(10) residual e = (z - pred) / se, frozen variance path s2f(k) = L + p^k (s2T1 - L)
   started at the end of `pre` (never updated, so a break cannot be absorbed), u = e / sqrt(s2f),
   w = the monotone piecewise-linear map of u onto normal scores defined by the sorted in-sample residuals
   eh / sqrt(s2h) (the bocpd `wu` reference), linearly extrapolated beyond the sample range. Under no break w is
   ~N(0, 1) and close to iid; the map is exactly invertible.
2. Donor (a real break series, post-break length >= min_post): pre = history ++ online[:tau]; the donor's change is
   its post-break w stream summarised as (a) a smooth Hermite-cubic fit of its quantile function (location, scale,
   skew, tails; the location is dropped by default) and (b) an AR(2) fit of the centred w (dependence change).
   Placebo donors (no-break series cut at a uniform fake break) give the same summaries with NO break: they measure
   how much sampling noise alone moves the change estimates (the calibration control).
3. Target: a real break-free continuation C after a uniform break position t of a re-cut (geometry as sbrt/synth.py).
   w_c = forward(C) under the target's own normal model (fitted on hist ++ online[:t]); then
       w'  = b^lam w_c + lam c He2(w_c) + lam d He3(w_c)   (the donor's smoothed quantile map, see Change)
       w'' = mean + AR(2) filter (lam_dep phi_d) of the centred w', rescaled to keep its variance
   and the inverse of the target's normal model turns w'' back into data. The order of the target's REAL
   innovations is kept (a monotone map), so the target's own vol clustering and tail shape survive; lam = 0 or an
   unchanged donor gives back C exactly (self-test). `lam` is the calibration knob (diagnostics/breakgen_gate.py).
4. Donor choice: from the parent's own cross-validation fold (so a variant never carries label information from a
   validation series; same rule as synth.py's splice donors), uniformly (match_k = 0) or among the match_k nearest
   donors in history-descriptor space (match_k > 0: a cheap conditional version; use only if the gate's pre-check
   finds the change predictable from the history).

Geometry: online length ~ U{10..999}, break position uniform, history length drawn from the ORIGINAL
history-length distribution (augment.draw_hist_len), so geometry cannot tell a variant from a real series.

Output format = sbrt.synth.make_synthetic: (parent_index, kind, lam, new_hist, new_online, new_tau).

Self-test:  python -m sbrt.breakgen
"""
import math

import numpy as np
from numba import njit

from sbrt import features as F
from sbrt import volstate as VS
from sbrt.augment import H_MAX, H_MIN, O_MAX, O_MIN, draw_hist_len
from sbrt.synth import break_free, fold_ids

AR = F.AR
VERSION = 3   # 2: smooth Hermite-cubic change map, location dropped; 3: + empirical-Bayes deconvolution (Deconv)


# ---------------------------------------------------------------------------------------------
def _normal_grid(n):
    from scipy.special import ndtri
    return ndtri((np.arange(n) + 0.5) / n)


def _ext_interp(x, xp, fp):
    """np.interp with linear extrapolation at both ends (slope from the last ~1% of the points)."""
    y = np.interp(x, xp, fp)
    n = len(xp)
    k = max(5, n // 100)
    k = min(k, n - 1)
    lo, hi = x < xp[0], x > xp[-1]
    if lo.any():
        s = (fp[k] - fp[0]) / max(xp[k] - xp[0], 1e-12)
        y[lo] = fp[0] + s * (x[lo] - xp[0])
    if hi.any():
        s = (fp[-1] - fp[-1 - k]) / max(xp[-1] - xp[-1 - k], 1e-12)
        y[hi] = fp[-1] + s * (x[hi] - xp[-1])
    return y


class NormalModel:
    """Normal behaviour of one break-free stream, forward (data -> w) and inverse (w -> data)."""

    def __init__(self, pre):
        H = VS.fit_history(np.asarray(pre, dtype=np.float64))
        self.mu, self.sd = float(H["P"][F.P_MU]), float(H["P"][F.P_SD])
        self.coef, self.se = np.ascontiguousarray(H["coef"], dtype=np.float64), float(H["se"])
        self.L, self.p, self.s2T1 = float(H["L"]), float(H["p"]), float(H["s2T1"])
        self.lags0 = np.ascontiguousarray(H["z"][::-1][:AR], dtype=np.float64)   # z[-1], z[-2], ...
        ref = np.sort(H["eh"] / np.sqrt(H["s2h"]))
        ref = ref + np.arange(len(ref)) * 1e-10          # strictly increasing -> exactly invertible map
        self.ref, self.grid = ref, _normal_grid(len(ref))
        P = H["P"]
        # history descriptors for matched donor draws
        self.desc = np.array([P[F.P_CTX], P[F.P_CTX + 1], P[F.P_CTX + 2], P[F.P_CTX + 3], H["a"], H["p"],
                              math.log(H["L"] / H["vbar"]), math.log(H["s2T1"] / H["L"])])

    def s2f(self, n):
        return self.L + self.p ** np.arange(n) * (self.s2T1 - self.L)

    def forward(self, x):
        u = _resid(np.asarray(x, dtype=np.float64), self.mu, self.sd, self.coef, self.se, self.lags0.copy())
        u = u / np.sqrt(self.s2f(len(u)))
        return _ext_interp(u, self.ref, self.grid)

    def inverse(self, w):
        u = _ext_interp(np.asarray(w, dtype=np.float64), self.grid, self.ref)
        e = u * np.sqrt(self.s2f(len(u)))
        return _recolour(e, self.mu, self.sd, self.coef, self.se, self.lags0.copy())


@njit(cache=True, nogil=True)
def _resid(x, mu, sd, coef, se, lags):
    out = np.empty(len(x))
    for t in range(len(x)):
        z = (x[t] - mu) / sd
        pred = 0.0
        for i in range(len(coef)):
            pred += coef[i] * lags[i]
        out[t] = (z - pred) / se
        for i in range(len(coef) - 1, 0, -1):
            lags[i] = lags[i - 1]
        lags[0] = z
    return out


@njit(cache=True, nogil=True)
def _recolour(e, mu, sd, coef, se, lags):
    out = np.empty(len(e))
    for t in range(len(e)):
        pred = 0.0
        for i in range(len(coef)):
            pred += coef[i] * lags[i]
        z = pred + se * e[t]
        out[t] = mu + sd * z
        for i in range(len(coef) - 1, 0, -1):
            lags[i] = lags[i - 1]
        lags[0] = z
    return out


@njit(cache=True, nogil=True)
def _ar2(y, f1, f2):
    v = np.empty(len(y))
    a, b = 0.0, 0.0
    for t in range(len(y)):
        v[t] = f1 * a + f2 * b + y[t]
        b = a
        a = v[t]
    return v


# ---------------------------------------------------------------------------------------------
def _monotone(b, c, d):
    """Shrink the skew/tail terms until T(w) = b w + c He2 + d He3 is increasing on |w| <= 6."""
    ww = np.linspace(-6.0, 6.0, 241)
    for _ in range(60):
        if (b + 2.0 * c * ww + 3.0 * d * (ww * ww - 1.0)).min() > 0.05 * b:
            return c, d
        c, d = 0.9 * c, 0.9 * d
    return 0.0, 0.0


class Deconv:
    """Empirical-Bayes deconvolution of donor change components (version 3).

    A donor's measured component m = true change + noise. The noise is what PLACEBO donors (no break) measure: mostly
    the no-break series' own variance wander against its frozen forecast (sd of log b ~0.10 whatever the post length,
    smoke 2026-10-20), and it is as large as most real changes, so transplanting m as it is adds a "noise break" to
    every variant (placebo transplants were MORE visible than real breaks). Per component k: noise = the placebo
    donors' own centred values (median removed), as a Gaussian KDE (the noise is heavy-tailed); the distribution of the true change = a discrete
    nonparametric prior on a grid fitted to the real donors by EM (NPMLE, Koenker-Mizera style). A transplant then
    uses a DRAW from each donor's posterior, component by component, so the variants carry the true-change
    distribution (not the under-dispersed posterior means). Components are deconvolved separately (a simplification).
    """

    def __init__(self, M_real, M_plac, n_grid=300, iters=400):
        M_real, M_plac = np.asarray(M_real), np.asarray(M_plac)
        self.bias = np.median(M_plac, axis=0)
        q75, q25 = np.percentile(M_plac, [75, 25], axis=0)
        self.sd = np.maximum((q75 - q25) / 1.349, 1e-6)
        self.grid, self.pi, self.nx, self.nf = [], [], [], []
        for k in range(M_real.shape[1]):
            # noise density = Gaussian KDE of the centred placebo values (heavy tails kept: a Gaussian noise model
            # lets the prior explain noise outliers as real changes), tabulated on a fine grid
            e = M_plac[:, k] - self.bias[k]
            h = 0.9 * min(e.std(), self.sd[k]) * len(e) ** -0.2
            span = max(np.abs(e).max(), np.abs(M_real[:, k] - self.bias[k]).max()) * 2.2 + 5 * h
            x = np.linspace(-span, span, 4001)
            f = np.zeros_like(x)
            for c0 in np.array_split(e, max(1, len(e) // 500)):
                f += np.exp(-0.5 * ((x[:, None] - c0[None, :]) / h) ** 2).sum(1)
            f = f / (len(e) * h * math.sqrt(2 * math.pi)) + 1e-300
            m = M_real[:, k] - self.bias[k]
            g = np.linspace(m.min() - 1e-9, m.max() + 1e-9, n_grid)
            lik = np.interp(m[:, None] - g[None, :], x, f)
            pi = np.full(n_grid, 1.0 / n_grid)
            for _ in range(iters):
                post = lik * pi
                post /= post.sum(1, keepdims=True) + 1e-300
                pi = post.mean(0)
            self.grid.append(g)
            self.pi.append(pi)
            self.nx.append(x)
            self.nf.append(f)

    def draw(self, m, rng):
        """One draw of the true components given a donor's measured components m."""
        out = np.empty(len(m))
        for k in range(len(m)):
            g, pi = self.grid[k], self.pi[k]
            w = pi * np.interp(m[k] - self.bias[k] - g, self.nx[k], self.nf[k])
            s = w.sum()
            out[k] = g[int(rng.choice(len(g), p=w / s))] if s > 0 else m[k] - self.bias[k]
        return out

    def prior_sd(self):
        return np.array([math.sqrt(max(float((pi * g * g).sum() - (pi * g).sum() ** 2), 0.0))
                         for g, pi in zip(self.grid, self.pi)])


class Change:
    """One donor's change in whitened space, as a SMOOTH map (version 2).

    The raw empirical quantile function of a donor's post-break stream carries its sampling noise; applied to a target
    as a fixed map over hundreds of points, that noise becomes a persistent, visible "break" (smoke run 2026-10-20:
    placebo donors with NO break gave oracle AUC 0.604 at L = 300 vs 0.555 for real breaks; mean 0.628 vs 0.510).
    So the quantile function is summarised by a Hermite cubic fitted on the central 98% of the Q-Q points:
        T(w) = a + b w + c (w^2 - 1) + d (w^3 - 3 w)        a location, b scale, c skew, d tails
    made monotone on |w| <= 6, plus an AR(2) fit of the time-ordered, centred stream (dependence)."""
    __slots__ = ("a", "b", "c", "d", "f1", "f2", "desc", "src", "npost")
    COMP = ("log b", "c", "d", "f1", "f2")

    def comp(self):
        return np.array([math.log(self.b), self.c, self.d, self.f1, self.f2])

    def with_comp(self, v):
        """A copy with components v (log b, c, d, f1, f2), made valid (monotone map, stationary AR(2))."""
        ch = object.__new__(Change)
        ch.a, ch.desc, ch.src, ch.npost = self.a, self.desc, self.src, self.npost
        ch.b = float(math.exp(v[0]))
        ch.c, ch.d = _monotone(ch.b, float(v[1]), float(v[2]))
        f2 = float(np.clip(v[4], -0.9, 0.9))
        ch.f1, ch.f2 = float(np.clip(v[3], -(1.0 - f2) * 0.98 + 1e-9, (1.0 - f2) * 0.98 - 1e-9)), f2
        return ch

    def __init__(self, w, desc, src):
        w = np.asarray(w, dtype=np.float64)
        n = len(w)
        g = _normal_grid(n)
        q = np.sort(w)
        k = max(1, int(0.01 * n))
        gg, qq = (g[k:n - k], q[k:n - k]) if n - 2 * k >= 20 else (g, q)
        X = np.column_stack([np.ones_like(gg), gg, gg * gg - 1.0, gg ** 3 - 3.0 * gg])
        a, b, c, d = np.linalg.lstsq(X, qq, rcond=None)[0]
        b = max(float(b), 0.05)
        c, d = float(c), float(d)
        c, d = _monotone(b, float(c), float(d))
        self.a, self.b, self.c, self.d = float(a), b, c, d
        cc = w - w.mean()                                 # dependence from the stream in its time order
        v = float(np.dot(cc, cc)) / n
        r1 = float(np.dot(cc[:-1], cc[1:])) / n / max(v, 1e-12)
        r2 = float(np.dot(cc[:-2], cc[2:])) / n / max(v, 1e-12)
        f2 = (r2 - r1 * r1) / max(1.0 - r1 * r1, 1e-6)
        f1 = r1 * (1.0 - f2)
        f2 = float(np.clip(f2, -0.9, 0.9))                # stationarity triangle (shrunk inside it)
        f1 = float(np.clip(f1, -(1.0 - f2) * 0.98 + 1e-9, (1.0 - f2) * 0.98 - 1e-9))
        self.f1, self.f2, self.desc, self.src, self.npost = f1, f2, desc, src, n

    def apply(self, wc, lam, lam_dep=None, keep_mean=False):
        """lam shrinks scale (in log), skew and tails; lam_dep (default lam) the dependence; the location is applied
        only with keep_mean (real breaks show no mean shifts: oracle AUC of the mean 0.50-0.51)."""
        lam_dep = lam if lam_dep is None else lam_dep
        if lam == 0.0 and lam_dep == 0.0:
            return wc.copy()
        w1 = (self.b ** lam) * wc + lam * self.c * (wc * wc - 1.0) + lam * self.d * (wc ** 3 - 3.0 * wc)
        if keep_mean:
            w1 = w1 + lam * self.a
        f1, f2 = lam_dep * self.f1, lam_dep * self.f2
        if f1 == 0.0 and f2 == 0.0:
            return w1
        m = float(w1.mean())
        v = _ar2(w1 - m, f1, f2)
        r1 = f1 / (1.0 - f2)
        g0 = 1.0 / max(1.0 - f1 * r1 - f2 * (f1 * r1 + f2), 1e-6)   # AR(2) variance gain
        return m + v / math.sqrt(g0)


def extract_change(hist, online, tau, src):
    pre = np.r_[hist, online[:tau]]
    M = NormalModel(pre)
    return Change(M.forward(online[tau:]), M.desc, src)


def donor_pool(series, tau, idx, mode="break", min_post=100, max_post=O_MAX, seed=0):
    """Change objects for the series idx. mode 'break': real breaks with >= min_post post-break points (first
    max_post used); 'placebo': no-break series cut at a uniform fake break leaving >= min_post points."""
    tau = np.asarray(tau)
    out = []
    for i in idx:
        _, h, o = series[i]
        t = int(tau[i])
        if mode == "break":
            if t < 0 or len(o) - t < min_post:
                continue
        elif mode == "placebo":
            if t >= 0 or len(o) < min_post + 1:
                continue
            t = int(np.random.default_rng([seed, 11, int(i)]).integers(0, len(o) - min_post + 1))
        else:
            raise ValueError(mode)
        out.append(extract_change(h, o[: t + max_post], t, int(i)))
    return out


def transplant(hist, online, t, ch, lam, lam_dep=None, keep_mean=False):
    """Apply donor change ch at position t of the real break-free (hist, online). Points before t untouched."""
    M = NormalModel(np.r_[hist, online[:t]])
    wc = M.forward(online[t:])
    new = online.astype(np.float64).copy()
    new[t:] = M.inverse(ch.apply(wc, lam, lam_dep, keep_mean))
    return new


# ---------------------------------------------------------------------------------------------
def make_breakgen(series, tau, n, seed=0, lam=1.0, p_break=0.5, folds=None, min_post=100, match_k=0,
                  donors="break", pools=None, hist_pool=None, lam_dep=None, keep_mean=False, deconv=None, ids=None):
    """n pseudo-series per real series, same output and geometry as synth.make_synthetic:
    (parent_index, kind, lam, new_hist, new_online, new_tau); kind 'transplant' / 'placebo' / '' (no break).
    History lengths follow hist_pool (default: the histories of `series`), like augment.py's trimmed re-cuts.
    deconv: {fold: Deconv} (fit_deconv) -> every transplant uses a posterior draw of its donor's true change.
    ids: global index of each of `series` (when called on a chunk): random streams and the no-self-donor rule use it,
    so the output does not depend on how the series are chunked."""
    tau = np.asarray(tau)
    folds = fold_ids(tau) if folds is None else np.asarray(folds)
    if pools is None:
        pools = {k: donor_pool(series, tau, np.flatnonzero(folds == k), donors, min_post, seed=seed)
                 for k in np.unique(folds)}
    descs = {}
    for k, pl in pools.items():
        D = np.array([c.desc for c in pl])
        mu, sd = D.mean(0), D.std(0) + 1e-9
        descs[k] = ((D - mu) / sd, mu, sd)
    kind_name = "transplant" if donors == "break" else donors
    if hist_pool is None:
        hist_pool = [len(h) for _, h, _ in series]
    hpool = np.sort(np.asarray(hist_pool))
    out = []
    ids = np.arange(len(series)) if ids is None else np.asarray(ids)
    for i, (_, h, o) in enumerate(series):
        gi = int(ids[i])
        rng = np.random.default_rng([seed, 17, gi])
        B = break_free(h, o, int(tau[i]))
        nB = len(B)
        if nB < H_MIN + O_MIN:
            continue
        pool = pools[folds[i]]
        for _ in range(n):
            L = int(rng.integers(O_MIN, O_MAX + 1))
            L = min(L, nB - H_MIN)
            # history length from the ORIGINAL distribution (a uniform draw is a giveaway for any classifier)
            nh = draw_hist_len(min(H_MAX, nB - L), hpool, rng)
            if nh is None:
                nh = int(rng.integers(H_MIN, min(H_MAX, nB - L) + 1))
            start = int(rng.integers(0, nB - nh - L + 1))
            hist, onl = B[start:start + nh], B[start + nh:start + nh + L]
            if rng.random() < p_break:
                t = int(rng.integers(0, L))
                if match_k > 0:
                    Dz, mu, sd = descs[folds[i]]
                    d0 = (NormalModel(np.r_[hist, onl[:t]]).desc - mu) / sd
                    near = np.argsort(((Dz - d0) ** 2).sum(1))
                    near = [j for j in near[: match_k + 1] if pool[j].src != gi][:match_k]
                    ch = pool[int(near[rng.integers(len(near))])]
                else:
                    ch = pool[int(rng.integers(len(pool)))]
                    if ch.src == gi:
                        ch = pool[(pool.index(ch) + 1) % len(pool)]
                if deconv is not None:
                    ch = ch.with_comp(deconv[folds[i]].draw(ch.comp(), rng))
                out.append((i, kind_name, lam, hist.copy(), transplant(hist, onl, t, ch, lam, lam_dep, keep_mean), t))
            else:
                out.append((i, "", 0.0, hist.copy(), onl.copy(), -1))
    return out


def fit_deconv(pools_real, pools_plac):
    """{fold: Deconv} from same-fold real and placebo donor pools (no label of another fold is used)."""
    return {k: Deconv([c.comp() for c in pools_real[k]], [c.comp() for c in pools_plac[k]]) for k in pools_real}


# ---------------------------------------------------------------------------------------------
def _selftest(n_series=600):
    import os
    import sys
    import time
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from sbrt.data import load_train

    S, tau = load_train()
    tau = np.asarray(tau)
    folds_all = fold_ids(tau)
    rng = np.random.default_rng(0)

    # 1. forward/inverse are exact inverses; lam = 0 reproduces the data
    for i in range(20):
        _, h, o = S[i]
        M = NormalModel(h)
        x = o if tau[i] < 0 else o[: max(int(tau[i]), 10)]
        assert np.allclose(M.inverse(M.forward(x)), x, atol=1e-7), "forward/inverse not inverse"
        assert np.allclose(transplant(h, x, len(x) // 2, Change(rng.standard_normal(300) * 1.5, M.desc, -1), 0.0), x,
                           atol=1e-7)
    print("forward/inverse exact; lam = 0 is the identity: OK")

    # 2. an unchanged donor (w exactly the normal grid, no dependence) changes almost nothing
    ch0 = Change(rng.permutation(_normal_grid(400)), np.zeros(8), -1)
    _, h, o = S[1]
    x = o if tau[1] < 0 else o[: max(int(tau[1]), 10)]
    y = transplant(h, x, 3, ch0, 1.0)
    print(f"null donor: max |change| / sd = {np.abs(y - x).max() / h.std():.4f} (small)")
    assert np.abs(y - x).max() / h.std() < 0.2

    # 3. a pure variance x2 donor doubles the post-break variance of the whitened stream
    chv = Change(rng.permutation(_normal_grid(400)) * math.sqrt(2.0), np.zeros(8), -1)
    M = NormalModel(np.r_[h, x[:3]])
    y = transplant(h, x, 3, chv, 1.0)
    r = M.forward(y[3:]).var() / M.forward(x[3:]).var()
    print(f"variance x2 donor: whitened variance ratio {r:.3f} (expect ~2)")
    assert 1.6 < r < 2.4

    # 4. a full generation run: pre-break data real and untouched, donors in-fold
    t0 = time.time()
    sub = slice(0, n_series)
    V = make_breakgen(S[sub], tau[sub], 2, seed=0, lam=1.0, folds=folds_all[sub])
    dt = time.time() - t0
    nb = 0
    for (p, kind, lam, h2, o2, t2) in V:
        B = break_free(S[p][1], S[p][2], int(tau[p]))
        pre = np.r_[h2, o2[:t2]] if t2 >= 0 else np.r_[h2, o2]
        k = np.flatnonzero(B == pre[0])
        assert any(np.array_equal(B[j:j + len(pre)], pre) for j in k), "pre-break data must be real"
        assert H_MIN <= len(h2) <= H_MAX and O_MIN <= len(o2) <= O_MAX and np.isfinite(o2).all()
        nb += t2 >= 0
    print(f"{len(V)} variants ({nb} with a break) from {n_series} series in {dt:.1f} s: pre-break data real: OK")
    pools = {k: donor_pool(S[sub], tau[sub], np.flatnonzero(folds_all[sub] == k)) for k in range(5)}
    for k, pl in pools.items():
        assert all(folds_all[c.src] == k for c in pl)
    print("donor pools per fold:", {k: len(v) for k, v in pools.items()}, "all in-fold: OK")
    V2 = make_breakgen(S[sub], tau[sub], 1, seed=0, lam=1.0, folds=folds_all[sub], match_k=10, pools=pools)
    print(f"matched draws (k = 10): {len(V2)} variants: OK")
    # deconvolution: placebo donors deconvolved against their own noise shrink to ~no change
    plac = {k: donor_pool(S[:3000], tau[:3000], np.flatnonzero(folds_all[:3000] == k), "placebo") for k in range(2)}
    real = {k: donor_pool(S[:3000], tau[:3000], np.flatnonzero(folds_all[:3000] == k), "break") for k in range(2)}
    Dp = Deconv([c.comp() for c in plac[0]], [c.comp() for c in plac[1]])
    Dr = Deconv([c.comp() for c in real[0]], [c.comp() for c in plac[1]])
    print("components        :", Change.COMP)
    print("measured sd  real :", np.round(np.std([c.comp() for c in real[0]], 0), 3))
    print("noise sd (placebo):", np.round(Dr.sd, 3))
    print("true-change sd    :", np.round(Dr.prior_sd(), 3), "(real donors, deconvolved)")
    print("placebo prior sd  :", np.round(Dp.prior_sd(), 3), "(should be << noise sd)")
    assert (Dp.prior_sd() < Dr.sd).all()
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    _selftest()
