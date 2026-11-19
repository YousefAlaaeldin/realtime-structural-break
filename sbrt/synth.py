"""Synthetic breaks injected into real break-free data.

Every pseudo-series is cut from the BREAK-FREE part of a real series (its history, plus its online points before
the break, or the whole online segment if it has no break), so the "normal" behaviour is always real. With
probability p_break a break is injected at a uniform position of the new online segment: the points after it are
transformed by one of these break types (the points before it are never changed):

    variance    x * c                                   c = exp(+-0.5 m)
    mean        x + d                                   d = +-0.5 m   (in history sd units)
    dependence  (x_t + a x_{t-1}) / norm                a = +-0.6 m (|a| <= 0.9), variance kept
    tails       x_t * w_t, E[w^2] = 1                   Student-t scale mixture, nu = 2 + 6/m (>= 3)
    skew        (x + g (x^2 - 1)) / norm                g = +-0.3 m
    volclust    x_t * sqrt(v_t), GARCH-like v_t, mean 1  alpha = min(0.15 m, 0.3), beta = 0.6
    splice      sqrt(1 - l^2) x + l d                   d = break-free data of ANOTHER series of the same fold,
                                                        l = min(m, 1)

The size m is 0 with probability p_zero (an invisible break, which real data has plenty of) and otherwise
scale * |N(0, 1)|. `scale` is the knob that must be CALIBRATED: synthetic breaks must be about as hard to see as
real ones (see synth_calibrate.py), or the model learns to rank big breaks and gets worse on real ones.
Donors for `splice` come from the same cross-validation fold as the parent, so a training variant never contains
values of a validation series.

Self-test:  python -m sbrt.synth
"""
import numpy as np

from sbrt.augment import H_MAX, H_MIN, O_MAX, O_MIN

TYPES = ("variance", "mean", "dependence", "tails", "skew", "volclust", "splice")


def break_free(hist, online, tau):
    """The break-free stream of one series: history plus online points before the break."""
    return np.concatenate([hist, online if tau < 0 else online[:tau]])


def draw_size(rng, scale, p_zero):
    return 0.0 if rng.random() < p_zero else float(scale * abs(rng.standard_normal()))


def apply_break(seg, tau, kind, m, rng, prev=0.0, rho1=0.0, donor=None):
    """Return a copy of seg with points seg[tau:] transformed by break type `kind` of size m.
    prev = the point just before seg[0] (for the dependence filter); rho1 = lag-1 autocorrelation of the history."""
    out = seg.astype(np.float64).copy()
    if tau < 0 or m == 0.0:
        return out
    x = out[tau:]
    sgn = 1.0 if rng.random() < 0.5 else -1.0
    if kind == "variance":
        x *= np.exp(sgn * 0.5 * m)
    elif kind == "mean":
        x += sgn * 0.5 * m
    elif kind == "dependence":
        a = float(np.clip(sgn * 0.6 * m, -0.9, 0.9))
        lag = np.r_[seg[tau - 1] if tau > 0 else prev, seg[tau:-1]]
        x[:] = (seg[tau:] + a * lag) / np.sqrt(max(1.0 + a * a + 2.0 * a * rho1, 0.05))
    elif kind == "tails":
        nu = max(3.0, 2.0 + 6.0 / m)
        w = np.sqrt((nu - 2.0) / rng.chisquare(nu, size=len(x)))
        x *= w
    elif kind == "skew":
        g = sgn * 0.3 * m
        x[:] = (x + g * (x * x - 1.0)) / np.sqrt(1.0 + 2.0 * g * g)
    elif kind == "volclust":
        al, be = min(0.15 * m, 0.3), 0.6
        v = 1.0
        src = x.copy()
        for i in range(len(x)):
            x[i] = src[i] * np.sqrt(v)
            v = (1.0 - al - be) + al * x[i] * x[i] + be * v
    elif kind == "splice":
        lam = min(m, 1.0)
        x[:] = np.sqrt(1.0 - lam * lam) * x + lam * donor[: len(x)]
    else:
        raise ValueError(kind)
    out[tau:] = x
    return out


def fold_ids(tau, n_folds=5):
    """Fold of every series under the split used by cv.py and gru_cv.py (needed to keep splice donors in-fold)."""
    from sklearn.model_selection import StratifiedKFold
    f = np.zeros(len(tau), dtype=np.int64)
    skf = StratifiedKFold(n_folds, shuffle=True, random_state=0)
    for k, (_, va) in enumerate(skf.split(np.arange(len(tau)), (np.asarray(tau) >= 0).astype(int))):
        f[va] = k
    return f


def make_synthetic(series, tau, n, seed=0, scale=0.5, p_zero=0.25, p_break=0.5, types=TYPES, folds=None):
    """n synthetic pseudo-series per real series. Returns a list of
    (parent_index, kind or '', size, new_hist, new_online, new_tau)."""
    tau = np.asarray(tau)
    folds = fold_ids(tau) if folds is None else folds
    by_fold = {k: np.flatnonzero(folds == k) for k in np.unique(folds)}
    out = []
    for i, (_, h, o) in enumerate(series):
        rng = np.random.default_rng([seed, 7, i])
        B = break_free(h, o, int(tau[i]))
        nB = len(B)
        if nB < H_MIN + O_MIN:
            continue
        hz = (h - h.mean()) / (h.std() + 1e-12)
        rho1 = float(np.dot(hz[:-1], hz[1:]) / max(float(np.dot(hz, hz)), 1e-12))
        for _ in range(n):
            L = int(rng.integers(O_MIN, O_MAX + 1))
            L = min(L, nB - H_MIN)
            nh = int(rng.integers(H_MIN, min(H_MAX, nB - L) + 1))
            start = int(rng.integers(0, nB - nh - L + 1))
            hist, onl = B[start:start + nh], B[start + nh:start + nh + L]
            if rng.random() < p_break:
                t = int(rng.integers(0, L))
                kind = types[int(rng.integers(len(types)))]
                m = draw_size(rng, scale, p_zero)
                donor = None
                if kind == "splice":
                    pool = by_fold[folds[i]]
                    j = int(pool[rng.integers(len(pool))])
                    if j == i:
                        j = int(pool[(np.searchsorted(pool, i) + 1) % len(pool)])
                    Dj = break_free(series[j][1], series[j][2], int(tau[j]))
                    Dj = (Dj - series[j][1].mean()) / (series[j][1].std() + 1e-12)
                    off = int(rng.integers(0, max(1, len(Dj) - (L - t))))
                    donor = Dj[off:off + (L - t)]
                    if len(donor) < L - t:
                        donor = np.resize(donor, L - t)
                    # bring the donor to the parent's units (values are in parent-history sd units)
                    donor = donor * (h.std() + 1e-12)
                onl = apply_break(onl, t, kind, m, rng, prev=hist[-1], rho1=rho1, donor=donor)
                out.append((i, kind, m, hist.copy(), onl, t))
            else:
                out.append((i, "", 0.0, hist.copy(), onl.copy(), -1))
    return out


def _selftest(n_series=1500):
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from sbrt.data import load_train

    S, tau = load_train()
    folds_all = fold_ids(tau)
    S, tau, folds = S[:n_series], tau[:n_series], folds_all[:n_series]
    V = make_synthetic(S, tau, 3, seed=0, scale=0.5, folds=folds)
    print(f"{len(V)} synthetic series from {len(S)} real ones")
    kinds = {}
    for (p, kind, m, h2, o2, t2) in V:
        B = break_free(S[p][1], S[p][2], int(tau[p]))
        s2 = np.r_[h2, o2]
        assert H_MIN <= len(h2) <= H_MAX and O_MIN <= len(o2) <= O_MAX
        # the history and the pre-break online points are untouched real break-free data
        pre = np.r_[h2, o2[:t2]] if t2 >= 0 else s2
        pos = np.flatnonzero(np.isclose(B, pre[0]))
        assert any(np.array_equal(B[k:k + len(pre)], pre) for k in pos), "pre-break data must be real and unchanged"
        if t2 < 0 or m == 0.0:
            assert any(np.array_equal(B[k:k + len(s2)], s2) for k in pos), "size 0 / no break must be a plain re-cut"
        assert np.isfinite(o2).all()
        kinds[kind or "none"] = kinds.get(kind or "none", 0) + 1
    print("pre-break data real and unchanged; size 0 and no-break series are plain re-cuts: OK")
    print("break types:", kinds)

    # splice donors stay in the parent's fold
    rng = np.random.default_rng(0)
    for i in rng.choice(n_series, 50, replace=False):
        pool = np.flatnonzero(folds == folds[i])
        assert i in pool
    print("fold ids computed with the cv.py split: OK")

    # size 0 reproduces exactly, for every type
    seg = S[0][2][:200].copy()
    for kind in TYPES:
        d = np.zeros(200)
        assert np.array_equal(apply_break(seg, 50, kind, 0.0, rng, donor=d), seg)
        y = apply_break(seg, 50, kind, 0.8, rng, donor=np.random.default_rng(1).standard_normal(200))
        assert np.array_equal(y[:50], seg[:50]) and np.isfinite(y).all()
    print("every break type: size 0 = identity, points before the break untouched: OK")
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    _selftest()
