"""Boundary re-cut augmentation: exact-label extra training series.

A series is one stream  s = history ++ online  with (at most) one break at absolute position
A = len(history) + tau. The history/online boundary is a choice of the data generator; any other
boundary B is equally valid as long as the new history s[:B] contains no post-break point (B <= A).
Moving the boundary by a shift m (new boundary B = nh - m):

    m > 0  boundary moves EARLIER: the last m history points become the start of the online segment
    m < 0  boundary moves LATER:   the first |m| online points join the history (needs |m| <= tau)

    new history = s[:nh - m]     new online = s[nh - m:]     new tau = tau + m   (or -1 = no break)

Nothing is synthetic: every value is real and the label is exact. Constraints keep the variants
inside the competition's ranges (history 1000-5000, online 10-999). The shift is drawn so the
variants follow the training distribution: online length ~ uniform and, for break series, the break
position ~ uniform within the online segment (density along the feasible line is proportional to
1 / new_online_length). Shifts smaller than `min_shift` are skipped (near-copies of the original).

No re-standardisation is needed: the feature kernel (make_params) and the GRU input builder both
standardise by the mean/std of whatever history they are given.

Self-test:   python -m sbrt.augment
"""
import numpy as np

H_MIN, H_MAX = 1000, 5000
O_MIN, O_MAX = 10, 999


def feasible_shifts(nh, no, tau, min_shift=20):
    """All admissible shifts m for one series (tau = -1 means no break), excluding |m| < min_shift."""
    lo = max(nh - H_MAX, O_MIN - no)          # new history <= 5000, new online >= 10
    hi = min(nh - H_MIN, O_MAX - no)          # new history >= 1000, new online <= 999
    if tau >= 0:
        lo = max(lo, -tau)                    # the break must stay in the online segment
    if hi < lo:
        return np.empty(0, dtype=np.int64)
    m = np.arange(lo, hi + 1, dtype=np.int64)
    return m[np.abs(m) >= min_shift]


def recut(hist, online, tau, m):
    """Re-partition one series at shift m. Returns (new_hist, new_online, new_tau)."""
    nh = len(hist)
    s = np.concatenate([hist, online])
    b = nh - m
    new_tau = -1 if tau < 0 else int(tau + m)
    if tau >= 0:
        assert b <= nh + tau, "a post-break point would enter the history"
        assert 0 <= new_tau < len(s) - b
    return s[:b].copy(), s[b:].copy(), new_tau


def draw_shifts(nh, no, tau, n, rng, min_shift=20):
    """Up to n distinct shifts, weighted so the variants follow the training distribution."""
    m = feasible_shifts(nh, no, tau, min_shift)
    if len(m) == 0:
        return m
    w = 1.0 / (no + m) if tau >= 0 else np.ones(len(m))
    k = min(n, len(m))
    return rng.choice(m, size=k, replace=False, p=w / w.sum())


def draw_hist_len(avail, pool, rng):
    """A history length <= avail drawn from the ORIGINAL history-length distribution (pool = sorted lengths), so
    trimmed variants do not shift the distribution of the history-length feature. None if nothing fits."""
    k = int(np.searchsorted(pool, avail, side="right"))
    lo = int(np.searchsorted(pool, H_MIN, side="left"))
    if k <= lo:
        return None
    return int(pool[rng.integers(lo, k)])


def make_variants(series, tau, n, seed=0, min_shift=20, trim_p=0.0, n_trim_only=0, min_trim=0.2):
    """series: list of (id, hist, online); tau: array (-1 = no break).
    Returns list of (parent_index, shift, new_hist, new_online, new_tau).

    History trimming (the labels do not change: only the START of the history moves):
      trim_p       probability that a boundary re-cut also drops the start of its history (new length drawn from the
                   original history-length distribution, at most the available length)
      n_trim_only  extra variants per series that keep the original boundary and online segment but use a shorter
                   history (at least `min_trim` of it removed, so they are not near-copies); shift reported as 0
    Trimming draws use their own random stream, so trim_p = 0 and n_trim_only = 0 reproduce the old variants exactly."""
    out = []
    pool = np.sort(np.array([len(h) for _, h, _ in series]))
    for i, (_, h, o) in enumerate(series):
        rng = np.random.default_rng([seed, i])
        rt = np.random.default_rng([seed, i, 7]) if (trim_p > 0 or n_trim_only > 0) else None
        for m in draw_shifts(len(h), len(o), int(tau[i]), n, rng, min_shift):
            nh_, no_, nt = recut(h, o, int(tau[i]), int(m))
            if trim_p > 0 and rt.random() < trim_p:
                L = draw_hist_len(len(nh_), pool, rt)
                if L is not None and L < len(nh_):
                    nh_ = nh_[-L:].copy()
            out.append((i, int(m), nh_, no_, nt))
        for _ in range(n_trim_only):
            L = draw_hist_len(int(len(h) * (1.0 - min_trim)), pool, rt)
            if L is not None:
                out.append((i, 0, h[-L:].copy(), o.copy(), int(tau[i])))
    return out


REV_VERSION = 1   # bump when reverse() / make_reversed() change: cv.py cache stubs carry it (shift-0 copies keep vlen)


def reverse(hist, online, tau):
    """Time-reversed copy of one series with the same geometry (history length, online length, tau).

    The series are time-reversible (HANDOFF: backcasting carried the same information as forecasting), so a
    regime played backwards is a plausible draw of the same regime. Each regime is reversed on its own, so the
    break stays at the same absolute position and the label is exact:
        no break:  r = reverse(history ++ online)
        break:     r = reverse(pre-break block) ++ reverse(post-break block),  pre-break block = s[:nh + tau]
    new history = r[:nh] (only pre-break values, since nh <= nh + tau), new online = r[nh:], new tau = tau.
    The new history ends with values from the START of the original history: a different history-end state
    than any boundary re-cut can give. Like the re-cuts, nothing is re-standardised (the feature builders
    standardise by whatever history they are given)."""
    nh = len(hist)
    s = np.concatenate([hist, online])
    if tau < 0:
        r = s[::-1].copy()
    else:
        a = nh + int(tau)
        r = np.concatenate([s[:a][::-1], s[a:][::-1]])
    return r[:nh].copy(), r[nh:].copy(), int(tau)


def make_reversed(series, tau, n, seed=0, min_shift=20):
    """n time-reversed variants per series: the reversed series at its original boundary (shift 0), plus n - 1
    boundary re-cuts OF the reversed series (drawn like make_variants, own random stream).
    Returns list of (parent_index, shift, new_hist, new_online, new_tau), ordered by parent."""
    out = []
    for i, (_, h, o) in enumerate(series):
        rh, ro, rt = reverse(h, o, int(tau[i]))
        out.append((i, 0, rh, ro, rt))
        if n > 1:
            rng = np.random.default_rng([seed, i, 17])
            for m in draw_shifts(len(rh), len(ro), rt, n - 1, rng, min_shift):
                nh_, no_, nt = recut(rh, ro, rt, int(m))
                out.append((i, int(m), nh_, no_, nt))
    return out


FLIP_VERSION = 1   # bump when flip() / make_flipped() change


def flip(hist, online, tau):
    """Sign-reflected copy of one series: (-history, -online), same geometry and tau (HANDOFF section 40).

    Negation is invertible and keeps every change in scale, tails and dependence, so the label is exact. The
    history stays exactly z-scored (mean 0, sd 1). Unlike reverse() it keeps the order of the volatility path;
    what it changes is every signed odd-moment statistic (skew, mean shift, leverage)."""
    return (-hist).copy(), (-online).copy(), int(tau)


def make_flipped(series, tau, n, seed=0, min_shift=20):
    """n sign-reflected variants per series: the flipped series at its original boundary (shift 0), plus n - 1
    boundary re-cuts OF the flipped series (drawn like make_variants, own random stream).
    Returns list of (parent_index, shift, new_hist, new_online, new_tau), ordered by parent."""
    out = []
    for i, (_, h, o) in enumerate(series):
        fh, fo, ft = flip(h, o, int(tau[i]))
        out.append((i, 0, fh, fo, ft))
        if n > 1:
            rng = np.random.default_rng([seed, i, 23])
            for m in draw_shifts(len(fh), len(fo), ft, n - 1, rng, min_shift):
                nh_, no_, nt = recut(fh, fo, ft, int(m))
                out.append((i, int(m), nh_, no_, nt))
    return out


def make_views(series, tau, shift):
    """Test-time views (cv.py --tta): every series with its boundary moved EARLIER by `shift` points (the last
    `shift` history points become the start of the online part), online part cut to O_MAX points. Causal: the
    view's online step v = t + shift uses only points up to the original step t. Series whose history would drop
    below H_MIN get no view. Labels are not used for scoring; new_tau is -1 when the break falls past the cut.
    Returns list of (parent_index, new_hist, new_online, new_tau)."""
    out = []
    for i, (_, h, o) in enumerate(series):
        if len(h) - shift < H_MIN:
            continue
        nh_, no_, nt = recut(h, o, int(tau[i]), int(shift))
        no_ = no_[:O_MAX].copy()
        if nt >= len(no_):
            nt = -1
        out.append((i, nh_, no_, nt))
    return out


EARLY = 100   # early-break re-cuts: break in the first EARLY online points, online cut to its first EARLY points


def make_early_variants(series, tau, n, used, seed=0, min_shift=20, early=EARLY):
    """Extra re-cuts aimed at the first `early` online steps (HANDOFF section 43).

    Break series: up to n re-cuts whose break lands in the first `early` online points (new tau in [0, early)),
    drawn uniformly over the feasible positions. No-break series: up to n ordinary re-cuts (uniform over feasible
    shifts), then subsampled to the number of break variants (break share 0.5). Every variant's online segment is
    cut to its first `early` points, so the source adds rows only at steps < early. Shifts in `used` (dict parent ->
    set of shifts already taken by the ordinary re-cuts) are excluded. Own random streams: make_variants is unchanged.
    Returns list of (parent_index, shift, new_hist, new_online, new_tau), ordered by parent."""
    brk, nob = [], []
    for i, (_, h, o) in enumerate(series):
        t = int(tau[i])
        m = feasible_shifts(len(h), len(o), t, min_shift)
        if i in used:
            m = m[~np.isin(m, np.fromiter(used[i], dtype=np.int64))]
        if t >= 0:
            m = m[t + m < early]
        if len(m) == 0:
            continue
        rng = np.random.default_rng([seed, i, 11])
        for s in np.sort(rng.choice(m, size=min(n, len(m)), replace=False)):
            nh_, no_, nt = recut(h, o, t, int(s))
            (brk if t >= 0 else nob).append((i, int(s), nh_, no_[:early].copy(), nt))
    if len(nob) > len(brk):
        keep = np.sort(np.random.default_rng([seed, 13]).choice(len(nob), len(brk), replace=False))
        nob = [nob[k] for k in keep]
    return sorted(brk + nob, key=lambda v: (v[0], v[1]))


def step_weights_mixed(y, step, is_orig, aug_rel=1.0):
    """TS-AUC-matched weights when augmented rows are mixed in.

    The per-step target is taken from ORIGINAL rows only: at step t the positives together weigh
    npos_o*nneg_o/2 and the negatives the same (class-balanced, proportional to the metric's own
    weight npos*nneg). That total is shared among all rows of the step and class, original rows with
    relative share 1 and augmented rows with relative share `aug_rel`. With no augmented rows this
    equals the old formula (positive -> nneg/2, negative -> npos/2), up to the final normalisation.
    """
    y = np.asarray(y).astype(np.int64)
    step = np.asarray(step).astype(np.int64)
    is_orig = np.asarray(is_orig, dtype=bool)
    T = int(step.max()) + 1
    npos_o = np.bincount(step[is_orig], weights=y[is_orig], minlength=T)
    n_o = np.bincount(step[is_orig], minlength=T).astype(float)
    target = npos_o * (n_o - npos_o) / 2.0                     # per class, per step
    rel = np.where(is_orig, 1.0, aug_rel)
    key = step * 2 + y
    denom = np.bincount(key, weights=rel, minlength=2 * T)
    w = target[step] * rel / np.maximum(denom[key], 1e-12)
    return w / max(w.mean(), 1e-12)


def describe(orig_no, orig_tau, var_no, var_tau, out=print):
    """Side-by-side distribution check: online length, break position, positive share per step,
    time since break among positives."""
    def q(a, p=(0.1, 0.25, 0.5, 0.75, 0.9)):
        return np.quantile(a, p).round(2) if len(a) else "-"
    b, vb = orig_tau >= 0, var_tau >= 0
    out("                          original                              variants")
    out(f"online length     {q(orig_no)}   {q(var_no)}")
    out(f"tau / online      {q(orig_tau[b] / orig_no[b])}   {q(var_tau[vb] / var_no[vb])}")
    out(f"share with break  {b.mean():.3f}                                   {vb.mean():.3f}")
    for t in (0, 50, 100, 300, 600):
        pa = ((orig_tau >= 0) & (orig_tau <= t))[orig_no > t].mean()
        pv = ((var_tau >= 0) & (var_tau <= t))[var_no > t].mean()
        out(f"positive share at step {t:3d}: original {pa:.3f}   variants {pv:.3f}")
    for t in (100, 300, 600):
        ia = (orig_no > t) & (orig_tau >= 0) & (orig_tau <= t)
        iv = (var_no > t) & (var_tau >= 0) & (var_tau <= t)
        out(f"time since break among positives at step {t}: original {q(t - orig_tau[ia], (0.25, 0.5, 0.75))}"
            f"   variants {q(t - var_tau[iv], (0.25, 0.5, 0.75))}")


def _selftest(n_series=3000):
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from sbrt.data import load_train
    from sbrt import features as F

    S, tau = load_train()
    S, tau = S[:n_series], tau[:n_series]
    V = make_variants(S, tau, 4, seed=0)
    print(f"{len(V)} variants from {len(S)} series ({len(V) / len(S):.2f} per series)")
    for (p, m, h2, o2, t2) in V:
        _, h, o = S[p]
        assert np.array_equal(np.r_[h2, o2], np.r_[h, o]), "re-cut must only move the boundary"
        assert H_MIN <= len(h2) <= H_MAX and O_MIN <= len(o2) <= O_MAX
        assert abs(m) >= 20
        if tau[p] < 0:
            assert t2 == -1
        else:
            A = len(h) + tau[p]                  # absolute break position
            assert len(h2) <= A and len(h2) + t2 == A
            y_new = np.arange(len(o2)) >= t2
            y_abs = (len(h2) + np.arange(len(o2))) >= A
            assert np.array_equal(y_new, y_abs)
    print("re-cut invariants: OK (same values, ranges, no post-break point in history, exact labels)")

    for i in range(5):
        _, h, o = S[i]
        h0, o0, _ = recut(h, o, int(tau[i]), 0)
        assert np.array_equal(F.run_series(h0, o0), F.run_series(h, o))
    print("shift 0 reproduces the original feature matrix: OK")

    VR = make_reversed(S, tau, 3, seed=0)
    for (p, m, h2, o2, t2) in VR:
        _, h, o = S[p]
        t = int(tau[p])
        assert H_MIN <= len(h2) <= H_MAX and O_MIN <= len(o2) <= O_MAX
        rh, ro, rt = reverse(h, o, t)
        if m == 0:
            assert len(h2) == len(h) and len(o2) == len(o) and t2 == t
            hh, oo, tt = reverse(h2, o2, t2)
            assert np.array_equal(hh, h) and np.array_equal(oo, o) and tt == t, "reversing twice must give the original"
        else:
            assert abs(m) >= 20 and np.array_equal(np.r_[h2, o2], np.r_[rh, ro])
        if t < 0:
            assert t2 == -1
        else:   # follow the original positions through the same reversal (and re-cut): exact-label check
            ih, io_, _ = reverse(np.arange(len(h)), np.arange(len(h), len(h) + len(o)), t)
            if m != 0:
                ih, io_, _ = recut(ih, io_, t, int(m))
            ri = np.r_[ih, io_]
            assert np.array_equal(np.r_[h, o][ri], np.r_[h2, o2])
            a = len(h) + t
            assert (ri[:len(h2) + t2] < a).all() and (ri[len(h2) + t2:] >= a).all(), "labels must stay exact"
    print(f"reversed variants: {len(VR)} from {len(S)} series; invariants OK (exact labels, involution, ranges)")

    VF = make_flipped(S, tau, 3, seed=0)
    for (p, m, h2, o2, t2) in VF:
        _, h, o = S[p]
        t = int(tau[p])
        assert H_MIN <= len(h2) <= H_MAX and O_MIN <= len(o2) <= O_MAX
        if m == 0:
            assert np.array_equal(h2, -h) and np.array_equal(o2, -o) and t2 == t
            hh, oo, tt = flip(h2, o2, t2)
            assert np.array_equal(hh, h) and np.array_equal(oo, o) and tt == t, "flipping twice must give the original"
        else:
            assert abs(m) >= 20 and np.array_equal(np.r_[h2, o2], -np.r_[h, o])
            if t >= 0:
                assert len(h2) + t2 == len(h) + t, "break must stay at the same absolute position"
        if t < 0:
            assert t2 == -1
    print(f"flipped variants: {len(VF)} from {len(S)} series; invariants OK (exact labels, involution, ranges)")

    for sh in (51, 150):
        VW = make_views(S, tau, sh)
        for (p, h2, o2, t2) in VW:
            _, h, o = S[p]
            assert len(h2) == len(h) - sh and len(h2) >= H_MIN and len(o2) <= O_MAX
            assert np.array_equal(np.r_[h2, o2], np.r_[h, o][:len(h2) + len(o2)])
            assert len(o2) == min(len(o) + sh, O_MAX)   # view step v <-> original step v - sh, nothing later
    print("test-time views: same values, boundary earlier by the shift, online cut to 999: OK")

    rng = np.random.default_rng(1)
    step = rng.integers(0, 50, 20000)
    y = (rng.random(20000) < 0.1 + step / 100).astype(np.int8)
    T = step.max() + 1
    npos = np.bincount(step, weights=y, minlength=T)
    n = np.bincount(step, minlength=T).astype(float)
    w_old = np.where(y == 1, (n - npos)[step] / 2.0, npos[step] / 2.0)
    w_old /= w_old.mean()
    w_new = step_weights_mixed(y, step, np.ones(len(y), bool))
    assert np.allclose(w_old, w_new), np.abs(w_old - w_new).max()
    y2 = np.r_[y, (rng.random(30000) < 0.5).astype(np.int8)]
    s2 = np.r_[step, rng.integers(0, 50, 30000)]
    io = np.r_[np.ones(20000, bool), np.zeros(30000, bool)]
    w2 = step_weights_mixed(y2, s2, io, aug_rel=0.5)
    tot = np.bincount(s2 * 2 + y2, weights=w2, minlength=2 * T).reshape(T, 2)
    ref = np.bincount(step * 2 + y, weights=w_old, minlength=2 * T).reshape(T, 2)
    assert np.allclose(tot / tot.sum(), ref / ref.sum())
    print("step weights: equal to the old formula without augmentation, per-step totals preserved with it: OK")

    no = np.array([len(s[2]) for s in S])
    vo = np.array([len(v[3]) for v in V])
    vt = np.array([v[4] for v in V])
    print()
    describe(no, tau, vo, vt)
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    _selftest()
