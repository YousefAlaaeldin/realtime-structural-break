"""Break SHAPE, location PERSISTENCE, detector AGREEMENT and symbol TRANSITIONS (HANDOFF section 20; owner's request
2026-10-18). Columns are appended only with cv.py --shape.

Why. Two facts from the project's own analysis: breaks are INSTANT switches (section 3, event study), and the main
error is break-free series whose variance WANDERS smoothly (GARCH-like). Every detector here compares "break" with
"no change"; none has a smooth-drift alternative, none looks at how its break-location estimate behaves over time,
and the location estimates of different detectors are never compared. All inputs are the bocpd.py whitened streams
(w = history-ECDF-whitened AR(10) residual, wu = the same after the frozen volatility forecast), so everything is
causal and ~N(0, 1) before a break.

12 features at online step t (cumulative sums, O(t) per step):
 A. step vs drift
    sh_step_ll       max over j of the variance-step log LR on w: 0.5 m (v - 1 - log v), v = mean w^2 over [j, t]
    sh_ramp_ll       max over a grid of drift rates b of the log LR of a smooth log-variance ramp log s2_i = b i
                     (from the history's level at the start of the online part)
    sh_step_vs_ramp  sh_step_ll - sh_ramp_ll: > 0 = the change looks like a jump, < 0 = like a drift
 B. location persistence (L* = argmax j of the variance step on w, tracked every step)
    sh_lstar_age     (t - L* + 1) / (t + 1): how far back the estimated break is
    sh_lock_frac     (t - last time L* jumped by more than max(3, 10% of t - L*)) / (t + 1): a real break locks L*
    sh_lsd50         sd of L* over the last 50 steps / (t + 1): noise keeps moving it
 C. agreement of four location estimates on different information
    sh_agree_wu      |L*(w) - L*(wu)| / (t + 1)          variance step after the volatility forecast
    sh_agree_dep     |L*(w) - L*(dep)| / (t + 1)         mean step of w_i w_(i-1) (dependence)
    sh_agree_tail    |L*(w) - L*(tail)| / (t + 1)        rate step of 1{|w| > 1.96} (tails)
    sh_dir_agree     sign(log v at L*(w)) * sign(log v at L*(wu)): +1 both streams say "variance up" (or down)
 D. symbol transitions (owner's "symbols repeat in a stable signal" idea, 2026-10-18)
    symbols = quartile of w (by the history's own ECDF); the history's 4 x 4 transition table (Dirichlet 1) is the
    reference; online transitions are tested against it given their from-symbol (so the test is about DEPENDENCE
    between consecutive symbols, not their frequencies, which the decile chi-square already has):
    sh_sym_chi       (chi2 - 12) / sqrt(24) over all online transitions
    sh_sym_chi100    the same over the last 100 transitions
    Prior art in the project (section 8): decile chi-square (unigram, in production), ordinal patterns (no signal),
    matrix profile / motif matching (no signal), MMD on pairs/triples (-0.0004, duplicates --dist). D is the one
    untried variant (conditional transitions on the whitened residual) and the most likely to be redundant.
Also considered and NOT built: the peakedness of the change profile (= the location-posterior entropy of the closed
--pshape, section 8).

DECISION RULE (written before any run): one family, one flag, the standing rule (HANDOFF section 5):
  1. cv.py trees seed 0 with --shape vs the current best seed-0 trees. Early stop: folds 0 and 1 both <= 0 -> CLOSED.
  2. 5-fold paired mean > 0: seed 1, seed-average comparison, trees-only blend.
  3. ADOPT if the blend gains >= +0.0005, >= 4/5 folds, > 2 SE. Production then needs a streaming version (the batch
     kernel is causal; a stream class is not built yet) and the GRU/SVM inputs.
  Reported, not a gate: single-feature TS-AUC of each column (self-test), and each sub-family's share of tree gain.
EXPECTATION (written before any run): ~20-25% chance to pass; blend +0.000 to +0.002.

Self-test (causality, batch == prefix runs, single-feature TS-AUC on series 5000-6999):  python -m sbrt.shape
"""
import math

import numpy as np
from numba import njit

from sbrt import bocpd as BO
from sbrt import evidence as EV
from sbrt import volstate as VS

VERSION = 1
FEATURE_NAMES = ["sh_step_ll", "sh_ramp_ll", "sh_step_vs_ramp",
                 "sh_lstar_age", "sh_lock_frac", "sh_lsd50",
                 "sh_agree_wu", "sh_agree_dep", "sh_agree_tail", "sh_dir_agree",
                 "sh_sym_chi", "sh_sym_chi100"]
N_SH = len(FEATURE_NAMES)
MINSEG = 5
TAIL = 1.96
P_TAIL = 0.05
RAMP_B = np.linspace(-0.01, 0.01, 41)       # drift of the log variance per step: up to +-3 over 300 steps
Q1, Q2, Q3 = -0.6744897501960817, 0.0, 0.6744897501960817   # normal quartiles (w is on the normal scale)


@njit(cache=True, nogil=True)
def _sym(w):
    if w < Q1:
        return 0
    if w < Q2:
        return 1
    if w < Q3:
        return 2
    return 3


@njit(cache=True, nogil=True)
def _chi(T, P):
    c = 0.0
    for a in range(4):
        na = 0.0
        for b in range(4):
            na += T[a, b]
        if na <= 0:
            continue
        for b in range(4):
            e = na * P[a, b]
            d = T[a, b] - e
            c += d * d / e
    return (c - 12.0) / math.sqrt(24.0)


@njit(cache=True, nogil=True)
def _best_var(C, t, j0):
    """argmax over j in [j0, t - MINSEG + 1] of the variance-step LR on cumulative squares C; (L*, LR, log v)."""
    best, bj, blv = 0.0, t, 0.0
    for j in range(j0, t - MINSEG + 2):
        m = t - j + 1
        v = (C[t + 1] - C[j]) / m
        if v < 1e-8:
            v = 1e-8
        lr = 0.5 * m * (v - 1.0 - math.log(v))
        if lr > best:
            best, bj, blv = lr, j, math.log(v)
    return bj, best, blv


@njit(cache=True, nogil=True)
def _run(w, wu, P, B, sub, out):
    n = len(w)
    Cw = np.zeros(n + 1)
    Cu = np.zeros(n + 1)
    Cp = np.zeros(n + 1)
    Ca = np.zeros(n + 1)
    R = np.zeros(len(B))
    T = np.zeros((4, 4))
    T100 = np.zeros((4, 4))
    ring = np.zeros(100, dtype=np.int64)       # transition codes a * 4 + b of the last 100 transitions
    ringL = np.zeros(50)
    prevL, lock = -1, 0
    r = 0
    for t in range(n):
        x, xu = w[t], wu[t]
        Cw[t + 1] = Cw[t] + x * x
        Cu[t + 1] = Cu[t] + xu * xu
        Cp[t + 1] = Cp[t] + (x * w[t - 1] if t >= 1 else 0.0)
        Ca[t + 1] = Ca[t] + (1.0 if abs(x) > TAIL else 0.0)
        for k in range(len(B)):
            R[k] += x * x * math.exp(-B[k] * t)
        if t >= 1:
            a, b = _sym(w[t - 1]), _sym(x)
            T[a, b] += 1.0
            nt = t - 1                          # index of this transition (0-based)
            if nt >= 100:
                old = ring[nt % 100]
                T100[old // 4, old % 4] -= 1.0
            ring[nt % 100] = a * 4 + b
            T100[a, b] += 1.0
        # variance-step location on w, every step (persistence needs its path)
        Lw, llw, lvw = _best_var(Cw, t, 0)
        if t >= MINSEG - 1:
            if prevL >= 0 and abs(Lw - prevL) > max(3.0, 0.1 * (t - Lw)):
                lock = t
            prevL = Lw
        else:
            lock = t
        ringL[t % 50] = Lw
        if t % sub != 0:
            continue
        o = out[r]
        r += 1
        if t < MINSEG - 1:
            for c in range(out.shape[1]):
                o[c] = 0.0
            continue
        Lu, llu, lvu = _best_var(Cu, t, 0)
        # dependence: mean step of p_i = w_i w_(i-1), i >= 1, null mean 0 and variance ~1
        bd, Ld = 0.0, t
        for j in range(1, t - MINSEG + 2):
            m = t - j + 1
            mu = (Cp[t + 1] - Cp[j]) / m
            lr = 0.5 * m * mu * mu
            if lr > bd:
                bd, Ld = lr, j
        # tails: Bernoulli rate step of 1{|w| > TAIL}, null rate P_TAIL
        bt, Lt = 0.0, t
        for j in range(0, t - MINSEG + 2):
            m = t - j + 1
            q = (Ca[t + 1] - Ca[j]) / m
            lr = 0.0
            if q > 0:
                lr += q * math.log(q / P_TAIL)
            if q < 1:
                lr += (1 - q) * math.log((1 - q) / (1 - P_TAIL))
            lr *= m
            if lr > bt:
                bt, Lt = lr, j
        # smooth ramp of the log variance from the history's level
        rb = 0.0
        tt = t * (t + 1) / 2.0
        for k in range(len(B)):
            v = -0.5 * (B[k] * tt + R[k] - Cw[t + 1])
            if v > rb:
                rb = v
        cnt = min(t + 1, 50)
        m1 = 0.0
        for i in range(cnt):
            m1 += ringL[i]
        m1 /= cnt
        s2 = 0.0
        for i in range(cnt):
            s2 += (ringL[i] - m1) ** 2
        den = t + 1.0
        o[0] = llw
        o[1] = rb
        o[2] = llw - rb
        o[3] = (t - Lw + 1) / den
        o[4] = (t - lock) / den
        o[5] = math.sqrt(s2 / cnt) / den
        o[6] = abs(Lw - Lu) / den
        o[7] = abs(Lw - Ld) / den
        o[8] = abs(Lw - Lt) / den
        o[9] = (1.0 if lvw >= 0 else -1.0) * (1.0 if lvu >= 0 else -1.0)
        o[10] = _chi(T, P)
        o[11] = _chi(T100, P)


def history_table(hist):
    """4 x 4 transition probabilities of the history's whitened-residual quartile symbols (Dirichlet 1)."""
    H = VS.fit_history(hist)
    eh = np.asarray(H["eh"], dtype=np.float64)
    ref = np.sort(eh)
    s = np.array([_sym(EV._whiten(e, ref)) for e in eh])
    T = np.ones((4, 4))
    np.add.at(T, (s[:-1], s[1:]), 1.0)
    return T / T.sum(1, keepdims=True)


def run_series_shape(hist, online, sub=1):
    """Rows for online steps 0, sub, 2 sub, ... (N_SH columns); causal: row t uses online[:t + 1] only."""
    online = np.asarray(online, dtype=np.float64)
    W = BO.whitened_online(hist, online)
    P = history_table(hist)
    out = np.zeros(((len(online) + sub - 1) // sub, N_SH))
    _run(np.ascontiguousarray(W[0]), np.ascontiguousarray(W[1]), P, RAMP_B, sub, out)
    return out


def _selftest(n_series=2000, offset=5000, threads=8):
    import time
    from joblib import Parallel, delayed
    from sbrt.data import load_train
    from sbrt.metric import ts_auc
    S, tau = load_train()
    # 1. causality: a truncated online segment gives exactly the prefix rows (and nothing after it changes them)
    rng = np.random.default_rng(0)
    for i in rng.choice(len(S), 20, replace=False):
        _, h, o = S[i]
        full = run_series_shape(h, o, 1)
        cut = int(rng.integers(1, len(o) + 1))
        part = run_series_shape(h, o[:cut], 1)
        assert np.array_equal(full[:cut], part), f"series {i}: prefix rows differ"
        alt = np.r_[o[:cut], rng.normal(0, 3, len(o) - cut)]
        assert np.array_equal(run_series_shape(h, alt, 1)[:cut], full[:cut]), f"series {i}: future leaked"
    eh_len = len(VS.fit_history(S[0][1])["eh"])
    print(f"causality OK on 20 series (prefix and altered-future checks); history residuals per series ~{eh_len}")
    # 2. single-feature TS-AUC on held-out series
    t0 = time.time()
    idx = range(offset, offset + n_series)
    mats = Parallel(n_jobs=threads, backend="threading")(delayed(run_series_shape)(S[i][1], S[i][2], 3) for i in idx)
    print(f"{n_series} series in {time.time() - t0:.0f}s")
    X = np.concatenate(mats)
    step = np.concatenate([np.arange(0, len(S[i][2]), 3) for i in idx])
    y = np.concatenate([((np.arange(0, len(S[i][2]), 3) >= tau[i]) & (tau[i] >= 0)) for i in idx]).astype(np.int8)
    print("single-feature TS-AUC (0.5 = none; < 0.5 = informative with the opposite sign):")
    for j, nm in enumerate(FEATURE_NAMES):
        a = ts_auc(X[:, j], y, step)
        print(f"  {nm:16s} {a:.4f}   |0.5 distance| {abs(a - 0.5):.4f}")
    print("self-test OK")


if __name__ == "__main__":
    _selftest()
