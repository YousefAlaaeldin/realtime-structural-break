"""Whole-distribution tests on the NBM's whitened prediction-error stream (cv.py --nbm-dist, HANDOFF section 41).

The question (teammate research G04, "conditional quantile coverage"): sbrt/nbm.py already turns each online point
into w_t = Phi^-1(ECDF_hist(v_t)), with v_t = (e_t - m_t) / s_t the NBM's standardised one-step error, i.e. a
conditional PIT of the point given the whole past. So far w only feeds a Gaussian BOCPD posterior (nb_ columns),
which reacts to its mean and variance. A change in the SHAPE of the conditional distribution (tails, asymmetry,
coverage of the conditional quantiles) with the same mean and variance is invisible to it. sbrt/disttest.py tests
shape, but only on the marginal AR residual e and the GARCH-standardised u, not on the NBM's conditional stream.

1. Reference: the history's own whitened values. Each in-history v is whitened against the sorted in-history v
   (itself included), so the reference sample is exactly Phi^-1((k + 0.5) / (m + 1)), k = 0 .. m - 1, with
   m = number of in-history v (nbm._init_core: rows >= LOSS_FROM + 1, so m = len(hist) - AR - LOSS_FROM - 1).
   Edges, reference CDF and weights come from disttest._ref_tables, as for e and u.
2. The same 8 statistics as disttest.py per variable, prefix nbd_: KS, AD, CvM, W1 over all online points; KS and AD
   over the last 100; KS and AD scanned over every start j (segment >= 20, max over j). disttest._seg does the maths.
3. Cost: no extra GRU pass. cv.py takes w from the NBM sums it already builds per fold (nbm.w_of); in production
   the stream would take the w that nbm_push returns. The scan is only evaluated on the kept rows (every SUB-th),
   counts at every step, so batch rows equal the streaming values exactly.

Self-test (seconds):  python -m sbrt.nbdist
"""
import os
import sys

import numpy as np
from numba import njit
from scipy.special import ndtri

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sbrt import disttest as DT   # noqa: E402
from sbrt import features as F    # noqa: E402
from sbrt import nbm as NBM       # noqa: E402

VERSION = 1
CAP = F.CAP
FEATURE_NAMES = [f"nbd_{s}" for s in DT.STATS]
N_NBD = len(FEATURE_NAMES)


def n_ref(hist_len):
    """Number of in-history whitened values the NBM keeps for a history of this length."""
    return max(int(hist_len) - F.AR - NBM.LOSS_FROM - 1, 2)


def make_params(hist_len):
    m = n_ref(hist_len)
    ref = ndtri((np.arange(m) + 0.5) / (m + 1.0))
    E, PR, W, DX = (a[None, :] for a in DT._ref_tables(ref))
    return (E, PR, W, DX, np.array([float(m)])), np.zeros((1, DT.NB, CAP + 1)), np.zeros(1)


@njit(cache=True, nogil=True)
def nbd_step(w, E, PR, W, DX, M, C, ST, out, do_scan):
    """One online value of the whitened stream: update the counts, write the 8 statistics into out."""
    t = int(ST[0])
    n = t + 1
    b = np.searchsorted(E[0], w)
    for k in range(DT.NB):
        C[0, k, n] = C[0, k, t]
    C[0, b, n] += 1.0
    DT._seg(C, 0, 0, n, PR, W, DX, M, out, 0, True)
    wl = 100 if n > 100 else n
    DT._seg(C, 0, n - wl, n, PR, W, DX, M, out, 4, False)
    if do_scan:
        tmp = np.zeros(2)
        kmax = 0.0
        amax = 0.0
        for j in range(0, n - DT.MINSEG + 1):
            DT._seg(C, 0, j, n, PR, W, DX, M, tmp, 0, False)
            if tmp[0] > kmax:
                kmax = tmp[0]
            if tmp[1] > amax:
                amax = tmp[1]
        out[6] = kmax
        out[7] = amax
    ST[0] = n * 1.0


@njit(cache=True, nogil=True)
def _run(w, E, PR, W, DX, M, C, ST, out, sub):
    r = 0
    row = np.zeros(out.shape[1])
    for i in range(len(w)):
        keep = i % sub == 0
        nbd_step(w[i], E, PR, W, DX, M, C, ST, row, keep)
        if keep:
            for c in range(out.shape[1]):
                out[r, c] = row[c]
            r += 1


def run_series(hist_len, w, sub=1):
    """Batch matrix (rows for online steps 0, sub, 2 sub, ..., N_NBD) from the whitened stream w of one series."""
    T, C, ST = make_params(hist_len)
    w = np.asarray(w, dtype=np.float64)
    if len(w) >= CAP:
        raise ValueError(f"online segment longer than the supported {CAP - 1} points")
    out = np.zeros(((len(w) + sub - 1) // sub, N_NBD))
    _run(w, *T, C, ST, out, sub)
    return out


def from_sums(hist_len, sums, sub=1):
    """cv.py: the 8 columns from the NBM sums of nbm.run_stream (w = first differences of the cumulative sum)."""
    return run_series(hist_len, NBM.w_of(sums), sub)


class NbdStream:
    """One series' streaming state (inference): push the w that nbm_push returns for each online point."""

    def __init__(self, hist_len):
        self.T, self.C, self.ST = make_params(hist_len)
        self.out = np.zeros(N_NBD)

    def push(self, w):
        if self.ST[0] >= CAP - 1:
            return self.out
        nbd_step(float(w), *self.T, self.C, self.ST, self.out, True)
        return self.out


# ---------------------------------------------------------------------------------------------
def _selftest():
    rng = np.random.default_rng(0)
    # 1. streaming == batch (all rows and every 3rd row), causal, deterministic, finite
    for it in range(20):
        nh = int(rng.integers(1000, 5001))
        no = int(rng.integers(10, 1000))
        w = rng.standard_normal(no)
        B = run_series(nh, w)
        st = NbdStream(nh)
        Sm = np.array([st.push(x).copy() for x in w])
        assert np.array_equal(B, Sm), "streaming != batch"
        assert np.array_equal(run_series(nh, w, 3), Sm[::3]), "sub-sampled batch != streaming"
        assert np.isfinite(B).all()
        k = no // 2
        w2 = w.copy()
        w2[k:] = rng.standard_normal(no - k) * 3.0
        assert np.array_equal(run_series(nh, w2)[:k], B[:k]), "not causal"
        assert np.array_equal(run_series(nh, w), B), "not deterministic"
    print("streaming == batch (also every 3rd row), causal, finite, deterministic: OK")

    # 2. the reference size matches what nbm.py keeps (len of NbmStream.ref), with the gate model if present
    if os.path.exists(NBM.WEIGHTS):
        from sbrt.data import load_train
        from sbrt import bocpd as BO
        M = NBM.load_model()
        S, _ = load_train()
        for i in range(5):
            h = S[i][1]
            st = NBM.NbmStream(h, M, (BO.KAPPA0, BO.ALPHA0, BO.BETA0))
            assert len(st.ref) == n_ref(len(h)), (len(st.ref), n_ref(len(h)))
            # the in-history whitened values are exactly the reference sample
            wh = np.array([NBM.EV._whiten(v, st.ref) for v in st.ref])
            ref = ndtri((np.arange(len(st.ref)) + 0.5) / (len(st.ref) + 1.0))
            assert np.allclose(np.sort(wh), ref, atol=1e-12)
        print("reference size and values = the NBM's in-history whitened values: OK")
    else:
        print(f"(no {NBM.WEIGHTS}: reference-size check against the NBM skipped)")

    # 3. sensitivity: same mean and variance, different shape (Student-t 3, unit variance) vs Gaussian
    nh, no = 2000, 600
    g = np.array([run_series(nh, rng.standard_normal(no))[-1] for _ in range(200)])
    tt = rng.standard_t(3, size=(200, no)) / np.sqrt(3.0)
    t3 = np.array([run_series(nh, x)[-1] for x in tt])
    print("mean statistic at step 600, Gaussian vs unit-variance t3 (shape-only change):")
    for j, nm in enumerate(FEATURE_NAMES):
        print(f"  {nm:12s} {g[:, j].mean():8.3f} {t3[:, j].mean():8.3f}")
    assert t3[:, 1].mean() > 2 * g[:, 1].mean(), "AD should react to a pure shape change"
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    _selftest()
