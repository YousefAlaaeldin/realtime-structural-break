"""Score-test features from the NBM (cv.py --nbm-score, HANDOFF section 43; teammate research card G02).

The question: sbrt/nbm.py turns each online point into ONE number, the standardised one-step error v_t, and the
nb_ features ask whether its mean / variance moved. A break can instead move the NBM's predictive distribution in a
direction that depends on the state (e.g. the scale is too small only when the state is in some region): that
averages out in v_t. A score (Lagrange-multiplier) test asks exactly this: is the gradient of the model's log-
likelihood, which has mean zero under the model, drifting away from zero in some direction?

1. Per point, the score of the Student-t prediction (location m_t, log scale ls_t, nu; z = (e_t - m_t) e^-ls_t):
       g_loc = (nu + 1) z / (nu + z^2)                  (location, in units of the scale)
       g_scl = (nu + 1) z^2 / (nu + z^2) - 1            (log scale)
       g_nu  = -d nll / d nu                            (tail shape)
   and the state-dependent directions g_loc * p_k, g_scl * p_k, p = the first K = 8 principal components of the NBM
   hidden state h_{t-1} that made the prediction (fold PCA basis as cv.py --nbm-hpca, standardised, clipped +-6).
   D = 3 + 2 K = 19 dimensions.
2. Reference: each series' own HISTORY scores (rows >= LOSS_FROM + 1, as nbm.py's whitening reference): mean mu and
   covariance Sigma (+ 1e-3 I); online scores are whitened, g~ = L^-1 (g - mu), Sigma = L L'.
3. Features (prefix nsc_, 9), from the cumulative sums C_n of g~ over the online part:
       cusum_loc, cusum_scl, cusum_nu       C_n[j] / sqrt(n), signed
       lm_base, lm_state                    ||C_n[base]||^2 / n (3 dims), ||C_n[state]||^2 / n (16 dims)
       lm_base100, lm_state100              the same over the last 100 points
       lm_base_scan, lm_state_scan          max over starts j (segment >= 20) of ||C_n - C_j||^2 / (n - j)
   The scans are evaluated only on kept rows (every SUB-th) in batch; the sums at every step, so batch == stream.
4. Cost: one more NBM GRU pass per series (no posterior). Numba, float64, one step function for batch and stream.

Commands:
    python -m sbrt.nbscore            # self-test (seconds): streaming == batch, causal, deterministic, null check
    python -m sbrt.nbscore test       # single-feature TS-AUC on held-out series 5000-6999 (gate model; ~5-10 min)
"""
import math
import os
import sys

import numpy as np
from numba import njit
from scipy.special import digamma

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sbrt import features as F    # noqa: E402
from sbrt import nbm as NBM       # noqa: E402

VERSION = 1
K = 8
D = 3 + 2 * K
AR = NBM.AR
CAP = F.CAP
MINSEG = 20
RIDGE = 1e-3
FEATURE_NAMES = ["nsc_cusum_loc", "nsc_cusum_scl", "nsc_cusum_nu", "nsc_lm_base", "nsc_lm_state",
                 "nsc_lm_base100", "nsc_lm_state100", "nsc_lm_base_scan", "nsc_lm_state_scan"]
N_NSC = len(FEATURE_NAMES)


def nu_consts(state):
    """nu of the NBM's Student-t (as NBM.nu()) and the nu-only part of d nll / d nu."""
    lnu = float(np.asarray(state["lnu"]))
    nu = 2.0 + min(max(math.exp(lnu), 0.1), 60.0)
    return nu, float(-0.5 * digamma((nu + 1.0) / 2.0) + 0.5 * digamma(nu / 2.0) + 0.5 / nu)


def basis_of(Hs, k=K):
    """PCA basis (mean, components k x H, projection sd) of hidden states Hs (rows = time steps), as cv.py hpca_fold."""
    mean = Hs.mean(0)
    _, _, Vt = np.linalg.svd(Hs - mean, full_matrices=False)
    comp = np.ascontiguousarray(Vt[:k])
    sd = ((Hs - mean) @ comp.T).std(0) + 1e-6
    return mean, comp, sd


@njit(cache=True, nogil=True)
def _score(e, m, ls, h, mean, comp, psd, nu, cnu, g):
    z = (e - m) * math.exp(-ls)
    z2 = z * z
    a = nu + z2
    g0 = (nu + 1.0) * z / a
    g1 = (nu + 1.0) * z2 / a - 1.0
    g[0] = g0
    g[1] = g1
    g[2] = -0.5 * math.log1p(z2 / nu) + (nu + 1.0) * z2 / (2.0 * nu * a) - cnu
    Kc = comp.shape[0]
    for k in range(Kc):
        s = 0.0
        for j in range(h.shape[0]):
            s += (h[j] - mean[j]) * comp[k, j]
        p = s / psd[k]
        p = min(max(p, -6.0), 6.0)
        g[3 + k] = g0 * p
        g[3 + Kc + k] = g1 * p


@njit(cache=True, nogil=True)
def _init(zh, coef, se, ctx, h, Wi, bi, Wih, bih, Whh, bhh, Wo1, bo1, Wo2, bo2, mean, comp, psd, nu, cnu,
          ST, LAG, MU, LINV):
    """Run the GRU over the history (as nbm._init_core); history score mean MU and whitening matrix LINV."""
    Hn = h.shape[0]
    a = np.zeros(Hn)
    gi = np.zeros(3 * Hn)
    gh = np.zeros(3 * Hn)
    hid = np.zeros(Hn)
    mo = np.zeros(2)
    xin = np.zeros(NBM.CIN)
    Dd = MU.shape[0]
    g = np.zeros(Dd)
    S1 = np.zeros(Dd)
    S2 = np.zeros((Dd, Dd))
    n_e = len(zh) - AR
    mp, lp, ep = 0.0, 0.0, 0.0
    cnt = 0
    for p in range(n_e):
        pred = 0.0
        for i in range(AR):
            pred += coef[i] * zh[AR + p - 1 - i]
        e = (zh[AR + p] - pred) / se
        e = min(max(e, -10.0), 10.0)
        if p >= NBM.LOSS_FROM + 1:
            _score(e, mp, lp, h, mean, comp, psd, nu, cnu, g)
            for i in range(Dd):
                S1[i] += g[i]
                for j in range(Dd):
                    S2[i, j] += g[i] * g[j]
            cnt += 1
        NBM._row(xin, zh[AR + p], e, ep, ctx)
        NBM._gru_step(xin, h, Wi, bi, Wih, bih, Whh, bhh, Wo1, bo1, Wo2, bo2, a, gi, gh, hid, mo)
        mp, lp, ep = mo[0], mo[1], e
    c = max(cnt, 1)
    for i in range(Dd):
        MU[i] = S1[i] / c
    Sig = np.zeros((Dd, Dd))
    for i in range(Dd):
        for j in range(Dd):
            Sig[i, j] = S2[i, j] / c - MU[i] * MU[j]
        Sig[i, i] += RIDGE
    L = np.linalg.cholesky(Sig)
    LI = np.linalg.inv(L)
    for i in range(Dd):
        for j in range(Dd):
            LINV[i, j] = LI[i, j]
    ST[0] = 0.0
    ST[1] = ep
    ST[2] = mp
    ST[3] = lp
    for i in range(AR):
        LAG[i] = zh[len(zh) - 1 - i]


@njit(cache=True, nogil=True)
def _q(C, a, b, lo, hi):
    s = 0.0
    for j in range(lo, hi):
        d = C[b, j] - C[a, j]
        s += d * d
    return s


@njit(cache=True, nogil=True)
def nsc_step(x, Qp, coef, ctx, h, ST, LAG, Wi, bi, Wih, bih, Whh, bhh, Wo1, bo1, Wo2, bo2, mean, comp, psd, nu,
             cnu, MU, LINV, C, out, do_scan, a, gi, gh, hid, mo, xin, g):
    """One online point: score of the current prediction, whitened, summed; 9 features into out; then the GRU step."""
    t = int(ST[0])
    n = t + 1
    z = (x - Qp[0]) / Qp[1]
    pred = 0.0
    for i in range(AR):
        pred += coef[i] * LAG[i]
    e = (z - pred) / Qp[2]
    e = min(max(e, -10.0), 10.0)
    _score(e, ST[2], ST[3], h, mean, comp, psd, nu, cnu, g)
    Dd = g.shape[0]
    for i in range(Dd):
        s = 0.0
        for j in range(i + 1):
            s += LINV[i, j] * (g[j] - MU[j])
        C[n, i] = C[t, i] + s
    rn = math.sqrt(n)
    out[0] = C[n, 0] / rn
    out[1] = C[n, 1] / rn
    out[2] = C[n, 2] / rn
    out[3] = _q(C, 0, n, 0, 3) / n
    out[4] = _q(C, 0, n, 3, Dd) / n
    lo = n - 100 if n > 100 else 0
    out[5] = _q(C, lo, n, 0, 3) / (n - lo)
    out[6] = _q(C, lo, n, 3, Dd) / (n - lo)
    if do_scan:
        bmax, smax = 0.0, 0.0
        for j in range(0, n - MINSEG + 1):
            vb = _q(C, j, n, 0, 3) / (n - j)
            vs = _q(C, j, n, 3, Dd) / (n - j)
            if vb > bmax:
                bmax = vb
            if vs > smax:
                smax = vs
        out[7] = bmax
        out[8] = smax
    NBM._row(xin, z, e, ST[1], ctx)
    NBM._gru_step(xin, h, Wi, bi, Wih, bih, Whh, bhh, Wo1, bo1, Wo2, bo2, a, gi, gh, hid, mo)
    ST[2] = mo[0]
    ST[3] = mo[1]
    ST[1] = e
    for i in range(AR - 1, 0, -1):
        LAG[i] = LAG[i - 1]
    LAG[0] = z
    ST[0] = n * 1.0


class NscStream:
    """One series' state. M = nbm.model_arrays(state); B = (mean, comp, psd) PCA basis; NU = nu_consts(state)."""

    def __init__(self, hist, M, B, NU):
        P, _, zh = F.make_params(np.asarray(hist, dtype=np.float64))
        self.M = M
        self.B = tuple(np.ascontiguousarray(b, dtype=np.float64) for b in B)
        self.nu, self.cnu = NU
        self.coef = np.ascontiguousarray(P[F.P_AR:F.P_AR + AR], dtype=np.float64)
        se = float(P[F.P_SE10])
        self.ctx = NBM._ctx(zh, self.coef, se)
        self.Qp = np.array([P[F.P_MU], P[F.P_SD], se])
        Hn = M[0].shape[0]
        self.h = np.zeros(Hn)
        self.ST = np.zeros(4)
        self.LAG = np.zeros(AR)
        self.MU = np.zeros(D)
        self.LINV = np.zeros((D, D))
        _init(np.ascontiguousarray(zh), self.coef, se, self.ctx, self.h, *M, *self.B, self.nu, self.cnu,
              self.ST, self.LAG, self.MU, self.LINV)
        self.C = np.zeros((CAP + 1, D))
        self.out = np.zeros(N_NSC)
        self.wk = (np.zeros(Hn), np.zeros(3 * Hn), np.zeros(3 * Hn), np.zeros(Hn), np.zeros(2), np.zeros(NBM.CIN),
                   np.zeros(D))

    def push(self, x, do_scan=True):
        if self.ST[0] >= CAP - 1:
            return self.out
        nsc_step(float(x), self.Qp, self.coef, self.ctx, self.h, self.ST, self.LAG, *self.M, *self.B, self.nu,
                 self.cnu, self.MU, self.LINV, self.C, self.out, do_scan, *self.wk)
        return self.out


@njit(cache=True, nogil=True)
def _run(online, Qp, coef, ctx, h, ST, LAG, Wi, bi, Wih, bih, Whh, bhh, Wo1, bo1, Wo2, bo2, mean, comp, psd, nu,
         cnu, MU, LINV, C, row, a, gi, gh, hid, mo, xin, g, sub, out):
    r = 0
    for i in range(len(online)):
        keep = i % sub == 0
        nsc_step(online[i], Qp, coef, ctx, h, ST, LAG, Wi, bi, Wih, bih, Whh, bhh, Wo1, bo1, Wo2, bo2, mean, comp,
                 psd, nu, cnu, MU, LINV, C, row, keep, a, gi, gh, hid, mo, xin, g)
        if keep:
            for c in range(row.shape[0]):
                out[r, c] = row[c]
            r += 1


def run_series(hist, online, M, B, NU, sub=1):
    """Batch matrix (rows for online steps 0, sub, 2 sub, ..., N_NSC) of one series."""
    st = NscStream(hist, M, B, NU)
    online = np.asarray(online, dtype=np.float64)
    if len(online) >= CAP:
        raise ValueError(f"online segment longer than the supported {CAP - 1} points")
    out = np.zeros(((len(online) + sub - 1) // sub, N_NSC))
    _run(online, st.Qp, st.coef, st.ctx, st.h, st.ST, st.LAG, *st.M, *st.B, st.nu, st.cnu, st.MU, st.LINV, st.C,
         st.out, *st.wk, sub, out)
    return out


def online_scores(hist, online, M, B, NU):
    """Whitened online scores g~ (n x D), for the null check: first differences of the cumulative sums."""
    st = NscStream(hist, M, B, NU)
    for x in np.asarray(online, dtype=np.float64):
        st.push(x, False)
    return np.diff(st.C[:len(online) + 1], axis=0)


# ---------------------------------------------------------------------------------------------
def _gate_model():
    import torch
    from sbrt.data import load_train
    state = torch.load(NBM.WEIGHTS, map_location="cpu")
    M = NBM.model_arrays(state)
    S, tau = load_train()
    Hs = np.concatenate([NBM.run_hidden(S[i][1], S[i][2], M) for i in range(0, 4500, 45)])
    return state, M, basis_of(Hs), nu_consts(state), S, tau


def _selftest():
    rng = np.random.default_rng(0)
    if not os.path.exists(NBM.WEIGHTS):
        sys.exit(f"needs the gate NBM {NBM.WEIGHTS} (python -m sbrt.nbm train)")
    state, M, B, NU, S, tau = _gate_model()
    print(f"gate NBM: nu {NU[0]:.2f}; PCA basis from 100 training series")
    # 1. streaming == batch (all rows and every 3rd row), causal, deterministic, finite
    for it in range(12):
        i = int(rng.integers(5000, 7000))
        h, o = S[i][1], S[i][2]
        Bm = run_series(h, o, M, B, NU)
        st = NscStream(h, M, B, NU)
        Sm = np.array([st.push(x).copy() for x in o])
        assert np.array_equal(Bm, Sm), "streaming != batch"
        assert np.array_equal(run_series(h, o, M, B, NU, 3), Sm[::3]), "sub-sampled batch != streaming"
        assert np.isfinite(Bm).all()
        k = len(o) // 2
        o2 = np.array(o, dtype=np.float64)
        o2[k:] = o2[k:] * 3.0 + 1.0
        assert np.array_equal(run_series(h, o2, M, B, NU)[:k], Bm[:k]), "not causal"
        assert np.array_equal(run_series(h, o, M, B, NU), Bm), "not deterministic"
    print("streaming == batch (also every 3rd row), causal, finite, deterministic: OK")
    # 2. hidden state used for the scores = nbm.run_hidden's (the prediction-making state h_{t-1})
    i = 5001
    Hh = NBM.run_hidden(S[i][1], S[i][2], M)
    st = NscStream(S[i][1], M, B, NU)
    st.push(S[i][2][0])
    assert np.allclose(st.h, Hh[0], atol=1e-12), "GRU state differs from nbm.run_hidden"
    print("GRU state after the first online point equals nbm.run_hidden: OK")
    # 3. null calibration on held-out no-break series: whitened online scores ~ mean 0, variance 1
    G = np.concatenate([online_scores(S[i][1], S[i][2], M, B, NU) for i in range(5000, 5600) if tau[i] < 0])
    print(f"null check, {len(G):,} online points of held-out no-break series (whitened by each history):")
    print("  mean  " + " ".join(f"{v:+.2f}" for v in G.mean(0)))
    print("  var   " + " ".join(f"{v:.2f}" for v in G.var(0)))


def _test(n_eval=2000):
    from joblib import Parallel, delayed
    from sbrt.metric import ts_auc
    state, M, B, NU, S, tau = _gate_model()
    ids = range(5000, 5000 + n_eval)
    R = Parallel(n_jobs=12, backend="threading")(delayed(run_series)(S[i][1], S[i][2], M, B, NU, 3) for i in ids)
    X = np.concatenate(R)
    step = np.concatenate([np.arange(0, len(S[i][2]), 3) for i in ids])
    y = np.concatenate([((np.arange(0, len(S[i][2]), 3) >= tau[i]) & (tau[i] >= 0)) for i in ids]).astype(np.int8)
    print(f"single-feature TS-AUC on held-out series 5000-{4999 + n_eval} (gate NBM; max(a, 1 - a)):")
    for j, nm in enumerate(FEATURE_NAMES):
        a = ts_auc(X[:, j], y, step)
        print(f"  {nm:20s} {max(a, 1 - a):.4f}{'  (reversed)' if a < 0.5 else ''}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        _test()
    else:
        _selftest()
