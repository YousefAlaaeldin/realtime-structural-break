"""Learned normal-behaviour model (NBM): a small GRU pretrained, self-supervised, on break-free HISTORIES to predict
the next AR residual's distribution; its online prediction errors become a new whitened stream for the BOCPD
posterior of sbrt/bocpd.py (HANDOFF sections 27 item 5 and 29: a normal-behaviour stream into the Bayesian machinery,
not an encoder initialisation of the classifier GRU).

Why: the main error source is no-break series whose online variance wanders (section 15.2). The AR residual stream
`w` compares each point with WHOLE-history constants; `wu` with a frozen GARCH path. The NBM conditions every
prediction on the whole past (history + online so far) with a learned, nonlinear model of scale and location.

1. Inputs per time step (11 channels; `e` is exactly the residual bocpd.py uses):
       z (clip 8), e (clip 8), min(e^2, 25) - 1, min(|e|, 8) - 0.8, e_t e_{t-1} (clip 25), 6 history descriptors
   Target: e_{t+1}. Output: location m and log scale (clamped +-4) of a Student-t (one learned nu).
   GRU: Linear 11->64 + GELU, GRU(64), head Linear 64->64 + GELU -> 2 (about 30k parameters).
2. Training (torch, GPU): full histories only (no online data, no labels), loss on positions >= 50.
   Gate model (`train`): series 0-4499, validation 4500-4999. cv.py --nbm: one model per fold, trained on that fold's
   ORIGINAL training histories (10% of them for early stopping); re-cut histories are not used.
3. Stream per online step t: v_t = (e_t - m_t) / s_t (prediction from data up to t-1; the GRU keeps consuming online
   points). Whitened by the ECDF of the same series' in-history v (rows >= 51), as bocpd.py whitens w (clip +-8).
   d_t = log s_t - log s_T (learned scale drift since the history's end).
4. Features (6, prefix nb_): BOCPD posterior on the whitened stream with an empirical-Bayes NIG prior fitted on
   post-break segments of training breaks (per fold in cv.py) -- logodds, tsb, lvr, mu -- plus lsd (d_t) and
   lsd_mean100 (mean d over the last 100 online points).
5. Runtime: numba, float64, one function `nbm_push` per online point, used by BOTH the batch builder (cv.py) and the
   streaming class (inference), so training features equal inference features exactly.

Commands:
    python -m sbrt.nbm train            # gate model (GPU, under a minute); data/nbm_v1.pt
    python -m sbrt.nbm test             # gate on held-out series 5000-6999 (numba runtime)
    python -m sbrt.nbm check            # streaming == batch, causality, numba vs torch agreement, timing
"""
import math
import os
import sys
import time

import numpy as np
from numba import njit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sbrt import bocpd as BO      # noqa: E402
from sbrt import evidence as EV   # noqa: E402
from sbrt import features as F    # noqa: E402
from sbrt import volstate as VS   # noqa: E402

VERSION = 1
CIN = 11
H = 64
AR = F.AR
CAP = F.CAP
LOSS_FROM = 50
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
WEIGHTS = os.path.join(DATA, f"nbm_v{VERSION}.pt")
FEATURE_NAMES = ["nb_logodds", "nb_tsb", "nb_lvr", "nb_mu", "nb_lsd", "nb_lsd_mean100"]
N_NB = len(FEATURE_NAMES)


# ---------------------------------------------------------------------------------------------
# training (torch)
def _ctx(zh, coef, se):
    nh = len(zh)
    rho1 = float(np.dot(zh[:-1], zh[1:]) / max(float(np.dot(zh, zh)), 1e-12))
    return np.array([math.log(nh) - 8.0, np.clip(np.mean(zh ** 3), -5, 5) / 2.0,
                     math.log1p(max(float(np.mean(zh ** 4)) - 3.0, 0.0)) - 1.0, rho1, coef[0], math.log(se)])


def build_seq(hist, online):
    """Torch inputs for history + online (one row per time step from AR on) and the residual e (e[p+1] = target of row p)."""
    Hh = VS.fit_history(np.asarray(hist, dtype=np.float64))
    P, coef, se, zh = Hh["P"], Hh["coef"], Hh["se"], Hh["z"]
    zo = (np.asarray(online, dtype=np.float64) - P[F.P_MU]) / P[F.P_SD]
    s = np.r_[zh, zo]
    n = len(s)
    Xl = np.column_stack([s[AR - 1 - i: n - 1 - i] for i in range(AR)])
    e = np.clip((s[AR:] - Xl @ coef) / se, -10.0, 10.0)
    ep = np.r_[0.0, e[:-1]]
    ctx = _ctx(zh, coef, se)
    X = np.column_stack([np.clip(s[AR:], -8, 8), np.clip(e, -8, 8), np.minimum(e * e, 25.0) - 1.0,
                         np.minimum(np.abs(e), 8.0) - 0.8, np.clip(e * ep, -25, 25),
                         np.tile(ctx, (len(e), 1))]).astype(np.float32)
    return X, e, len(zh) - AR      # rows 0..nh-AR-1 are history


def _torch():
    import torch
    import torch.nn as nn

    class NBM(nn.Module):
        def __init__(self, hidden=H, multi=1):
            super().__init__()
            self.multi = multi
            self.inp = nn.Sequential(nn.Linear(CIN, hidden), nn.GELU())
            self.gru = nn.GRU(hidden, hidden, batch_first=True)
            # multi > 1 (HANDOFF 37/41): heads for e_{t+1} .. e_{t+multi}, outputs ordered m1, ls1, m2, ls2, ...;
            # only rows 0-1 (the 1-step head) are kept after training, so the runtime never sees the other heads
            self.out = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 2 * multi))
            self.lnu = nn.Parameter(torch.tensor(math.log(5.0 - 2.0)))

        def forward(self, x):
            o, _ = self.gru(self.inp(x))
            r = self.out(o)
            if self.multi > 1:
                r = r.view(x.shape[0], x.shape[1], self.multi, 2)      # (B, L, multi) each
            return r[..., 0], r[..., 1].clamp(-4, 4)

        def nu(self):
            return 2.0 + torch.exp(self.lnu).clamp(0.1, 60.0)

    return torch, NBM


def t_nll(torch, y, m, ls, nu):
    z = (y - m) * torch.exp(-ls)
    return (ls + 0.5 * (nu + 1) * torch.log1p(z * z / nu)
            - torch.lgamma((nu + 1) / 2) + torch.lgamma(nu / 2) + 0.5 * torch.log(nu * math.pi))


def _pad(seqs, dev, torch):
    L = -(-max(len(x) for x in seqs) // 512) * 512   # round up: repeated shapes let the CUDA cache reuse blocks
    X = np.zeros((len(seqs), L, seqs[0].shape[1]), np.float32)
    for i, x in enumerate(seqs):
        X[i, :len(x)] = x
    return torch.as_tensor(X, device=dev)


def prep_histories(hs, multi=1):
    """(inputs, next-residual target) pairs for pretraining (model independent: build once, reuse across a sweep).
    multi > 1: targets (L, multi) with column j = e_{t+1+j}; the last multi-1 rows are dropped (as nbm_sweep3.prep)."""
    out = []
    for h in hs:
        X, e, _ = build_seq(h, np.zeros(0))
        if multi == 1:
            out.append((X[:-1], e[1:].astype(np.float32)))
        else:
            L = len(e) - multi
            out.append((X[:L], np.column_stack([e[1 + j: 1 + j + L] for j in range(multi)]).astype(np.float32)))
    return out


def train_model(hists_tr, hists_va, epochs=12, batch=48, lr=2e-3, seed=0, verbose=True, hidden=H, wd=1e-4,
                loss_from=LOSS_FROM, return_nll=False, prepared=False, multi=1):
    """Pretrain on lists of histories; returns the best state_dict (by validation NLL) on the CPU
    (and the best validation NLL when return_nll). The hidden size is read back from the weights at inference.
    multi > 1 trains heads for 1..multi steps ahead (loss and best epoch on the NLL averaged over the horizons,
    exactly as diagnostics/nbm_sweep3.py) and returns only the 1-step head: the state dict has the multi = 1 shape."""
    torch, NBM = _torch()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr = hists_tr if prepared else prep_histories(hists_tr, multi)
    va = hists_va if prepared else prep_histories(hists_va, multi)
    net = NBM(hidden, multi).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    steps = epochs * ((len(tr) + batch - 1) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)

    def batches(D, shuffle):
        order = np.argsort([len(d[1]) for d in D])
        bs = [order[i:i + batch] for i in range(0, len(order), batch)]
        if shuffle:
            rng.shuffle(bs)
        for b in bs:
            x = _pad([D[i][0] for i in b], dev, torch)
            if multi == 1:
                y = _pad([D[i][1][:, None] for i in b], dev, torch)[..., 0]
                msk = torch.zeros_like(y)
            else:
                y = _pad([D[i][1] for i in b], dev, torch)                # (B, L, multi)
                msk = torch.zeros(y.shape[:2], device=dev)
            for k, i in enumerate(b):
                msk[k, loss_from:len(D[i][1])] = 1.0
            yield x, y, msk
            del x, y, msk
            if dev.type == "cuda":
                # batch shapes vary; with Windows sysmem fallback the cache would otherwise spill into system RAM
                # (seen at hidden 192: 15.5 GB VRAM + 27 GB RAM, near-stalled)
                torch.cuda.empty_cache()

    def nll(y, m, ls):
        v = t_nll(torch, y, m, ls, net.nu())
        return v if multi == 1 else v.mean(-1)                         # average over the horizons

    def evaluate(D):
        net.eval()
        tot, cnt = 0.0, 0.0
        with torch.no_grad():
            for x, y, msk in batches(D, False):
                m, ls = net(x)
                tot += float((nll(y, m, ls) * msk).sum())
                cnt += float(msk.sum())
        net.train()
        return tot / cnt

    best, best_state = evaluate(va), None
    if verbose:
        print(f"NBM pretraining: {sum(len(d[1]) for d in tr):,} train / {sum(len(d[1]) for d in va):,} valid points "
              f"on {dev}; valid NLL before {best:.4f}", flush=True)
    best = 1e9
    for ep in range(epochs):
        t1 = time.time()
        for x, y, msk in batches(tr, True):
            m, ls = net(x)
            loss = (nll(y, m, ls) * msk).sum() / msk.sum()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
        v = evaluate(va)
        if verbose:
            print(f"  epoch {ep + 1:2d}  valid NLL {v:.4f}  nu {float(net.nu().detach()):.2f}  ({time.time() - t1:.0f}s)",
                  flush=True)
        if v < best:
            best, best_state = v, {k: t.detach().cpu().clone() for k, t in net.state_dict().items()}
    if multi > 1:   # keep the 1-step head only (rows m1, ls1): the multi = 1 shape that model_arrays / NbmStream use
        best_state["out.2.weight"] = best_state["out.2.weight"][:2].clone()
        best_state["out.2.bias"] = best_state["out.2.bias"][:2].clone()
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return (best_state, best) if return_nll else best_state


def train():
    from sbrt.data import load_train
    import torch
    S, _ = load_train()
    state = train_model([s[1] for s in S[:4500]], [s[1] for s in S[4500:5000]])
    torch.save(state, WEIGHTS)
    print(f"saved {WEIGHTS}")


# ---------------------------------------------------------------------------------------------
# runtime (numba, float64): the same code for batch features and streaming inference
def model_arrays(state):
    """Tuple of float64 C-contiguous arrays from a torch state_dict (or a dict of numpy arrays)."""
    g = lambda k: np.ascontiguousarray(np.asarray(state[k].numpy() if hasattr(state[k], "numpy") else state[k],
                                                  dtype=np.float64))
    return (g("inp.0.weight"), g("inp.0.bias"), g("gru.weight_ih_l0"), g("gru.bias_ih_l0"),
            g("gru.weight_hh_l0"), g("gru.bias_hh_l0"), g("out.0.weight"), g("out.0.bias"),
            g("out.2.weight"), g("out.2.bias"))


def load_model(path=WEIGHTS):
    import torch
    return model_arrays(torch.load(path, map_location="cpu"))


@njit(cache=True, nogil=True)
def _gelu(x):
    return 0.5 * x * (1.0 + math.erf(x / math.sqrt(2.0)))


@njit(cache=True, nogil=True)
def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


@njit(cache=True, nogil=True)
def _gru_step(xin, h, Wi, bi, Wih, bih, Whh, bhh, Wo1, bo1, Wo2, bo2, a, gi, gh, hid, mo):
    """One time step: updates the hidden state h in place; mo[0], mo[1] = predicted location, log scale."""
    Hn = h.shape[0]
    for i in range(Hn):
        s = bi[i]
        for j in range(xin.shape[0]):
            s += Wi[i, j] * xin[j]
        a[i] = _gelu(s)
    for i in range(3 * Hn):
        s1 = bih[i]
        s2 = bhh[i]
        for j in range(Hn):
            s1 += Wih[i, j] * a[j]
            s2 += Whh[i, j] * h[j]
        gi[i] = s1
        gh[i] = s2
    for i in range(Hn):
        r = _sigmoid(gi[i] + gh[i])
        z = _sigmoid(gi[Hn + i] + gh[Hn + i])
        n = math.tanh(gi[2 * Hn + i] + r * gh[2 * Hn + i])
        h[i] = (1.0 - z) * n + z * h[i]
    for i in range(Hn):
        s = bo1[i]
        for j in range(Hn):
            s += Wo1[i, j] * h[j]
        hid[i] = _gelu(s)
    for k in range(2):
        s = bo2[k]
        for j in range(Hn):
            s += Wo2[k, j] * hid[j]
        mo[k] = s
    if mo[1] > 4.0:
        mo[1] = 4.0
    elif mo[1] < -4.0:
        mo[1] = -4.0


# Q: per-series constants; ST: scalar state; LAG: last AR z values (most recent first)
Q_MU, Q_SD, Q_SE, Q_LST = 0, 1, 2, 3
Q_LEN = 4
S_T, S_EP, S_M, S_LS = 0, 1, 2, 3
S_LEN = 4


@njit(cache=True, nogil=True)
def _row(xin, z, e, ep, ctx):
    xin[0] = min(max(z, -8.0), 8.0)
    xin[1] = min(max(e, -8.0), 8.0)
    xin[2] = min(e * e, 25.0) - 1.0
    xin[3] = min(abs(e), 8.0) - 0.8
    xin[4] = min(max(e * ep, -25.0), 25.0)
    for k in range(6):
        xin[5 + k] = ctx[k]


@njit(cache=True, nogil=True)
def _init_core(zh, coef, se, ctx, h, Wi, bi, Wih, bih, Whh, bhh, Wo1, bo1, Wo2, bo2, vref, ST, LAG, Q):
    """Run the GRU over the history; fill the in-history v (unsorted, rows >= LOSS_FROM+1) and the state."""
    Hn = h.shape[0]
    a = np.zeros(Hn)
    gi = np.zeros(3 * Hn)
    gh = np.zeros(3 * Hn)
    hid = np.zeros(Hn)
    mo = np.zeros(2)
    xin = np.zeros(CIN)
    n_e = len(zh) - AR
    mp, lp, ep, lst = 0.0, 0.0, 0.0, 0.0
    nv = 0
    for p in range(n_e):
        pred = 0.0
        for i in range(AR):
            pred += coef[i] * zh[AR + p - 1 - i]
        e = (zh[AR + p] - pred) / se
        e = min(max(e, -10.0), 10.0)
        if p >= LOSS_FROM + 1:
            vref[nv] = (e - mp) * math.exp(-lp)
            nv += 1
        if p == n_e - 1:
            lst = lp            # log scale of the prediction for the last history residual
        _row(xin, zh[AR + p], e, ep, ctx)
        _gru_step(xin, h, Wi, bi, Wih, bih, Whh, bhh, Wo1, bo1, Wo2, bo2, a, gi, gh, hid, mo)
        mp, lp, ep = mo[0], mo[1], e
    Q[Q_LST] = lst
    ST[S_T] = 0.0
    ST[S_EP] = ep
    ST[S_M] = mp
    ST[S_LS] = lp
    for i in range(AR):
        LAG[i] = zh[len(zh) - 1 - i]
    return nv


@njit(cache=True, nogil=True)
def nbm_push(x, Q, coef, ctx, ref, h, ST, LAG, C1, C2, CD, Wi, bi, Wih, bih, Whh, bhh, Wo1, bo1, Wo2, bo2,
             prior, LOGH, LP, buf, out, do_post, a, gi, gh, hid, mo, xin):
    """One online point: update the whitened stream sums and (if do_post) write the 6 features into out."""
    t = int(ST[S_T])
    z = (x - Q[Q_MU]) / Q[Q_SD]
    pred = 0.0
    for i in range(AR):
        pred += coef[i] * LAG[i]
    e = (z - pred) / Q[Q_SE]
    e = min(max(e, -10.0), 10.0)
    v = (e - ST[S_M]) * math.exp(-ST[S_LS])
    w = EV._whiten(v, ref)
    w = min(max(w, -8.0), 8.0)
    d = ST[S_LS] - Q[Q_LST]
    n = t + 1
    C1[0, n] = C1[0, t] + w
    C2[0, n] = C2[0, t] + w * w
    CD[n] = CD[t] + d
    if do_post:
        BO._posterior(C1, C2, 0, n, prior, LOGH, LP, buf, out, 0)
        # out[0] logodds, out[1] logbf (overwritten), out[2] tsb, out[3] lvr, out[4] mu  ->  reorder
        lo = n - 100 if n > 100 else 0
        lsd_m = (CD[n] - CD[lo]) / (n - lo)
        out[1] = out[2]
        out[2] = out[3]
        out[3] = out[4]
        out[4] = CD[n] - CD[t]      # = d up to rounding; written this way so batch rows match bit for bit
        out[5] = lsd_m
    _row(xin, z, e, ST[S_EP], ctx)
    _gru_step(xin, h, Wi, bi, Wih, bih, Whh, bhh, Wo1, bo1, Wo2, bo2, a, gi, gh, hid, mo)
    ST[S_M] = mo[0]
    ST[S_LS] = mo[1]
    ST[S_EP] = e
    for i in range(AR - 1, 0, -1):
        LAG[i] = LAG[i - 1]
    LAG[0] = z
    ST[S_T] = n * 1.0
    return w


class NbmStream:
    """One series' NBM state. Use .push(x) per online point (inference); the batch builder uses the same calls."""

    def __init__(self, hist, M, prior):
        Hh = VS.fit_history(np.asarray(hist, dtype=np.float64))
        P, coef, se, zh = Hh["P"], Hh["coef"], Hh["se"], Hh["z"]
        self.M = M
        self.coef = np.ascontiguousarray(coef, dtype=np.float64)
        self.ctx = _ctx(zh, coef, se)
        self.Q = np.zeros(Q_LEN)
        self.Q[Q_MU], self.Q[Q_SD], self.Q[Q_SE] = P[F.P_MU], P[F.P_SD], se
        Hn = M[0].shape[0]          # hidden size from the weights
        self.h = np.zeros(Hn)
        self.ST = np.zeros(S_LEN)
        self.LAG = np.zeros(AR)
        vref = np.zeros(max(len(zh), 1))
        nv = _init_core(np.ascontiguousarray(zh), self.coef, float(se), self.ctx, self.h, *M, vref, self.ST, self.LAG, self.Q)
        self.ref = np.sort(vref[:nv])
        self.C1 = np.zeros((1, CAP + 1))
        self.C2 = np.zeros((1, CAP + 1))
        self.CD = np.zeros(CAP + 1)
        self.prior = np.asarray(prior, dtype=np.float64)
        self.buf = np.zeros(CAP + 1)
        self.out = np.zeros(N_NB)
        self.wk = (np.zeros(Hn), np.zeros(3 * Hn), np.zeros(3 * Hn), np.zeros(Hn), np.zeros(2), np.zeros(CIN))

    def push(self, x, do_post=True):
        if self.ST[S_T] >= CAP - 1:
            return self.out
        nbm_push(float(x), self.Q, self.coef, self.ctx, self.ref, self.h, self.ST, self.LAG, self.C1, self.C2, self.CD,
                 *self.M, self.prior, BO.LOGH, BO.LP, self.buf, self.out, do_post, *self.wk)
        return self.out


@njit(cache=True, nogil=True)
def _post_rows(C1, C2, CD, n_pts, prior, LOGH, LP, buf, sub, out):
    """Features at online steps 0, sub, 2 sub, ... from the stored sums (identical to do_post inside nbm_push)."""
    tmp = np.zeros(5)
    r = 0
    for t in range(0, n_pts, sub):
        n = t + 1
        BO._posterior(C1, C2, 0, n, prior, LOGH, LP, buf, tmp, 0)
        lo = n - 100 if n > 100 else 0
        out[r, 0] = tmp[0]
        out[r, 1] = tmp[2]
        out[r, 2] = tmp[3]
        out[r, 3] = tmp[4]
        out[r, 4] = CD[n] - CD[n - 1]
        out[r, 5] = (CD[n] - CD[lo]) / (n - lo)
        r += 1


def run_stream(hist, online, M):
    """Pass 1 (expensive, prior-free): the GRU over history + online. Returns the sums (C1, C2, CD) and w."""
    st = NbmStream(hist, M, (BO.KAPPA0, BO.ALPHA0, BO.BETA0))
    online = np.asarray(online, dtype=np.float64)
    if len(online) >= CAP:
        raise ValueError(f"online segment longer than the supported {CAP - 1} points")
    for x in online:
        st.push(x, False)
    n = len(online)
    return st.C1[:, :n + 1].copy(), st.C2[:, :n + 1].copy(), st.CD[:n + 1].copy()


@njit(cache=True, nogil=True)
def _hidden_run(online, Q, coef, ctx, ref, h, ST, LAG, C1, C2, CD, Wi, bi, Wih, bih, Whh, bhh, Wo1, bo1, Wo2, bo2,
                prior, LOGH, LP, buf, out, a, gi, gh, hid, mo, xin, Hs):
    for i in range(len(online)):
        nbm_push(online[i], Q, coef, ctx, ref, h, ST, LAG, C1, C2, CD, Wi, bi, Wih, bih, Whh, bhh, Wo1, bo1, Wo2, bo2,
                 prior, LOGH, LP, buf, out, False, a, gi, gh, hid, mo, xin)
        for k in range(h.shape[0]):
            Hs[i, k] = h[k]


def run_hidden(hist, online, M, mean=None, comp=None):
    """The GRU hidden state after each online point (it has consumed points <= t: causal), shape (n, H); with
    mean / comp (K x H) the PCA projection (n, K) float32 instead (HANDOFF section 24, gru_cv.py --nbm-hpca)."""
    st = NbmStream(hist, M, (BO.KAPPA0, BO.ALPHA0, BO.BETA0))
    online = np.asarray(online, dtype=np.float64)
    if len(online) >= CAP:
        raise ValueError(f"online segment longer than the supported {CAP - 1} points")
    Hs = np.empty((len(online), len(st.h)))
    # one compiled loop (no per-point Python call, GIL released): same nbm_push calls as NbmStream.push
    _hidden_run(online, st.Q, st.coef, st.ctx, st.ref, st.h, st.ST, st.LAG, st.C1, st.C2, st.CD, *st.M, st.prior,
                BO.LOGH, BO.LP, st.buf, st.out, *st.wk, Hs)
    if comp is None:
        return Hs
    return ((Hs - mean) @ comp.T).astype(np.float32)



N_GATE = 3
GATE_NAMES = ["nbg_upd", "nbg_rst", "nbg_ldh"]


@njit(cache=True, nogil=True)
def _gates(h, hprev, gi, gh, g):
    """Gate activity of the step that just ran (gi / gh still hold its pre-activations): mean share of the memory
    overwritten (1 - update gate), mean reset gate, log size of the state change."""
    Hn = h.shape[0]
    u, r, d = 0.0, 0.0, 0.0
    for i in range(Hn):
        r += _sigmoid(gi[i] + gh[i])
        u += 1.0 - _sigmoid(gi[Hn + i] + gh[Hn + i])
        d += (h[i] - hprev[i]) ** 2
    g[0] = u / Hn
    g[1] = r / Hn
    g[2] = math.log(math.sqrt(d) + 1e-4)


@njit(cache=True, nogil=True)
def _history_rec(zh, coef, se, ctx, h, Wi, bi, Wih, bih, Whh, bhh, Wo1, bo1, Wo2, bo2, Hrec, Grec):
    """The GRU over the history exactly as _init_core runs it, recording the state and gate activity after each
    history residual (row p = after the point zh[AR + p])."""
    Hn = h.shape[0]
    a = np.zeros(Hn)
    gi = np.zeros(3 * Hn)
    gh = np.zeros(3 * Hn)
    hid = np.zeros(Hn)
    mo = np.zeros(2)
    xin = np.zeros(CIN)
    hp = np.zeros(Hn)
    ep = 0.0
    for p in range(len(zh) - AR):
        pred = 0.0
        for i in range(AR):
            pred += coef[i] * zh[AR + p - 1 - i]
        e = (zh[AR + p] - pred) / se
        e = min(max(e, -10.0), 10.0)
        _row(xin, zh[AR + p], e, ep, ctx)
        for k in range(Hn):
            hp[k] = h[k]
        _gru_step(xin, h, Wi, bi, Wih, bih, Whh, bhh, Wo1, bo1, Wo2, bo2, a, gi, gh, hid, mo)
        ep = e
        for k in range(Hn):
            Hrec[p, k] = h[k]
        _gates(h, hp, gi, gh, Grec[p])


@njit(cache=True, nogil=True)
def _hidden_run_g(online, Q, coef, ctx, ref, h, ST, LAG, C1, C2, CD, Wi, bi, Wih, bih, Whh, bhh, Wo1, bo1, Wo2, bo2,
                  prior, LOGH, LP, buf, out, a, gi, gh, hid, mo, xin, Hs, Gs):
    hp = np.zeros(h.shape[0])
    for i in range(len(online)):
        for k in range(h.shape[0]):
            hp[k] = h[k]
        nbm_push(online[i], Q, coef, ctx, ref, h, ST, LAG, C1, C2, CD, Wi, bi, Wih, bih, Whh, bhh, Wo1, bo1, Wo2, bo2,
                 prior, LOGH, LP, buf, out, False, a, gi, gh, hid, mo, xin)
        for k in range(h.shape[0]):
            Hs[i, k] = h[k]
        _gates(h, hp, gi, gh, Gs[i])


def run_hidden_full(hist, online, M):
    """History AND online hidden states and gate activity (HANDOFF section 46, gru_cv.py --nbm-hist / --nbm-gates).
    Returns Hh (n_e, H), Gh (n_e, 3) for the history residuals (row p = after history point AR + p, n_e =
    len(hist) - AR) and Ho (n, H), Go (n, 3) after each online point (causal). Ho equals run_hidden()."""
    st = NbmStream(hist, M, (BO.KAPPA0, BO.ALPHA0, BO.BETA0))
    online = np.asarray(online, dtype=np.float64)
    if len(online) >= CAP:
        raise ValueError(f"online segment longer than the supported {CAP - 1} points")
    Hn = len(st.h)
    Hh = VS.fit_history(np.asarray(hist, dtype=np.float64))
    zh = np.ascontiguousarray(Hh["z"])
    n_e = len(zh) - AR
    Hrec, Grec = np.empty((n_e, Hn)), np.empty((n_e, N_GATE))
    h0 = np.zeros(Hn)
    _history_rec(zh, st.coef, float(Hh["se"]), st.ctx, h0, *st.M, Hrec, Grec)
    if not np.array_equal(h0, st.h):
        raise RuntimeError("run_hidden_full: recorded history state differs from NbmStream's")
    Ho, Go = np.empty((len(online), Hn)), np.empty((len(online), N_GATE))
    _hidden_run_g(online, st.Q, st.coef, st.ctx, st.ref, st.h, st.ST, st.LAG, st.C1, st.C2, st.CD, *st.M, st.prior,
                  BO.LOGH, BO.LP, st.buf, st.out, *st.wk, Ho, Go)
    return Hrec, Grec, Ho, Go

def features_from_sums(sums, prior, sub=1):
    """Pass 2 (cheap): the 6 features at online steps 0, sub, ... for a prior."""
    C1, C2, CD = sums
    n = C1.shape[1] - 1
    out = np.zeros(((n + sub - 1) // sub, N_NB))
    _post_rows(C1, C2, CD, n, np.asarray(prior, dtype=np.float64), BO.LOGH, BO.LP, np.zeros(n + 1), sub, out)
    return out


def run_series_nbm(hist, online, M, prior, sub=1):
    return features_from_sums(run_stream(hist, online, M), prior, sub)


def fit_prior_w(ws, tau, max_post=300, min_post=100):
    """Empirical-Bayes NIG prior from post-break segments of whitened streams (same estimator as bocpd.fit_prior)."""
    v, r = [], []
    for w, t in zip(ws, tau):
        if t < 0 or len(w) - t < min_post:
            continue
        x = w[t:t + max_post]
        s2 = float(x.var())
        v.append(s2)
        r.append(float(x.mean()) ** 2 / max(s2, 1e-6) - 1.0 / len(x))
    v = np.array(v)
    a0 = v.mean() ** 2 / max(v.var(), 1e-6) + 2.0
    return (1.0 / max(float(np.mean(r)), 1e-3), a0, v.mean() * (a0 - 1.0)), len(v)


def sums_parallel(items, M, threads=8):
    from joblib import Parallel, delayed
    return Parallel(n_jobs=threads, backend="threading")(delayed(run_stream)(h, o, M) for h, o in items)


def w_of(sums):
    return np.diff(sums[0][0])


# ---------------------------------------------------------------------------------------------
def check(n=20):
    """streaming == batch (all rows and every 3rd), causal, deterministic; numba vs torch; timing."""
    from sbrt.data import load_train
    S, tau = load_train()
    M = load_model()
    prior = (448.0, 14.88, 15.26)
    rng = np.random.default_rng(0)
    for i in rng.choice(np.arange(5000, 7000), n, replace=False):
        _, h, o = S[i]
        B = run_series_nbm(h, o, M, prior)
        assert np.array_equal(B, run_series_nbm(h, o, M, prior)), "two runs differ"
        st = NbmStream(h, M, prior)
        Sm = np.array([st.push(x).copy() for x in o])
        assert np.array_equal(B, Sm), f"streaming != batch (max diff {np.abs(B - Sm).max()})"
        assert np.array_equal(run_series_nbm(h, o, M, prior, 3), Sm[::3]), "sub-sampled batch != streaming"
        assert np.isfinite(B).all()
        if len(o) > 40:
            o2 = o.copy()
            o2[35:] = rng.standard_normal(len(o) - 35) * 3
            assert np.array_equal(run_series_nbm(h, o2, M, prior)[:35], B[:35]), "future points changed earlier features"
    print(f"streaming == batch (also every 3rd row), causal, finite, deterministic: OK ({n} series)")
    # numba (float64) vs torch (float32, GPU or CPU): the location / log-scale path
    torch, NBM = _torch()
    sd = torch.load(WEIGHTS, map_location="cpu")
    net = NBM(sd["inp.0.weight"].shape[0])
    net.load_state_dict(sd)
    net.eval()
    worst = 0.0
    for i in range(5000, 5010):
        _, h, o = S[i]
        X, e, nh = build_seq(h, o)
        with torch.no_grad():
            _, ls = net(torch.as_tensor(X[None]))
        st = NbmStream(h, M, prior)
        lsn = []
        for x in o:
            lsn.append(st.ST[S_LS])
            st.push(x, False)
        worst = max(worst, float(np.abs(np.array(lsn) - ls[0].numpy()[nh - 1:nh - 1 + len(o)]).max()))
    print(f"numba float64 vs torch float32 predicted log scale, max abs diff over 10 series: {worst:.2e}")
    # timing
    t0 = time.time()
    k = 0
    for i in range(7000, 7050):
        _, h, o = S[i]
        run_stream(h, o, M)
        k += len(h) + len(o)
    dt = time.time() - t0
    print(f"timing: {dt / 50 * 1000:.0f} ms per series single-threaded ({dt / k * 1e6:.1f} us per GRU step incl. history)")
    print("ALL CHECKS PASSED")


def selftest(offset=5000, n_series=2000, threads=8):
    from scipy.stats import rankdata

    from sbrt.data import load_train
    from sbrt.metric import ts_auc

    S_all, tau_all = load_train()
    M = load_model()
    t0 = time.time()
    fit_items = [(s[1], s[2]) for s in S_all[:5000]]
    sf = sums_parallel(fit_items, M, threads)
    prior, nfit = fit_prior_w([w_of(q) for q in sf], tau_all[:5000])
    print(f"EB prior on the nb stream from {nfit} breaks: kappa0 {prior[0]:.0f}, alpha0 {prior[1]:.2f}, beta0 "
          f"{prior[2]:.2f} (mean post-break var {prior[2] / (prior[1] - 1):.2f})  ({time.time() - t0:.0f}s)")
    S, tau = S_all[offset:offset + n_series], tau_all[offset:offset + n_series]
    se = sums_parallel([(s[1], s[2]) for s in S], M, threads)
    nbw = np.concatenate([w_of(q) for q, t in zip(se, tau) if t < 0])
    print(f"held-out no-break whitened nb stream: mean {nbw.mean():+.3f} var {nbw.var():.3f}")
    NB = np.concatenate([features_from_sums(q, prior, 3) for q in se])
    from joblib import Parallel, delayed
    Bo = np.concatenate(Parallel(n_jobs=threads, backend="threading")(delayed(BO.run_series_bocpd)(s[1], s[2], 3) for s in S))
    V = np.concatenate([VS.run_series_vol(s[1], s[2])[::3] for s in S])
    print(f"features built ({time.time() - t0:.0f}s total)")
    step = np.concatenate([np.arange(0, len(s[2]), 3) for s in S])
    sid = np.repeat(np.arange(n_series), [(len(s[2]) + 2) // 3 for s in S])
    y = ((tau[sid] >= 0) & (step >= tau[sid])).astype(np.int8)
    vr = np.full(n_series, np.nan)
    for i, (s, t) in enumerate(zip(S, tau)):
        if t >= 0 and len(s[2]) - t >= 30:
            vr[i] = BO.whitened_online(s[1], s[2])[0][t:t + 300].var()
    same = (vr > 0.85) & (vr < 1.15)
    in_slice = (tau[sid] < 0) | same[sid]
    buckets = ((0, 100), (100, 300), (300, 700), (700, 10000))

    def row(nm, x, extra=""):
        r = [ts_auc(x, y, step)] + [ts_auc(x[(step >= lo) & (step < hi)], y[(step >= lo) & (step < hi)],
                                          step[(step >= lo) & (step < hi)]) for lo, hi in buckets]
        r.append(ts_auc(x[in_slice], y[in_slice], step[in_slice]))
        print(f"  {nm:18s} " + "  ".join(f"{v:7.4f}" for v in r) + extra)

    print("\nsingle-feature TS-AUC              all     0-100  100-300  300-700    700+   var-same")
    for c in (0, 3, 5, 8):
        row(BO.FEATURE_NAMES[c], Bo[:, c])
    rng = np.random.default_rng(0)
    sel = rng.choice(len(y), min(len(y), 150000), replace=False)
    Z = np.column_stack([Bo[sel], V[sel]])
    names = list(BO.FEATURE_NAMES) + list(VS.FEATURE_NAMES)
    R = np.column_stack([rankdata(Z[:, j]) for j in range(Z.shape[1])])
    R = (R - R.mean(0)) / (R.std(0) + 1e-12)
    for c, nm in enumerate(FEATURE_NAMES):
        r = rankdata(NB[sel, c])
        r = (r - r.mean()) / (r.std() + 1e-12)
        cc = np.abs(R.T @ r) / len(r)
        j = int(np.argmax(cc))
        row(nm, NB[:, c], f"   {cc[j]:.2f} {names[j]}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "test"
    {"train": train, "check": check, "test": selftest}[cmd]()
