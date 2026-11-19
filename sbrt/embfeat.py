"""Embedding-drift features for cv.py --femb / --emb (HANDOFF section 33). Same maths as
diagnostics/embedding_gate.py (emb) and diagnostics/embedding_feat_gate.py (femb), packaged for per-fold use.

Detector: embed windows of L points; anchor = normalised mean embedding of the history windows (last REF points,
stride HSTRIDE); at every SUB-th online step, d = standardised drop of cos(window ending now, anchor) against the
series' own in-history cosines (leave-out anchors without overlapping neighbours); lag = the same for
cos(window now, window L earlier); CUSUMs (allowance 0.5); BOCPD posterior on the non-overlapping block stream of d.
  emb   1-D dilated CNN on 4 residual channels (z, e, e^2-1, |e|-0.8) -> 32-dim unit vector; one net for both L.
  femb  15 window statistics, standardised with training constants -> MLP 15-64-64-16 -> unit vector; one net per L.
Both trained contrastively (InfoNCE, temperature 0.1, batch 512 series; positives = two non-overlapping history
windows of the same series) on the fold's ORIGINAL training histories only; 2 seeds, features averaged over seeds.
The block-stream EB prior (uses labels) is fitted on the fold's original training breaks only.
18 columns per kind: <kind><L>_<col> for L in LS, col in COLS.
Causal: every value at online step t uses points up to t only (checked in diagnostics/embedding_gate.py).
"""
import time

import numpy as np
from numba import njit

from . import features as F
from . import nbm as NB
from . import volstate as VS

VERSION = 1
AR = F.AR
SUB = 3
LS = (30, 120)
HSTRIDE = 6
REF = 1500
SEEDS = (0, 1)
TEMP = 0.1
BATCH = 512
COLS = ["d", "dmean", "dcus", "lag", "lagcus", "pb_logodds", "pb_tsb", "pb_lvr", "pb_mu"]
FALLBACK_PRIOR = (100.0, 10.0, 9.0)
CHUNK = 400


def names(kind):
    return [f"{kind}{L}_{c}" for L in LS for c in COLS]


# ---------------------------------------------------------------------------------------------
# series preparation

@njit(cache=True)
def fwd_resid(z, coef, se, start):
    N, p = len(z), len(coef)
    out = np.full(N, np.nan)
    for i in range(max(start, p), N):
        s = 0.0
        for k in range(p):
            s += coef[k] * z[i - 1 - k]
        e = (z[i] - s) / se
        out[i] = min(max(e, -10.0), 10.0)
    return out


def chans(z, e):
    e0 = np.nan_to_num(e)
    return np.column_stack([np.clip(z, -8, 8), np.clip(e0, -8, 8), np.minimum(e0 * e0, 25.0) - 1.0,
                            np.minimum(np.abs(e0), 8.0) - 0.8]).astype(np.float32)


def prep(hist, online, full_hist=False):
    H = VS.fit_history(hist)
    P, zh, coef, se = H["P"], H["z"], H["coef"], H["se"]
    zo = (np.asarray(online, dtype=np.float64) - P[F.P_MU]) / P[F.P_SD]
    z = np.r_[zh, zo]
    e = fwd_resid(z, coef, se, AR)
    T = len(zh)
    h0 = 0 if full_hist else max(0, T - (REF + 2 * max(LS) + AR + HSTRIDE))
    return dict(T=T, n=len(zo), h0=h0, Ch=chans(z, e)[h0:])


def grid(p, L):
    T, n, h0 = p["T"], p["n"], p["h0"]
    lo = max(h0, AR) - T + L - 1
    sh = np.arange(-1, max(lo, -REF) - 1, -HSTRIDE)[::-1]
    so = np.arange(0, n, SUB)
    sl = so - L
    S = np.unique(np.r_[sh, so, sl])
    S = S[S >= lo]                                         # very short histories: drop undefined windows
    return S, np.searchsorted(S, sh), np.searchsorted(S, so), np.searchsorted(S, np.maximum(sl, S[0]))


def windows(p, S, L):
    from numpy.lib.stride_tricks import sliding_window_view
    sw = sliding_window_view(p["Ch"], L, axis=0)
    return np.ascontiguousarray(sw[p["T"] + S - p["h0"] - L + 1])


def stats(W):
    z, e = W[:, 0].astype(np.float64), W[:, 1].astype(np.float64)
    zc = z - z.mean(1, keepdims=True)
    vz = (zc * zc).mean(1) + 1e-6
    e2 = (e * e).mean(1) + 1e-6
    ae = np.abs(e)
    out = [np.log(vz), np.log((np.diff(z, axis=1) ** 2).mean(1) + 1e-6), np.log(e2),
           np.log((e ** 4).mean(1) / e2 ** 2 + 1e-6), np.clip((e ** 3).mean(1) / e2 ** 1.5, -5, 5)]
    for k in range(1, 6):
        out.append((zc[:, k:] * zc[:, :-k]).mean(1) / vz)
    q = e * e - e2[:, None]
    out.append((q[:, 1:] * q[:, :-1]).mean(1) / ((q * q).mean(1) + 1e-6))
    qs = np.quantile(ae, [0.5, 0.9], axis=1)
    out += [np.log(qs[0] + 1e-3), np.log(qs[1] + 1e-3), np.log(ae.max(1) + 1e-3), z.mean(1)]
    return np.column_stack(out).astype(np.float32)


# ---------------------------------------------------------------------------------------------
# cosine machinery

@njit(cache=True)
def _cusum(x, k):
    out = np.empty(len(x))
    s = 0.0
    for i in range(len(x)):
        s = max(0.0, s + x[i] - k)
        out[i] = s
    return out


def whiten(e, ref):
    from scipy.special import ndtri
    n = len(ref)
    k = np.searchsorted(ref, e)
    w = ndtri((k + 0.5) / (n + 1.0))
    hi, lo = e > ref[-1], e < ref[0]
    w[hi] = ndtri((n + 0.5) / (n + 1.0)) + (e[hi] - ref[-1])
    w[lo] = ndtri(0.5 / (n + 1.0)) - (ref[0] - e[lo])
    return np.clip(w, -8.0, 8.0)


def cos_streams(E, ih, io, il, L):
    k = L // HSTRIDE
    if len(ih) < 2 * k + 3:                                # history too short for a null: neutral values
        return np.zeros(len(io)), np.zeros(len(io)), np.zeros(0)
    Eh = E[ih]
    sm = Eh.sum(0)
    anchor = sm / (np.linalg.norm(sm) + 1e-12)
    N = len(Eh)
    CS = np.vstack([np.zeros((1, Eh.shape[1])), np.cumsum(Eh, 0)])
    w = np.arange(N)
    A = sm[None, :] - (CS[np.minimum(w + k, N)] - CS[np.maximum(w - k + 1, 0)])
    ch = (Eh * A).sum(1) / (np.linalg.norm(A, axis=1) + 1e-12)
    mu, sd = ch.mean(), ch.std() + 1e-6
    co = E[io] @ anchor
    cl = (Eh[k:] * Eh[:-k]).sum(1)
    mul, sdl = cl.mean(), cl.std() + 1e-6
    lo = (E[io] * E[il]).sum(1)
    return -(co - mu) / sd, -(lo - mul) / sdl, -(ch - mu) / sd


def block_stream(d, dh, L):
    idx = np.arange(L - 3, len(d) * SUB, L) // SUB
    idx = idx[idx < len(d)]
    return whiten(d[idx], np.sort(dh)) if (len(idx) and len(dh)) else np.zeros(0)


def block_tau(tau, L):
    return -1 if tau < 0 else int(np.ceil((tau + 2) / L))


def fit_block_prior(ws, taub, max_post, min_post=3):
    v, r = [], []
    for w, t in zip(ws, taub):
        if t < 0 or len(w) - t < min_post:
            continue
        x = w[t:t + max_post]
        s2 = max(float(x.var()), 0.25)
        v.append(s2)
        r.append(float(x.mean()) ** 2 / s2 - 1.0 / len(x))
    if len(v) < 20:
        return FALLBACK_PRIOR, len(v)
    v = np.array(v)
    a0 = v.mean() ** 2 / max(v.var(), 1e-6) + 2.0
    return (1.0 / max(float(np.mean(r)), 1e-3), a0, v.mean() * (a0 - 1.0)), len(v)


def post_rows(w, n, L, prior):
    rows = np.arange(0, n, SUB)
    out = np.zeros((len(rows), 4))
    if len(w) == 0:
        return out
    C1 = np.r_[0.0, np.cumsum(w)][None, :]
    C2 = np.r_[0.0, np.cumsum(w * w)][None, :]
    Fb = NB.features_from_sums((C1, C2, np.zeros(len(w) + 1)), prior, 1)[:, :4]
    m = (rows + 3) // L
    ok = m >= 1
    out[ok] = Fb[np.minimum(m[ok], len(w)) - 1]
    return out


def rows_from_streams(d, dl, w, n, L, prior):
    cnt = np.arange(1, len(d) + 1)
    return np.column_stack([d, np.cumsum(d) / cnt, _cusum(d, 0.5), dl, _cusum(dl, 0.5), post_rows(w, n, L, prior)])


# ---------------------------------------------------------------------------------------------
# training (contrastive, InfoNCE)

def _nets():
    import torch
    import torch.nn as nn
    import torch.nn.functional as Fn

    class CNN(nn.Module):
        def __init__(self, width=64):
            super().__init__()
            self.c = nn.Sequential(nn.Conv1d(4, width, 5, padding=2), nn.GELU(),
                                   nn.Conv1d(width, width, 5, padding=4, dilation=2), nn.GELU(),
                                   nn.Conv1d(width, width, 5, padding=8, dilation=4), nn.GELU())
            self.h = nn.Sequential(nn.Linear(2 * width, 128), nn.GELU(), nn.Linear(128, 32))

        def forward(self, x):
            h = self.c(x)
            return Fn.normalize(self.h(torch.cat([h.mean(2), h.std(2)], 1)), dim=1)

    class MLP(nn.Module):
        def __init__(self, cin):
            super().__init__()
            self.f = nn.Sequential(nn.Linear(cin, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 16))

        def forward(self, x):
            return Fn.normalize(self.f(x), dim=1)

    return torch, CNN, MLP


def _train(net, sample, n_series, steps, lr, seed, dev, Ls):
    """sample(ids, L, rng) -> (2B, ...) tensor of paired views; returns the best-validation state."""
    torch, _, _ = _nets()
    import torch.nn.functional as Fn
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_series)
    va, tr = perm[: max(2, n_series // 10)], perm[max(2, n_series // 10):]
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)

    def nce(x):
        z = net(x)
        B = z.shape[0] // 2
        lg = z[:B] @ z[B:].T / TEMP
        lab = torch.arange(B, device=dev)
        return 0.5 * (Fn.cross_entropy(lg, lab) + Fn.cross_entropy(lg.T, lab))

    def valid():
        net.eval()
        r = np.random.default_rng(999)
        with torch.no_grad():
            v = sum(float(nce(sample(r.permutation(va)[:BATCH], L, r))) for L in Ls for _ in range(3))
        net.train()
        return v

    best, best_state = 1e18, None
    B = min(BATCH, len(tr))
    for it in range(steps):
        loss = nce(sample(rng.choice(tr, B, replace=False), Ls[it % len(Ls)], rng))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if (it + 1) % max(1, steps // 6) == 0 or it == steps - 1:
            v = valid()
            if v < best:
                best, best_state = v, {k: t.detach().clone() for k, t in net.state_dict().items()}
    net.load_state_dict(best_state)
    net.eval()
    return net


def train_cnn(hist_chans, seed, steps, dev, width=64):
    torch, CNN, _ = _nets()
    lens = np.array([len(h) for h in hist_chans])
    offs = np.r_[0, np.cumsum(lens)[:-1]]
    X = torch.as_tensor(np.concatenate(hist_chans), device=dev)
    ar = {L: torch.arange(L, device=dev) for L in LS}

    def sample(ids, L, r):
        Tl = lens[ids]
        a = AR + (r.random(len(ids)) * (Tl - 2 * L - AR + 1)).astype(np.int64)
        b = a + L + (r.random(len(ids)) * (Tl - L - (a + L) + 1)).astype(np.int64)
        sw = r.random(len(ids)) < 0.5
        a, b = np.where(sw, b, a), np.where(sw, a, b)
        st = torch.as_tensor(np.r_[offs[ids] + a, offs[ids] + b], device=dev)
        return X[st[:, None] + ar[L][None, :]].permute(0, 2, 1)

    net = _train(CNN(width).to(dev), sample, len(hist_chans), steps, 1e-3, seed, dev, LS)
    del X
    return net


def train_mlp(stat_lists, L, seed, steps):
    """stat_lists: per training series, standardised stats of its history grid windows (stride HSTRIDE)."""
    torch, _, MLP = _nets()
    k = L // HSTRIDE
    lens = np.array([len(x) for x in stat_lists])
    offs = np.r_[0, np.cumsum(lens)[:-1]]
    X = torch.as_tensor(np.concatenate(stat_lists))

    def sample(ids, L_, r):
        n = lens[ids]
        a = (r.random(len(ids)) * (n - k)).astype(np.int64)
        b = a + k + (r.random(len(ids)) * (n - a - k)).astype(np.int64)
        sw = r.random(len(ids)) < 0.5
        a, b = np.where(sw, b, a), np.where(sw, a, b)
        return X[torch.as_tensor(np.r_[offs[ids] + a, offs[ids] + b])]

    return _train(MLP(X.shape[1]), sample, len(stat_lists), steps, 2e-3, seed, torch.device("cpu"), (L,))


# ---------------------------------------------------------------------------------------------
# per-fold features

def fold_features(kind, S, tau, tr_s, extra_items, steps, threads, say=print, width=64):
    """kind 'emb' or 'femb'. S: list of (id, hist, online); extra_items: list (per extra source) of (hist, online)
    lists. Returns [orig matrix] + [one matrix per extra source], each (rows, 18) float32 in cv.py row order."""
    import torch
    from joblib import Parallel, delayed
    t0 = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(max(1, min(threads, 8)))
    tr_s = np.asarray(tr_s)
    keep = [s for s in tr_s if len(S[s][1]) >= 2 * max(LS) + AR + 50]
    if kind == "emb":
        full = Parallel(n_jobs=threads, backend="threading")(delayed(prep)(S[i][1], S[i][2], True) for i in keep)
        hc = [p["Ch"][:p["T"]] for p in full]
        del full
        nets = [train_cnn(hc, seed, steps, dev, width) for seed in SEEDS]
        del hc

        def embed(Ws, L):          # list over seeds of list over series
            sizes = np.cumsum([len(W) for W in Ws])[:-1]
            Xa = np.concatenate(Ws)
            outs = []
            with torch.no_grad():
                for net in nets:
                    o = np.empty((len(Xa), 32), np.float32)
                    for k in range(0, len(Xa), 16384):
                        o[k:k + 16384] = net(torch.as_tensor(Xa[k:k + 16384], device=dev)).cpu().numpy()
                    outs.append(np.split(o, sizes))
            return outs
    else:
        preps = Parallel(n_jobs=threads, backend="threading")(delayed(prep)(S[i][1], S[i][2]) for i in keep)
        std, nets = {}, {}
        for L in LS:
            def hs(p, L=L):
                g = grid(p, L)
                return stats(windows(p, g[0][g[1]], L))
            Xs = Parallel(n_jobs=threads, backend="threading")(delayed(hs)(p) for p in preps)
            R = np.concatenate(Xs)
            mu, sd = R.mean(0), R.std(0) + 1e-6
            std[L] = (mu, sd)
            Xs = [((x - mu) / sd).astype(np.float32) for x in Xs if len(x) > L // HSTRIDE + 1]
            nets[L] = [train_mlp(Xs, L, seed, steps) for seed in SEEDS]
        del preps

        def embed(Ws, L):
            mu, sd = std[L]
            St = Parallel(n_jobs=threads, backend="threading")(delayed(stats)(W) for W in Ws)
            sizes = np.cumsum([len(x) for x in St])[:-1]
            Xa = torch.as_tensor(((np.concatenate(St) - mu) / sd).astype(np.float32))
            with torch.no_grad():
                return [np.split(net(Xa).numpy(), sizes) for net in nets[L]]
    t1 = time.time()

    sources = [[(q[1], q[2]) for q in S]] + list(extra_items)
    streams = []      # per source: dict (L, seed) -> list of (d, dl, w, n)
    for items in sources:
        st = {(L, s): [] for L in LS for s in range(len(SEEDS))}
        for c0 in range(0, len(items), CHUNK):
            part = Parallel(n_jobs=threads, backend="threading")(delayed(prep)(h, o) for h, o in items[c0:c0 + CHUNK])
            for L in LS:
                gs = [grid(p, L) for p in part]
                per_seed = embed([windows(p, g[0], L) for p, g in zip(part, gs)], L)
                for s, Es in enumerate(per_seed):
                    for E, g, p in zip(Es, gs, part):
                        d, dl, dh = cos_streams(E, g[1], g[2], g[3], L)
                        st[(L, s)].append((d.astype(np.float32), dl.astype(np.float32), block_stream(d, dh, L), p["n"]))
        streams.append(st)
    priors = {}
    for L in LS:
        for s in range(len(SEEDS)):
            o = streams[0][(L, s)]
            priors[(L, s)] = fit_block_prior([o[i][2] for i in tr_s], [block_tau(tau[i], L) for i in tr_s],
                                             max_post=max(3, 300 // L))[0]
    out = []
    for st in streams:
        cols = []
        for L in LS:
            acc = None
            for s in range(len(SEEDS)):
                M = np.concatenate([rows_from_streams(d.astype(np.float64), dl.astype(np.float64), w, n, L,
                                                      priors[(L, s)]) for d, dl, w, n in st[(L, s)]])
                acc = M if acc is None else acc + M
            cols.append(acc / len(SEEDS))
        out.append(np.column_stack(cols).astype(np.float32))
    del nets, embed
    if dev.type == "cuda":
        torch.cuda.empty_cache()                           # XGBoost trains on the same GPU right after
    say(f"{kind}: nets trained on {len(keep)} training histories in {t1 - t0:.0f}s; features for "
        f"{sum(len(x) for x in sources):,} series in {time.time() - t1:.0f}s; block priors "
        + ", ".join(f"L{L}: kappa0 {priors[(L, 0)][0]:.2f}" for L in LS))
    return out
