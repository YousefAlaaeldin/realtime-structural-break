"""Embedding drift (owner's idea, 2026-10-15): embed the history as a vector, embed the online points as they arrive,
and read a break from the cosine similarity drifting away.

Design (the reasoning, in short):
- An embedding of a window is a learned summary statistic; cosine to the history embedding is a two-sample test with
  a learned kernel (MMD in disguise; section 21 found the fixed-kernel MMD redundant with --dist). So the question is
  NOT "does the cosine move at a break" (it will) but "does a LEARNED embedding see something the existing features
  and the fixed-kernel tests do not". Hence a hand-made control embedding through identical machinery, and the MMD
  columns in the incremental base.
- The literal signal "cos(now, previous embedding)" is TRANSIENT: it dips while one window straddles the break and
  recovers once both windows sit in the new regime, while the label stays 1 for the rest of the series. Overlapping
  consecutive windows give cos ~ 1 anyway. So both are built: (a) distance to a fixed HISTORY ANCHOR (persists), and
  (b) the non-overlapping lag version cos(e_t, e_{t-L}), accumulated by a CUSUM.
- Section 3: scanning over start times beats fixed windows; every winner so far (bo_, nb_) is a whitened stream fed to
  the BOCPD posterior. So non-overlapping block cosines are whitened by the series' own in-history cosines and fed
  to the same posterior (nbm.features_from_sums), giving pb_logodds/tsb/lvr/mu.

Arms (same cosine machinery for both):
  ctl  hand-made embedding: 15 window statistics (log var z, log var dz, log mean e^2, log kurtosis e, skew e,
       acf z 1-5, acf e^2 lag 1, log q50 / q90 / max |e|, mean z), standardised with fit-set constants.
  emb  learned: 1-D dilated CNN (4 -> 64 x3, mean+std pooling -> 128 -> 32, L2-normalised) on the 4 residual channels
       of backcast_nn_gate.chans (z, e, e^2-1, |e|-0.8). Trained CONTRASTIVELY (InfoNCE, temperature 0.1) on fit-set
       HISTORIES only: positives = two non-overlapping windows of the same series, negatives = the other series in
       the batch (512). No labels, no online data, no per-series descriptors (constant per series, they would let it
       identify series without looking at the dynamics). One net for both window lengths. 2 seeds, features averaged.
Per arm and window length L, 9 columns at every 3rd online step:
  d       -(cos(window ending now, anchor) - mu) / sd; anchor = normalised mean of history windows; mu, sd = the
          series' in-history cosines with a leave-window-out anchor
  dmean   running mean of d;  dcus  CUSUM max(0, S + d - 0.5)
  lag     the same for cos(window now, window L earlier), null from in-history lag pairs;  lagcus  its CUSUM
  pb_*    BOCPD posterior (logodds, tsb, lvr, mu) on the non-overlapping block stream of d, ECDF-whitened by the
          history d values; EB prior fitted on the fit set's post-break blocks
L VALUES (fixed before any run): 30 and 120.

Expectation (written before any run): breaks are subtle (about 0.60 even with the boundary known); a window
embedding is a summary of the same variance / roughness / tails the features already read, and the fixed-kernel
version (MMD) added nothing. Expected best single emb feature about 0.57-0.59, rank correlation > 0.7 with vol_/nb_,
and KE2 killed. A PASS would be a surprise.

KILL CRITERIA (written before any run):
  KE1 (is the learning worth anything): best single emb column (seed-mean) beats the best single ctl column by
      >= +0.003 overall TS-AUC (direction-free: max(auc, 1 - auc)).
  LightGBM incremental, 4 folds by series x 3 seeds; base = bo + vol + nb + step + MMD (12). Project bar: >= +0.003,
  > 2 SE, >= 3/4 folds. (Amended after the first smoke run, before any full run: ctl moved out of the base into its
  own test, so a gain carried by the hand-made embedding is not reported as "closed".)
  KE2c: base + ctl (18) vs base. Pass -> the cheap hand-made version goes to a paired cv.py run (no net needed).
  KE2 (learned): base + ctl + emb (18) vs base + ctl. Pass -> torch/numba runtime + paired cv.py run.
  Instrument check: the base should score about 0.614 or more (same refs and eval set as the backcast gates); far
  below that, distrust the verdicts.

Run (GPU; my estimate 35-55 min: about 5-10 min training + embedding, the rest 36 LightGBM fits):
    python diagnostics/embedding_gate.py
    python diagnostics/embedding_gate.py --fit 0:400 --eval 2000:2600 --folds 2 --lgb-seeds 1 --steps 300   # smoke
Out: cv_out/embedding_gate.txt, cv_out/embedding_gate.npz (features), data/embnet_s<seed>_<fit>.pt
"""
import argparse
import os
import sys
import time

import numpy as np
from numba import njit

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "diagnostics"))
os.chdir(HERE)

import _backcast_common as C   # noqa: E402
import backcast_gate as G      # noqa: E402
import backcast_nn_gate as BN  # noqa: E402
from sbrt import features as F   # noqa: E402
from sbrt import nbm as NB       # noqa: E402
from sbrt import volstate as VS  # noqa: E402

AR = F.AR
LS = (30, 120)          # window lengths, multiples of 6 (history grid stride) and of SUB
HSTRIDE = 6             # history null grid stride
REF = 1500              # history points used for the anchor / null
SEEDS = (0, 1)
DIM = 32
BATCH = 512
TEMP = 0.1
COLS = ["d", "dmean", "dcus", "lag", "lagcus", "pb_logodds", "pb_tsb", "pb_lvr", "pb_mu"]
FALLBACK_PRIOR = (100.0, 10.0, 9.0)


# ---------------------------------------------------------------------------------------------
# data

def prep(hist, online, full_hist):
    H = VS.fit_history(hist)
    P, zh, coef, se = H["P"], H["z"], H["coef"], H["se"]
    zo = (np.asarray(online, dtype=np.float64) - P[F.P_MU]) / P[F.P_SD]
    z = np.r_[zh, zo]
    e = G.fwd_resid(z, coef, se, AR)          # causal: e_t uses z_{t-10..t} only
    T = len(zh)
    h0 = 0 if full_hist else max(0, T - (REF + 2 * max(LS) + AR + HSTRIDE))
    return dict(T=T, n=len(zo), h0=h0, Ch=BN.chans(z, e)[h0:])


def grid(p, L):
    """Window-end positions s (relative to the end of the history: s = -1 is the last history point, s = t is online
    step t). Returns the sorted unique s, and index arrays: history null grid, online rows, lag partners of rows."""
    T, n, h0 = p["T"], p["n"], p["h0"]
    lo = max(h0, AR) - T + L - 1                       # smallest s whose window is fully defined
    sh = np.arange(-1, max(lo, -REF) - 1, -HSTRIDE)[::-1]
    so = np.arange(0, n, C.SUB)
    sl = so - L
    S = np.unique(np.r_[sh, so, sl])
    return S, np.searchsorted(S, sh), np.searchsorted(S, so), np.searchsorted(S, sl)


def windows(p, S, L):
    from numpy.lib.stride_tricks import sliding_window_view
    sw = sliding_window_view(p["Ch"], L, axis=0)        # (N - L + 1, 4, L), window starting at each index
    start = p["T"] + S - p["h0"] - L + 1
    return np.ascontiguousarray(sw[start])


# ---------------------------------------------------------------------------------------------
# control embedding: hand-made window statistics

def stats(W):
    z, e = W[:, 0].astype(np.float64), W[:, 1].astype(np.float64)
    zc = z - z.mean(1, keepdims=True)
    vz = (zc * zc).mean(1) + 1e-6
    e2 = (e * e).mean(1) + 1e-6
    ae = np.abs(e)
    out = [np.log(vz), np.log((np.diff(z, axis=1) ** 2).mean(1) + 1e-6), np.log(e2),
           np.log((e ** 4).mean(1) / e2 ** 2 + 1e-6),
           np.clip((e ** 3).mean(1) / e2 ** 1.5, -5, 5)]
    for k in range(1, 6):
        out.append((zc[:, k:] * zc[:, :-k]).mean(1) / vz)
    q = e * e - e2[:, None]
    out.append((q[:, 1:] * q[:, :-1]).mean(1) / ((q * q).mean(1) + 1e-6))
    qs = np.quantile(ae, [0.5, 0.9], axis=1)
    out += [np.log(qs[0] + 1e-3), np.log(qs[1] + 1e-3), np.log(ae.max(1) + 1e-3), z.mean(1)]
    return np.column_stack(out).astype(np.float32)


def control_embedder(pf, L, threads):
    """returns embed(list of window arrays) -> list of unit vectors; standardisation constants from fit histories."""
    from joblib import Parallel, delayed
    def hist_stats(p):
        S, ih, _, _ = grid(p, L)
        return stats(windows(p, S[ih], L))
    raw = Parallel(n_jobs=threads, backend="threading")(delayed(hist_stats)(p) for p in pf)
    R = np.concatenate(raw)
    mu, sd = R.mean(0), R.std(0) + 1e-6

    def embed(Ws):
        out = Parallel(n_jobs=threads, backend="threading")(delayed(stats)(W) for W in Ws)
        res = []
        for X in out:
            X = (X - mu) / sd
            res.append(X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9))
        return res
    return embed


# ---------------------------------------------------------------------------------------------
# learned embedding

def _torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as Fn

    class Enc(nn.Module):
        def __init__(self):
            super().__init__()
            self.c = nn.Sequential(nn.Conv1d(4, 64, 5, padding=2), nn.GELU(),
                                   nn.Conv1d(64, 64, 5, padding=4, dilation=2), nn.GELU(),
                                   nn.Conv1d(64, 64, 5, padding=8, dilation=4), nn.GELU())
            self.h = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Linear(128, DIM))

        def forward(self, x):
            h = self.c(x)
            return Fn.normalize(self.h(torch.cat([h.mean(2), h.std(2)], 1)), dim=1)

    return torch, Enc


def train_encoder(pf, seed, steps, say):
    torch, Enc = _torch()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    hs = [p["Ch"][:p["T"] - p["h0"]] for p in pf]      # fit preps keep the full history (h0 = 0)
    lens = np.array([len(h) for h in hs])
    offs = np.r_[0, np.cumsum(lens)[:-1]]
    X = torch.as_tensor(np.concatenate(hs), device=dev)
    nS = len(hs)
    perm = rng.permutation(nS)
    va, tr = perm[: max(2, nS // 10)], perm[max(2, nS // 10):]
    net = Enc().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps, pct_start=0.1)
    ar = {L: torch.arange(L, device=dev) for L in LS}

    def batch(ids, L, r):
        Tl = lens[ids]
        a = AR + (r.random(len(ids)) * (Tl - 2 * L - AR + 1)).astype(np.int64)
        b = a + L + (r.random(len(ids)) * (Tl - L - (a + L) + 1)).astype(np.int64)
        sw = r.random(len(ids)) < 0.5
        a, b = np.where(sw, b, a), np.where(sw, a, b)
        st = torch.as_tensor(np.r_[offs[ids] + a, offs[ids] + b], device=dev)
        x = X[st[:, None] + ar[L][None, :]].permute(0, 2, 1)
        return x

    def nce(x):
        z = net(x)
        B = z.shape[0] // 2
        lg = z[:B] @ z[B:].T / TEMP
        lab = torch.arange(B, device=dev)
        loss = 0.5 * (torch.nn.functional.cross_entropy(lg, lab) + torch.nn.functional.cross_entropy(lg.T, lab))
        return loss, (lg.argmax(1) == lab).float().mean()

    def valid():
        net.eval()
        r = np.random.default_rng(999)
        out = {}
        with torch.no_grad():
            for L in LS:
                ls, ac = [], []
                for _ in range(5):
                    ids = r.permutation(va)[:BATCH]
                    l_, a_ = nce(batch(ids, L, r))
                    ls.append(float(l_))
                    ac.append(float(a_))
                out[L] = (np.mean(ls), np.mean(ac))
        net.train()
        return out

    best, best_state, t0 = 1e9, None, time.time()
    B = min(BATCH, len(tr))
    for it in range(steps):
        L = LS[it % len(LS)]
        loss, _ = nce(batch(rng.choice(tr, B, replace=False), L, rng))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if (it + 1) % max(1, steps // 6) == 0 or it == steps - 1:
            v = valid()
            tot = sum(v[L][0] for L in LS)
            if tot < best:
                best, best_state = tot, {k: t.detach().cpu().clone() for k, t in net.state_dict().items()}
                best_v = v
    say(f"    encoder seed {seed}: {len(tr)} train series, {steps} steps, {time.time() - t0:.0f}s; best valid "
        + "  ".join(f"L{L} InfoNCE {best_v[L][0]:.3f} top-1 {best_v[L][1]:.3f} (chance {1 / min(BATCH, len(va)):.3f})"
                    for L in LS))
    net.load_state_dict(best_state)
    net.eval()
    del X
    return net, best_state


def net_embedder(net):
    torch, _ = _torch()
    dev = next(net.parameters()).device

    def embed(Ws):
        sizes = [len(W) for W in Ws]
        Xa = np.concatenate(Ws)
        out = np.empty((len(Xa), DIM), np.float32)
        with torch.no_grad():
            for k in range(0, len(Xa), 16384):
                out[k:k + 16384] = net(torch.as_tensor(Xa[k:k + 16384], device=dev)).float().cpu().numpy()
        return np.split(out, np.cumsum(sizes)[:-1])
    return embed


# ---------------------------------------------------------------------------------------------
# cosine machinery (identical for both arms)

@njit(cache=True)
def _cusum(x, k):
    out = np.empty(len(x))
    s = 0.0
    for i in range(len(x)):
        s = max(0.0, s + x[i] - k)
        out[i] = s
    return out


def cos_streams(E, ih, io, il, L):
    """d at online rows, lag d at online rows, history d values (null for whitening)."""
    Eh = E[ih]
    sm = Eh.sum(0)
    anchor = sm / (np.linalg.norm(sm) + 1e-12)
    # leave-out anchors: drop the window AND its overlapping grid neighbours (|offset| < L / HSTRIDE); keeping
    # near-copies of a window in its own anchor would bias the null cosines upward vs a fresh online window.
    # (A prequential null, anchor = earlier non-overlapping windows only, was tried in the smoke run: no better.)
    N, k = len(Eh), L // HSTRIDE
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
    """non-overlapping blocks: rows at online steps t = L-3, 2L-3, ... (window t-L+1..t), whitened by history d."""
    idx = np.arange(L - 3, len(d) * C.SUB, L) // C.SUB
    idx = idx[idx < len(d)]
    return G.whiten(d[idx], np.sort(dh)) if len(idx) else np.zeros(0)


def block_tau(tau, L):
    """index of the first block whose window starts at or after the break (block k starts at online step kL - 2)."""
    return -1 if tau < 0 else int(np.ceil((tau + 2) / L))


def fit_block_prior(ws, taub, max_post, min_post):
    """nbm.fit_prior_w with the per-series variance floored at 0.25: with only a few blocks per series the raw
    estimator gives near-zero variances and a degenerate kappa0 (seen in the smoke run)."""
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
    rows = np.arange(0, n, C.SUB)
    out = np.zeros((len(rows), 4))
    if len(w) == 0:
        return out
    C1 = np.r_[0.0, np.cumsum(w)][None, :]
    C2 = np.r_[0.0, np.cumsum(w * w)][None, :]
    CD = np.zeros(len(w) + 1)
    Fb = NB.features_from_sums((C1, C2, CD), prior, 1)[:, :4]
    m = (rows + 3) // L                                   # completed blocks at online step t
    ok = m >= 1
    out[ok] = Fb[np.minimum(m[ok], len(w)) - 1]
    return out


def series_rows(E, g, n, L, prior):
    S, ih, io, il = g
    d, dl, dh = cos_streams(E, ih, io, il, L)
    w = block_stream(d, dh, L)
    cnt = np.arange(1, len(d) + 1)
    return np.column_stack([d, np.cumsum(d) / cnt, _cusum(d, 0.5), dl, _cusum(dl, 0.5), post_rows(w, n, L, prior)])


def arm_features(embed, pf, tau_f, pe, tau_e, L, say, chunk=200):
    """prior from the fit series' block streams, then the 9 columns for the eval series."""
    ws = []
    for c0 in range(0, len(pf), chunk):
        part = pf[c0:c0 + chunk]
        gs = [grid(p, L) for p in part]
        Es = embed([windows(p, g[0], L) for p, g in zip(part, gs)])
        for E, g in zip(Es, gs):
            d, _, dh = cos_streams(E, g[1], g[2], g[3], L)
            ws.append(block_stream(d, dh, L))
    taub = [block_tau(t, L) for t in tau_f]
    prior, nb_ = fit_block_prior(ws, taub, max_post=max(3, 300 // L), min_post=3)
    say(f"      L{L}: block prior from {nb_} breaks: kappa0 {prior[0]:.1f} alpha0 {prior[1]:.2f} beta0 {prior[2]:.2f}")
    rows, cd, cw = [], [], []
    for c0 in range(0, len(pe), chunk):
        part = pe[c0:c0 + chunk]
        gs = [grid(p, L) for p in part]
        Es = embed([windows(p, g[0], L) for p, g in zip(part, gs)])
        for j, (E, g, p) in enumerate(zip(Es, gs, part)):
            rows.append(series_rows(E, g, p["n"], L, prior))
            if tau_e[c0 + j] < 0:                         # calibration on no-break series
                d, _, dh = cos_streams(E, g[1], g[2], g[3], L)
                cd.append(d[L // C.SUB:])
                cw.append(block_stream(d, dh, L))
    cd, cw = np.concatenate(cd), np.concatenate(cw)
    say(f"      L{L}: no-break calibration (target ~0 / ~1): d at t >= L mean {cd.mean():+.3f} var {cd.var():.3f}; "
        f"whitened blocks mean {cw.mean():+.3f} var {cw.var():.3f}")
    return np.concatenate(rows), prior


# ---------------------------------------------------------------------------------------------

def mmd_reference(S_all, ev, threads, say):
    from joblib import Parallel, delayed
    from sbrt import mmdstream as MM
    e0, e1 = ev
    path = os.path.join(HERE, "data", f"emb_mmd_{e0}_{e1}.npz")
    if os.path.exists(path):
        say(f"MMD twin: cache {os.path.basename(path)}")
        return np.load(path)["M"]
    t1 = time.time()
    M = np.concatenate(Parallel(n_jobs=threads, backend="threading")(
        delayed(MM.run_series_mmd)(S_all[i][1], S_all[i][2], C.SUB) for i in range(e0, e1))).astype(np.float32)
    np.savez(path, M=M)
    say(f"MMD twin: {time.time() - t1:.0f}s (cached)")
    return M


def causality_check(embed, S_all, ev, prior, L, say):
    worst = 0.0
    for i in range(ev[0], ev[0] + 5):
        h, o = S_all[i][1], S_all[i][2]
        cut = max(3, int(len(o) * 0.6))
        pa, pb = prep(h, o, False), prep(h, o[:cut], False)
        ga, gb = grid(pa, L), grid(pb, L)
        Ea, Eb = embed([windows(pa, ga[0], L), windows(pb, gb[0], L)])
        ra = series_rows(Ea, ga, pa["n"], L, prior)
        rb = series_rows(Eb, gb, pb["n"], L, prior)
        worst = max(worst, float(np.abs(ra[:len(rb)] - rb).max()))
    say(f"  causality (5 series, online truncated at 60%, L{L}): max diff on the shared rows {worst:.2e}")
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", default="0:2000")
    ap.add_argument("--eval", default="2000:10000")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--lgb-seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=3000, help="encoder training steps")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default="embedding_gate")
    args = ap.parse_args()
    from joblib import Parallel, delayed
    from scipy.stats import spearmanr

    from sbrt.data import load_train

    say = C.Log(os.path.join(HERE, "cv_out", f"{args.out}.txt"))
    fit = tuple(map(int, args.fit.split(":")))
    ev = tuple(map(int, args.eval.split(":")))
    S_all, tau_all = load_train()
    t0 = time.time()
    torch, _ = _torch()
    say(f"embedding gate | fit {fit[0]}-{fit[1] - 1} | eval {ev[0]}-{ev[1] - 1} | L {LS} | seeds {SEEDS} | "
        f"device {'cuda' if torch.cuda.is_available() else 'CPU (slow)'}")
    idx = list(range(fit[0], fit[1])) + list(range(ev[0], ev[1]))
    nfit = fit[1] - fit[0]
    preps = Parallel(n_jobs=args.threads)(delayed(prep)(S_all[i][1], S_all[i][2], k < nfit)
                                          for k, i in enumerate(idx))
    pf, pe = preps[:nfit], preps[nfit:]
    tau_f = tau_all[fit[0]:fit[1]]
    tau_e = tau_all[ev[0]:ev[1]]
    say(f"series prepared: {time.time() - t0:.0f}s")

    Bo, V, NBF = C.references(S_all, tau_all, fit, ev, args.threads, say)
    MM = mmd_reference(S_all, ev, args.threads, say)
    step, sid, y, in_slice = C.layout(S_all, tau_all, ev)
    say(f"rows {len(y):,} | eval series {len(pe)}")

    feats = {}
    say("\ncontrol arm (hand-made window statistics)")
    for L in LS:
        t1 = time.time()
        feats[("ctl", L)], _ = arm_features(control_embedder(pf, L, args.threads), pf, tau_f, pe, tau_e, L, say)
        say(f"      L{L}: {time.time() - t1:.0f}s")
    say("\nlearned arm (contrastive CNN)")
    per_seed = {L: [] for L in LS}
    for seed in SEEDS:
        net, state = train_encoder(pf, seed, args.steps, say)
        torch.save(state, os.path.join(HERE, "data", f"embnet_s{seed}_{fit[0]}_{fit[1]}.pt"))
        emb = net_embedder(net)
        for L in LS:
            t1 = time.time()
            Fm, prior = arm_features(emb, pf, tau_f, pe, tau_e, L, say)
            per_seed[L].append(Fm)
            say(f"      L{L}: {time.time() - t1:.0f}s")
            if seed == SEEDS[0]:
                causality_check(emb, S_all, ev, prior, L, say)
        del net
    for L in LS:
        feats[("emb", L)] = np.mean(per_seed[L], 0)
    for k, v in feats.items():
        assert len(v) == len(y), (k, len(v), len(y))

    # single features
    say("\nsingle-feature TS-AUC (* = flipped: reported as 1 - auc)      " + C.HDR + "   max |rho| with bo/vol/nb")
    say(f"  {'nb_logodds (reference)':26s}    {C.fmt(C.auc_row(NBF[:, 0], y, step, in_slice))}")
    REF_COLS = np.column_stack([Bo, V, NBF])
    ref_names = [f"bo{i}" for i in range(Bo.shape[1])] + [f"vol{i}" for i in range(V.shape[1])] + \
                [f"nb{i}" for i in range(NBF.shape[1])]
    sub = np.random.default_rng(0).permutation(len(y))[:200000]
    best = {}
    from sbrt.metric import ts_auc
    for (arm, L), Fm in feats.items():
        for c, nm in enumerate(COLS):
            r = C.auc_row(Fm[:, c], y, step, in_slice)
            flip = r[0] < 0.5
            if flip:
                r = 1.0 - r
            rho = [abs(spearmanr(Fm[sub, c], REF_COLS[sub, j])[0]) for j in range(REF_COLS.shape[1])]
            j = int(np.nanargmax(rho))
            extra = ""
            if arm == "emb":
                extra = "   seeds " + " / ".join(f"{max(a, 1 - a):.4f}" for a in
                                                  (ts_auc(p_[:, c], y, step) for p_ in per_seed[L]))
            say(f"  {arm}{L}_{nm:12s}{'*' if flip else ' '}            {C.fmt(r)}   {rho[j]:.2f} {ref_names[j]}{extra}")
            if r[0] > best.get(arm, (0, ""))[0]:
                best[arm] = (r[0], f"{arm}{L}_{nm}")

    ke1 = best["emb"][0] - best["ctl"][0]
    say(f"\nKE1: best emb {best['emb'][1]} {best['emb'][0]:.4f} vs best ctl {best['ctl'][1]} {best['ctl'][0]:.4f}: "
        f"{ke1:+.4f} -> {'learned embedding ahead' if ke1 >= 0.003 else 'killed (learning adds nothing over the control)'}")

    say(f"\nIncremental: LightGBM {args.folds}-fold by series, {args.lgb_seeds} seeds per fold")
    ctl = np.column_stack([feats[("ctl", L)] for L in LS])
    emb = np.column_stack([feats[("emb", L)] for L in LS])
    base0 = np.column_stack([Bo, V, NBF, step, MM]).astype(np.float32)
    sets = {"base": base0, "base+ctl": np.column_stack([base0, ctl]).astype(np.float32),
            "base+ctl+emb": np.column_stack([base0, ctl, emb]).astype(np.float32)}
    sc = {}
    res = C.incremental(sets, y, step, sid, args.folds, args.lgb_seeds, args.threads, say, scores_out=sc)
    dd = np.array(sc["base+ctl+emb"]) - np.array(sc["base+ctl"])
    r_emb = (dd.mean(), dd.std(ddof=1) / np.sqrt(len(dd)), int((dd > 0).sum()))
    say(f"  {'+emb over base+ctl':14s}: {r_emb[0]:+.4f} +- {r_emb[1]:.4f}  ({r_emb[2]}/{len(dd)} folds)")
    r_ctl = res["base+ctl"]
    ke2c = C.passes_project_bar(r_ctl, args.folds)
    ke2 = C.passes_project_bar(r_emb, args.folds)
    np.savez(os.path.join(HERE, "cv_out", f"{args.out}.npz"), ctl=ctl.astype(np.float32),
             emb=emb.astype(np.float32), step=step, y=y, sid=sid)

    say("\nVERDICTS (criteria written in the docstring before any run)")
    if ev[1] - ev[0] < 4000:
        say("  SMOKE RUN (< 4,000 eval series): pipeline check only; do not act on it.")
    b0 = np.mean(sc["base"])
    if b0 < 0.60:
        say(f"  WARNING: base {b0:.4f} is far below nb_logodds alone; the LightGBM instrument is not trustworthy here.")
    say(f"  KE1 {'pass' if ke1 >= 0.003 else 'killed'} ({ke1:+.4f})")
    say(f"  KE2c hand-made embedding over base: {r_ctl[0]:+.4f} +- {r_ctl[1]:.4f} ({r_ctl[2]}/{args.folds}) -> "
        f"{'PASSES (cheap version goes on: paired cv.py run, no net needed)' if ke2c else 'killed'}")
    say(f"  KE2 learned embedding over base+ctl: {r_emb[0]:+.4f} +- {r_emb[1]:.4f} ({r_emb[2]}/{args.folds}) -> "
        f"{'PASSES (learned version goes on: runtime + paired cv.py run)' if ke2 else 'killed'}")
    if not (ke2 or ke2c):
        say("  -> embedding drift is CLOSED (both the hand-made and the learned embedding)")
    say(f"\ntotal {time.time() - t0:.0f}s; log {say.path}")


if __name__ == "__main__":
    main()
