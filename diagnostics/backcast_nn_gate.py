"""B with a better modeller: a pooled neural network that predicts the PAST, against the same network predicting the
FUTURE (HANDOFF section 32; follows the linear backcast gate of section 31).

The question is NOT "does a neural backcaster beat the linear one" (it will, as the NBM beat AR(10), by modelling the
scale). It is "does the network run BACKWARD beat the SAME network run FORWARD": only then is backcasting adding
anything, since forward scale modelling is already in nb_.

Model (identical for both directions, only the time direction differs):
  per series, history z-scored; residual of the linear AR(10) of that direction (forward e_f: from the 10 points
  before; backward e_b: from the 10 points after; both fitted on the series' own history).
  input = the K nearest points on the model's side (nearest first), 4 channels each: z, e, min(e^2,25)-1,
          min(|e|,8)-0.8 (clipped), plus the 6 history descriptors of the NBM (nbm._ctx)        -> 4K + 6 numbers
  output = location m and log scale ls (clamped +-4) of a Student-t for the residual of the target point (one learned nu)
  net = MLP 4K+6 -> 128 -> 128 -> 2 (GELU), AdamW lr 2e-3, wd 1e-4, OneCycle, batch 2048 (fixed shape),
        6 epochs x 1000 steps, best epoch by validation NLL.
  training data = HISTORIES of the fit series only (break-free, no labels, no online data); 10% of those series
  for validation. Pooled across series (one network per direction, K and seed).
Stream: v = (e - m) / exp(ls), whitened by the ECDF of the same series' in-history v (history-only arrays, so no
online data leaks into the reference), plus lsd = ls - mean ls over the last 300 history targets. Features as the
NBM: BOCPD posterior with an EB prior fitted on the fit set -> logodds, tsb, lvr, mu, lsd, lsd_mean100.

Timing: the backward value of online point s needs points up to s + K + 10 (the input residuals need 10 more), so
it is D = K + 10 steps late. Three variants per K:
  fw_   forward net, on time                        (practical comparison)
  fwD_  forward net, the same stream delayed by D    (equal-timing twin: tests DIRECTION, not lateness)
  bk_   backward net, D late
Seeds: 2 per network (single neural runs move about +-0.005 on this gate, section 29); features averaged over the
seeds for each variant alike; per-seed single-feature numbers are printed too.

K VALUES (fixed before any run): 20 and 50. No other values.

KILL CRITERIA (written before any run), per K:
  KN1 (direction, the interpretation): seed-mean bk_logodds - fwD_logodds < +0.003 overall AND < +0.005 on var-same
      -> the backward direction carries nothing the forward one does not.
  KN2 (practical, the decision): LightGBM incremental, 4 folds by series, 3 seeds averaged per fold;
      base = bo + vol + nb + step + fw_ (6) + fwD_ (6); test = + bk_ (6). Project bar: >= +0.003, > 2 SE, >= 3/4 folds.
  B (neural) goes on (numba port of the net + a paired cv.py run) only if KN2 passes for K = 20 or K = 50.
  Also printed: validation NLL of the backward vs forward nets (equal NLL = the series look the same reversed).

Run (GPU; my estimate 35-55 min: about 10 min training + inference, the rest the 48 LightGBM fits):
    python diagnostics/backcast_nn_gate.py
    python diagnostics/backcast_nn_gate.py --fit 0:400 --eval 2000:2600 --folds 2 --lgb-seeds 1 --steps 150   # smoke
Out: cv_out/backcast_nn_gate.txt, cv_out/backcast_nn_gate_K<K>.npz (features), data/bknn_*.pt (nets)
"""
import argparse
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "diagnostics"))
os.chdir(HERE)

import _backcast_common as C   # noqa: E402
import backcast_gate as G      # noqa: E402
from sbrt import features as F  # noqa: E402
from sbrt import nbm as NB      # noqa: E402
from sbrt import volstate as VS  # noqa: E402

AR = F.AR
KS = (20, 50)
SEEDS = (0, 1)
HID = 128
BATCH = 2048
REF_MAX = 1500
FEAT = ["logodds", "tsb", "lvr", "mu", "lsd", "lsd_mean100"]


# ---------------------------------------------------------------------------------------------
# data

def chans(z, e):
    e0 = np.nan_to_num(e)
    return np.column_stack([np.clip(z, -8, 8), np.clip(e0, -8, 8), np.minimum(e0 * e0, 25.0) - 1.0,
                            np.minimum(np.abs(e0), 8.0) - 0.8]).astype(np.float32)


def prep_series(hist, online, full_hist):
    """Channel arrays per direction. Eval series keep only the history tail they need (h0 / f0 = absolute index of
    each stored array's first element; residuals are computed on the whole history before trimming)."""
    H = VS.fit_history(hist)
    P, zh, coef, se = H["P"], H["z"], H["coef"], H["se"]
    zo = (np.asarray(online, dtype=np.float64) - P[F.P_MU]) / P[F.P_SD]
    b, seb = G.back_fit(zh, AR)
    z = np.r_[zh, zo]
    ef_h, eb_h = G.fwd_resid(zh, coef, se, AR), G.bwd_resid(zh, b, seb)
    ef, eb = G.fwd_resid(z, coef, se, AR), G.bwd_resid(z, b, seb)
    T = len(zh)
    h0 = 0 if full_hist else max(0, T - (REF_MAX + 2 * max(KS) + 2 * AR + 10))
    f0 = max(0, T - (max(KS) + 2 * AR + 10))
    return dict(T=T, n=len(zo), h0=h0, f0=f0, ctx=NB._ctx(zh, coef, se).astype(np.float32),
                hist={"f": (chans(zh, ef_h)[h0:], np.nan_to_num(ef_h).astype(np.float32)[h0:]),
                      "b": (chans(zh, eb_h)[h0:], np.nan_to_num(eb_h).astype(np.float32)[h0:])},
                full={"f": (chans(z, ef)[f0:], np.nan_to_num(ef).astype(np.float32)[f0:]),
                      "b": (chans(z, eb)[f0:], np.nan_to_num(eb).astype(np.float32)[f0:])})


def hist_targets(d, T, K):
    """valid target positions inside a history-only array (window and residuals all defined)."""
    return np.arange(AR + K, T) if d == "f" else np.arange(0, T - AR - K)


def online_targets(d, T, n, K):
    return T + (np.arange(0, n) if d == "f" else np.arange(0, max(n - AR - K, 0)))


def offsets(d, K):
    return -np.arange(1, K + 1) if d == "f" else np.arange(1, K + 1)


def windows(Ch, idx, offs, ctx):
    if len(idx) == 0:
        return np.zeros((0, Ch.shape[1] * len(offs) + len(ctx)), np.float32)
    X = Ch[idx[:, None] + offs[None, :]].reshape(len(idx), -1)
    return np.concatenate([X, np.broadcast_to(ctx, (len(idx), len(ctx)))], 1)


# ---------------------------------------------------------------------------------------------
# model

def _torch():
    import torch
    import torch.nn as nn

    class MLP(nn.Module):
        def __init__(self, cin):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(cin, HID), nn.GELU(), nn.Linear(HID, HID), nn.GELU(), nn.Linear(HID, 2))
            self.lnu = nn.Parameter(torch.tensor(math.log(5.0 - 2.0)))

        def forward(self, x):
            r = self.net(x)
            return r[:, 0], r[:, 1].clamp(-4, 4)

        def nu(self):
            return 2.0 + torch.exp(self.lnu).clamp(0.1, 60.0)

    return torch, MLP


def train_net(preps, d, K, seed, steps, epochs, say):
    torch, MLP = _torch()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    nS = len(preps)
    va_s = set(rng.permutation(nS)[: max(1, nS // 10)].tolist())
    Chs, Es, ctxs, tr_idx, va_idx = [], [], [], [], []
    off = 0
    for i, p in enumerate(preps):
        Ch, E = p["hist"][d]
        t = hist_targets(d, p["T"], K) - p["h0"] + off
        (va_idx if i in va_s else tr_idx).append(t)
        Chs.append(Ch)
        Es.append(E)
        ctxs.append(np.repeat(p["ctx"][None], len(E), 0))
        off += len(E)
    Ch = torch.as_tensor(np.concatenate(Chs), device=dev)
    E = torch.as_tensor(np.concatenate(Es), device=dev)
    CT = torch.as_tensor(np.concatenate(ctxs), device=dev)
    tr = np.concatenate(tr_idx)
    va = np.concatenate(va_idx)
    va = va[np.random.default_rng(99).permutation(len(va))[:100 * BATCH]]
    va = va[: (len(va) // BATCH) * BATCH]
    offs = torch.as_tensor(offsets(d, K), device=dev)
    net = MLP(4 * K + 6).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=epochs * steps, pct_start=0.1)

    def batch(ix):
        ix = torch.as_tensor(ix, device=dev)
        x = Ch[ix[:, None] + offs[None, :]].reshape(len(ix), -1)
        return torch.cat([x, CT[ix]], 1), E[ix]

    def valid():
        net.eval()
        tot = 0.0
        with torch.no_grad():
            for k in range(0, len(va), BATCH):
                x, y = batch(va[k:k + BATCH])
                m, ls = net(x)
                tot += float(NB.t_nll(torch, y, m, ls, net.nu()).sum())
        net.train()
        return tot / len(va)

    best, best_state = 1e9, None
    t0 = time.time()
    for ep in range(epochs):
        for _ in range(steps):
            x, y = batch(tr[rng.integers(0, len(tr), BATCH)])
            m, ls = net(x)
            loss = NB.t_nll(torch, y, m, ls, net.nu()).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
        v = valid()
        if v < best:
            best, best_state = v, {k: t.detach().cpu().clone() for k, t in net.state_dict().items()}
    say(f"    net {d} K{K} seed {seed}: {len(tr):,} train targets, best valid NLL {best:.4f} ({time.time() - t0:.0f}s)")
    net.load_state_dict(best_state)
    net.eval()
    del Ch, E, CT
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return net, best, best_state


def predict(net, X):
    torch, _ = _torch()
    dev = next(net.parameters()).device
    m_out, l_out = np.empty(len(X)), np.empty(len(X))
    B = 8 * BATCH
    with torch.no_grad():
        for k in range(0, len(X), B):
            xb = X[k:k + B]
            pad = B - len(xb)
            if pad:
                xb = np.concatenate([xb, np.zeros((pad, X.shape[1]), np.float32)])
            m, ls = net(torch.as_tensor(xb, device=dev))
            m_out[k:k + B] = m.cpu().numpy()[:B - pad]
            l_out[k:k + B] = ls.cpu().numpy()[:B - pad]
    return m_out, l_out


def streams(net, preps, d, K):
    """per series: whitened online stream w and scale drift lsd (online s = 0.. for 'f'; 0..n-D-1 for 'b')."""
    out = []
    for c0 in range(0, len(preps), 200):
        part = preps[c0:c0 + 200]
        Xs, meta = [], []
        for p in part:
            Chh, Eh = p["hist"][d]
            ih = hist_targets(d, p["T"], K)[-REF_MAX:] - p["h0"]
            Chf, Ef = p["full"][d]
            io = online_targets(d, p["T"], p["n"], K) - p["f0"]
            Xs.append(windows(Chh, ih, offsets(d, K), p["ctx"]))
            Xs.append(windows(Chf, io, offsets(d, K), p["ctx"]))
            meta.append((Eh[ih], Ef[io]))
        m, ls = predict(net, np.concatenate(Xs).astype(np.float32))
        k = 0
        for eh, eo in meta:
            mh, lh = m[k:k + len(eh)], ls[k:k + len(eh)]
            k += len(eh)
            mo, lo = m[k:k + len(eo)], ls[k:k + len(eo)]
            k += len(eo)
            vh = (eh - mh) * np.exp(-lh)
            vo = (eo - mo) * np.exp(-lo)
            w = G.whiten(vo, np.sort(vh)) if len(vo) else np.zeros(0)
            out.append((w, lo - lh[-300:].mean()))
    return out


def feats(st, prior):
    w, dd = st
    m = len(w)
    if m == 0:
        return np.zeros((0, 6))
    C1 = np.r_[0.0, np.cumsum(w)][None, :]
    C2 = np.r_[0.0, np.cumsum(w * w)][None, :]
    CD = np.r_[0.0, np.cumsum(dd)]
    return NB.features_from_sums((C1, C2, CD), prior, 1)


def rows_from(Fm, n, delay):
    """feature rows at online steps 0, SUB, ... from a stream whose count at step t is t - delay + 1."""
    rows = np.arange(0, n, C.SUB)
    out = np.zeros((len(rows), 6))
    ok = rows >= delay
    if ok.any():
        out[ok] = Fm[rows[ok] - delay]
    return out


# ---------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", default="0:2000")
    ap.add_argument("--eval", default="2000:10000")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--lgb-seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=1000, help="training steps per epoch")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default="backcast_nn_gate")
    args = ap.parse_args()
    from joblib import Parallel, delayed

    from sbrt.data import load_train
    from sbrt.metric import ts_auc

    say = C.Log(os.path.join(HERE, "cv_out", f"{args.out}.txt"))
    fit = tuple(map(int, args.fit.split(":")))
    ev = tuple(map(int, args.eval.split(":")))
    S_all, tau_all = load_train()
    t0 = time.time()
    torch, _ = _torch()
    say(f"backcast NN gate | fit {fit[0]}-{fit[1] - 1} | eval {ev[0]}-{ev[1] - 1} | K {KS} | seeds {SEEDS} | "
        f"device {'cuda' if torch.cuda.is_available() else 'CPU (slow)'}")
    idx = list(range(fit[0], fit[1])) + list(range(ev[0], ev[1]))
    nfit = fit[1] - fit[0]
    preps = Parallel(n_jobs=args.threads)(delayed(prep_series)(S_all[i][1], S_all[i][2], k < nfit)
                                          for k, i in enumerate(idx))
    pf, pe = preps[:nfit], preps[nfit:]
    tau_f = tau_all[fit[0]:fit[1]]
    say(f"series prepared: {time.time() - t0:.0f}s")

    Bo, V, NBF = C.references(S_all, tau_all, fit, ev, args.threads, say)
    step, sid, y, in_slice = C.layout(S_all, tau_all, ev)
    say(f"rows {len(y):,} | eval series {len(pe)}")
    nb_row = C.auc_row(NBF[:, 0], y, step, in_slice)
    say("\nsingle-feature TS-AUC            " + C.HDR)
    say(f"  nb_logodds (reference)        {C.fmt(nb_row)}")

    verdicts = []
    for K in KS:
        D = K + AR
        say(f"\n===== K = {K} (backward delay D = {D}) =====")
        per = {v: [] for v in ("fw", "fwD", "bk")}     # per seed: eval feature rows (rows x 6)
        nll = {"f": [], "b": []}
        for seed in SEEDS:
            for d in ("f", "b"):
                net, v, state = train_net(pf, d, K, seed, args.steps, args.epochs, say)
                nll[d].append(v)
                torch.save(state, os.path.join(HERE, "data", f"bknn_{d}_K{K}_s{seed}_{fit[0]}_{fit[1]}.pt"))
                sf = streams(net, pf, d, K)
                prior, nb_ = NB.fit_prior_w([s_[0] for s_ in sf], tau_f)
                se_ = streams(net, pe, d, K)
                Fs = [feats(s_, prior) for s_ in se_]
                if d == "f":
                    per["fw"].append(np.concatenate([rows_from(Fm, p["n"], 0) for Fm, p in zip(Fs, pe)]))
                    per["fwD"].append(np.concatenate([rows_from(Fm, p["n"], D) for Fm, p in zip(Fs, pe)]))
                else:
                    per["bk"].append(np.concatenate([rows_from(Fm, p["n"], D) for Fm, p in zip(Fs, pe)]))
                del net
        say(f"  validation NLL (seed mean): forward {np.mean(nll['f']):.4f}  backward {np.mean(nll['b']):.4f}")
        mean = {v: np.mean(per[v], 0) for v in per}
        rows = {}
        for v in ("fw", "fwD", "bk"):
            for c, nm in enumerate(FEAT):
                r = C.auc_row(mean[v][:, c], y, step, in_slice)
                rows[(v, nm)] = r
                if nm in ("logodds", "lvr", "lsd"):
                    extra = ""
                    if nm == "logodds":
                        extra = "   seeds " + " / ".join(f"{ts_auc(p_[:, 0], y, step):.4f}" for p_ in per[v])
                    say(f"  {v + '_' + nm:28s}  {C.fmt(r)}{extra}")
        g_all = rows[("bk", "logodds")][0] - rows[("fwD", "logodds")][0]
        g_vs = rows[("bk", "logodds")][5] - rows[("fwD", "logodds")][5]
        kn1 = g_all < 0.003 and g_vs < 0.005
        say(f"  bk - fwD (logodds): all {g_all:+.4f}, var-same {g_vs:+.4f}; bk - fw: all "
            f"{rows[('bk', 'logodds')][0] - rows[('fw', 'logodds')][0]:+.4f}")
        a = C.boot_diff(mean["bk"][:, 0], mean["fwD"][:, 0], y, step, sid)
        say(f"  bk - fwD logodds, series-bootstrap: {a[0]:+.4f} +- {a[1]:.4f}")
        say(f"\n  Incremental (K {K}): LightGBM {args.folds}-fold by series, {args.lgb_seeds} seeds per fold")
        base = np.column_stack([Bo, V, NBF, step, mean["fw"], mean["fwD"]]).astype(np.float32)
        sets = {"base": base, "base+bk": np.column_stack([base, mean["bk"]]).astype(np.float32)}
        res = C.incremental(sets, y, step, sid, args.folds, args.lgb_seeds, args.threads, say)
        kn2 = C.passes_project_bar(res["base+bk"], args.folds)
        verdicts.append((K, kn1, kn2, res["base+bk"]))
        np.savez(os.path.join(HERE, "cv_out", f"{args.out}_K{K}.npz"), fw=mean["fw"].astype(np.float32),
                 fwD=mean["fwD"].astype(np.float32), bk=mean["bk"].astype(np.float32), step=step, y=y, sid=sid)

    say("\nVERDICTS (criteria written in the docstring before any run)")
    if ev[1] - ev[0] < 4000:
        say("  SMOKE RUN (< 4,000 eval series): pipeline check only; do not act on it.")
    for K, kn1, kn2, r in verdicts:
        say(f"  K {K}: KN1 direction {'NO advantage (killed)' if kn1 else 'backward ahead of equal-timing forward'}; "
            f"KN2 incremental {r[0]:+.4f} +- {r[1]:.4f} ({r[2]}/{args.folds}) -> {'PASSES' if kn2 else 'killed'}")
    go = any(v[2] for v in verdicts)
    say(f"  -> {'B (neural) goes on: numba port + paired cv.py run' if go else 'B (neural) is CLOSED'}")
    say(f"\ntotal {time.time() - t0:.0f}s; log {say.path}")


if __name__ == "__main__":
    main()
