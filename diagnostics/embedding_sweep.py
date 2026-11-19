"""Embedding drift: encoder hyper-parameter sweep + tunable alarm thresholds (owner's request, 2026-10-15; HANDOFF
section 33). Follows embedding_gate.py (learned embedding +0.0019 +- 0.0014 over the base, 1.4 SE: not shown).

The detector, as the owner framed it: embed the history (anchor = mean embedding of the history windows), embed the
window ending at each new point, d_t = standardised drop of cos(window_t, anchor) against the series' own in-history
cosines; raise an alarm when the drift crosses a tunable threshold.
Thresholds (two knobs, both swept):
  k  CUSUM allowance: S_t = max(0, S_{t-1} + d_t - k)  (drift below k per step is ignored)
  h  alarm level: the alarm fires the first time S_t >= h and stays on. Feature `since` = steps since the first alarm
     (+1), 0 before it.
Why both: TS-AUC ranks series at each step, so a continuous score already contains every threshold at once; a hard
threshold can only add through MEMORY (the alarm stays on after the drift fades). So `cus_k` (continuous) and
`since_k_h` (alarm with memory) are both scored; k also changes the statistic itself.

SWEEP (stage 1): one encoder per config (seed 0), trained on fit-set histories exactly as embedding_gate.py but on its
own window length only. Scored on SWEEP series (eval series below --split): direction-free single-feature TS-AUC of
d, dmean, lag, cus_k (k grid), lagcus_k, since_k_h (k x h grid). Config score = its best feature.
  base: L 120, width 64, dim 32, temperature 0.1, 3000 steps, lr 1e-3; one change at a time:
  L 30 / 60 / 240; width 128; dim 16 / 64; temperature 0.05 / 0.2; 10000 steps; lr 3e-3.
CONFIRM (stage 2, untouched series >= --split): best config (2 seeds averaged) with its best k and (k, h); 10 columns
(d, dmean, cus_k*, lag, lagcus_k*, pb_logodds/tsb/lvr/mu, since_k*_h*). Compared with the UNSWEPT gate features
(embedding_gate.npz emb, 18 columns) on the same rows. LightGBM 8 folds x 3 seeds, base = bo + vol + nb + step + MMD.

RULES (written before any run):
  S1 (did the sweep help, interpretation): best single swept column on the confirm series beats the best single
     unswept emb column by >= +0.003.
  S2 (decision): base + swept vs base on the confirm series: >= +0.003, > 2 SE, >= 6/8 folds -> paired cv.py run.
  Selection happens only on the sweep series, so the confirm numbers are not inflated by the search.
Expectation (written before any run): series-identity retrieval is already 98.7% at L 120, so the encoder is not the
bottleneck (the objective is: it learns what tells SERIES apart, which a subtle break barely moves). Sweep gains of
+0.002-0.005 single-feature at most; the threshold features about equal to cus/dmean; S2 killed.

Run (GPU + CPU; my estimate 25-40 min: sweep about 8-12 min, confirm about 3 min, 72 LightGBM fits the rest):
    python diagnostics/embedding_sweep.py
    python diagnostics/embedding_sweep.py --fit 0:400 --eval 2000:2600 --split 2300 --configs base,L30,d16 --steps-scale 0.1 --folds 2 --lgb-seeds 1 --emb-npz cv_out/embedding_gate_smoke.npz --out embedding_sweep_smoke   # smoke
Out: cv_out/embedding_sweep.txt, cv_out/embedding_sweep.csv (every config x feature), cv_out/embedding_sweep.npz
"""
import argparse
import math
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
import embedding_gate as EG    # noqa: E402

BASE = dict(L=120, width=64, dim=32, temp=0.1, steps=3000, lr=1e-3)
CONFIGS = {
    "base": {},
    "L30": dict(L=30), "L60": dict(L=60), "L240": dict(L=240),
    "w128": dict(width=128),
    "d16": dict(dim=16), "d64": dict(dim=64),
    "t0.05": dict(temp=0.05), "t0.2": dict(temp=0.2),
    "s10k": dict(steps=10000),
    "lr3e-3": dict(lr=3e-3),
}
BATCH = 512


# ---------------------------------------------------------------------------------------------
# encoder

def make_net(width, dim):
    import torch
    import torch.nn as nn
    import torch.nn.functional as Fn

    class Enc(nn.Module):
        def __init__(self):
            super().__init__()
            self.c = nn.Sequential(nn.Conv1d(4, width, 5, padding=2), nn.GELU(),
                                   nn.Conv1d(width, width, 5, padding=4, dilation=2), nn.GELU(),
                                   nn.Conv1d(width, width, 5, padding=8, dilation=4), nn.GELU())
            self.h = nn.Sequential(nn.Linear(2 * width, 128), nn.GELU(), nn.Linear(128, dim))

        def forward(self, x):
            h = self.c(x)
            return Fn.normalize(self.h(torch.cat([h.mean(2), h.std(2)], 1)), dim=1)

    return Enc()


def train(pf, cfg, seed, steps, say, tag):
    import torch
    import torch.nn.functional as Fn
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    L = cfg["L"]
    hs = [p["Ch"][:p["T"] - p["h0"]] for p in pf]
    lens = np.array([len(h) for h in hs])
    offs = np.r_[0, np.cumsum(lens)[:-1]]
    X = torch.as_tensor(np.concatenate(hs), device=dev)
    nS = len(hs)
    perm = rng.permutation(nS)
    va, tr = perm[: max(2, nS // 10)], perm[max(2, nS // 10):]
    net = make_net(cfg["width"], cfg["dim"]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=steps, pct_start=0.1)
    ar = torch.arange(L, device=dev)

    def batch(ids, r):
        Tl = lens[ids]
        a = EG.AR + (r.random(len(ids)) * (Tl - 2 * L - EG.AR + 1)).astype(np.int64)
        b = a + L + (r.random(len(ids)) * (Tl - L - (a + L) + 1)).astype(np.int64)
        sw = r.random(len(ids)) < 0.5
        a, b = np.where(sw, b, a), np.where(sw, a, b)
        st = torch.as_tensor(np.r_[offs[ids] + a, offs[ids] + b], device=dev)
        return X[st[:, None] + ar[None, :]].permute(0, 2, 1)

    def nce(x):
        z = net(x)
        B = z.shape[0] // 2
        lg = z[:B] @ z[B:].T / cfg["temp"]
        lab = torch.arange(B, device=dev)
        return 0.5 * (Fn.cross_entropy(lg, lab) + Fn.cross_entropy(lg.T, lab)), (lg.argmax(1) == lab).float().mean()

    def valid():
        net.eval()
        r = np.random.default_rng(999)
        with torch.no_grad():
            res = [nce(batch(r.permutation(va)[:BATCH], r)) for _ in range(5)]
        net.train()
        return np.mean([float(a) for a, _ in res]), np.mean([float(b) for _, b in res])

    best, best_state, best_v, t0 = 1e9, None, None, time.time()
    B = min(BATCH, len(tr))
    for it in range(steps):
        loss, _ = nce(batch(rng.choice(tr, B, replace=False), rng))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if (it + 1) % max(1, steps // 6) == 0 or it == steps - 1:
            v = valid()
            if v[0] < best:
                best, best_v = v[0], v
                best_state = {k: t.detach().cpu().clone() for k, t in net.state_dict().items()}
    net.load_state_dict(best_state)
    net.eval()
    del X
    say(f"    {tag} seed {seed}: {steps} steps {time.time() - t0:.0f}s; valid InfoNCE {best_v[0]:.3f} top-1 "
        f"{best_v[1]:.3f} (chance {1 / min(BATCH, len(va)):.3f})")
    return net


def embedder(net, dim):
    import torch
    dev = next(net.parameters()).device

    def embed(Ws):
        sizes = [len(W) for W in Ws]
        Xa = np.concatenate(Ws)
        out = np.empty((len(Xa), dim), np.float32)
        with torch.no_grad():
            for k in range(0, len(Xa), 16384):
                out[k:k + 16384] = net(torch.as_tensor(Xa[k:k + 16384], device=dev)).float().cpu().numpy()
        return np.split(out, np.cumsum(sizes)[:-1])
    return embed


# ---------------------------------------------------------------------------------------------
# thresholds

@njit(cache=True)
def cusum_multi(x, starts, k):
    out = np.empty(len(x))
    for s in range(len(starts) - 1):
        S = 0.0
        for i in range(starts[s], starts[s + 1]):
            S = max(0.0, S + x[i] - k)
            out[i] = S
    return out


@njit(cache=True)
def since_alarm(S, starts, h, sub):
    out = np.zeros(len(S))
    for s in range(len(starts) - 1):
        first = -1
        for i in range(starts[s], starts[s + 1]):
            if first < 0 and S[i] >= h:
                first = i
            if first >= 0:
                out[i] = (i - first) * sub + 1.0
    return out


def mean_multi(x, starts):
    out = np.empty(len(x))
    for s in range(len(starts) - 1):
        a, b = starts[s], starts[s + 1]
        out[a:b] = np.cumsum(x[a:b]) / np.arange(1, b - a + 1)
    return out


def threshold_feats(d, dl, starts, ks, hs):
    F = {"d": d, "dmean": mean_multi(d, starts), "lag": dl}
    for k in ks:
        S = cusum_multi(d, starts, k)
        F[f"cus_{k:g}"] = S
        F[f"lagcus_{k:g}"] = cusum_multi(dl, starts, k)
        for h in hs:
            F[f"since_{k:g}_{h:g}"] = since_alarm(S, starts, h, C.SUB)
    return F


def dirfree(a):
    return max(a, 1.0 - a)


# ---------------------------------------------------------------------------------------------

def streams(embed, preps, L, chunk=200):
    """d and lag-d at online rows, concatenated over the series, plus series starts."""
    ds, dls = [], []
    for c0 in range(0, len(preps), chunk):
        part = preps[c0:c0 + chunk]
        gs = [EG.grid(p, L) for p in part]
        Es = embed([EG.windows(p, g[0], L) for p, g in zip(part, gs)])
        for E, g in zip(Es, gs):
            d, dl, _ = EG.cos_streams(E, g[1], g[2], g[3], L)
            ds.append(d)
            dls.append(dl)
    starts = np.r_[0, np.cumsum([len(d) for d in ds])].astype(np.int64)
    return np.concatenate(ds), np.concatenate(dls), starts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", default="0:2000")
    ap.add_argument("--eval", default="2000:10000")
    ap.add_argument("--split", type=int, default=5000, help="eval series < split: sweep; >= split: confirm")
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--ks", default="0,0.25,0.5,1,1.5,2", help="CUSUM allowances")
    ap.add_argument("--hs", default="2,5,10,20,40", help="alarm levels")
    ap.add_argument("--steps-scale", type=float, default=1.0)
    ap.add_argument("--folds", type=int, default=8)
    ap.add_argument("--lgb-seeds", type=int, default=3)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--emb-npz", default="cv_out/embedding_gate.npz", help="unswept gate features (comparison)")
    ap.add_argument("--out", default="embedding_sweep")
    args = ap.parse_args()
    import torch
    from joblib import Parallel, delayed

    from sbrt.data import load_train
    from sbrt.metric import ts_auc

    say = C.Log(os.path.join(HERE, "cv_out", f"{args.out}.txt"))
    fit = tuple(map(int, args.fit.split(":")))
    ev = tuple(map(int, args.eval.split(":")))
    ks = [float(v) for v in args.ks.split(",")]
    hs = [float(v) for v in args.hs.split(",")]
    names = args.configs.split(",")
    S_all, tau_all = load_train()
    t0 = time.time()
    say(f"embedding sweep | fit {fit[0]}-{fit[1] - 1} | sweep {ev[0]}-{args.split - 1} | confirm {args.split}-{ev[1] - 1}"
        f" | configs {names} | k {ks} | h {hs} | device {'cuda' if torch.cuda.is_available() else 'CPU'}")
    idx = list(range(fit[0], fit[1])) + list(range(ev[0], ev[1]))
    nfit = fit[1] - fit[0]
    preps = Parallel(n_jobs=args.threads)(delayed(EG.prep)(S_all[i][1], S_all[i][2], k < nfit)
                                          for k, i in enumerate(idx))
    pf, pe = preps[:nfit], preps[nfit:]
    nsw = args.split - ev[0]
    psw, pcf = pe[:nsw], pe[nsw:]
    tau_f, tau_cf = tau_all[fit[0]:fit[1]], tau_all[args.split:ev[1]]
    step, sid, y, in_slice = C.layout(S_all, tau_all, ev)
    msw, mcf = sid < nsw, sid >= nsw
    y_sw, st_sw = y[msw], step[msw]
    say(f"series prepared: {time.time() - t0:.0f}s | rows sweep {msw.sum():,} confirm {mcf.sum():,}")

    # ---------------- stage 1: sweep
    say("\nSTAGE 1: sweep (seed 0, scored on the sweep series)")
    rows, nets, best_cfg = [], {}, None
    for nm in names:
        cfg = dict(BASE, **CONFIGS[nm])
        steps = max(50, int(cfg["steps"] * args.steps_scale))
        net = train(pf, cfg, 0, steps, say, nm)
        nets[nm] = (cfg, steps, {k: t.cpu() for k, t in net.state_dict().items()})
        t1 = time.time()
        d, dl, starts = streams(embedder(net, cfg["dim"]), psw, cfg["L"])
        assert len(d) == msw.sum()
        F = threshold_feats(d, dl, starts, ks, hs)
        res = {f: dirfree(ts_auc(x, y_sw, st_sw)) for f, x in F.items()}
        for f, a in res.items():
            rows.append((nm, f, a))
        bk = max((f for f in res if f.startswith("cus_")), key=res.get)
        bs = max((f for f in res if f.startswith("since_")), key=res.get)
        bl = max((f for f in res if f.startswith("lagcus_")), key=res.get)
        top = max(res, key=res.get)
        say(f"      {nm:7s} d {res['d']:.4f}  dmean {res['dmean']:.4f}  lag {res['lag']:.4f}  best {bk} {res[bk]:.4f}  "
            f"best {bl} {res[bl]:.4f}  best {bs} {res[bs]:.4f}  -> top {top} {res[top]:.4f}  ({time.time() - t1:.0f}s)")
        if best_cfg is None or res[top] > best_cfg[1]:
            best_cfg = (nm, res[top], res)
        if nm == names[0]:
            say(f"      threshold grid for {nm} (since-alarm TS-AUC, rows k, columns h {hs}):")
            for k in ks:
                say(f"        k {k:4g}: " + "  ".join(f"{res[f'since_{k:g}_{h:g}']:.4f}" for h in hs)
                    + f"   | continuous cus_{k:g} {res[f'cus_{k:g}']:.4f}")
        del net
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    import csv
    with open(os.path.join(HERE, "cv_out", f"{args.out}.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["config", "feature", "ts_auc_dirfree_sweep"])
        w.writerows(rows)

    nm, top_auc, res = best_cfg
    kd = float(max((f for f in res if f.startswith("cus_")), key=res.get).split("_")[1])
    kl = float(max((f for f in res if f.startswith("lagcus_")), key=res.get).split("_")[1])
    bs = max((f for f in res if f.startswith("since_")), key=res.get)
    ka, ha = float(bs.split("_")[1]), float(bs.split("_")[2])
    say(f"\nselected on the sweep series: config {nm} (top {top_auc:.4f}); cus k {kd:g}; lagcus k {kl:g}; "
        f"alarm k {ka:g} h {ha:g}")

    # ---------------- stage 2: confirm
    say(f"\nSTAGE 2: confirm on series {args.split}-{ev[1] - 1} (never used for selection), seeds 0 + 1")
    cfg, steps, state0 = nets[nm]
    per = []
    for seed in (0, 1):
        net = make_net(cfg["width"], cfg["dim"]).to("cuda" if torch.cuda.is_available() else "cpu")
        if seed == 0:
            net.load_state_dict(state0)
            net.eval()
        else:
            net = train(pf, cfg, seed, steps, say, nm)
        Fm, _ = EG.arm_features(embedder(net, cfg["dim"]), pf, tau_f, pcf, tau_cf, cfg["L"], say)
        starts = np.r_[0, np.cumsum([(p["n"] + C.SUB - 1) // C.SUB for p in pcf])].astype(np.int64)
        d, dl = np.ascontiguousarray(Fm[:, 0]), np.ascontiguousarray(Fm[:, 3])
        Fm[:, 2] = cusum_multi(d, starts, kd)
        Fm[:, 4] = cusum_multi(dl, starts, kl)
        since = since_alarm(cusum_multi(d, starts, ka), starts, ha, C.SUB)
        per.append(np.column_stack([Fm, since]))
        del net
    swept = np.mean(per, 0)
    cols = ["d", "dmean", f"cus_{kd:g}", "lag", f"lagcus_{kl:g}", "pb_logodds", "pb_tsb", "pb_lvr", "pb_mu",
            f"since_{ka:g}_{ha:g}"]
    y_cf, st_cf, sl_cf = y[mcf], step[mcf], in_slice[mcf]
    sid_cf = sid[mcf] - nsw
    assert len(swept) == len(y_cf)
    z = np.load(os.path.join(HERE, args.emb_npz))
    assert np.array_equal(z["step"], step), "emb npz rows do not match this eval range"
    default = z["emb"][mcf]
    say("\nsingle-feature TS-AUC on the confirm series (direction-free)   " + C.HDR)
    b_sw = max(dirfree(ts_auc(swept[:, c], y_cf, st_cf)) for c in range(swept.shape[1]))
    b_df = max(dirfree(ts_auc(default[:, c], y_cf, st_cf)) for c in range(default.shape[1]))
    for c, cn in enumerate(cols):
        r = C.auc_row(swept[:, c], y_cf, st_cf, sl_cf)
        r = 1.0 - r if r[0] < 0.5 else r
        say(f"  swept_{cn:18s}  {C.fmt(r)}")
    say(f"  best swept {b_sw:.4f} vs best unswept gate emb {b_df:.4f}: {b_sw - b_df:+.4f}")

    Bo, V, NBF = C.references(S_all, tau_all, fit, ev, args.threads, say)
    MM = EG.mmd_reference(S_all, ev, args.threads, say)
    base = np.column_stack([Bo, V, NBF, step, MM])[mcf].astype(np.float32)
    sets = {"base": base, "+unswept": np.column_stack([base, default]).astype(np.float32),
            "+swept": np.column_stack([base, swept]).astype(np.float32)}
    say(f"\nIncremental on the confirm series: LightGBM {args.folds}-fold by series, {args.lgb_seeds} seeds per fold")
    sc = {}
    C.incremental(sets, y_cf, st_cf, sid_cf, args.folds, args.lgb_seeds, args.threads, say, scores_out=sc)
    dd = np.array(sc["+swept"]) - np.array(sc["+unswept"])
    say(f"  +swept vs +unswept: {dd.mean():+.4f} +- {dd.std(ddof=1) / np.sqrt(len(dd)):.4f} ({(dd > 0).sum()}/{len(dd)})")
    np.savez(os.path.join(HERE, "cv_out", f"{args.out}.npz"), swept=swept.astype(np.float32), cols=np.array(cols),
             step=st_cf, y=y_cf, sid=sid_cf)

    d = np.array(sc["+swept"]) - np.array(sc["base"])
    m, se, npos = d.mean(), d.std(ddof=1) / np.sqrt(len(d)), int((d > 0).sum())
    s2 = m >= 0.003 and m > 2 * se and npos >= math.ceil(0.75 * len(d))
    say("\nVERDICTS (rules in the docstring, written before any run)")
    if ev[1] - args.split < 2000:
        say("  SMOKE RUN: pipeline check only; do not act on it.")
    if np.mean(sc["base"]) < 0.60:
        say(f"  WARNING: base {np.mean(sc['base']):.4f} is far below nb_logodds alone; instrument not trustworthy.")
    say(f"  S1 sweep vs unswept (best single, confirm series): {b_sw - b_df:+.4f} -> "
        f"{'the sweep helped' if b_sw - b_df >= 0.003 else 'no real gain from the sweep'}")
    say(f"  S2 base + swept vs base: {m:+.4f} +- {se:.4f} ({npos}/{len(d)} folds) -> "
        f"{'PASSES: paired cv.py run next' if s2 else 'killed'}")
    say(f"\ntotal {time.time() - t0:.0f}s; log {say.path}")


if __name__ == "__main__":
    main()
