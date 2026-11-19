"""Sub-test of the embedding gate (HANDOFF section 33): a LEARNED embedding of window STATISTICS (owner's question,
2026-10-15: "embedding on the features").

Three arms, identical cosine machinery (embedding_gate.cos_streams / series_rows: d, dmean, dcus, lag, lagcus, pb_*):
  ctl   the 15 window statistics of embedding_gate.stats, standardised, fixed metric        (from embedding_gate.npz)
  femb  NEW: the same 15 standardised statistics -> MLP 15 -> 64 -> 64 -> 16 (GELU), L2-normalised, trained
        contrastively exactly like the raw-window encoder (InfoNCE, temperature 0.1, batch 512 series, positives = two
        non-overlapping history windows of the same series, fit-set HISTORIES only, no labels). One net per L, 2 seeds
        averaged. It learns WHICH combinations of statistics tell one series from another.
  emb   the raw-window CNN of embedding_gate.py                                              (from embedding_gate.npz)
Why it is informative: femb > ctl -> learning the metric helps; emb > femb -> the raw points carry something the
statistics miss; all equal -> embeddings are not the lever.
(Embedding the existing STREAMING features is not tested: they already compare online with history and the trees
see them with labels; an unsupervised projection of them adds no information.)

Expectation (written before any run): femb about ctl; KF2 and KF3 killed.

KILL CRITERIA (written before any run). Same rows, refs and LightGBM protocol as embedding_gate.py (4 folds x 3 seeds,
base = bo + vol + nb + step + MMD); project bar >= +0.003, > 2 SE, >= 3/4 folds:
  KF1 (interpretation): best single femb column beats best single ctl column by >= +0.003 (direction-free).
  KF2 (decision for femb): base + ctl + femb vs base + ctl meets the bar -> femb goes to a paired cv.py run.
  KF3 (interpretation): base + ctl + femb + emb vs base + ctl + femb meets the bar -> the raw-window net sees more.

Needs the finished main gate (its cv_out/embedding_gate.npz and the MMD cache). CPU only (tiny nets), so it can run
while the GPU is busy. My estimate 30-45 min (48 LightGBM fits; the arm itself about 3-5 min).
    python diagnostics/embedding_feat_gate.py
    python diagnostics/embedding_feat_gate.py --fit 0:400 --eval 2000:2600 --folds 2 --lgb-seeds 1 --steps 300 --emb-npz cv_out/embedding_gate_smoke.npz --out embedding_feat_gate_smoke   # smoke
Out: cv_out/embedding_feat_gate.txt (+ .npz)
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "diagnostics"))
os.chdir(HERE)

import _backcast_common as C   # noqa: E402
import embedding_gate as EG    # noqa: E402

LS = EG.LS
SEEDS = (0, 1)
FDIM = 16
BATCH = 512


def hist_stats(p, L):
    S, ih, _, _ = EG.grid(p, L)
    return EG.stats(EG.windows(p, S[ih], L))


def train_stats_net(Xs, L, seed, steps, say):
    """Xs: per fit series, standardised stats of its history grid windows (stride HSTRIDE)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as Fn
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    k = L // EG.HSTRIDE                                   # grid offset at which windows stop overlapping
    nS = len(Xs)
    perm = rng.permutation(nS)
    va, tr = perm[: max(2, nS // 10)], perm[max(2, nS // 10):]
    lens = np.array([len(x) for x in Xs])
    offs = np.r_[0, np.cumsum(lens)[:-1]]
    X = torch.as_tensor(np.concatenate(Xs))
    net = nn.Sequential(nn.Linear(X.shape[1], 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, FDIM))
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps, pct_start=0.1)

    def batch(ids, r):
        n = lens[ids]
        a = (r.random(len(ids)) * (n - k)).astype(np.int64)
        b = a + k + (r.random(len(ids)) * (n - a - k)).astype(np.int64)
        sw = r.random(len(ids)) < 0.5
        a, b = np.where(sw, b, a), np.where(sw, a, b)
        return X[torch.as_tensor(np.r_[offs[ids] + a, offs[ids] + b])]

    def nce(x):
        z = Fn.normalize(net(x), dim=1)
        B = z.shape[0] // 2
        lg = z[:B] @ z[B:].T / EG.TEMP
        lab = torch.arange(B)
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
                best_state = {kk: t.detach().clone() for kk, t in net.state_dict().items()}
    net.load_state_dict(best_state)
    net.eval()
    say(f"    stats net L{L} seed {seed}: {steps} steps, {time.time() - t0:.0f}s; best valid InfoNCE {best_v[0]:.3f} "
        f"top-1 {best_v[1]:.3f} (chance {1 / min(BATCH, len(va)):.3f})")
    return net


def stats_net_embedder(net, mu, sd, threads):
    import torch
    from joblib import Parallel, delayed

    def embed(Ws):
        out = Parallel(n_jobs=threads, backend="threading")(delayed(EG.stats)(W) for W in Ws)
        res = []
        with torch.no_grad():
            for X in out:
                z = net(torch.as_tensor((X - mu) / sd)).numpy()
                res.append(z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9))
        return res
    return embed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", default="0:2000")
    ap.add_argument("--eval", default="2000:10000")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--lgb-seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=2000, help="stats-net training steps")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--emb-npz", default="cv_out/embedding_gate.npz", help="ctl / emb features of the main gate")
    ap.add_argument("--out", default="embedding_feat_gate")
    args = ap.parse_args()
    import torch
    from joblib import Parallel, delayed

    from sbrt.data import load_train
    from sbrt.metric import ts_auc

    torch.set_num_threads(4)
    say = C.Log(os.path.join(HERE, "cv_out", f"{args.out}.txt"))
    fit = tuple(map(int, args.fit.split(":")))
    ev = tuple(map(int, args.eval.split(":")))
    S_all, tau_all = load_train()
    t0 = time.time()
    say(f"embedding sub-test: learned metric on window statistics | fit {fit[0]}-{fit[1] - 1} | eval {ev[0]}-{ev[1] - 1}"
        f" | L {LS} | seeds {SEEDS} | CPU")
    idx = list(range(fit[0], fit[1])) + list(range(ev[0], ev[1]))
    nfit = fit[1] - fit[0]
    preps = Parallel(n_jobs=args.threads)(delayed(EG.prep)(S_all[i][1], S_all[i][2], k < nfit)
                                          for k, i in enumerate(idx))
    pf, pe = preps[:nfit], preps[nfit:]
    tau_f, tau_e = tau_all[fit[0]:fit[1]], tau_all[ev[0]:ev[1]]
    say(f"series prepared: {time.time() - t0:.0f}s")

    Bo, V, NBF = C.references(S_all, tau_all, fit, ev, args.threads, say)
    MM = EG.mmd_reference(S_all, ev, args.threads, say)
    step, sid, y, in_slice = C.layout(S_all, tau_all, ev)
    z = np.load(os.path.join(HERE, args.emb_npz))
    assert len(z["step"]) == len(step) and np.array_equal(z["step"], step), "emb npz rows do not match this eval set"
    ctl, emb = z["ctl"], z["emb"]
    say(f"rows {len(y):,} | ctl / emb from {args.emb_npz}")

    say("\nfemb arm (contrastive MLP on the 15 window statistics)")
    femb_parts = []
    for L in LS:
        Xs = Parallel(n_jobs=args.threads, backend="threading")(delayed(hist_stats)(p, L) for p in pf)
        R = np.concatenate(Xs)
        mu, sd = R.mean(0), R.std(0) + 1e-6
        Xs = [((x - mu) / sd).astype(np.float32) for x in Xs]
        per = []
        for seed in SEEDS:
            net = train_stats_net(Xs, L, seed, args.steps, say)
            t1 = time.time()
            Fm, _ = EG.arm_features(stats_net_embedder(net, mu, sd, args.threads), pf, tau_f, pe, tau_e, L, say)
            per.append(Fm)
            say(f"      L{L} seed {seed}: features {time.time() - t1:.0f}s")
        femb_parts.append(np.mean(per, 0))
    femb = np.column_stack(femb_parts)

    names = [f"{L}_{c}" for L in LS for c in EG.COLS]
    say("\nsingle-feature TS-AUC (* = flipped)        " + C.HDR)
    best = {}
    for arm, M in (("ctl", ctl), ("femb", femb), ("emb", emb)):
        for c, nm in enumerate(names):
            r = C.auc_row(M[:, c], y, step, in_slice)
            flip = r[0] < 0.5
            if flip:
                r = 1.0 - r
            if arm == "femb":
                say(f"  {arm}{nm:16s}{'*' if flip else ' '}       {C.fmt(r)}")
            if r[0] > best.get(arm, (0, ""))[0]:
                best[arm] = (r[0], arm + nm)
    for arm in best:
        say(f"  best {arm}: {best[arm][1]} {best[arm][0]:.4f}")
    kf1 = best["femb"][0] - best["ctl"][0]

    say(f"\nIncremental: LightGBM {args.folds}-fold by series, {args.lgb_seeds} seeds per fold")
    base = np.column_stack([Bo, V, NBF, step, MM]).astype(np.float32)
    sets = {"base": base,
            "+ctl": np.column_stack([base, ctl]).astype(np.float32),
            "+ctl+femb": np.column_stack([base, ctl, femb]).astype(np.float32),
            "+ctl+femb+emb": np.column_stack([base, ctl, femb, emb]).astype(np.float32)}
    sc = {}
    C.incremental(sets, y, step, sid, args.folds, args.lgb_seeds, args.threads, say, scores_out=sc)

    def paired(a, b):
        d = np.array(sc[a]) - np.array(sc[b])
        return d.mean(), d.std(ddof=1) / np.sqrt(len(d)), int((d > 0).sum())
    r2, r3 = paired("+ctl+femb", "+ctl"), paired("+ctl+femb+emb", "+ctl+femb")
    say(f"  femb over +ctl:        {r2[0]:+.4f} +- {r2[1]:.4f}  ({r2[2]}/{args.folds} folds)")
    say(f"  emb over +ctl+femb:    {r3[0]:+.4f} +- {r3[1]:.4f}  ({r3[2]}/{args.folds} folds)")
    kf2, kf3 = C.passes_project_bar(r2, args.folds), C.passes_project_bar(r3, args.folds)
    np.savez(os.path.join(HERE, "cv_out", f"{args.out}.npz"), femb=femb.astype(np.float32), step=step, y=y, sid=sid)

    say("\nVERDICTS (criteria written in the docstring before any run)")
    if ev[1] - ev[0] < 4000:
        say("  SMOKE RUN (< 4,000 eval series): pipeline check only; do not act on it.")
    b0 = np.mean(sc["base"])
    if b0 < 0.60:
        say(f"  WARNING: base {b0:.4f} is far below nb_logodds alone; the LightGBM instrument is not trustworthy here.")
    say(f"  KF1 learned metric vs fixed (single): {kf1:+.4f} -> {'femb ahead' if kf1 >= 0.003 else 'no gain from learning the metric'}")
    say(f"  KF2 femb over +ctl: {r2[0]:+.4f} +- {r2[1]:.4f} ({r2[2]}/{args.folds}) -> "
        f"{'PASSES (paired cv.py run)' if kf2 else 'killed'}")
    say(f"  KF3 raw-window emb over +ctl+femb: {r3[0]:+.4f} +- {r3[1]:.4f} ({r3[2]}/{args.folds}) -> "
        f"{'raw points add beyond the statistics' if kf3 else 'nothing beyond the statistics'}")
    say(f"\ntotal {time.time() - t0:.0f}s; log {say.path}")


if __name__ == "__main__":
    main()
