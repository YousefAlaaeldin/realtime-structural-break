"""Hyperparameter sweep for the NBM pretraining (sbrt/nbm.py), scored on what matters downstream.

Every configuration is pretrained on the histories of series 0-4499 (validation 4500-4999, as the gate model) and
scored on two things:
  1. valid NLL (nats per point): how well it predicts normal behaviour -- the pretraining objective;
  2. the gate: single-feature TS-AUC of nb_logodds on held-out series 5000-6999 (prior fitted on 0-4999's breaks),
     by step range and on breaks with unchanged variance. THIS is the selection criterion: a better normal-behaviour
     model is not automatically a better break detector (e.g. one that adapts faster to a break loses signal).
Seed repeats of the base configuration (base_s1, base_s2) measure the noise floor: treat a difference smaller than
the base seeds' spread as no difference. The winner must still pass the paired 5-fold CV (cv.py --nbm); the sweep
selects on series 5000-6999, so its own numbers are slightly optimistic.

Evaluation runs on the GPU in float32 (agrees with the numba runtime to about 2e-7 in the log scale).

    python diagnostics/nbm_sweep.py --list               # show the configurations
    python diagnostics/nbm_sweep.py                      # run all not yet in diagnostics/nbm_sweep.csv (resumable)
    python diagnostics/nbm_sweep.py --only h128,lr4e-3   # run some
    python diagnostics/nbm_sweep.py --save h128          # re-train one config and save it as data/nbm_v1_<name>.pt
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sbrt import bocpd as BO      # noqa: E402
from sbrt import evidence as EV   # noqa: E402
from sbrt import nbm as NB        # noqa: E402

BASE = dict(hidden=64, lr=2e-3, epochs=12, batch=48, wd=1e-4, loss_from=50, seed=0)
CONFIGS = {   # name -> changes from BASE
    "base": {},
    "base_s1": dict(seed=1),
    "base_s2": dict(seed=2),
    # capacity
    "h32": dict(hidden=32),
    "h128": dict(hidden=128),
    "h192": dict(hidden=192),
    # optimisation
    "lr1e-3": dict(lr=1e-3),
    "lr4e-3": dict(lr=4e-3),
    "ep25": dict(epochs=25),
    "ep40": dict(epochs=40),
    "bs24": dict(batch=24),
    "bs96": dict(batch=96),
    "wd1e-2": dict(wd=1e-2),
    # loss only after a longer warm-up of the hidden state
    "lf200": dict(loss_from=200),
    # combinations worth trying if capacity or longer training helps
    "h128_ep25": dict(hidden=128, epochs=25),
    "h128_ep25_lr1e-3": dict(hidden=128, epochs=25, lr=1e-3),
    # round 2 (2026-10-15), after round 1: smaller batch and higher lr both helped at width 64 (monotone: bs96 < 48 < 24,
    # lr 1e-3 < 2e-3 < 4e-3), and width 128 / 25 epochs with lr 1e-3 was best overall. Questions: does the trend
    # continue, do the two optimisation changes stack at the cheap width, and do the leaders repeat on another seed?
    "bs12": dict(batch=12),
    "lr8e-3": dict(lr=8e-3),
    "bs24_lr4e-3": dict(batch=24, lr=4e-3),
    "bs24_ep25": dict(batch=24, epochs=25),
    "bs24_s1": dict(batch=24, seed=1),
    "h128_bs24": dict(hidden=128, batch=24),
    "h128_ep25_lr1e-3_s1": dict(hidden=128, epochs=25, lr=1e-3, seed=1),
}
OUT_CSV = os.path.join("diagnostics", "nbm_sweep.csv")
FIELDS = ["name", "hidden", "lr", "epochs", "batch", "wd", "loss_from", "seed", "params", "train_s", "valid_nll",
          "nb_logodds", "lo_0_100", "lo_100_300", "lo_300_700", "lo_700p", "lo_varsame", "nb_lvr", "nb_lsd"]


def gpu_sums(net, built, torch, dev, batch=64):
    """Same stream as NbmStream (whitened prediction errors, log-scale drift) in float32 on the GPU, as sums."""
    out = [None] * len(built)
    order = np.argsort([len(b[1]) for b in built])
    with torch.no_grad():
        for k in range(0, len(order), batch):
            b = order[k:k + batch]
            x = NB._pad([built[i][0] for i in b], dev, torch)
            m, ls = net(x)
            m, ls = m.cpu().numpy().astype(np.float64), ls.cpu().numpy().astype(np.float64)
            del x
            if dev.type == "cuda":
                torch.cuda.empty_cache()
            for r, i in enumerate(b):
                X, e, nh = built[i]
                mp = np.r_[0.0, m[r, :len(e) - 1]]
                sp = np.r_[0.0, ls[r, :len(e) - 1]]
                v = (e - mp) * np.exp(-sp)
                ref = np.sort(v[NB.LOSS_FROM + 1:nh])
                w = np.clip(np.array([EV._whiten(float(q), ref) for q in v[nh:]]), -8, 8)
                d = sp[nh:] - sp[nh - 1]
                out[i] = (np.r_[0.0, np.cumsum(w)][None], np.r_[0.0, np.cumsum(w * w)][None], np.r_[0.0, np.cumsum(d)])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=str, default="")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--save", type=str, default="")
    args = ap.parse_args()
    if args.list:
        for k, v in CONFIGS.items():
            print(f"  {k:18s} {dict(BASE, **v)}")
        return
    import torch
    from sbrt.data import load_train
    from sbrt.metric import ts_auc

    S_all, tau_all = load_train()
    if args.save:
        cfg = dict(BASE, **CONFIGS[args.save])
        state = NB.train_model([s[1] for s in S_all[:4500]], [s[1] for s in S_all[4500:5000]],
                               epochs=cfg["epochs"], batch=cfg["batch"], lr=cfg["lr"], seed=cfg["seed"],
                               hidden=cfg["hidden"], wd=cfg["wd"], loss_from=cfg["loss_from"])
        path = os.path.join("data", f"nbm_v{NB.VERSION}_{args.save}.pt")
        torch.save(state, path)
        print(f"saved {path}")
        return

    done = set()
    if os.path.exists(OUT_CSV):
        with open(OUT_CSV, newline="") as fh:
            done = {r["name"] for r in csv.DictReader(fh)}
    names = [n for n in (args.only.split(",") if args.only else CONFIGS) if n]
    bad = [n for n in names if n not in CONFIGS]
    if bad:
        sys.exit(f"unknown configs {bad}; see --list")
    todo = [n for n in names if n not in done]
    if not todo:
        print(f"nothing to do: all requested configs are in {OUT_CSV}")
        return
    print(f"configs to run: {todo} (already done: {sorted(done & set(names))})")

    # model-independent preparation, once
    t0 = time.time()
    tr = NB.prep_histories([s[1] for s in S_all[:4500]])
    va = NB.prep_histories([s[1] for s in S_all[4500:5000]])
    fit_built = [NB.build_seq(s[1], s[2]) for s in S_all[:5000]]
    S, tau = S_all[5000:7000], tau_all[5000:7000]
    ev_built = [NB.build_seq(s[1], s[2]) for s in S]
    step = np.concatenate([np.arange(0, len(s[2]), 3) for s in S])
    sid = np.repeat(np.arange(len(S)), [(len(s[2]) + 2) // 3 for s in S])
    y = ((tau[sid] >= 0) & (step >= tau[sid])).astype(np.int8)
    vr = np.full(len(S), np.nan)
    for i, (s, t) in enumerate(zip(S, tau)):
        if t >= 0 and len(s[2]) - t >= 30:
            vr[i] = BO.whitened_online(s[1], s[2])[0][t:t + 300].var()
    in_slice = (tau[sid] < 0) | ((vr > 0.85) & (vr < 1.15))[sid]
    print(f"prepared inputs in {time.time() - t0:.0f}s")

    def auc(x, lo=0, hi=10 ** 6, mask=None):
        mk = (step >= lo) & (step < hi)
        if mask is not None:
            mk &= mask
        return ts_auc(x[mk], y[mk], step[mk])

    torch_, NBM_ = NB._torch()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    new_file = not os.path.exists(OUT_CSV)
    for name in todo:
        cfg = dict(BASE, **CONFIGS[name])
        t1 = time.time()
        state, vnll = NB.train_model(tr, va, epochs=cfg["epochs"], batch=cfg["batch"], lr=cfg["lr"], seed=cfg["seed"],
                                     verbose=False, hidden=cfg["hidden"], wd=cfg["wd"], loss_from=cfg["loss_from"],
                                     return_nll=True, prepared=True)
        t_train = time.time() - t1
        net = NBM_(cfg["hidden"]).to(dev)
        net.load_state_dict(state)
        net.eval()
        params = sum(p.numel() for p in net.parameters())
        sf = gpu_sums(net, fit_built, torch, dev)
        prior, _ = NB.fit_prior_w([np.diff(q[0][0]) for q in sf], tau_all[:5000])
        se = gpu_sums(net, ev_built, torch, dev)
        Fm = np.concatenate([NB.features_from_sums(q, prior, 3) for q in se])
        del net, sf, se
        r = dict(name=name, **cfg, params=params, train_s=round(t_train), valid_nll=round(vnll, 5),
                 nb_logodds=round(auc(Fm[:, 0]), 4), lo_0_100=round(auc(Fm[:, 0], 0, 100), 4),
                 lo_100_300=round(auc(Fm[:, 0], 100, 300), 4), lo_300_700=round(auc(Fm[:, 0], 300, 700), 4),
                 lo_700p=round(auc(Fm[:, 0], 700), 4), lo_varsame=round(auc(Fm[:, 0], mask=in_slice), 4),
                 nb_lvr=round(auc(Fm[:, 2]), 4), nb_lsd=round(auc(Fm[:, 4]), 4))
        with open(OUT_CSV, "a", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=FIELDS)
            if new_file:
                wr.writeheader()
                new_file = False
            wr.writerow(r)
        print(f"{name:18s} params {params:7,d}  train {t_train:4.0f}s  valid NLL {vnll:.4f}  nb_logodds {r['nb_logodds']:.4f}"
              f"  (100-300 {r['lo_100_300']:.4f}, 300-700 {r['lo_300_700']:.4f}, var-same {r['lo_varsame']:.4f})"
              f"  | {time.time() - t1:.0f}s total", flush=True)

    # summary, best first, with the seed noise floor
    with open(OUT_CSV, newline="") as fh:
        rows = list(csv.DictReader(fh))
    base = [float(r["nb_logodds"]) for r in rows if r["name"].startswith("base")]
    print("\nall configs so far, by nb_logodds (the selection criterion):")
    for r in sorted(rows, key=lambda r: -float(r["nb_logodds"])):
        print(f"  {r['name']:18s} nb_logodds {float(r['nb_logodds']):.4f}  valid NLL {float(r['valid_nll']):.4f}  "
              f"var-same {float(r['lo_varsame']):.4f}  params {int(r['params']):,}")
    if len(base) > 1:
        print(f"\nnoise floor: base seeds nb_logodds {min(base):.4f}-{max(base):.4f} (spread {max(base) - min(base):.4f}); "
              f"a config must beat the base mean {np.mean(base):.4f} by clearly more than that spread to count.")


if __name__ == "__main__":
    main()
