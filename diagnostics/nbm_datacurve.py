"""NBM data learning curve (HANDOFF section 15, 2026-10-17): is the normal-behaviour model data-limited?

If it is, pretraining it on more break-free data (the 2025 edition's break-free stretches, the break-free online parts
of the training series) is new information for the strongest feature family. Unsupervised pretraining uses no labels,
so the failure of the labelled 2025 runs ("2025-looking series break less often") cannot recur. If the curve is flat,
the idea is closed without touching the 2025 data.

Setup (diagnostics/nbm_sweep.py's gate, production NBM settings bs24_ep25: width 64, batch 24, lr 2e-3, wd 1e-4, loss
from position 50):
  pool       the histories of series 0-4499 and 7000-9999: 7,500, about one CV fold's pretraining set, so the slope is
             measured where the production NBM operates; validation (early stopping) 4500-4999 as in the sweep
  arms       100% / 50% / 25% of the pool (whole series; a different random subset per seed) x seeds 0 / 1 / 2
  updates    matched: epochs = 25 / fraction, so every arm takes the same number of optimiser steps (within 1%; OneCycle
             over all of them). Otherwise the small arms would lose to under-training, not to less data
  gate       the sweep's: single-feature TS-AUC of nb_logodds on held-out series 5000-6999 (prior fitted on the breaks of
             0-4999); also nb_lvr (the top tree feature), valid NLL and the learned Student-t nu
The sweep's seed noise: one seed vs another of the same setting differs by 0.004-0.006 in nb_logodds.

DECISION RULE (written 2026-10-17 before any run): DATA-LIMITED only if the 100% arm's 3-seed mean nb_logodds beats the
50% arm's by >= 0.006 AND by > 2 SE (SE from the two arms' seed spreads); otherwise CLOSED (more pretraining data will
not help). DATA-LIMITED is not adoption: it opens the build of NBM pretraining on fold histories + 2025 break-free
stretches + the fold's break-free online parts in cv.py, then the paired trees run and the blend (HANDOFF section 5).

    python diagnostics/nbm_datacurve.py            # the 9 runs not yet in diagnostics/nbm_datacurve.csv (resumable)
    python diagnostics/nbm_datacurve.py --smoke    # tiny end-to-end check (writes nothing)
"""
import argparse
import csv
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
os.chdir(os.path.dirname(HERE))

import numpy as np  # noqa: E402

from sbrt import bocpd as BO  # noqa: E402
from sbrt import nbm as NB  # noqa: E402
from nbm_sweep import gpu_sums  # noqa: E402

CFG = dict(hidden=64, lr=2e-3, epochs=25, batch=24, wd=1e-4, loss_from=50)
FRACS = (1.0, 0.5, 0.25)
SEEDS = (0, 1, 2)
OUT_CSV = os.path.join("diagnostics", "nbm_datacurve.csv")
FIELDS = ["name", "frac", "seed", "n_hist", "train_points", "epochs", "steps", "train_s", "valid_nll", "nu",
          "nb_logodds", "lo_0_100", "lo_100_300", "lo_300_700", "lo_700p", "lo_varsame", "nb_lvr", "nb_lsd"]


def summary(rows):
    by = {}
    for r in rows:
        by.setdefault(float(r["frac"]), []).append(r)
    print("\narm      seeds  nb_logodds mean +- sd    nb_lvr mean     valid NLL      nu")
    stats = {}
    for f in sorted(by, reverse=True):
        x = np.array([float(r["nb_logodds"]) for r in by[f]])
        sd = x.std(ddof=1) if len(x) > 1 else float("nan")
        stats[f] = (x.mean(), sd, len(x))
        print(f"{100 * f:4.0f}%    {len(x):3d}    {x.mean():.4f} +- {sd:.4f}       "
              f"{np.mean([float(r['nb_lvr']) for r in by[f]]):.4f}        "
              f"{np.mean([float(r['valid_nll']) for r in by[f]]):.5f}    {np.mean([float(r['nu']) for r in by[f]]):.2f}")
    for lo in (0.5, 0.25):
        if 1.0 in stats and lo in stats:
            (ma, sa, na), (mb, sb, nb) = stats[1.0], stats[lo]
            se = np.sqrt(sa ** 2 / na + sb ** 2 / nb) if na > 1 and nb > 1 else float("nan")
            print(f"100% minus {100 * lo:.0f}%: {ma - mb:+.4f} +- {se:.4f} (SE)")
    if all(f in stats and stats[f][2] >= 3 for f in (1.0, 0.5)):
        (ma, sa, na), (mb, sb, nb) = stats[1.0], stats[0.5]
        d, se = ma - mb, np.sqrt(sa ** 2 / na + sb ** 2 / nb)
        ok = d >= 0.006 and d > 2 * se
        print(f"\nDECISION (rule written before the run: 100% - 50% >= 0.006 and > 2 SE): "
              + ("DATA-LIMITED -> build NBM pretraining with more break-free data (HANDOFF section 15)" if ok
                 else "CLOSED (more pretraining data will not help the NBM)"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end check; writes nothing")
    args = ap.parse_args()
    import torch
    from sbrt.data import load_train
    from sbrt.metric import ts_auc

    S_all, tau_all = load_train()
    if args.smoke:
        pool_ids, va_ids = np.r_[0:150, 7000:7150], np.arange(4500, 4560)
        fit_ids, ev_ids, ep_base = np.arange(0, 400), np.arange(5000, 5200), 2
    else:
        pool_ids, va_ids = np.r_[0:4500, 7000:10000], np.arange(4500, 5000)
        fit_ids, ev_ids, ep_base = np.arange(0, 5000), np.arange(5000, 7000), CFG["epochs"]

    done = set()
    if not args.smoke and os.path.exists(OUT_CSV):
        with open(OUT_CSV, newline="") as fh:
            done = {r["name"] for r in csv.DictReader(fh)}
    todo = [(f, s) for f in FRACS for s in SEEDS if f"lc{round(100 * f)}_s{s}" not in done]
    print(f"runs to do: {[f'lc{round(100 * f)}_s{s}' for f, s in todo]} (done: {sorted(done)})")

    t0 = time.time()
    pool = NB.prep_histories([S_all[i][1] for i in pool_ids])
    va = NB.prep_histories([S_all[i][1] for i in va_ids])
    fit_built = [NB.build_seq(S_all[i][1], S_all[i][2]) for i in fit_ids]
    S, tau = [S_all[i] for i in ev_ids], tau_all[ev_ids]
    ev_built = [NB.build_seq(s[1], s[2]) for s in S]
    step = np.concatenate([np.arange(0, len(s[2]), 3) for s in S])
    sid = np.repeat(np.arange(len(S)), [(len(s[2]) + 2) // 3 for s in S])
    y = ((tau[sid] >= 0) & (step >= tau[sid])).astype(np.int8)
    vr = np.full(len(S), np.nan)
    for i, (s, t) in enumerate(zip(S, tau)):
        if t >= 0 and len(s[2]) - t >= 30:
            vr[i] = BO.whitened_online(s[1], s[2])[0][t:t + 300].var()
    in_slice = (tau[sid] < 0) | ((vr > 0.85) & (vr < 1.15))[sid]
    print(f"prepared inputs in {time.time() - t0:.0f}s: pool {len(pool):,} histories "
          f"({sum(len(p[1]) for p in pool):,} points), validation {len(va)}, gate {len(S)} series")

    def auc(x, lo=0, hi=10 ** 6, mask=None):
        mk = (step >= lo) & (step < hi)
        if mask is not None:
            mk &= mask
        return ts_auc(x[mk], y[mk], step[mk])

    torch_, NBM_ = NB._torch()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    if not args.smoke and os.path.exists(OUT_CSV):
        with open(OUT_CSV, newline="") as fh:
            rows = list(csv.DictReader(fh))
    new_file = not os.path.exists(OUT_CSV)
    for frac, seed in todo:
        name = f"lc{round(100 * frac)}_s{seed}"
        if frac < 1.0:
            rng = np.random.default_rng(1000 + seed)
            sub = np.sort(rng.choice(len(pool), int(round(frac * len(pool))), replace=False))
        else:
            sub = np.arange(len(pool))
        tr = [pool[j] for j in sub]
        epochs = int(round(ep_base / frac))
        steps = epochs * ((len(tr) + CFG["batch"] - 1) // CFG["batch"])
        t1 = time.time()
        state, vnll = NB.train_model(tr, va, epochs=epochs, batch=CFG["batch"], lr=CFG["lr"], seed=seed, verbose=False,
                                     hidden=CFG["hidden"], wd=CFG["wd"], loss_from=CFG["loss_from"], return_nll=True,
                                     prepared=True)
        t_train = time.time() - t1
        net = NBM_(CFG["hidden"]).to(dev)
        net.load_state_dict(state)
        net.eval()
        nu = float(net.nu().detach())
        sf = gpu_sums(net, fit_built, torch, dev)
        prior, _ = NB.fit_prior_w([np.diff(q[0][0]) for q in sf], tau_all[fit_ids])
        se = gpu_sums(net, ev_built, torch, dev)
        Fm = np.concatenate([NB.features_from_sums(q, prior, 3) for q in se])
        del net, sf, se
        r = dict(name=name, frac=frac, seed=seed, n_hist=len(tr), train_points=sum(len(p[1]) for p in tr),
                 epochs=epochs, steps=steps, train_s=round(t_train), valid_nll=round(vnll, 5), nu=round(nu, 3),
                 nb_logodds=round(auc(Fm[:, 0]), 4), lo_0_100=round(auc(Fm[:, 0], 0, 100), 4),
                 lo_100_300=round(auc(Fm[:, 0], 100, 300), 4), lo_300_700=round(auc(Fm[:, 0], 300, 700), 4),
                 lo_700p=round(auc(Fm[:, 0], 700), 4), lo_varsame=round(auc(Fm[:, 0], mask=in_slice), 4),
                 nb_lvr=round(auc(Fm[:, 2]), 4), nb_lsd=round(auc(Fm[:, 4]), 4))
        rows.append(r)
        if not args.smoke:
            with open(OUT_CSV, "a", newline="") as fh:
                wr = csv.DictWriter(fh, fieldnames=FIELDS)
                if new_file:
                    wr.writeheader()
                    new_file = False
                wr.writerow(r)
        print(f"{name:9s} {len(tr):5d} histories  {epochs:3d} epochs ({steps:,} steps)  train {t_train:4.0f}s  "
              f"valid NLL {vnll:.5f}  nu {nu:.2f}  nb_logodds {r['nb_logodds']:.4f}  nb_lvr {r['nb_lvr']:.4f}  "
              f"(300-700 {r['lo_300_700']:.4f}, var-same {r['lo_varsame']:.4f})  | {time.time() - t1:.0f}s total",
              flush=True)
    summary(rows)
    if args.smoke:
        print("(smoke run: numbers meaningless; nothing written)")


if __name__ == "__main__":
    main()
