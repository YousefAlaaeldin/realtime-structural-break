"""Tune the augmented GRU: every setting on 5 folds x 2 seeds, paired against the matching baseline seed.

Run:    python gru_tune.py --list                         # show the settings and what has finished
        python gru_tune.py                                # run all settings (resumable: finished runs are skipped)
        python gru_tune.py --configs lr1e3,drop3          # only some settings
        python gru_tune.py --group capacity --prefix cap  # bigger GRUs and input-channel weighting (gates)
        python gru_tune.py --fixed "--folds 5 --epochs 25 --feat-top 20 --aug 2 --synth 2 --threads 4 --device cuda" \
                           --base0 cv_out/oof_gru_aug2_synth2.npz --base1 cv_out/oof_gru_aug2_synth2_s1.npz --prefix tunesyn
Output: cv_out/gru_tune_results.csv and a ranked table; per-run logs cv_out/gru_summary_<prefix>_<setting>_s<seed>.txt

Why 2 seeds x 5 folds: a single GRU run moves by about 0.01 per fold with the seed, so one-fold or one-seed
comparisons cannot see gains of a few thousandths. A setting is worth keeping only if BOTH seeds show a paired gain
of at least +0.003 over their baseline seed and the two-seed average (and its blend with the trees) improves.
Each run takes about 1.5-3 minutes on the GPU; the full list is about 30-45 minutes.
"""
import argparse
import csv
import os
import re
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sbrt.metric import ts_auc  # noqa: E402

CONFIGS = {
    "lr1e3": "--lr 0.001",
    "lr5e4": "--lr 0.0005 --patience 6",
    "drop3": "--dropout 0.3",
    "wd1e3": "--wd 0.001",
    "lr1e3_drop3_wd1e3": "--lr 0.001 --dropout 0.3 --wd 0.001",
    "h96_lr1e3": "--hidden 96 --lr 0.001",
    "b128": "--batch 128",
    "lr1e3_ep40_pat6": "--lr 0.001 --epochs 40 --patience 6",
}
# bigger models and channel weighting (--group capacity)
CAPACITY = {
    "l2_h96": "--layers 2 --hidden 96 --dropout 0.2 --lr 0.001",
    "l2_h128": "--layers 2 --hidden 128 --dropout 0.3 --lr 0.001",
    "l3_h96": "--layers 3 --hidden 96 --dropout 0.3 --lr 0.0007 --patience 6",
    "gate_prior": "--gate prior",
    "gate_static": "--gate static",
    "gate_dynamic": "--gate dynamic",
    "gate_static_f40": "--gate static --feat-top 40",
    "l2_h96_gate_static": "--layers 2 --hidden 96 --dropout 0.2 --lr 0.001 --gate static",
}
GROUPS = {"tune": list(CONFIGS), "capacity": list(CAPACITY), "all": list(CONFIGS) + list(CAPACITY)}
CONFIGS.update(CAPACITY)

ap = argparse.ArgumentParser()
ap.add_argument("--configs", type=str, default="", help="comma-separated subset of: " + ",".join(CONFIGS))
ap.add_argument("--group", choices=list(GROUPS), default="tune", help="which list to run when --configs is empty")
ap.add_argument("--fixed", type=str, default="--folds 5 --epochs 25 --feat-top 20 --aug 2 --threads 4 --device cuda")
ap.add_argument("--base0", type=str, default=os.path.join("cv_out", "oof_gru_aug2.npz"), help="baseline, seed 0")
ap.add_argument("--base1", type=str, default=os.path.join("cv_out", "oof_gru_aug2_s1.npz"), help="baseline, seed 1")
ap.add_argument("--blend", type=str, default=os.path.join("cv_out", "oof_aug2_gpu.npz"), help="tree OOF to blend with")
ap.add_argument("--prefix", type=str, default="tune")
ap.add_argument("--list", action="store_true")
args = ap.parse_args()

names = [c for c in args.configs.split(",") if c] or GROUPS[args.group]
bad = [c for c in names if c not in CONFIGS]
if bad:
    sys.exit(f"unknown settings {bad}; choose from {list(CONFIGS)}")
BASE = {0: args.base0, 1: args.base1}


def tag(name, seed):
    return f"{args.prefix}_{name}_s{seed}"


def oof_path(t):
    return os.path.join(HERE, "cv_out", f"oof_gru_{t}.npz")


def parse(t):
    p = os.path.join(HERE, "cv_out", f"gru_summary_{t}.txt")
    if not os.path.exists(p):
        return None, None
    txt = open(p, encoding="utf-8").read()
    h = re.search(r"HEADLINE\s+GRU mean TS-AUC = ([0-9.]+)", txt)
    pr = re.search(r"PAIRED vs .*?: ([+-][0-9.]+) \+- ([0-9.nan]+) .*?, (\d)/(\d) folds better", txt)
    return (float(h.group(1)) if h else None), (pr.groups() if pr else None)


def avg_and_blend(files):
    zs = [np.load(f) for f in files]
    y, s = zs[0]["y"], zs[0]["step"]
    g = sum(z["oof"] / z["oof"].std() for z in zs)
    two = ts_auc(g, y, s)
    best = (float("nan"), None)
    if os.path.exists(args.blend):
        b = np.load(args.blend)["oof"]
        b = b / b.std()
        g = g / g.std()
        best = max((ts_auc((1 - w) * b + w * g, y, s), w) for w in (0.3, 0.4, 0.5, 0.6))
    return two, best


if args.list:
    for nm in names:
        done = [s for s in (0, 1) if os.path.exists(oof_path(tag(nm, s)))]
        print(f"{nm:20s} {CONFIGS[nm]:40s} finished seeds: {done}")
    sys.exit(0)

for b in BASE.values():
    if not os.path.exists(b):
        sys.exit(f"baseline {b} not found: run the baseline GRU for both seeds first (see HANDOFF.md)")

env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE")
for nm in names:
    for seed in (0, 1):
        t = tag(nm, seed)
        if os.path.exists(oof_path(t)):
            print(f"skip {t} (finished)")
            continue
        cmd = ([sys.executable, os.path.join(HERE, "gru_cv.py")] + args.fixed.split() + CONFIGS[nm].split()
               + ["--seed", str(seed), "--compare", BASE[seed], "--tag", t])
        print(f"run  {t}: {' '.join(cmd[2:])}", flush=True)
        r = subprocess.run(cmd, cwd=HERE, env=env, capture_output=True, text=True)
        h, pr = parse(t)
        if not (os.path.exists(oof_path(t)) and h is not None):
            print(f"  FAILED (exit code {r.returncode}):", " | ".join((r.stderr or r.stdout).strip().splitlines()[-3:]))
            continue
        if r.returncode != 0:  # e.g. a crash while the process shuts down, after all results were written
            print(f"  note: exit code {r.returncode} after the run finished and saved its results (ignored)")
        print(f"  headline {h:.4f}   paired {pr[0]} +- {pr[1]} ({pr[2]}/{pr[3]} folds better)", flush=True)

rows = []
two, (bl, bw) = avg_and_blend([BASE[0], BASE[1]])
rows.append(dict(setting="BASELINE", args="", s0="", s1="", paired0="", paired1="", two_seed=f"{two:.4f}",
                 blend=f"{bl:.4f}", blend_w=bw))
for nm in names:
    t0, t1 = tag(nm, 0), tag(nm, 1)
    if not (os.path.exists(oof_path(t0)) and os.path.exists(oof_path(t1))):
        continue
    h0, p0 = parse(t0)
    h1, p1 = parse(t1)
    two, (bl, bw) = avg_and_blend([oof_path(t0), oof_path(t1)])
    rows.append(dict(setting=nm, args=CONFIGS[nm], s0=f"{h0:.4f}", s1=f"{h1:.4f}",
                     paired0=f"{p0[0]} ({p0[2]}/5)" if p0 else "", paired1=f"{p1[0]} ({p1[2]}/5)" if p1 else "",
                     two_seed=f"{two:.4f}", blend=f"{bl:.4f}", blend_w=bw))
out = os.path.join(HERE, "cv_out", "gru_tune_results.csv")
with open(out, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
rows[1:] = sorted(rows[1:], key=lambda r: -float(r["two_seed"]))
print(f"\n{'setting':20s} {'seed0':>7s} {'seed1':>7s} {'paired s0':>14s} {'paired s1':>14s} {'2-seed':>7s} {'+trees':>7s}")
for r in rows:
    print(f"{r['setting']:20s} {r['s0']:>7s} {r['s1']:>7s} {r['paired0']:>14s} {r['paired1']:>14s} "
          f"{r['two_seed']:>7s} {r['blend']:>7s}  (GRU weight {r['blend_w']})")
print(f"\nsaved {out}. Keep a setting only if BOTH paired columns are >= +0.003 and the 2-seed and +trees columns beat "
      f"the BASELINE row.")
