"""Hyper-parameter sweep for gru_cv.py. Runs one training per setting (sequentially, on the GPU) and keeps a ranked table.

Run:   python gru_sweep.py                                  # one-at-a-time sweep of the sensitivity knobs, 1 fold
       python gru_sweep.py --knobs ramp,mem_bias --mode grid     # every combination of those knobs
       python gru_sweep.py --mode random --trials 12             # 12 random settings from the whole search space
       python gru_sweep.py --confirm 2                           # then re-run the 2 best on all 5 folds
       python gru_sweep.py --list                                # just show what would run
Output: cv_out/gru_sweep_results.csv (appended after every run, so it is safe to stop and resume),
        cv_out/gru_summary_<tag>.txt per run (full log of that run)

Modes
  oat     take the base setting and change ONE knob at a time (default). Cheapest way to see which knobs matter.
  grid    Cartesian product of the chosen knobs' values.
  random  random draws from the full search space.
Already-finished settings are skipped on re-run (use --rerun to repeat them).

Reading the table: a single fold has a noise level of roughly 0.005 TS-AUC. Differences smaller than that are not
evidence. Use --confirm to re-run the leaders on all 5 folds before believing a gain.
"""
import argparse
import csv
import hashlib
import itertools
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

BASE = dict(hidden=64, layers=1, dropout=0.1, lr=2e-3, wd=1e-4, batch=64, burn=100,
            ramp=0, pos_boost=1.0, mem_bias=1.0, focal=0.0, tails=0, feat_top=0, seed=0)
NEW_KNOBS = ("tails", "feat_top", "seed")  # added later: left out of the run id while 0, so older results keep their ids
DEFAULT_BASE = dict(BASE)  # labels are always written relative to these defaults
SPACE = dict(
    ramp=[0, 15, 30, 60],
    mem_bias=[0.0, 1.0, 2.0, 3.0, 5.0],
    hidden=[32, 64, 128],
    layers=[1, 2],
    dropout=[0.0, 0.1, 0.3],
    lr=[1e-3, 2e-3, 4e-3],
    wd=[1e-5, 1e-4, 1e-3],
    pos_boost=[0.5, 1.0, 2.0],
    focal=[0.0, 1.0, 2.0],
    burn=[50, 100, 200],
    batch=[32, 64, 128],
    tails=[0, 1],
    feat_top=[0, 20, 40],
    seed=[0, 1, 2, 3, 4],
)
DEFAULT_OAT_KNOBS = "ramp,mem_bias,hidden,dropout,lr,pos_boost,tails,feat_top"
FLAG = {k: "--" + k.replace("_", "-") for k in BASE}

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["oat", "grid", "random"], default="oat")
ap.add_argument("--knobs", type=str, default="", help="comma list of knobs (default: a sensible set for the mode)")
ap.add_argument("--trials", type=int, default=12, help="random mode: number of settings")
ap.add_argument("--max-runs", type=int, default=40, help="safety cap on the number of runs")
ap.add_argument("--folds", type=int, default=1)
ap.add_argument("--epochs", type=int, default=15)
ap.add_argument("--fixed", type=str, default="", help="extra arguments passed to every gru_cv.py run, e.g. '--no-ctx'")
ap.add_argument("--confirm", type=int, default=0, help="afterwards re-run the top N on all 5 folds")
ap.add_argument("--blend", type=str, default="", help="oof file passed to the confirm runs, e.g. cv_out/oof_loc.npz")
ap.add_argument("--results", type=str, default=os.path.join("cv_out", "gru_sweep_results.csv"))
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--rerun", action="store_true")
ap.add_argument("--list", action="store_true")
ap.add_argument("--parallel", type=int, default=1,
                help="run this many settings AT THE SAME TIME on the one GPU (the model is small, so 2-3 usually "
                     "finish sooner than one after another)")
ap.add_argument("--cpu-threads", type=int, default=4, help="CPU threads each run may use (keeps the CPU cool)")
ap.add_argument("--allow-cpu", action="store_true", help="do not insist on a GPU (very slow)")
ap.add_argument("--base", type=str, default="",
                help="change the centre of the sweep, e.g. --base feat_top=20,hidden=128 (one-at-a-time mode then "
                     "varies each knob around THIS setting)")
ap.add_argument("--values", action="append", default=[],
                help="replace a knob's candidate values, e.g. --values feat_top=10/15/30 (repeatable)")
args = ap.parse_args()


def _typed(k, s):
    return int(float(s)) if isinstance(DEFAULT_BASE[k], int) else float(s)


for _kv in [x for x in args.base.split(",") if x]:
    _k, _v = _kv.split("=")
    BASE[_k] = _typed(_k, _v)
for _spec in args.values:
    _k, _vs = _spec.split("=")
    SPACE[_k] = [_typed(_k, x) for x in _vs.split("/")]

COLS = ["tag", "label", "folds", "headline", "std", "b0_100", "b100_300", "b300_600", "b600_plus", "seconds", "status"]


def label_of(cfg):
    diff = [f"{k}={cfg[k]:g}" for k in DEFAULT_BASE if cfg[k] != DEFAULT_BASE[k]]
    return " ".join(diff) if diff else "base"


def tag_of(cfg, folds):
    items = {k: v for k, v in cfg.items() if not (k in NEW_KNOBS and v == 0)}
    key = repr(sorted(items.items())) + f"|{folds}|{args.epochs}|{args.fixed}"
    return "sw_" + hashlib.md5(key.encode()).hexdigest()[:8]


def make_configs():
    rng = np.random.default_rng(args.seed)
    cfgs = []
    if args.mode == "oat":
        knobs = (args.knobs or DEFAULT_OAT_KNOBS).split(",")
        cfgs.append(dict(BASE))
        for k in knobs:
            for v in SPACE[k]:
                if v != BASE[k]:
                    c = dict(BASE)
                    c[k] = v
                    cfgs.append(c)
    elif args.mode == "grid":
        knobs = (args.knobs or "ramp,mem_bias").split(",")
        for combo in itertools.product(*[SPACE[k] for k in knobs]):
            c = dict(BASE)
            c.update(dict(zip(knobs, combo)))
            cfgs.append(c)
    else:
        knobs = (args.knobs or ",".join(SPACE)).split(",")
        seen = set()
        tries = 0
        while len(cfgs) < args.trials and tries < 1000:
            tries += 1
            c = dict(BASE)
            for k in knobs:
                c[k] = SPACE[k][int(rng.integers(len(SPACE[k])))]
            key = repr(sorted(c.items()))
            if key not in seen:
                seen.add(key)
                cfgs.append(c)
    return cfgs[: args.max_runs]


def read_results(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def append_row(path, row):
    new = not os.path.exists(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        if new:
            w.writeheader()
        w.writerow({c: row.get(c, "") for c in COLS})


def run_one(cfg, folds, tag, blend=""):
    cmd = [sys.executable, os.path.join(HERE, "gru_cv.py"), "--folds", str(folds), "--epochs", str(args.epochs),
           "--tag", tag]
    for k, v in cfg.items():
        if k == "tails":  # on/off flag, not a value
            if v:
                cmd.append(FLAG[k])
        else:
            cmd += [FLAG[k], str(v)]
    cmd += ["--device", "auto" if args.allow_cpu else "cuda"]
    if blend:
        cmd += ["--blend", blend]
    if args.fixed:
        cmd += args.fixed.split()
    th = str(args.cpu_threads)
    env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE", OMP_NUM_THREADS=th, MKL_NUM_THREADS=th)
    t0 = time.time()
    p = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True, env=env)
    dt = time.time() - t0
    summ = os.path.join(HERE, "cv_out", f"gru_summary_{tag}.txt")
    text = open(summ, encoding="utf-8").read() if os.path.exists(summ) else ""
    m = re.search(r"HEADLINE\s+GRU mean TS-AUC = ([0-9.]+)\s+\(std over folds ([0-9.]+)", text)
    buckets = re.findall(r"steps \[\s*\d+,\s*\d+\):\s+([0-9.]+)", text)
    if p.returncode != 0 or not m:
        tail = (p.stderr or p.stdout or "").strip().splitlines()[-4:]
        return dict(status="FAILED: " + " | ".join(tail)[:300], seconds=f"{dt:.0f}")
    row = dict(headline=m.group(1), std=m.group(2), seconds=f"{dt:.0f}", status="ok")
    for name, v in zip(["b0_100", "b100_300", "b300_600", "b600_plus"], buckets):
        row[name] = v
    return row


def show(rows, title):
    ok = [r for r in rows if r.get("status") == "ok"]
    ok.sort(key=lambda r: -float(r["headline"]))
    print(f"\n{title}")
    print(f"{'rank':>4s} {'TS-AUC':>7s} {'<100':>7s} {'100-300':>8s} {'300-600':>8s} {'600+':>7s} {'min':>5s}  setting")
    for i, r in enumerate(ok, 1):
        print(f"{i:4d} {float(r['headline']):7.4f} {r.get('b0_100', ''):>7s} {r.get('b100_300', ''):>8s} "
              f"{r.get('b300_600', ''):>8s} {r.get('b600_plus', ''):>7s} {float(r['seconds']) / 60:5.1f}  "
              f"{r['label']} (folds={r['folds']})")
    bad = [r for r in rows if r.get("status") != "ok"]
    for r in bad:
        print(f"  FAILED  {r['label']}: {r['status']}")
    return ok


def check_gpu():
    """Refuse to start a multi-hour sweep on the CPU by accident."""
    if args.allow_cpu:
        return
    env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE")
    r = subprocess.run([sys.executable, "-c", "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 3)"],
                       capture_output=True, text=True, env=env)
    if r.returncode != 0:
        sys.exit("No CUDA-capable PyTorch found, so the sweep would run on the CPU (hours). Install the GPU build:\n"
                 "  pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130\n"
                 "or pass --allow-cpu to run anyway.")
    print("GPU check passed (torch.cuda.is_available() == True)")


def main():
    cfgs = make_configs()
    done = {r["tag"]: r for r in read_results(args.results)}
    todo = [(c, tag_of(c, args.folds)) for c in cfgs]
    print(f"mode {args.mode}: {len(todo)} setting(s), {args.folds} fold(s), {args.epochs} epochs each, "
          f"{args.parallel} at a time.")
    if args.list:
        for c, t in todo:
            print(f"  {t}  {label_of(c)}" + ("   [already done]" if t in done and not args.rerun else ""))
        return
    check_gpu()
    t_start = time.time()
    lock = threading.Lock()

    def work(n, c, t):
        with lock:
            print(f"[{n}/{len(todo)}] running: {label_of(c)} ...", flush=True)
        row = run_one(c, args.folds, t)
        row.update(tag=t, label=label_of(c), folds=args.folds)
        with lock:
            append_row(args.results, row)
            if row["status"] == "ok":
                print(f"[{n}/{len(todo)}] done: {label_of(c)}  TS-AUC {row['headline']}  "
                      f"({float(row['seconds']) / 60:.1f} min)", flush=True)
            else:
                print(f"[{n}/{len(todo)}] FAILED: {label_of(c)}  {row['status']}", flush=True)
        return row

    pending = []
    for n, (c, t) in enumerate(todo, 1):
        if t in done and not args.rerun and done[t].get("status") == "ok":
            print(f"[{n}/{len(todo)}] skip (done): {label_of(c)}")
        else:
            pending.append((n, c, t))
    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as ex:
        list(ex.map(lambda a: work(*a), pending))

    allrows = [r for r in read_results(args.results) if r["tag"] in {t for _, t in todo}]
    ok = show(allrows, f"RESULTS (this sweep, {args.folds} fold(s); noise per fold ~0.005 -> differences below that are not evidence)")
    if ok:
        base_rows = [r for r in ok if r["label"] == label_of(BASE)]
        if base_rows:
            b = float(base_rows[0]["headline"])
            best = float(ok[0]["headline"])
            print(f"\ncentre ({label_of(BASE)}) = {b:.4f}; best = {best:.4f} ({best - b:+.4f}). "
                  + ("Gain is within single-fold noise." if best - b < 0.005 else "Worth confirming on 5 folds (--confirm)."))

    if args.confirm and ok:
        print(f"\nConfirming the top {args.confirm} on all 5 folds ...")
        conf_rows = []
        by_label = {r["label"]: r for r in ok}
        cfg_by_label = {label_of(c): c for c in cfgs}
        for r in ok[: args.confirm]:
            c = cfg_by_label[r["label"]]
            t = tag_of(c, 5) + "_cv5"
            print(f"  5-fold run: {r['label']} ...", flush=True)
            row = run_one(c, 5, t, args.blend)
            row.update(tag=t, label=r["label"], folds=5)
            append_row(args.results, row)
            conf_rows.append(row)
        show(conf_rows, "CONFIRMATION (5 folds; compare with LightGBM 0.6155 +- 0.0066)")
        if args.blend:
            print(f"(blend tables with {args.blend} are in cv_out/gru_summary_<tag>.txt of each confirmation run)")
    print(f"\nTotal time {(time.time() - t_start) / 60:.0f} min. All results: {args.results}")


if __name__ == "__main__":
    main()
