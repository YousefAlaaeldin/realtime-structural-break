"""Inference throughput with several worker processes at once, as in the cloud (HANDOFF 48).

The first cloud run (blend-v3-trees2, INFER_PARALLELISM 4, 16 vCPUs) took over 2 h of inference for ~5M online points,
against ~14 min expected from the single-process timing. Suspect: numpy's BLAS starts one thread per CPU in EVERY
worker process and those threads busy-wait between the per-point SVM products, starving the workers. infer() now
limits BLAS to one thread per worker (main._limit_blas) and INFER_PARALLELISM is 12.

This script runs P worker processes at once, each streaming its own training series through main.SeriesScorer (the
exact per-point path of infer()), with BLAS at its default thread count or limited to one thread, pinned to the first
C logical CPUs to mimic the cloud's 16 vCPUs. Workers compile (numba) and warm up first, then start together.
Locally numpy uses MKL (Intel OpenMP threads, which also spin after each call); the cloud uses OpenBLAS.

Run:  python diagnostics/infer_speed.py                         P = 1, 4, 8, 12 x BLAS default / 1, 16 CPUs (~5 min)
      python diagnostics/infer_speed.py --workers 4,12,16 --per-worker 20 --cpus 0      (0 = no pinning)
Read: points/s per arm and the projected time for the cloud's 5.04M points (on this machine's cores; cloud vCPUs are
slower). Expected if the diagnosis is right: BLAS-default arms get slower per worker as P grows, BLAS-1 arms scale.
Output: cv_out/infer_speed.txt
"""
import argparse
import multiprocessing as mp
import os
import sys
import time

here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, here)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

CLOUD_POINTS = 5_036_517        # online points in X_train (X_test.parquet has about the same size)


def worker(rank, items, blas1, cpus, barrier, q):
    if cpus:
        import psutil
        psutil.Process().cpu_affinity(list(range(cpus)))
    import threadpoolctl
    import main
    lim = threadpoolctl.threadpool_limits(limits=1) if blas1 else None
    M = main.load_models(os.path.join(here, "resources"))
    h, o = items[0]
    sc = main.SeriesScorer(h, M)
    for x in o[:50]:                                    # numba compilation + warm-up, not timed
        sc.push(float(x))
    barrier.wait()
    t0, n = time.perf_counter(), 0
    for h, o in items:
        sc = main.SeriesScorer(h, M)
        for x in o:
            sc.push(float(x))
        n += len(o)
    q.put((rank, n, time.perf_counter() - t0))
    if lim is not None:
        lim.restore_original_limits()


def run_arm(series, P, blas1, cpus, per):
    ctx = mp.get_context("spawn")
    barrier, q = ctx.Barrier(P), ctx.Queue()
    procs = [ctx.Process(target=worker, args=(r, series[r * per:(r + 1) * per], blas1, cpus, barrier, q))
             for r in range(P)]
    for p in procs:
        p.start()
    res = [q.get() for _ in range(P)]
    for p in procs:
        p.join()
    pts = sum(n for _, n, _ in res)
    wall = max(s for _, _, s in res)
    per_us = sorted(s / n * 1e6 for _, n, s in res)
    return pts, wall, per_us


def main_():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", default="1,4,8,12")
    ap.add_argument("--blas", default="default,1", help="arms: default and/or 1 (BLAS threads)")
    ap.add_argument("--per-worker", type=int, default=12, help="training series per worker")
    ap.add_argument("--cpus", type=int, default=16, help="pin workers to the first N logical CPUs (0 = no pinning)")
    ap.add_argument("--out", default="infer_speed.txt", help="file name in cv_out/")
    args = ap.parse_args()
    from sbrt.data import load_train
    S, _ = load_train()
    Ps = [int(v) for v in args.workers.split(",")]
    series = [(h, o) for _, h, o in S[:max(Ps) * args.per_worker]]
    lines = [f"inference throughput, {args.per_worker} training series per worker, pinned to "
             f"{args.cpus or os.cpu_count()} of {os.cpu_count()} logical CPUs",
             f"{'workers':>7} {'BLAS':>8} {'points':>8} {'wall s':>7} {'points/s':>9} {'us/pt median':>12} "
             f"{'us/pt worst':>11} {'5.04M pts':>10}"]
    print(lines[0], "\n" + lines[1], flush=True)
    for P in Ps:
        for b in args.blas.split(","):
            pts, wall, per_us = run_arm(series, P, b == "1", args.cpus, args.per_worker)
            row = (f"{P:>7} {b:>8} {pts:>8,} {wall:>7.1f} {pts / wall:>9,.0f} {per_us[len(per_us) // 2]:>12.0f} "
                   f"{per_us[-1]:>11.0f} {CLOUD_POINTS / (pts / wall) / 60:>8.1f} m")
            lines.append(row)
            print(row, flush=True)
    import numpy as np
    from threadpoolctl import threadpool_info
    blas = sorted({(d["internal_api"], d["num_threads"]) for d in threadpool_info()})
    lines.insert(1, f"numpy {np.__version__}, BLAS/OpenMP libraries {blas}")
    out = os.path.join(here, "cv_out", args.out)
    open(out, "w").write("\n".join(lines) + "\n")
    print("saved", out)


if __name__ == "__main__":
    main_()
