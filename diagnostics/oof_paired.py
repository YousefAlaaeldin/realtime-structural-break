"""Paired per-fold comparison of two out-of-fold score sets (trees only, no blend), e.g. a seed average vs one seed.

Each side may list several OOF files; they are combined like one blend.py member (sum of score / its std), so
`--a s0 s1 --ref s0` is "average of two tree seeds vs seed 0". Folds are the CV folds (StratifiedKFold(5, shuffle,
random_state=0) over series, stratified by break / no break, exactly as cv.py and blend.py).

    python diagnostics/oof_paired.py --a cv_out/oof_X.npz cv_out/oof_X_s1.npz --ref cv_out/oof_X.npz
"""
import argparse
import os
import sys

import numpy as np
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sbrt.data import load_train   # noqa: E402
from sbrt.metric import ts_auc     # noqa: E402

BUCKETS = ((0, 100), (100, 300), (300, 600), (600, 10000))


def load_group(files):
    zs = [np.load(f) for f in files]
    for z in zs[1:]:
        if not (np.array_equal(z["sid"], zs[0]["sid"]) and np.array_equal(z["step"], zs[0]["step"])):
            sys.exit("files have different row layouts")
    return sum(z["oof"] / z["oof"].std() for z in zs), zs[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", nargs="+", required=True)
    ap.add_argument("--ref", nargs="+", required=True)
    args = ap.parse_args()
    A, za = load_group(args.a)
    R, zr = load_group(args.ref)
    if not (np.array_equal(za["sid"], zr["sid"]) and np.array_equal(za["step"], zr["step"])):
        sys.exit("--a and --ref have different row layouts")
    y, step, sid = za["y"], za["step"], za["sid"]
    tau = load_train()[1]
    d = []
    for f, (_, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(np.arange(len(tau)),
                                                                                    (tau >= 0).astype(int))):
        m = np.isin(sid, va)
        a, r = ts_auc(A[m], y[m], step[m]), ts_auc(R[m], y[m], step[m])
        d.append(a - r)
        print(f"fold {f}: {r:.4f} -> {a:.4f}   paired {a - r:+.4f}")
    d = np.array(d)
    print(f"pooled TS-AUC: ref {ts_auc(R, y, step):.4f}  ->  a {ts_auc(A, y, step):.4f}")
    print(f"PAIRED (mean of per-fold differences): {d.mean():+.4f} +- {d.std(ddof=1) / np.sqrt(len(d)):.4f}, "
          f"{int((d > 0).sum())}/{len(d)} folds better")
    for lo, hi in BUCKETS:
        mk = (step >= lo) & (step < hi)
        print(f"  steps [{lo:4d}, {hi:5d}): ref {ts_auc(R[mk], y[mk], step[mk]):.4f}  a {ts_auc(A[mk], y[mk], step[mk]):.4f}")


if __name__ == "__main__":
    main()
