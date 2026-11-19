"""Logistic regression on the same per-step streaming features as cv.py -- a linear-boundary baseline.

Run:    python logreg_cv.py                                  # 5 folds, 2 penalty strengths (~15-25 min)
        python logreg_cv.py --folds 2 --sub 4                # quick look (~3 min)
        python logreg_cv.py --blend cv_out/oof_fine.npz      # also test a blend with a LightGBM run from cv.py
Needs:  the feature cache that cv.py builds (data/feat_<N>.npz). Run cv.py once first (any settings).
Output: cv_out/logreg_summary.txt, cv_out/oof_logreg.npz

What it answers
  1. How far does a LINEAR decision boundary get on these features? (LightGBM reference: ~0.6155)
  2. Does it make DIFFERENT mistakes from the boosted trees, i.e. does blending the two help? (--blend)

Preprocessing: heavy-tailed features are compressed with sign(x)*log1p(|x|), standardised on the training
fold, clipped to +-5. Rows are weighted like the metric (same per-step class balancing as cv.py).
The folds are identical to cv.py's, so out-of-fold scores from the two scripts line up row by row.
"""
import argparse
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from sbrt.data import load_train
from sbrt import features as F
from sbrt.metric import ts_auc

ap = argparse.ArgumentParser()
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--Cs", type=str, default="0.001,0.01", help="comma list of inverse regularisation strengths")
ap.add_argument("--sub", type=int, default=2, help="use every SUB-th cached training row (rows are near-duplicates)")
ap.add_argument("--max-iter", type=int, default=300)
ap.add_argument("--drop", type=str, default="", help="feature-name prefixes to remove (leading ~ = contains), e.g. ctx_")
ap.add_argument("--cache", type=str, default="", help="feature cache path (default data/feat_<N_FEAT>.npz)")
ap.add_argument("--blend", type=str, default="", help="oof .npz written by cv.py to test a blend with")
args = ap.parse_args()

OUT = "cv_out"
os.makedirs(OUT, exist_ok=True)


class Tee:
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")

    def write(self, s):
        sys.__stdout__.write(s)
        self.f.write(s)

    def flush(self):
        sys.__stdout__.flush()
        self.f.flush()


sys.stdout = Tee(os.path.join(OUT, "logreg_summary.txt"))
warnings.filterwarnings("ignore", category=UserWarning)  # lbfgs convergence chatter


def step_weights(y, step):
    T = int(step.max()) + 1
    npos = np.bincount(step, weights=y, minlength=T)
    n = np.bincount(step, minlength=T).astype(float)
    nneg = n - npos
    w = np.where(y == 1, nneg[step] / 2.0, npos[step] / 2.0)
    return w / max(w.mean(), 1e-12)


def signed_log(X, chunk=200_000):
    out = np.empty(X.shape, dtype=np.float32)
    for a in range(0, len(X), chunk):
        blk = X[a:a + chunk]
        out[a:a + chunk] = np.sign(blk) * np.log1p(np.abs(blk))
    return out


def per_step_z(s, step, mask):
    """Standardise scores within every online step (across series) so two models can be averaged."""
    T = int(step.max()) + 1
    cnt = np.maximum(np.bincount(step[mask], minlength=T), 1)
    m = np.bincount(step[mask], weights=s[mask], minlength=T) / cnt
    v = np.bincount(step[mask], weights=s[mask] ** 2, minlength=T) / cnt - m ** 2
    return (s - m[step]) / np.sqrt(np.maximum(v[step], 1e-12))


def main():
    cache = args.cache or os.path.join("data", f"feat_{F.N_FEAT}.npz")
    if not os.path.exists(cache):
        sys.exit(f"feature cache {cache} not found -> run cv.py once first (it builds and caches the features)")
    z = np.load(cache)
    X, y, sid, step = z["X"], z["y"], z["sid"], z["step"]
    names = F.FEATURE_NAMES if X.shape[1] == len(F.FEATURE_NAMES) else [f"f{j}" for j in range(X.shape[1])]
    drop = [p for p in args.drop.split(",") if p]
    keep = [j for j, nm in enumerate(names)
            if not any((p[1:] in nm) if p.startswith("~") else nm.startswith(p) for p in drop)]
    if len(keep) < len(names):
        X = X[:, keep]
        print(f"dropped {len(names) - len(keep)} features; {len(keep)} remain")
    names = [names[j] for j in keep]
    _, tau = load_train()
    n_series = int(sid.max()) + 1
    has_break = (tau >= 0).astype(int)
    Cs = [float(c) for c in args.Cs.split(",") if c]
    print(f"rows {X.shape[0]:,} x features {X.shape[1]} | positive rate {y.mean():.3f} | series {n_series}")
    print(f"settings: {vars(args)}")

    skf = StratifiedKFold(5, shuffle=True, random_state=0)  # identical to cv.py
    scores = {C: [] for C in Cs}
    oof = {C: np.full(len(y), np.nan) for C in Cs}
    coefs = {}
    for f, (tr_s, va_s) in enumerate(skf.split(np.arange(n_series), has_break)):
        if f >= args.folds:
            break
        t0 = time.time()
        tr = np.flatnonzero(np.isin(sid, tr_s))
        va = np.flatnonzero(np.isin(sid, va_s))
        w = step_weights(y[tr], step[tr])
        tr = tr[:: args.sub]
        w = w[:: args.sub]
        A = signed_log(X[tr])
        mu, sd = A.mean(axis=0), A.std(axis=0) + 1e-6
        A = np.clip((A - mu) / sd, -5, 5)
        B = np.clip((signed_log(X[va]) - mu) / sd, -5, 5)
        for C in Cs:
            t1 = time.time()
            m = LogisticRegression(C=C, solver="lbfgs", max_iter=args.max_iter, tol=1e-4)
            m.fit(A, y[tr], sample_weight=w)
            s = m.decision_function(B)
            sc = ts_auc(s, y[va], step[va])
            scores[C].append(sc)
            oof[C][va] = s
            coefs[C] = m.coef_[0].copy()
            print(f"fold {f}  C={C:<8g} TS-AUC {sc:.4f}  ({m.n_iter_[0]} iterations, {time.time() - t1:.0f}s)", flush=True)
        print(f"  fold {f} done in {time.time() - t0:.0f}s", flush=True)

    print("\nMean TS-AUC over folds:")
    for C in Cs:
        print(f"  C={C:<8g} {np.mean(scores[C]):.4f}  (std {np.std(scores[C]):.4f})")
    bestC = max(Cs, key=lambda c: np.mean(scores[c]))
    print(f"\nHEADLINE  logistic regression TS-AUC = {np.mean(scores[bestC]):.4f} at C={bestC:g}"
          "   (LightGBM reference from cv.py: about 0.6155)")

    done = ~np.isnan(oof[bestC])
    print("\nOOF TS-AUC by online-step bucket:")
    for lo, hi in ((0, 100), (100, 300), (300, 600), (600, 10000)):
        mk = done & (step >= lo) & (step < hi)
        if mk.sum():
            print(f"  steps [{lo:4d}, {hi:5d}):  {ts_auc(oof[bestC][mk], y[mk], step[mk]):.4f}")

    cf = coefs[bestC]
    order = np.argsort(-np.abs(cf))
    print("\nLargest standardised coefficients (last fold; + = more break-like when high):")
    for j in order[:15]:
        print(f"  {names[j]:26s} {cf[j]:+.3f}")

    np.savez(os.path.join(OUT, "oof_logreg.npz"), oof=oof[bestC], y=y, step=step, sid=sid)
    print(f"\nsaved {OUT}/oof_logreg.npz")

    if args.blend:
        g = np.load(args.blend)
        og = g["oof"]
        both = done & ~np.isnan(og)
        if both.sum() == 0:
            print("\nblend: no overlapping out-of-fold rows (run both scripts on the same folds)")
        else:
            zg = per_step_z(og, step, both)
            zl = per_step_z(oof[bestC], step, both)
            print(f"\nBLEND with {args.blend} on {int(both.sum()):,} rows (scores standardised within each step):")
            for wl in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0):
                sc = ts_auc((1 - wl) * zg[both] + wl * zl[both], y[both], step[both])
                tag = "  <- LightGBM only" if wl == 0.0 else ("  <- logistic only" if wl == 1.0 else "")
                print(f"  weight on logistic {wl:.1f}: TS-AUC {sc:.4f}{tag}")
            print("A blend only helps if some mixture beats BOTH ends by more than the fold noise (~0.003).")


if __name__ == "__main__":
    main()
