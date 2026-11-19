"""Calibrate the size of synthetic breaks (sbrt/synth.py) against the real ones. No model training.

Run:    python synth_calibrate.py                       # scales 0.15 0.25 0.35 0.5 0.75 1.0
        python synth_calibrate.py --scales 0.2 0.3 0.4
Output: cv_out/synth_calibration.txt

For real breaks and for synthetic breaks at each size scale, it computes the same "oracle" table as eda.py
section 5: with the break position KNOWN, how well does each single statistic of the L points after the break
separate break segments from no-break segments (two-sided AUC; 0.5 = invisible)? Synthetic breaks are useful
training data only if they are about as hard to see as real ones, so pick the scale whose average AUC is closest
to the real average (printed at the bottom), then pass it to cv.py / gru_cv.py as --synth-scale.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.stats import kurtosis, rankdata, skew

from sbrt.data import load_train
from sbrt.synth import TYPES, fold_ids, make_synthetic

ap = argparse.ArgumentParser()
ap.add_argument("--scales", type=float, nargs="+", default=[0.15, 0.25, 0.35, 0.5, 0.75, 1.0])
ap.add_argument("--n-series", type=int, default=3000, help="real series used as parents for the synthetic ones")
ap.add_argument("--per-series", type=int, default=3)
ap.add_argument("--p-zero", type=float, default=0.25)
ap.add_argument("--types", type=str, default="all", help="comma list of break types, or all")
args = ap.parse_args()


class Tee:
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")

    def write(self, s):
        sys.__stdout__.write(s)
        self.f.write(s)

    def flush(self):
        sys.__stdout__.flush()
        self.f.flush()


os.makedirs("cv_out", exist_ok=True)
sys.stdout = Tee(os.path.join("cv_out", "synth_calibration.txt"))
LS = (30, 100, 300)


def auc(a, b):
    r = rankdata(np.r_[a, b])
    return (r[: len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))


def stat_vec(h, z):
    c = lambda x, k: np.corrcoef(x[:-k], x[k:])[0, 1]
    return {
        "mean (|.|)": z.mean() * np.sqrt(len(z)),
        "log var": np.log(z.var() + 1e-9),
        "mean |z|": np.abs(z).mean() - np.abs(h).mean(),
        "ac1": c(z, 1) - c(h, 1),
        "ac2": c(z, 2) - c(h, 2),
        "skew": skew(z) - skew(h),
        "kurt": kurtosis(z) - kurtosis(h),
        "tail>2": (np.abs(z) > 2).mean() - (np.abs(h) > 2).mean(),
        "vol clustering": c(np.abs(z), 1) - c(np.abs(h), 1),
        "log var(diff)": np.log(np.diff(z).var() + 1e-9) - np.log(np.diff(h).var() + 1e-9),
    }


def oracle(items):
    """items: iterable of (hist, online, tau). Returns {stat: {L: auc}}."""
    items = list(items)
    table = {}
    for L in LS:
        bk, nbk = [], []
        for h, o, t in items:
            mu, sd = h.mean(), h.std() + 1e-12          # same standardisation as the platform
            hz, oz = (h - mu) / sd, (o - mu) / sd
            if t >= 0 and len(o) - t >= L:
                bk.append(stat_vec(hz, oz[t:t + L]))
            elif t < 0 and len(o) >= L:
                nbk.append(stat_vec(hz, oz[:L]))
        if len(bk) < 30 or len(nbk) < 30:
            continue
        for key in bk[0]:
            a = np.array([d[key] for d in bk])
            b = np.array([d[key] for d in nbk])
            a, b = a[np.isfinite(a)], b[np.isfinite(b)]
            med = np.median(b)
            table.setdefault(key, {})[L] = auc(np.abs(a - med), np.abs(b - med))
    return table


def summary(table):
    return {L: float(np.mean([row[L] for row in table.values() if L in row])) for L in LS}


S, tau = load_train()
folds = fold_ids(tau)
print("REAL breaks (all 10,000 series)")
real = oracle((s[1], s[2], int(t)) for s, t in zip(S, tau))
print("%-16s %8s %8s %8s" % ("statistic", "L=30", "L=100", "L=300"))
for key, row in sorted(real.items(), key=lambda kv: -max(kv[1].values())):
    print("%-16s %8.3f %8.3f %8.3f" % (key, row[30], row[100], row[300]))
rs = summary(real)
print("%-16s %8.3f %8.3f %8.3f" % ("AVERAGE", rs[30], rs[100], rs[300]))

sub = slice(0, args.n_series)
res = {}
for sc in args.scales:
    types = TYPES if args.types == "all" else tuple(t.strip() for t in args.types.split(","))
    V = make_synthetic(S[sub], tau[sub], args.per_series, seed=0, scale=sc, p_zero=args.p_zero, folds=folds[sub], types=types)
    tab = oracle((v[3], v[4], v[5]) for v in V)
    res[sc] = (tab, summary(tab))
    print(f"\nSYNTHETIC breaks, scale {sc:g} ({len(V):,} pseudo-series, size 0 for {args.p_zero:.0%} of breaks)")
    for key, row in sorted(tab.items(), key=lambda kv: -max(kv[1].values())):
        print("%-16s %8.3f %8.3f %8.3f      real: %.3f %.3f %.3f" % (key, row[30], row[100], row[300],
                                                                     real[key][30], real[key][100], real[key][300]))
    s = res[sc][1]
    print("%-16s %8.3f %8.3f %8.3f      real: %.3f %.3f %.3f" % ("AVERAGE", s[30], s[100], s[300], rs[30], rs[100], rs[300]))

print("\nAVERAGE oracle AUC over the 10 statistics (closer to real = better calibrated)")
print("%-10s %8s %8s %8s %10s" % ("scale", "L=30", "L=100", "L=300", "distance"))
print("%-10s %8.3f %8.3f %8.3f" % ("real", rs[30], rs[100], rs[300]))
best, bestd = None, 1e9
for sc, (_, s) in res.items():
    d = sum(abs(s[L] - rs[L]) for L in LS)
    print("%-10g %8.3f %8.3f %8.3f %10.4f" % (sc, s[30], s[100], s[300], d))
    if d < bestd:
        best, bestd = sc, d
print(f"\nRECOMMENDED --synth-scale {best:g}  (also consider one step larger: the model may learn faster from slightly "
      f"clearer examples, and cross-validation on real series decides)")
print(f"break types used: {args.types}")
