"""Are variance DECREASES at breaks more common than chance? (they are not). Run: python diagnostics/decrease_check.py"""
import os, sys
import numpy as np
from scipy.stats import rankdata
HERE = r"C:\Users\youse\R projects\AIDA breakpoint detection benchmark"
sys.path.insert(0, HERE); os.chdir(HERE)
from sbrt.data import load_train
S, tau = load_train()
rng = np.random.default_rng(0)


def auc(a, b):
    r = rankdata(np.r_[a, b]); return (r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))


for L in (100, 300):
    P = {k: [] for k in ("lv", "la", "q")}; Nn = {k: [] for k in ("lv", "la", "q")}
    for i, (_, h, o) in enumerate(S):
        t = tau[i]
        if t < 0:
            if len(o) < L + 10: continue
            t = rng.integers(0, len(o) - L)
            D = Nn
        else:
            if len(o) - t < L: continue
            D = P
        z = o[t:t + L]
        D["lv"].append(np.log(z.var() / h.var()))
        D["la"].append(np.mean(np.log(np.abs(z) + 1e-3)) - np.mean(np.log(np.abs(h) + 1e-3)))   # robust log-scale
        hs = np.sort(np.abs(h)); D["q"].append(np.mean(np.searchsorted(hs, np.abs(z)) / len(hs)) - 0.5)  # ECDF of |x|
    print(f"\nL={L}: {len(P['lv'])} break segments, {len(Nn['lv'])} no-break segments")
    for k in P:
        p, n = np.array(P[k]), np.array(Nn[k])
        qs = [5, 25, 50, 75, 95]
        print(f"  {k}: negatives pct {np.round(np.percentile(n, qs), 3)}  positives pct {np.round(np.percentile(p, qs), 3)}")
        print(f"      AUC up (pos>neg) {auc(p, n):.3f}   two-sided |x-med_neg| {auc(np.abs(p - np.median(n)), np.abs(n - np.median(n))):.3f}")
        lo_p = (p < np.percentile(n, 10)).mean(); hi_p = (p > np.percentile(n, 90)).mean()
        print(f"      share of positives below the negatives' 10th pct {lo_p:.3f}, above the 90th {hi_p:.3f}  (0.10 each = no signal)")
