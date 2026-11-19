"""How much of the natural (no-break) variance wandering in the online segment is predictable from the END of
the history (volatility clustering), and does conditioning on it sharpen break detection?"""
import os, sys
import numpy as np
from scipy.stats import rankdata
HERE = r"C:\Users\youse\R projects\AIDA breakpoint detection benchmark"
sys.path.insert(0, HERE); os.chdir(HERE)
from sbrt.data import load_train
S, tau = load_train()


def auc(a, b):
    r = rankdata(np.r_[a, b]); return (r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))


def lv(x):
    return np.log(np.mean(x ** 2) + 1e-12)   # history is z-scored: mean 0, var 1


# Segment = online[0:L] for negatives, online[tau:tau+L] for break series whose break is at the very start
# (tau < 20) so that the segment start is close to the history end for both classes.
for L in (50, 100, 200):
    rows = []
    for i, (_, h, o) in enumerate(S):
        t = tau[i]
        if len(o) < L: continue
        if t < 0: lab = 0
        elif t < 20 and len(o) - t >= L: lab = 1
        else: continue
        s = 0 if lab == 0 else t
        z = o[s:s + L]
        rows.append((lab, lv(z), lv(h[-50:]), lv(h[-200:]), lv(h[-500:]), lv(np.diff(z)) - lv(np.diff(h))))
    R = np.array(rows); y = R[:, 0] == 1; neg = ~y
    print(f"\nL={L}: {y.sum()} early-break series, {neg.sum()} no-break series")
    for j, nm in [(2, "last 50"), (3, "last 200"), (4, "last 500")]:
        c = np.corrcoef(R[neg, 1], R[neg, j])[0, 1]
        print(f"  no-break: corr(online log-var, history {nm} log-var) = {c:+.3f}")
    # residualise on the history tail with a fit on negatives only
    X = np.c_[np.ones(neg.sum()), R[neg, 2], R[neg, 3], R[neg, 4]]
    beta = np.linalg.lstsq(X, R[neg, 1], rcond=None)[0]
    pred = np.c_[np.ones(len(R)), R[:, 2], R[:, 3], R[:, 4]] @ beta
    res = R[:, 1] - pred
    print(f"  no-break spread (sd) of online log-var: raw {R[neg,1].std():.3f}  after conditioning on history tail {res[neg].std():.3f}")
    for nm, s in [("raw log-var vs history (=0)", R[:, 1]), ("conditioned on history tail", res)]:
        m = np.median(s[neg])
        print(f"  {nm:30s} AUC up {auc(s[y], s[neg]):.3f}   two-sided {auc(np.abs(s[y]-m), np.abs(s[neg]-m)):.3f}")
