"""Breakdown of the best blend (oof_aug2_evc + both augmented GRU seeds) by step, time since break, size and kind of change.
Run: python diagnostics/error_breakdown.py   (HANDOFF.md section 15.2)"""
import os, sys
import numpy as np
from scipy.stats import kurtosis, skew
HERE = r"C:\Users\youse\R projects\AIDA breakpoint detection benchmark"
sys.path.insert(0, HERE); os.chdir(HERE)
from sbrt.data import load_train
from sbrt.metric import ts_auc

S, tau = load_train()
a = np.load("cv_out/oof_aug2_evc.npz")
g0 = np.load("cv_out/oof_gru_aug2.npz")["oof"]; g1 = np.load("cv_out/oof_gru_aug2_s1.npz")["oof"]
y, step, sid = a["y"], a["step"], a["sid"]
g = g0 / g0.std() + g1 / g1.std()
sc = 0.5 * a["oof"] / a["oof"].std() + 0.5 * g / g.std()
print("blend", round(ts_auc(sc, y, step), 4), "trees", round(ts_auc(a["oof"], y, step), 4))
print("sid range", sid.min(), sid.max(), "n series", len(S))

# per-series descriptors from history, and post-break change sizes for break series
N = len(S)
nh = np.array([len(s[1]) for s in S]); no = np.array([len(s[2]) for s in S])
ku = np.array([kurtosis(s[1]) for s in S]); sk = np.array([skew(s[1]) for s in S])
r1 = np.array([np.corrcoef(s[1][:-1], s[1][1:])[0, 1] for s in S])
def ac(x, k=1): return np.corrcoef(x[:-k], x[k:])[0, 1]
dlv = np.full(N, np.nan); dr1 = np.full(N, np.nan); dku = np.full(N, np.nan); dabs = np.full(N, np.nan); dmean = np.full(N, np.nan)
for i, (_, h, o) in enumerate(S):
    t = tau[i]
    if t >= 0 and len(o) - t >= 30:
        z = o[t:]
        dlv[i] = np.log(z.var() / h.var()); dr1[i] = ac(z) - ac(h); dku[i] = kurtosis(z) - kurtosis(h)
        dmean[i] = abs(z.mean() - h.mean()) / h.std() * np.sqrt(len(z))
        dabs[i] = np.log(np.diff(z).var() / np.diff(h).var())

def sub_auc(mask_series):
    m = mask_series[sid]
    return ts_auc(sc[m], y[m], step[m]), m.sum()

print("\n== subgroups (positives and negatives both restricted to the subgroup) ==")
groups = {
    "all": np.ones(N, bool),
    "kurt>3": ku > 3, "kurt<=3": ku <= 3,
    "|rho1|>0.3": np.abs(r1) > 0.3, "|rho1|<=0.3": np.abs(r1) <= 0.3,
    "rho1>0.9": r1 > 0.9,
    "hist<2000": nh < 2000, "hist>=4000": nh >= 4000,
    "|skew|>1": np.abs(sk) > 1,
}
for k, msk in groups.items():
    v, n = sub_auc(msk)
    print(f"{k:14s} series {msk.sum():5d}  TS-AUC {v:.4f}")

print("\n== step buckets ==")
_, curve = ts_auc(sc, y, step, return_curve=True)
curve = np.array(curve)
W = curve[:, 2].sum()
for lo, hi in [(0, 30), (30, 100), (100, 300), (300, 700), (700, 1000)]:
    m = (curve[:, 0] >= lo) & (curve[:, 0] < hi)
    print(f"steps {lo:4d}-{hi:4d}: AUC {np.average(curve[m,1], weights=curve[m,2]):.4f}  weight share {curve[m,2].sum()/W:.3f}")

print("\n== positives by time since break (vs all negatives at the same step) ==")
tsb = np.where(y == 1, step - tau[sid], -1)
for lo, hi in [(0, 10), (10, 30), (30, 100), (100, 300), (300, 1000)]:
    m = (y == 0) | ((tsb >= lo) & (tsb < hi))
    print(f"tsb {lo:4d}-{hi:4d}: TS-AUC {ts_auc(sc[m], y[m], step[m]):.4f}  share of positive rows {((tsb>=lo)&(tsb<hi)).sum()/max((y==1).sum(),1):.3f}")

print("\n== positives by size of the actual change (post-break vs history), vs all negatives ==")
for name, d, edges in [("log var ratio", dlv, [-9, -0.3, -0.1, 0.1, 0.3, 9]),
                       ("log diff-var ratio", dabs, [-9, -0.3, -0.1, 0.1, 0.3, 9]),
                       ("d rho1", dr1, [-9, -0.1, -0.03, 0.03, 0.1, 9]),
                       ("d kurt", dku, [-99, -1, -0.3, 0.3, 1, 99]),
                       ("mean z*sqrt(n)", dmean, [0, 1, 2, 3, 5, 999])]:
    print(name)
    for lo, hi in zip(edges[:-1], edges[1:]):
        bs = (d >= lo) & (d < hi)
        m = (y == 0) | ((y == 1) & bs[sid])
        print(f"   [{lo:6.2f},{hi:6.2f}): {bs.sum():5d} break series  TS-AUC {ts_auc(sc[m], y[m], step[m]):.4f}")

# how many breaks look like 'nothing changed' on all five axes
quiet = (np.abs(dlv) < 0.1) & (np.abs(dabs) < 0.1) & (np.abs(dr1) < 0.03) & (np.abs(dku) < 0.5) & (dmean < 2)
nb = np.isfinite(dlv).sum()
print(f"\nbreak series with >=30 post points: {nb};  'quiet' on all 5 axes: {quiet.sum()} ({quiet.sum()/nb:.1%})")
m = (y == 0) | ((y == 1) & quiet[sid])
print(f"quiet breaks vs negatives TS-AUC {ts_auc(sc[m], y[m], step[m]):.4f}")
m = (y == 0) | ((y == 1) & (~quiet & np.isfinite(dlv))[sid])
print(f"non-quiet breaks vs negatives TS-AUC {ts_auc(sc[m], y[m], step[m]):.4f}")
