"""How similar are two blend members? Mean per-step Spearman correlation of their OOF scores (all series, and
no-break series only), weighted like the metric (n_pos x n_neg per step). Several files per side = seeds, averaged
after dividing by their sd (as blend.py does).

Run: python diagnostics/member_corr.py --a cv_out/oof_tcn_x.npz cv_out/oof_tcn_x_s1.npz --b cv_out/oof_gru_y.npz ...
Reading: ~0.9 or more = the new member mostly repeats the old one; expect little from blending.
"""
import argparse

import numpy as np
from scipy.stats import rankdata

ap = argparse.ArgumentParser()
ap.add_argument("--a", nargs="+", required=True)
ap.add_argument("--b", nargs="+", required=True)
args = ap.parse_args()


def load(files):
    zs = [np.load(f) for f in files]
    return sum(z["oof"] / z["oof"].std() for z in zs), zs[0]


a, za = load(args.a)
b, zb = load(args.b)
if not (np.array_equal(za["sid"], zb["sid"]) and np.array_equal(za["step"], zb["step"])):
    raise SystemExit("different row layouts")
y, step = za["y"], za["step"]
order = np.argsort(step, kind="stable")
bounds = np.flatnonzero(np.diff(step[order])) + 1
num = {"all": 0.0, "no-break": 0.0}
den = {"all": 0.0, "no-break": 0.0}
for idx in np.split(order, bounds):
    npos = int(y[idx].sum())
    w = npos * (len(idx) - npos)
    if w == 0:
        continue
    for k, m in (("all", np.ones(len(idx), bool)), ("no-break", y[idx] == 0)):
        if m.sum() < 10:
            continue
        r = np.corrcoef(rankdata(a[idx][m]), rankdata(b[idx][m]))[0, 1]
        num[k] += w * r
        den[k] += w
for k in num:
    print(f"metric-weighted per-step Spearman, {k:8s} series: {num[k] / den[k]:.3f}")
