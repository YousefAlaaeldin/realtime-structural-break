"""Oracle analysis: if we KNEW the break position and had >=150 post-break points, how separable
are break vs no-break series by simple statistics of that segment vs history?"""
import sys
sys.path.insert(0, ".")
import numpy as np
from scipy.stats import rankdata, skew, kurtosis
from sbrt.data import load_train

S, tau = load_train()


def auc(a, b):
    r = rankdata(np.r_[a, b])
    return (r[: len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))


def stats(h, seg):
    n = len(seg)
    z = seg  # history is already z-scored
    out = {}
    out["mean"] = z.mean() * np.sqrt(n)
    out["logvar"] = np.log(z.var() + 1e-9)
    out["abs_mean"] = np.abs(z).mean() - np.abs(h).mean()
    out["ac1"] = np.corrcoef(z[:-1], z[1:])[0, 1] - np.corrcoef(h[:-1], h[1:])[0, 1]
    out["ac2"] = np.corrcoef(z[:-2], z[2:])[0, 1] - np.corrcoef(h[:-2], h[2:])[0, 1]
    out["skew"] = skew(z) - skew(h)
    out["kurt"] = kurtosis(z) - kurtosis(h)
    out["tail2"] = (np.abs(z) > 2).mean() - (np.abs(h) > 2).mean()
    out["vol_clust"] = np.corrcoef(np.abs(z[:-1]), np.abs(z[1:]))[0, 1] - np.corrcoef(np.abs(h[:-1]), np.abs(h[1:]))[0, 1]
    # variance-of-diffs (roughness)
    out["dvar"] = np.log(np.diff(z).var() + 1e-9) - np.log(np.diff(h).var() + 1e-9)
    return out


L = 150
brk, nob = [], []
for (i, h, o), t in zip(S, tau):
    if t >= 0:
        seg = o[t:]
        if len(seg) >= L:
            brk.append(stats(h, seg[:L]))
    else:
        if len(o) >= L:
            nob.append(stats(h, o[:L]))
print("n break", len(brk), "n nobreak", len(nob))
for k in brk[0]:
    a = np.array([d[k] for d in brk]); b = np.array([d[k] for d in nob])
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    print("%-10s signed AUC %.3f   abs AUC %.3f" % (k, auc(a, b), auc(np.abs(a - np.median(b)), np.abs(b - np.median(b)))))
