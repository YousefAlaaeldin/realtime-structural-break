"""Time-Stratified AUC, matching the competition definition."""
import numpy as np
from scipy.stats import rankdata


def ts_auc(scores, labels, steps, return_curve=False):
    """scores/labels/steps: flat arrays, one entry per (series, online step).
    steps[i] is the online step index of row i. AUC is computed per step across
    series, then averaged with weight n_pos * n_neg."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(np.int8)
    steps = np.asarray(steps)
    order = np.argsort(steps, kind="stable")
    scores, labels, steps = scores[order], labels[order], steps[order]
    bounds = np.flatnonzero(np.diff(steps)) + 1
    starts = np.r_[0, bounds]
    ends = np.r_[bounds, len(steps)]
    num = den = 0.0
    curve = []
    for s, e in zip(starts, ends):
        l = labels[s:e]
        npos = int(l.sum())
        nneg = len(l) - npos
        if npos == 0 or nneg == 0:
            continue
        r = rankdata(scores[s:e])
        auc = (r[l == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)
        w = npos * nneg
        num += w * auc
        den += w
        curve.append((steps[s], auc, w))
    res = num / den if den else 0.5
    return (res, curve) if return_curve else res
