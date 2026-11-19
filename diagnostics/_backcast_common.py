"""Shared pieces for the backcasting follow-ups (backcast_a_control.py, backcast_nn_gate.py; HANDOFF section 32):
reference features (bo_, vol_, nb_) with a disk cache, labels / slices, metric-matched weights, a bootstrap SE for
TS-AUC differences, and a seed-averaged K-fold LightGBM incremental test."""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from sbrt import bocpd as BO      # noqa: E402
from sbrt import nbm as NB        # noqa: E402
from sbrt import volstate as VS   # noqa: E402

SUB = 3
BUCKETS = ((0, 100), (100, 300), (300, 700), (700, 10 ** 6))


class Log:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, "w", encoding="utf-8")
        self.path = path

    def __call__(self, s=""):
        print(s, flush=True)
        self.f.write(s + "\n")
        self.f.flush()


def references(S_all, tau_all, fit, ev, threads, say):
    """bo_ (10), vol_ (13), nb_ (6, gate model data/nbm_v1.pt, prior fitted on the fit set) for the eval series,
    rows every SUB-th online step. Cached in data/backcast_refs_<fit>_<eval>.npz."""
    f0, f1 = fit
    e0, e1 = ev
    path = os.path.join(HERE, "data", f"backcast_refs_{f0}_{f1}_{e0}_{e1}.npz")
    if os.path.exists(path):
        z = np.load(path)
        say(f"reference features (bo, vol, nb): cache {os.path.basename(path)}")
        return z["Bo"], z["V"], z["NBF"]
    from joblib import Parallel, delayed
    t1 = time.time()
    pairs = [(S_all[i][1], S_all[i][2]) for i in range(e0, e1)]
    Bo = np.concatenate(Parallel(n_jobs=threads, backend="threading")(
        delayed(BO.run_series_bocpd)(h, o, SUB) for h, o in pairs))
    V = np.concatenate([VS.run_series_vol(h, o)[::SUB] for h, o in pairs])
    M = NB.load_model()
    sf = NB.sums_parallel([(S_all[i][1], S_all[i][2]) for i in range(f0, f1)], M, threads)
    nb_prior, _ = NB.fit_prior_w([NB.w_of(q) for q in sf], tau_all[f0:f1])
    se_ = NB.sums_parallel(pairs, M, threads)
    NBF = np.concatenate([NB.features_from_sums(q, nb_prior, SUB) for q in se_])
    np.savez(path, Bo=Bo.astype(np.float32), V=V.astype(np.float32), NBF=NBF.astype(np.float32))
    say(f"reference features (bo, vol, nb): {time.time() - t1:.0f}s (cached)")
    return Bo, V, NBF


def layout(S_all, tau_all, ev):
    """step, sid, y and the var-same slice for the eval series (rows every SUB-th online step)."""
    e0, e1 = ev
    pairs = [(S_all[i][1], S_all[i][2]) for i in range(e0, e1)]
    tau = tau_all[e0:e1]
    step = np.concatenate([np.arange(0, len(o), SUB) for _, o in pairs])
    sid = np.repeat(np.arange(len(pairs)), [(len(o) + SUB - 1) // SUB for _, o in pairs])
    y = ((tau[sid] >= 0) & (step >= tau[sid])).astype(np.int8)
    vr = np.full(len(pairs), np.nan)
    for i, ((h, o), t) in enumerate(zip(pairs, tau)):
        if t >= 0 and len(o) - t >= 30:
            vr[i] = BO.whitened_online(h, o)[0][t:t + 300].var()
    same = (vr > 0.85) & (vr < 1.15)
    in_slice = (tau[sid] < 0) | same[sid]
    return step, sid, y, in_slice


def auc_row(x, y, step, in_slice):
    from sbrt.metric import ts_auc
    r = [ts_auc(x, y, step)]
    for lo, hi in BUCKETS:
        m = (step >= lo) & (step < hi)
        r.append(ts_auc(x[m], y[m], step[m]))
    r.append(ts_auc(x[in_slice], y[in_slice], step[in_slice]))
    return np.array(r)


def fmt(r):
    return "  ".join(f"{v:7.4f}" for v in r)


HDR = "all    0-100  100-300  300-700    700+  var-same"


def boot_diff(xa, xb, y, step, sid, n_boot=10, mask=None):
    """TS-AUC(xa) - TS-AUC(xb) and its series-bootstrap SE (rows of a series move together)."""
    from sbrt.metric import ts_auc
    m = np.ones(len(y), bool) if mask is None else mask
    d = ts_auc(xa[m], y[m], step[m]) - ts_auc(xb[m], y[m], step[m])
    nS = int(sid.max()) + 1
    starts = np.r_[0, np.cumsum(np.bincount(sid, minlength=nS))]
    bs = []
    for b in range(n_boot):
        pick = np.random.default_rng(100 + b).integers(0, nS, nS)
        idx = np.concatenate([np.arange(starts[i], starts[i + 1]) for i in pick])
        idx = idx[m[idx]]
        bs.append(ts_auc(xa[idx], y[idx], step[idx]) - ts_auc(xb[idx], y[idx], step[idx]))
    return d, float(np.std(bs))


def weights(y, step):
    wt = np.zeros(len(y))
    sb = step // 30
    for g in np.unique(sb):
        m = sb == g
        npos, nneg = int(y[m].sum()), int(m.sum() - y[m].sum())
        if npos and nneg:
            wt[m & (y == 1)] = nneg / 2.0
            wt[m & (y == 0)] = npos / 2.0
    return wt / wt.mean()


def incremental(sets, y, step, sid, folds, seeds, threads, say, scores_out=None):
    """K-fold by series; per fold the prediction is the average over `seeds` LightGBM seeds. Returns per-set fold
    scores and prints paired differences vs sets['base']."""
    import lightgbm as lgb
    from sbrt.metric import ts_auc
    wt = weights(y, step)
    nS = int(sid.max()) + 1
    fold_of = np.random.default_rng(7).permutation(nS) % folds
    scores = {k: [] for k in sets}
    for f in range(folds):
        va = fold_of[sid] == f
        for k, X in sets.items():
            p = np.zeros(int(va.sum()))
            for s in range(seeds):
                params = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=500,
                              feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                              num_threads=threads, verbose=-1, seed=s)
                bst = lgb.train(params, lgb.Dataset(X[~va], y[~va], weight=wt[~va]), num_boost_round=300)
                p += bst.predict(X[va]) / seeds
            scores[k].append(ts_auc(p, y[va], step[va]))
        say(f"  fold {f}: " + "  ".join(f"{k} {scores[k][-1]:.4f}" for k in sets))
    out = {}
    for k in sets:
        if k == "base":
            continue
        d = np.array(scores[k]) - np.array(scores["base"])
        se = d.std(ddof=1) / np.sqrt(len(d))
        out[k] = (d.mean(), se, int((d > 0).sum()))
        say(f"  {k:14s} - base: {d.mean():+.4f} +- {se:.4f}  ({(d > 0).sum()}/{len(d)} folds)   "
            f"[base {np.mean(scores['base']):.4f}]")
    if scores_out is not None:
        scores_out.update(scores)
    return out


def passes_project_bar(res, folds):
    """The project rule: >= +0.003, > 2 SE, positive in >= folds - 1 folds."""
    m, se, npos = res
    return m >= 0.003 and m > 2 * se and npos >= folds - 1
