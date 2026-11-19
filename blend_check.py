import sys, os, numpy as np
sys.path.insert(0, os.getcwd())
from sbrt.metric import ts_auc
a = np.load('cv_out/oof_loc.npz'); g = np.load('cv_out/oof_gru_sw_2348ae47_cv5.npz')
y, step, sid = a['y'], a['step'], a['sid']
L, G = a['oof'], g['oof']
print("LGB", round(ts_auc(L, y, step), 4), "GRU", round(ts_auc(G, y, step), 4))

# 1. blends usable at inference
print("\n-- raw logit average (no standardisation)")
for w in (0.3, 0.4, 0.5, 0.6, 0.7):
    print(f"  w={w}: {ts_auc((1-w)*L + w*G, y, step):.4f}")
print("-- scaled by each model's GLOBAL std (two shipped constants)")
Ls, Gs = L / L.std(), G / G.std()
for w in (0.4, 0.5, 0.6):
    print(f"  w={w}: {ts_auc((1-w)*Ls + w*Gs, y, step):.4f}")
print("-- per-step constants (mean/std of each model at step t, from OOF, shipped as a table)")
T = step.max() + 1
def per_step_std(x):
    n = np.bincount(step, minlength=T).astype(float)
    m = np.bincount(step, weights=x, minlength=T) / np.maximum(n, 1)
    v = np.bincount(step, weights=x * x, minlength=T) / np.maximum(n, 1) - m * m
    return (x - m[step]) / np.sqrt(np.maximum(v[step], 1e-12))
Lp, Gp = per_step_std(L), per_step_std(G)
for w in (0.4, 0.5, 0.6):
    print(f"  w={w}: {ts_auc((1-w)*Lp + w*Gp, y, step):.4f}")

# 2. temporal post-processing (per series, in step order; OOF rows are every 3rd step)
order = np.lexsort((step, sid)); inv = np.empty_like(order); inv[order] = np.arange(len(order))
def per_series(x, fn):
    xs = x[order]; ss = sid[order]
    bounds = np.r_[0, np.flatnonzero(np.diff(ss)) + 1, len(ss)]
    out = np.empty_like(xs)
    for s, e in zip(bounds[:-1], bounds[1:]):
        out[s:e] = fn(xs[s:e])
    return out[inv]
def ema(alpha):
    def f(v):
        o = np.empty_like(v); acc = v[0]
        for i, t in enumerate(v):
            acc = alpha * t + (1 - alpha) * acc; o[i] = acc
        return o
    return f
B = 0.5 * Ls + 0.5 * Gs
for name, x in (("LGB", L), ("blend", B)):
    print(f"\n-- temporal post-processing of {name}: base {ts_auc(x, y, step):.4f}")
    cm = per_series(x, np.maximum.accumulate)
    print(f"  running max            : {ts_auc(cm, y, step):.4f}")
    for lam in (0.25, 0.5, 0.75):
        print(f"  {lam}*x + {1-lam}*runmax : {ts_auc(lam*x + (1-lam)*cm, y, step):.4f}")
    for al in (0.7, 0.5, 0.3, 0.15):
        print(f"  EMA alpha={al} (per 3 steps): {ts_auc(per_series(x, ema(al)), y, step):.4f}")
    # running mean of the last k OOF rows (k*3 steps)
    for k in (3, 10):
        def rm(v, k=k):
            c = np.cumsum(np.r_[0.0, v]); i = np.arange(1, len(v) + 1); lo = np.maximum(i - k, 0)
            return (c[i] - c[lo]) / (i - lo)
        print(f"  mean of last {k} rows     : {ts_auc(per_series(x, rm), y, step):.4f}")

# 3. do histories repeat / overlap across series?
z = np.load('data/train_cache.npz', allow_pickle=True)
S = list(z['series']); tau = z['tau']
from collections import defaultdict
W = 64
seen = defaultdict(set)
for i, (_, h, o) in enumerate(S):
    r = np.round(h, 6)
    for off in range(0, len(r) - W, 97):          # sample windows at a stride
        seen[r[off:off + W].tobytes()].add(i)
multi = [v for v in seen.values() if len(v) > 1]
pairs = set()
for v in multi:
    v = sorted(v)
    for j in v[1:]:
        pairs.add((v[0], j))
ser = set(i for v in multi for i in v)
print(f"\n-- history overlap: {len(multi)} shared 64-point windows, {len(ser)} series involved, {len(pairs)} pairs (of {len(S)})")
# constant / discrete series
uniq = np.array([len(np.unique(np.round(h, 6))) / len(h) for _, h, _ in S])
print(f"   series with unique-value ratio < 0.5: {(uniq < 0.5).sum()}")
