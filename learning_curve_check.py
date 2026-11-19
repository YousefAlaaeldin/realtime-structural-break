"""Fold 0+1 LightGBM: (a) learning curve 25/50/100% of training series, (b) +30 extra history descriptors."""
import sys, os, time, numpy as np, lightgbm as lgb
sys.path.insert(0, os.getcwd())
from sklearn.model_selection import StratifiedKFold
from sbrt.metric import ts_auc
z = np.load('data/feat_270.npz'); X, y, sid, step = z['X'], z['y'], z['sid'], z['step']
d = np.load('data/train_cache.npz', allow_pickle=True); S = list(d['series']); tau = d['tau']
n_series = len(S)

def hist_desc(h):
    h = np.asarray(h, float); zz = (h - h.mean()) / (h.std() + 1e-12); n = len(zz)
    out = []
    for k in range(1, 11):
        out.append(float(np.dot(zz[:-k], zz[k:]) / n))                     # acf 1..10
    z2 = zz * zz - 1
    for k in (1, 2, 5):
        out.append(float(np.dot(z2[:-k], z2[k:]) / np.dot(z2, z2)))        # ARCH effect
    for b in (10, 50):
        m = n // b; bm = zz[:m * b].reshape(m, b).mean(1)
        out.append(float(np.log(bm.var() * b + 1e-9)))                     # variance ratio
    for b in (100,):
        m = n // b; bs = zz[:m * b].reshape(m, b).std(1)
        out.append(float(np.log(bs.std() + 1e-9)))                         # heteroskedasticity
        out.append(float(np.log(bs.max() / (bs.min() + 1e-9))))
    out += [float(np.mean(np.abs(zz) > 2)), float(np.mean(np.abs(zz) > 3)), float(np.abs(zz).max())]
    out += list(np.quantile(zz, [0.01, 0.05, 0.5, 0.95, 0.99]))
    out.append(float(np.mean(np.sign(zz[1:]) != np.sign(zz[:-1]))))
    p = np.abs(np.fft.rfft(zz[: (n // 2) * 2])) ** 2; p = p[1:]; e = len(p)
    tot = p.sum() + 1e-12
    out += [float(np.log(p[: e // 8].sum() / tot + 1e-9)), float(np.log(p[e // 8: e // 2].sum() / tot + 1e-9)),
            float(np.log(p[e // 2:].sum() / tot + 1e-9))]
    out.append(float(n))
    dz = np.diff(zz); out.append(float(np.log(dz.var() + 1e-9)))
    out.append(float(np.mean(dz ** 4) / (dz.var() ** 2 + 1e-9)))
    return out

t0 = time.time()
H = np.array([hist_desc(s[1]) for s in S], np.float32)
print(f"{H.shape[1]} history descriptors in {time.time()-t0:.0f}s")
Xe = np.hstack([X, H[sid]])

P = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_data_in_leaf=2000, feature_fraction=0.5,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, seed=0, deterministic=True, force_row_wise=True,
         num_threads=16, verbose=-1)
def w_of(yy, st):
    T = st.max() + 1; npos = np.bincount(st, weights=yy, minlength=T); n = np.bincount(st, minlength=T).astype(float)
    w = np.where(yy == 1, (n - npos)[st] / 2, npos[st] / 2); return w / w.mean()
def run(XX, tr, va, rounds=250):
    m = lgb.train(P, lgb.Dataset(XX[tr], y[tr], weight=w_of(y[tr], step[tr])), rounds)
    return ts_auc(m.predict(XX[va], raw_score=True), y[va], step[va])

skf = StratifiedKFold(5, shuffle=True, random_state=0)
rng = np.random.default_rng(0)
for f, (trs, vas) in enumerate(skf.split(np.arange(n_series), (tau >= 0).astype(int))):
    if f >= 2: break
    va = np.isin(sid, vas)
    res = {}
    for frac in (0.25, 0.5, 1.0):
        sub = rng.choice(trs, int(frac * len(trs)), replace=False)
        tr = np.isin(sid, sub); t0 = time.time()
        res[f"base {int(frac*100)}%"] = run(X, tr, va)
    tr = np.isin(sid, trs)
    res["base + 30 history descriptors"] = run(Xe, tr, va)
    for k, v in res.items():
        print(f"fold {f}  {k:34s} {v:.4f}", flush=True)
