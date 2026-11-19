"""History-calibrated convolution exceedance on the WHITENED streams: is there break information that the blend
misses? (owner's idea 2026-10-19, HANDOFF section 28; MiniRocket-style, after the ICT lab's study 033b)

Idea. Fixed MiniRocket filters (84 patterns, length 9, weights +2 / -1, zero-sum) run over the bocpd.py whitened
streams w (history-ECDF-whitened AR(10) residual) and wu (the same after the frozen volatility forecast). Variance is
already removed from these streams, so a filter can only react to a change of SHAPE / DEPENDENCE: the target is the
variance-unchanged breaks the blend misses (blend 0.56-0.57 there vs 0.84 when variance rises, section 15.2).
Everything about "normal" is learned from the series' OWN history ("pretrained by the history"):
  - responses r_t = sum_k a_k x_(t - (8 - k) d), causal (current and past points only), dilations d = 1, 2, 4, 8
  - per series / channel / filter / dilation: thresholds = history quantiles 0.1 and 0.9 of r
  - indicator hi = r > q90, lo = r < q10; PPV = trailing share over W = 32, 128, 512 points (the window may reach
    back into the history at early online steps; still causal)
  - the same PPV over every history window gives a mean m and sd s (autocorrelated indicators handled empirically);
    feature z = (PPV_t - m) / max(s, 0.5 sqrt(p (1 - p) / W))
Online nothing is refitted. Two column sets:
  agg_  per (channel, dilation, tail, W) across the 84 filters: mean z, mean z^2, max |z|      = 144 columns
  f_    per filter, W = 128, dilations 1 and 4, both tails and channels                         = 672 columns
        (single-feature screening only; the best of 672 is optimistic by selection)
Prior art (section 8 / 20): ordinal patterns (no signal), MMD pairs/triples (duplicates --dist), --shape symbol
transitions (0.18% of tree gain), matrix profile (variance in disguise), TCN member (+0.0004 in-sample). The new part
is fixed multi-scale pattern exceedance calibrated per series on its own history, on variance-whitened streams.

Checks (series 5000-6999 by default; their blend scores are out-of-fold, so this is not the test):
  1. single-feature TS-AUC of every agg_ column (and the f_ distribution), on all series and on three slices of the
     break series by the actual post-break log variance ratio (raw values, >= 30 post points, as error_breakdown.py):
     UP >= 0.3, FLAT |.| < 0.1, DOWN <= -0.1 (positives restricted to the slice, all negatives kept); blend and best
     --dist column on the same slices for reference.
  2. redundancy: within-step rank correlation of the top agg_ columns with the blend and with every --dist column.
  3. INCREMENTAL (the gate): LightGBM on [blend score, step] vs [blend score, step, 144 agg_], grouped by series,
     3 repeats x 5 folds over the same folds, paired per fold, on all series and on the FLAT slice.
     Blend = the current best (CV 0.6534): trees pair 0.4 / GRU hpca8 pair 0.5 / SVM hpca8 0.1, each group / its sd.

KILL RULE (written before any run):
  PASS only if the incremental gain on all series is >= +0.001 AND > 2 SE (15 folds), or on the FLAT slice >= +0.005
  AND > 2 SE. Otherwise CLOSED. PASS -> streaming agg_ columns into cv.py (`--conv`) and the standing rule (section 5).
  The 15 folds share series across repeats, so the SE is optimistic; a borderline pass needs a second look.
EXPECTATION (written before any run): 10-15% to pass; blend +0.000 to +0.001 if built. Single-feature AUCs of the
  agg_ columns ~0.50-0.53 on all series (no variance in them); the question is only the FLAT slice.

Run:   python diagnostics/conv_oracle.py [n_series] [threads]      (default 2000 series from 5000, 12 processes)
       python diagnostics/conv_oracle.py --selftest                  (causality + a 40-series smoke, < 1 min)
Out:   cv_out/conv_oracle.txt
"""
import itertools
import os
import sys
import time

import numpy as np
from scipy.special import ndtri

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
os.chdir(HERE)

from sbrt import bocpd as BO  # noqa: E402
from sbrt import volstate as VS  # noqa: E402
from sbrt.metric import ts_auc  # noqa: E402

SUB = 3                          # rows at online steps 0, 3, 6, ... (= the OOF files)
OFFSET = 5000
DILS = (1, 2, 4, 8)
WINS = (32, 128, 512)
QS = (0.1, 0.9)
TAILS = ("lo", "hi")
CHANS = ("w", "wu")
F_WIN, F_DILS = 128, (1, 4)
FILT = np.full((84, 9), -1.0)
for _i, _c in enumerate(itertools.combinations(range(9), 3)):
    FILT[_i, list(_c)] = 2.0
AGGS = ("mz", "mz2", "maxabs")
AGG_NAMES = [f"agg_{c}_d{d}_{tl}_w{W}_{a}" for c in CHANS for d in DILS for tl in TAILS for W in WINS for a in AGGS]
F_NAMES = [f"f_{c}_d{d}_{tl}_{k:02d}" for c in CHANS for d in F_DILS for tl in TAILS for k in range(84)]
BLEND = (("trees", ["cv_out/oof_aug2_df_bo_nbm_bs24_ep25.npz", "cv_out/oof_aug2_df_bo_nbm_bs24_ep25_s1.npz"], 0.4),
         ("gru", ["cv_out/oof_gru_aug2_vol_bo_nbm_hpca8.npz", "cv_out/oof_gru_aug2_vol_bo_nbm_hpca8_s1.npz"], 0.5),
         ("svm", ["cv_out/oof_svm_nbm_hpca8.npz"], 0.1))
OUT = "cv_out/conv_oracle.txt"


def _whiten_vec(e, ref):
    """Vectorised evidence._whiten (history ECDF -> normal scale, linear tails), clipped like bocpd.py."""
    n = len(ref)
    k = np.searchsorted(ref, e)
    w = ndtri((k + 0.5) / (n + 1.0))
    hi, lo = e > ref[-1], e < ref[0]
    w[hi] = ndtri((n + 0.5) / (n + 1.0)) + (e[hi] - ref[-1])
    w[lo] = ndtri(0.5 / (n + 1.0)) - (ref[0] - e[lo])
    return np.clip(w, -8.0, 8.0)


def streams(hist, online):
    """(2, nh + no) whitened streams w, wu: history in-sample (bocpd.py refs), online exactly as bocpd.py."""
    H = VS.fit_history(hist)
    eh = np.asarray(H["eh"], np.float64)
    uh = eh / np.sqrt(H["s2h"])
    hs = np.stack([_whiten_vec(eh, np.sort(eh)), _whiten_vec(uh, np.sort(uh))])
    on = BO.whitened_online(hist, online)
    return np.concatenate([hs, on], axis=1), hs.shape[1]


def conv_series(hist, online):
    """Rows for online steps 0, SUB, 2 SUB, ...: (agg float32 [R, 144], f float32 [R, 672]). Causal."""
    online = np.asarray(online, np.float64)
    X, nh = streams(hist, online)
    rows = nh + np.arange(0, len(online), SUB)       # global positions of the kept online steps
    R = len(rows)
    agg = np.empty((R, len(AGG_NAMES)), np.float32)
    fz = np.empty((R, len(F_NAMES)), np.float32)
    ai = fi = 0
    for c in range(2):
        x = X[c]
        for d in DILS:
            span = 8 * d
            n = len(x)
            lag = np.stack([x[span - (8 - k) * d: n - (8 - k) * d] for k in range(9)], axis=1)   # rows t = span..n-1
            r = lag @ FILT.T                                   # [n - span, 84]
            hr = r[:nh - span]                                 # responses fully inside the history
            qs = np.quantile(hr, QS, axis=0)                   # [2, 84]
            for ti, tl in enumerate(TAILS):
                ind = (r < qs[0]) if tl == "lo" else (r > qs[1])
                p0 = 0.1
                cs = np.concatenate([np.zeros((1, 84)), np.cumsum(ind, axis=0, dtype=np.float64)])
                for W in WINS:
                    # history windows: end positions (in r coordinates) W-1 .. nh-span-1
                    hw = (cs[W:nh - span + 1] - cs[:nh - span + 1 - W]) / W
                    m, s = hw.mean(0), hw.std(0)
                    s = np.maximum(s, 0.5 * np.sqrt(p0 * (1 - p0) / W))
                    e = rows - span + 1                        # cs index one past the current response
                    ppv = (cs[e] - cs[e - W]) / W
                    z = (ppv - m) / s
                    agg[:, ai] = z.mean(1)
                    agg[:, ai + 1] = (z * z).mean(1)
                    agg[:, ai + 2] = np.abs(z).max(1)
                    ai += 3
                    if W == F_WIN and d in F_DILS:
                        fz[:, fi:fi + 84] = z
                        fi += 84
    assert ai == agg.shape[1] and fi == fz.shape[1]
    return agg, fz


def _within_step_rank(v, step):
    import pandas as pd
    return pd.Series(v).groupby(step).rank(pct=True).to_numpy()


def _blend(sid_keep):
    zs = []
    ref = None
    for _, files, wt in BLEND:
        g = None
        for f in files:
            z = np.load(f)
            if ref is None:
                ref = (z["sid"], z["step"], z["y"])
            assert np.array_equal(z["sid"], ref[0]) and np.array_equal(z["step"], ref[1])
            g = z["oof"] / z["oof"].std() if g is None else g + z["oof"] / z["oof"].std()
        zs.append(wt * g / g.std())
    sc = sum(zs)
    m = np.isin(ref[0], sid_keep)
    return sc[m], ref[0][m], ref[1][m], ref[2][m]


def _selftest():
    from sbrt.data import load_train
    S, tau = load_train()
    rng = np.random.default_rng(0)
    for i in rng.choice(len(S), 10, replace=False):
        _, h, o = S[i]
        a, f = conv_series(h, o)
        cut = int(rng.integers(1, len(o))) if len(o) > 1 else 1
        a2, f2 = conv_series(h, o[:cut])
        r = len(a2)
        assert np.array_equal(a[:r], a2) and np.array_equal(f[:r], f2), f"series {i}: prefix rows differ"
        alt = np.r_[o[:cut], rng.normal(0, 3, len(o) - cut)]
        a3, _ = conv_series(h, alt)
        assert np.array_equal(a3[:r], a[:r]), f"series {i}: future leaked"
        assert np.all(np.isfinite(a)) and np.all(np.isfinite(f))
    print("causality OK on 10 series (prefix and altered-future checks)")
    t0 = time.time()
    idx = range(OFFSET, OFFSET + 40)
    mats = [conv_series(S[i][1], S[i][2]) for i in idx]
    print(f"40 series in {time.time() - t0:.1f}s; agg {mats[0][0].shape}, f {mats[0][1].shape}")
    A = np.concatenate([m[0] for m in mats])
    # null sanity: before any break the agg means should sit near 0 (z) / ~1 (z^2)
    step = np.concatenate([np.arange(0, len(S[i][2]), SUB) for i in idx])
    y = np.concatenate([((np.arange(0, len(S[i][2]), SUB) >= tau[i]) & (tau[i] >= 0)) for i in idx])
    for nm in ("agg_w_d1_hi_w128_mz", "agg_w_d1_hi_w128_mz2", "agg_wu_d4_lo_w512_mz2"):
        j = AGG_NAMES.index(nm)
        print(f"  {nm:26s} pre-break mean {A[~y, j].mean():+.3f} sd {A[~y, j].std():.3f}   "
              f"post-break mean {A[y, j].mean():+.3f}")
    sid_keep = np.arange(OFFSET, OFFSET + 40)
    sc, sid, st, yy = _blend(sid_keep)
    assert len(sc) == len(A), f"row count {len(A)} vs OOF {len(sc)}"
    assert np.array_equal(st, step) and np.array_equal(yy.astype(bool), y)
    print(f"rows align with the OOF files ({len(sc)} rows); smoke blend TS-AUC {ts_auc(sc, yy, st):.4f}")
    print("self-test OK")


def main(n_series=2000, threads=12):
    import lightgbm as lgb
    from joblib import Parallel, delayed
    from sbrt import disttest as DT
    from sbrt.data import load_train

    lines = []

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    S, tau = load_train()
    idx = np.arange(OFFSET, OFFSET + n_series)
    t0 = time.time()
    mats = Parallel(n_jobs=threads)(delayed(conv_series)(S[i][1], S[i][2]) for i in idx)
    out(f"conv features: {n_series} series in {time.time() - t0:.0f}s")
    t0 = time.time()
    dmats = Parallel(n_jobs=threads)(delayed(DT.run_series_dist)(S[i][1], S[i][2], SUB) for i in idx)
    out(f"--dist reference features in {time.time() - t0:.0f}s")
    A = np.concatenate([m[0] for m in mats])
    Fz = np.concatenate([m[1] for m in mats])
    D = np.concatenate(dmats)
    del mats, dmats
    sc, sid, step, y = _blend(idx)
    y = y.astype(np.int8)
    assert len(sc) == len(A) == len(D), "rows do not align with the OOF files"

    # slices by the actual post-break log variance ratio (error_breakdown.py definition)
    dlv = np.full(len(S), np.nan)
    for i in idx:
        _, h, o = S[i]
        t = tau[i]
        if t >= 0 and len(o) - t >= 30:
            dlv[i] = np.log(o[t:].var() / h.var())
    slices = {"UP": dlv >= 0.3, "FLAT": np.abs(dlv) < 0.1, "DOWN": dlv <= -0.1}
    masks = {"ALL": np.ones(len(y), bool)}
    for k, b in slices.items():
        masks[k] = (y == 0) | ((y == 1) & b[sid])
        out(f"slice {k:4s}: {int(b[idx].sum())} break series")

    def aucs(v):
        return {k: ts_auc(v[m], y[m], step[m]) for k, m in masks.items()}

    hdr = "  ".join(f"{k:>6s}" for k in masks)
    out(f"\n== reference single scores (TS-AUC on {hdr}) ==")
    ab = aucs(sc)
    out(f"{'blend (current best)':30s} " + "  ".join(f"{ab[k]:.4f}" for k in masks))
    dres = []
    for j, nm in enumerate(DT.FEATURE_NAMES):
        a = aucs(D[:, j])
        dres.append((max(abs(a['FLAT'] - 0.5), abs(a['ALL'] - 0.5)), nm, a))
    dres.sort(reverse=True)
    for _, nm, a in dres[:5]:
        out(f"{nm:30s} " + "  ".join(f"{a[k]:.4f}" for k in masks))

    out(f"\n== agg_ single-feature TS-AUC (top 25 by distance from 0.5 on FLAT; columns {hdr}) ==")
    res = []
    for j, nm in enumerate(AGG_NAMES):
        a = aucs(A[:, j])
        res.append((abs(a["FLAT"] - 0.5), j, nm, a))
    res.sort(reverse=True)
    for _, _, nm, a in res[:25]:
        out(f"{nm:30s} " + "  ".join(f"{a[k]:.4f}" for k in masks))
    allfar = np.array([abs(r[3]["ALL"] - 0.5) for r in res])
    out(f"agg_ |AUC - 0.5| on ALL: median {np.median(allfar):.4f}, max {allfar.max():.4f}")

    out("\n== f_ per-filter screen (W = 128; best of 672 is optimistic) ==")
    fr = []
    for j, nm in enumerate(F_NAMES):
        m = masks["FLAT"]
        fr.append((abs(ts_auc(Fz[m, j], y[m], step[m]) - 0.5), nm))
    fr.sort(reverse=True)
    far = np.array([r[0] for r in fr])
    out(f"FLAT |AUC - 0.5|: median {np.median(far):.4f}, 90th pct {np.quantile(far, 0.9):.4f}, max {far.max():.4f}")
    out("top 10: " + ", ".join(f"{nm} {d + 0.5:.4f}" for d, nm in fr[:10]))
    del Fz

    out("\n== redundancy: within-step rank correlation of the top 10 agg_ (FLAT order) ==")
    rb = _within_step_rank(sc, step)
    rD = np.stack([_within_step_rank(D[:, j], step) for j in range(D.shape[1])], 1)
    for _, j, nm, _ in res[:10]:
        ra = _within_step_rank(A[:, j], step)
        cb = np.corrcoef(ra, rb)[0, 1]
        cd = np.array([np.corrcoef(ra, rD[:, k])[0, 1] for k in range(rD.shape[1])])
        k = int(np.nanargmax(np.abs(cd)))
        out(f"{nm:30s} blend {cb:+.3f}   max |dist| {cd[k]:+.3f} ({DT.FEATURE_NAMES[k]})")
    del rD

    out("\n== INCREMENTAL (the gate): LightGBM [blend, step] vs [blend, step, 144 agg_], 3 x 5 grouped folds ==")
    base = np.column_stack([sc, step]).astype(np.float32)
    full = np.column_stack([base, A])
    prm = dict(objective="binary", learning_rate=0.05, num_leaves=15, min_data_in_leaf=1000,
               feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0,
               num_threads=threads, verbose=-1)
    dl = {k: [] for k in ("ALL", "FLAT")}
    for rep in range(3):
        perm = np.random.default_rng(100 + rep).permutation(idx)
        fold_of = dict(zip(perm, np.arange(len(perm)) % 5))
        fo = np.array([fold_of[s] for s in sid])
        for f in range(5):
            tr, te = fo != f, fo == f
            p = []
            for X in (base, full):
                bst = lgb.train(dict(prm, seed=rep * 10 + f), lgb.Dataset(X[tr], y[tr]), num_boost_round=300)
                p.append(bst.predict(X[te], raw_score=True))
            for k in dl:
                m = masks[k][te]
                a0 = ts_auc(p[0][m], y[te][m], step[te][m])
                a1 = ts_auc(p[1][m], y[te][m], step[te][m])
                dl[k].append(a1 - a0)
            out(f"  rep {rep} fold {f}: ALL {dl['ALL'][-1]:+.4f}   FLAT {dl['FLAT'][-1]:+.4f}")
    verdict = []
    for k, bar in (("ALL", 0.001), ("FLAT", 0.005)):
        d = np.array(dl[k])
        se = d.std(ddof=1) / np.sqrt(len(d))
        ok = d.mean() >= bar and d.mean() > 2 * se
        verdict.append(ok)
        out(f"{k:4s}: {d.mean():+.4f} +- {se:.4f}, {int((d > 0).sum())}/{len(d)} folds > 0   "
            f"(bar >= +{bar} and > 2 SE: {'PASS' if ok else 'no'})")
    out(f"\nVERDICT: {'PASS -> build streaming --conv for cv.py' if any(verdict) else 'CLOSED'}")
    with open(OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"written {OUT}")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        a = [int(v) for v in sys.argv[1:3]]
        main(*a)
