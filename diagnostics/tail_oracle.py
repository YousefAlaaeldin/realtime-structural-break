"""Is there break information in the TAILS beyond their rate? (oracle, break position known; section 33)

The 2026-10-20 literature sweep (reports/measurement_coverage_2026-10-20.md) found every measurement axis of hctsa /
catch22 / tsfresh / TSFEL already in production or closed, except three tail-centred families. The models see tail
RATES (flags at 1-4 sigma, AD on e and u) but not how extremes CLUSTER in time, how the tails DEPEND on each other
across lags, or the tail SHAPE at fixed scale. This oracle checks all three.

Same oracle setup and same baseline as longrange_oracle.py / info_oracle.py, so the numbers are comparable:
    break series     segment = online[tau : tau + L]
    no-break series  segment = online[s : s + L], s uniform over the valid starts
    L = 100, 300, 600 (600: only series with enough online points)
Every statistic is computed on the segment AND on history windows of the same length L (stride L/2) and taken as
segment minus the history-window mean (estimator bias cancels).

All three families are SCALE-FREE by construction: thresholds and quantiles are the segment's (or the history
window's) OWN empirical ones, and the Hill estimate only measures shape. A pure variance change does not move them;
the question is whether anything in the tails changes AT FIXED VARIANCE (our weak slice).
Caveat for v: the EWMA enters a break segment carrying the pre-break volatility, so for its first ~20-40 points v is
inflated / deflated and its extremes cluster at the segment start. v's clustering columns are therefore partly
variance detectors on ALL rows; the variance-matched slice is the clean read for v.
Two variables:
    e   AR(10) residual (history-fitted), as in the baseline
    v   e divided by a causal EWMA volatility (half-life 20, run over history + online): removes ordinary volatility
        clustering, so clustering left in v's extremes is not "variance in disguise"

Family tl_ (per variable):
  tl_eiv_{v}     intervals estimator of the extremal index (Ferro-Segers) of the top-10% |x| exceedances
                 (1 = isolated extremes, < 1 = extremes arrive in clusters)
  tl_gapcv_{v}   coefficient of variation of the gaps between those exceedances (1 = Poisson, > 1 = clustered)
  tl_follow_{v}  P(exceedance at t+1 | exceedance at t) - 0.1 (cluster tendency at lag 1)
  tl_qlo1_{v}, tl_qhi1_{v}   quantilogram at lag 1 for the lower / upper 10% quantile (tail dependence)
  tl_qlo_{v},  tl_qhi_{v}    sum of the quantilogram over lags 1-5 (lower / upper)
  tl_hill_{v}    Hill tail index of |x| on the top 10% (min 5 points)
  tl_hasym_{v}   Hill of the upper tail minus Hill of the lower tail (tail asymmetry)
Baseline b_ / ctx_ exactly as longrange_oracle.py: variance, difference variance, ac 1/2/5, residual ac1, kurtosis,
skew, 3-sigma tails, mean, volatility clustering, KS of e, history context. Kurtosis, tail3 and vc1 are the closest
existing statistics; the rank-correlation column shows how much a new statistic just repeats them.

Reports per L: single-statistic AUC (magnitude of the difference) on all series and on the VARIANCE-MATCHED slice
(|log var ratio of e| < 0.1), the largest rank correlation with a baseline statistic on all rows AND on the matched
slice only (section 28 lesson: exceedance statistics collapsed into variance detectors; on the slice b_lv is flat by
construction, so b_lvd / b_tail3 / b_kurt are the informative matches), and the INCREMENTAL AUC of a small LightGBM,
baseline vs baseline + X (3 x 5-fold, paired over 15 folds) for X = the whole tl_ family and each subfamily alone:
EXC (eiv / gapcv / follow, 6 cols), QG (quantilogram, 8 cols), HILL (hill / hasym, 4 cols). Subfamilies are tested
separately because 18 columns on a few thousand rows can bury one real signal under overfitting noise.

KILL RULE (written before any run; same as section 34 / longrange_oracle.py):
  WHOLE family: PASS if, at some L, the incremental gain on all series is > +0.0066 AND > 3 SE, or on the
  variance-matched slice >= +0.005 AND > 2 SE.
  SUBFAMILY (three extra looks, so stricter): PASS if, at some L, a subfamily alone gains on all series > +0.0066 AND
  > 3 SE, or on the variance-matched slice >= +0.005 AND > 3 SE.
  Overall PASS if the whole family or any subfamily passes. (+0.0066 = the MMD result that passed this oracle and then added nothing as streaming
  features.) PASS -> build streaming tl_ features and gate them with cv.py like the others. FAIL -> close; that
  completes the literature list of "properties of the points".
  Expectation (stated before running): FAIL. The prior is about +0.000 to +0.001 on the blend: tail rates and AD
  already see much of the tails, volatility clustering at lags 2-20 had no signal (section 20 of the old file), and
  the history-calibrated exceedance oracle (section 28) collapsed into variance detectors. Tail statistics need many
  points, so any signal should show at L = 300 / 600 rather than at L = 100.

Run:  python diagnostics/tail_oracle.py [n_series] [threads]     (default all 10,000 series, 8 processes)
Out:  cv_out/tail_oracle.txt
"""
import os
import sys
import time

import numpy as np
from scipy.signal import lfilter
from scipy.stats import kurtosis, ks_2samp, rankdata, skew

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "diagnostics"))
os.chdir(HERE)

from info_oracle import _ac, _auc, _corr  # noqa: E402  (same helpers as section 20)

AR = 10
LENS = (100, 300, 600)
VAR_SLICE = 0.1
P_EXC = 0.10          # top share of |x| treated as extremes
Q_LO, Q_HI = 0.10, 0.90
QLAGS = 5
HL_VOL = 20           # half-life of the EWMA volatility used for v


def _exc_stats(x):
    """Extremal index (intervals estimator), gap CV and follow probability of the top-P_EXC |x| exceedances."""
    a = np.abs(x)
    n = len(a)
    k = max(int(round(P_EXC * n)), 5)
    thr = np.partition(a, n - k)[n - k]
    idx = np.flatnonzero(a >= thr)
    if len(idx) < 4:
        return np.nan, np.nan, np.nan
    T = np.diff(idx).astype(np.float64)
    N = len(idx)
    if T.max() <= 2:
        ei = 2.0 * T.sum() ** 2 / ((N - 1) * (T ** 2).sum())
    else:
        den = ((N - 1) * ((T - 1) * (T - 2)).sum())
        ei = 2.0 * ((T - 1).sum()) ** 2 / den if den > 0 else np.nan
    ei = min(ei, 1.0) if np.isfinite(ei) else np.nan
    gapcv = T.std() / T.mean()
    exc = a >= thr
    follow = exc[1:][exc[:-1]].mean() - P_EXC
    return float(ei), float(gapcv), float(follow)


def _quantilogram(x, q, lags):
    """Linton-Whang quantilogram: corr of psi_q(x_t) = 1{x_t < q-quantile} - q across lags."""
    thr = np.quantile(x, q)
    psi = (x < thr).astype(np.float64) - q
    d = (psi ** 2).sum()
    if d <= 0:
        return [np.nan] * len(lags)
    return [float((psi[k:] * psi[:-k]).sum() / d) for k in lags]


def _hill(a):
    """Hill tail index (mean log excess) of the top P_EXC of a (>= 0), at least 5 points."""
    a = a[a > 0]
    n = len(a)
    k = max(int(round(P_EXC * n)), 5)
    if n < k + 2:
        return np.nan
    s = np.sort(a)[::-1]
    return float(np.mean(np.log(s[:k])) - np.log(s[k]))


def tl_stats(e, v):
    out = {}
    for name, x in (("e", e), ("v", v)):
        ei, gcv, fol = _exc_stats(x)
        out[f"tl_eiv_{name}"] = ei
        out[f"tl_gapcv_{name}"] = gcv
        out[f"tl_follow_{name}"] = fol
        lags = range(1, QLAGS + 1)
        lo = _quantilogram(x, Q_LO, lags)
        hi = _quantilogram(-x, 1 - Q_HI, lags)   # upper tail: x > 0.9-quantile  <=>  -x < its 0.1-quantile
        out[f"tl_qlo1_{name}"] = lo[0]
        out[f"tl_qhi1_{name}"] = hi[0]
        out[f"tl_qlo_{name}"] = float(np.sum(lo))
        out[f"tl_qhi_{name}"] = float(np.sum(hi))
        xc = x - np.median(x)
        out[f"tl_hill_{name}"] = _hill(np.abs(xc))
        hu, hl = _hill(xc[xc > 0]), _hill(-xc[xc < 0])
        out[f"tl_hasym_{name}"] = hu - hl if np.isfinite(hu) and np.isfinite(hl) else np.nan
    return out


def series_rows(i, h, o, t, seed):
    rng = np.random.default_rng(seed)
    h = np.asarray(h, np.float64)
    o = np.asarray(o, np.float64)
    mu, sd = h.mean(), h.std() + 1e-12
    zh, zo = (h - mu) / sd, (o - mu) / sd
    nh = len(h)
    Xl = np.column_stack([zh[AR - 1 - k: nh - 1 - k] for k in range(AR)])
    coef = np.linalg.lstsq(Xl, zh[AR:], rcond=None)[0]
    e = lfilter(np.r_[1.0, -coef], [1.0], np.r_[zh, zo])
    se = max(float(e[2 * AR:nh].std()), 1e-8)
    e = np.clip(e / se, -10, 10)
    # causal EWMA volatility (variance known up to t-1), started at the history variance (= 1)
    lam = 0.5 ** (1.0 / HL_VOL)
    e2 = np.minimum(e ** 2, 25.0)
    s2 = lfilter([1 - lam], [1.0, -lam], e2, zi=[lam * 1.0])[0]
    s2 = np.r_[1.0, s2[:-1]]
    vv = e / np.sqrt(np.maximum(s2, 1e-4))
    eh, eo = e[2 * AR:nh], e[nh:]
    vh, vo = vv[2 * AR:nh], vv[nh:]
    zh2 = zh[2 * AR:]

    H = dict(ac={k: _ac(zh, k) for k in (1, 2, 5)}, eac1=_ac(eh, 1), kurt=kurtosis(eh), skew=skew(eh),
             tail3=float(np.mean(np.abs(eh) > 3)), vc1=_corr(np.abs(eh[:-1]), np.abs(eh[1:])),
             lvd=np.log(np.diff(zh).var()))
    ctx = dict(ctx_kurt=kurtosis(zh), ctx_rho1=_ac(zh, 1), ctx_logn=np.log(nh))
    out = []
    for L in LENS:
        if t >= 0:
            if len(o) - t < L:
                continue
            st, lab = t, 1
        else:
            if len(o) < L:
                continue
            st, lab = int(rng.integers(0, len(o) - L + 1)), 0
        if len(zh2) < L:
            continue
        zs, es, vs = zo[st:st + L], eo[st:st + L], vo[st:st + L]
        r = dict(ctx)
        r["b_lv"] = float(np.log(es.var() / eh.var()))
        r["b_lvd"] = float(np.log(np.diff(zs).var()) - H["lvd"])
        for k in (1, 2, 5):
            r[f"b_ac{k}"] = _ac(zs, k) - H["ac"][k]
        r["b_eac1"] = _ac(es, 1) - H["eac1"]
        r["b_kurt"] = float(kurtosis(es) - H["kurt"])
        r["b_skew"] = float(skew(es) - H["skew"])
        r["b_tail3"] = float(np.mean(np.abs(es) > 3) - H["tail3"])
        r["b_mean"] = float(zs.mean() * np.sqrt(L))
        r["b_vc1"] = _corr(np.abs(es[:-1]), np.abs(es[1:])) - H["vc1"]
        r["b_ks"] = float(ks_2samp(es, eh).statistic)
        # tail family: segment vs the mean over history windows of the same length (bias cancels)
        seg = tl_stats(es, vs)
        stride = max(L // 2, 1)
        starts = range(0, len(eh) - L + 1, stride)
        hw = [tl_stats(eh[a:a + L], vh[a:a + L]) for a in starts]
        for k, val in seg.items():
            vals = np.array([w[k] for w in hw], np.float64)
            vals = vals[np.isfinite(vals)]
            ref = vals.mean() if len(vals) else np.nan
            r[k] = val - ref if np.isfinite(val) and np.isfinite(ref) else np.nan
        out.append((L, lab, r))
    return out


def main():
    from joblib import Parallel, delayed
    from sklearn.model_selection import StratifiedKFold
    import lightgbm as lgb

    from sbrt.data import load_train

    n_series = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    threads = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    log = open(os.path.join("cv_out", "tail_oracle.txt" if not n_series else "tail_oracle_smoke.txt"), "w",
               encoding="utf-8")

    def say(s=""):
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    S, tau = load_train()
    if n_series:
        S, tau = S[:n_series], tau[:n_series]
    t0 = time.time()
    res = Parallel(n_jobs=threads, backend="loky", batch_size=8)(
        delayed(series_rows)(i, s[1], s[2], int(tau[i]), 1000 + i) for i, s in enumerate(S))
    say(f"statistics for {len(S)} series in {time.time() - t0:.0f}s")

    verdict = []
    for L in LENS:
        rows = [(lab, r) for rr in res for (LL, lab, r) in rr if LL == L]
        if len(rows) < 100:
            say(f"\nL = {L}: only {len(rows)} segments, skipped")
            continue
        y = np.array([lab for lab, _ in rows])
        names = list(rows[0][1].keys())
        X = np.array([[r.get(n, np.nan) for n in names] for _, r in rows], dtype=np.float64)
        col = {n: j for j, n in enumerate(names)}
        vm = np.abs(X[:, col["b_lv"]]) < VAR_SLICE
        base = [n for n in names if n.startswith(("b_", "ctx_"))]
        bnames = [n for n in base if n.startswith("b_")]
        fam = [n for n in names if n.startswith("tl_")]
        say(f"\n{'=' * 100}\nL = {L}: {int(y.sum())} break segments, {int((1 - y).sum())} no-break segments; "
            f"variance-matched slice (|log var ratio| < {VAR_SLICE}): {int(y[vm].sum())} / {int((1 - y[vm]).sum())}")
        Xf = np.nan_to_num(X)
        R = np.column_stack([rankdata(Xf[:, col[n]]) for n in bnames])
        R = (R - R.mean(0)) / (R.std(0) + 1e-12)
        Rv = np.column_stack([rankdata(Xf[vm][:, col[n]]) for n in bnames])
        Rv = (Rv - Rv.mean(0)) / (Rv.std(0) + 1e-12)
        say(f"\n  single-statistic AUC of |difference|          all    var-matched   (signed: all)   "
            f"max |rank corr| w/ baseline: all rows | var-matched rows")
        for grp, cols in (("baseline", bnames), ("tl_ (new)", fam)):
            say(f"  -- {grp}")
            for n in cols:
                x = Xf[:, col[n]]
                a, av, asg = _auc(np.abs(x), y), _auc(np.abs(x[vm]), y[vm]), _auc(x, y)
                if grp == "baseline":
                    say(f"    {n:14s} {a:.4f}   {av:.4f}       ({asg:.4f})")
                else:
                    rr = rankdata(x)
                    rr = (rr - rr.mean()) / (rr.std() + 1e-12)
                    cc = np.abs(R.T @ rr) / len(rr)
                    j = int(np.argmax(cc))
                    rv = rankdata(x[vm])
                    rv = (rv - rv.mean()) / (rv.std() + 1e-12)
                    cv_ = np.abs(Rv.T @ rv) / len(rv)
                    jv = int(np.argmax(cv_))
                    say(f"    {n:14s} {a:.4f}   {av:.4f}       ({asg:.4f})    {cc[j]:.2f} {bnames[j]:8s} | "
                        f"{cv_[jv]:.2f} {bnames[jv]}")

        params = dict(objective="binary", learning_rate=0.05, num_leaves=15, min_data_in_leaf=50,
                      feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1,
                      deterministic=True, force_row_wise=True, num_threads=threads, seed=0)

        def cv_auc(cols):
            Xc = X[:, [col[n] for n in cols]]      # NaN allowed: LightGBM handles missing values
            out = []
            for seed in (0, 1, 2):
                oof = np.zeros(len(y))
                for tr, va in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xc, y):
                    m = lgb.train(params, lgb.Dataset(Xc[tr], y[tr]), 300)
                    oof[va] = m.predict(Xc[va])
                    out.append((_auc(oof[va], y[va]), _auc(oof[va][vm[va]], y[va][vm[va]])))
            return np.array(out)

        b = cv_auc(base)
        subs = [("tl_ (all)", fam, False),
                ("EXC", [n for n in fam if n.startswith(("tl_eiv_", "tl_gapcv_", "tl_follow_"))], True),
                ("QG", [n for n in fam if n.startswith(("tl_qlo", "tl_qhi"))], True),
                ("HILL", [n for n in fam if n.startswith(("tl_hill_", "tl_hasym_"))], True)]
        say(f"\n  incremental value (LightGBM, 3 x 5-fold; paired vs baseline over 15 folds)")
        say(f"    baseline ({len(base)} stats):           all {b[:, 0].mean():.4f}   var-matched {b[:, 1].mean():.4f}")
        for lab_, cols, is_sub in subs:
            c = cv_auc(base + cols)
            d = c - b
            se = d.std(0, ddof=1) / np.sqrt(len(d))
            say(f"    + {lab_:10s} ({len(cols):2d} stats):   all {c[:, 0].mean():.4f} ({d[:, 0].mean():+.4f} +- "
                f"{se[0]:.4f}, {int((d[:, 0] > 0).sum())}/15)   var-matched {c[:, 1].mean():.4f} "
                f"({d[:, 1].mean():+.4f} +- {se[1]:.4f}, {int((d[:, 1] > 0).sum())}/15)")
            k_vm = 3 if is_sub else 2
            p_all = d[:, 0].mean() > 0.0066 and d[:, 0].mean() > 3 * se[0]
            p_vm = d[:, 1].mean() >= 0.005 and d[:, 1].mean() > k_vm * se[1]
            verdict.append((L, lab_, p_all, p_vm))

    say("\nVERDICT (kill rule in the docstring: all > +0.0066 and > 3 SE, or var-matched >= +0.005 and > 2 SE "
        "(subfamilies: > 3 SE), at some L)")
    for L, lab_, pa, pv in verdict:
        say(f"  L = {L} {lab_:10s}: all {'PASS' if pa else 'fail'}   var-matched {'PASS' if pv else 'fail'}")
    ok = any(pa or pv for _, _, pa, pv in verdict)
    say(f"  => {'PASS: build streaming tl_ features and gate them' if ok else 'FAIL: no usable tail information beyond the baseline'}")
    if n_series:
        say("  (subset run: too few series to read; use the full run)")
    log.close()


if __name__ == "__main__":
    main()
