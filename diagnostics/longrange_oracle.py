"""Is there break information in the SLOW, LONG-RANGE structure of a series? (oracle, break position known)

Everything the models measure about dependence stops at about lag 16 (lag products up to 10, variograms up to 16,
AR(10) residuals). This checks the one family of "properties of the points" never tested: multi-scale variance and
long memory. A break that leaves point-to-point behaviour alone but makes the series wander more slowly, or remember
its past longer, would be invisible to the current features.

Same oracle setup and same baseline as info_oracle.py (section 20), so the numbers are comparable:
    break series     segment = online[tau : tau + L]
    no-break series  segment = online[s : s + L], s uniform over the valid starts
    L = 100, 300, 600 (600: only series with enough online points; fewer of them)
Every statistic is computed on the segment AND on windows of the same length L from the series' own history
(stride L/2), and the difference is taken against the history-window mean, so the small-sample bias of each
estimator (which depends on L) cancels. Variables: z (the series) and e (AR(10) residual, history-fitted).

Family ms_ (all scale-free: a pure variance change does not move them):
  ms_vr{m}_{v}   log variance ratio at scale m (overlapping sums of m points: var(sum) / (m var); 1 = no memory,
                 > 1 = persistence, < 1 = mean reversion); m = 4, 8, 16, 32, 64 (m <= L/5)
  ms_dfa_{v}     DFA-1 scaling exponent (long-memory / Hurst-type), scales 4 .. L/5
  ms_lf_{v}      share of periodogram power in the lowest 10% of frequencies (log)
  ms_ac{k}_z     autocorrelation of z at lag 20 and 50 (lag <= L/5)
  ms_q_e         sum of squared residual autocorrelations at lags 11-50 (dependence the AR(10) did not remove)
Baseline b_ / ctx_ exactly as info_oracle.py (variance, difference variance, ac 1/2/5, residual ac1, kurtosis, skew,
3-sigma tails, mean, volatility clustering, KS of e, history context).

Reports per L: single-statistic AUC (magnitude of the difference) on all series and on the VARIANCE-MATCHED slice,
the largest rank correlation with a baseline statistic, and the INCREMENTAL AUC of a small LightGBM, baseline vs
baseline + ms_ (3 x 5-fold, paired over 15 folds).

KILL RULE (written before the full run):
  PASS only if, at some L, the incremental gain on all series is > +0.0066 AND > 3 SE, or on the variance-matched
  slice >= +0.005 AND > 2 SE. The bar +0.0066 is the MMD result at L = 300 (section 20), which passed this oracle and
  then added nothing as streaming features (section 21); a family that does not beat that precedent is not worth a
  streaming build. PASS -> build streaming ms_ features (windows / scans) and gate them like the others.
  Expectation (stated before running): low to moderate. Section 15.2 found the misses are mostly breaks with
  unchanged variance and that the variance-matched slice is near-undetectable even with tau known (0.53-0.59);
  long-range statistics need many points, so any signal should show at L = 300/600 rather than L = 100.

Run:  python diagnostics/longrange_oracle.py [n_series] [threads]     (default all 10,000 series, 8 processes)
Out:  cv_out/longrange_oracle.txt
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
SCALES = (4, 8, 16, 32, 64)


def _vr(x, m):
    """Lo-MacKinlay variance ratio with overlapping sums of m points (log)."""
    x = x - x.mean()
    v = x.var()
    if v <= 0 or len(x) < 5 * m:
        return np.nan
    c = np.r_[0.0, np.cumsum(x)]
    s = c[m:] - c[:-m]
    return float(np.log(max(s.var() / (m * v), 1e-6)))


def _dfa(x):
    """DFA-1 exponent over log-spaced scales 4 .. len/5."""
    y = np.cumsum(x - x.mean())
    n = len(y)
    hi = n // 5
    if hi < 8:
        return np.nan
    scales = np.unique(np.round(np.exp(np.linspace(np.log(4), np.log(hi), 8))).astype(int))
    F = []
    for s in scales:
        k = n // s
        Y = y[:k * s].reshape(k, s)
        t = np.arange(s) - (s - 1) / 2.0
        tt = (t * t).sum()
        b = (Y * t).sum(1) / tt
        a = Y.mean(1)
        res = Y - a[:, None] - b[:, None] * t
        F.append(np.sqrt((res ** 2).mean()))
    F = np.array(F)
    if np.any(F <= 0):
        return np.nan
    return float(np.polyfit(np.log(scales), np.log(F), 1)[0])


def _lf(x):
    p = np.abs(np.fft.rfft(x - x.mean())[1:]) ** 2
    if p.sum() <= 0:
        return np.nan
    k = max(1, int(round(0.1 * len(p))))
    return float(np.log(max(p[:k].sum() / p.sum(), 1e-9)))


def _q(e, L):
    if L < 5 * 50:
        return np.nan
    return float(L * sum(_ac(e, k) ** 2 for k in range(11, 51)))


def ms_stats(zs, es, L):
    out = {}
    for v, x in (("z", zs), ("e", es)):
        for m in SCALES:
            if m <= L // 5:
                out[f"ms_vr{m}_{v}"] = _vr(x, m)
        out[f"ms_dfa_{v}"] = _dfa(x)
        out[f"ms_lf_{v}"] = _lf(x)
    for k in (20, 50):
        if k <= L // 5:
            out[f"ms_ac{k}_z"] = _ac(zs, k)
    out["ms_q_e"] = _q(es, L)
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
    eh, eo = e[2 * AR:nh], e[nh:]
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
        zs, es = zo[st:st + L], eo[st:st + L]
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
        # long-range family: segment vs the mean over history windows of the same length (bias cancels)
        seg = ms_stats(zs, es, L)
        stride = max(L // 2, 1)
        starts = range(0, len(zh2) - L + 1, stride)
        hw = [ms_stats(zh2[a:a + L], eh[a:a + L], L) for a in starts]
        for k, v in seg.items():
            vals = np.array([w[k] for w in hw], np.float64)
            vals = vals[np.isfinite(vals)]
            ref = vals.mean() if len(vals) else np.nan   # e.g. ms_q_e is undefined for L = 100: stays NaN
            r[k] = v - ref if np.isfinite(v) and np.isfinite(ref) else np.nan
        out.append((L, lab, r))
    return out


def main():
    from joblib import Parallel, delayed
    from sklearn.model_selection import StratifiedKFold
    import lightgbm as lgb

    from sbrt.data import load_train

    n_series = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    threads = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    log = open(os.path.join("cv_out", "longrange_oracle.txt"), "w", encoding="utf-8")

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
        fam = [n for n in names if n.startswith("ms_")]
        say(f"\n{'=' * 100}\nL = {L}: {int(y.sum())} break segments, {int((1 - y).sum())} no-break segments; "
            f"variance-matched slice (|log var ratio| < {VAR_SLICE}): {int(y[vm].sum())} / {int((1 - y[vm]).sum())}")
        Xf = np.nan_to_num(X)
        R = np.column_stack([rankdata(Xf[:, col[n]]) for n in bnames])
        R = (R - R.mean(0)) / (R.std(0) + 1e-12)
        say(f"\n  single-statistic AUC of |difference|          all    var-matched   (signed: all)   max |rank corr| w/ baseline")
        for grp, cols in (("baseline", bnames), ("ms_ (new)", fam)):
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
                    say(f"    {n:14s} {a:.4f}   {av:.4f}       ({asg:.4f})    {cc[j]:.2f} {bnames[j]}")

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
        c = cv_auc(base + fam)
        d = c - b
        se = d.std(0, ddof=1) / np.sqrt(len(d))
        say(f"\n  incremental value (LightGBM, 3 x 5-fold; paired vs baseline over 15 folds)")
        say(f"    baseline ({len(base)} stats):        all {b[:, 0].mean():.4f}   var-matched {b[:, 1].mean():.4f}")
        say(f"    + ms_ ({len(fam):2d} stats):         all {c[:, 0].mean():.4f} ({d[:, 0].mean():+.4f} +- {se[0]:.4f}, "
            f"{int((d[:, 0] > 0).sum())}/15)   var-matched {c[:, 1].mean():.4f} ({d[:, 1].mean():+.4f} +- "
            f"{se[1]:.4f}, {int((d[:, 1] > 0).sum())}/15)")
        p_all = d[:, 0].mean() > 0.0066 and d[:, 0].mean() > 3 * se[0]
        p_vm = d[:, 1].mean() >= 0.005 and d[:, 1].mean() > 2 * se[1]
        verdict.append((L, p_all, p_vm))

    say("\nVERDICT (kill rule in the docstring: all > +0.0066 and > 3 SE, or var-matched >= +0.005 and > 2 SE, at some L)")
    for L, pa, pv in verdict:
        say(f"  L = {L}: all {'PASS' if pa else 'fail'}   var-matched {'PASS' if pv else 'fail'}")
    ok = any(pa or pv for _, pa, pv in verdict)
    say(f"  => {'PASS: build streaming ms_ features and gate them' if ok else 'FAIL: no usable long-range information beyond the baseline'}")
    if n_series:
        say("  (subset run: too few series to read; use the full run)")
    log.close()


if __name__ == "__main__":
    main()
