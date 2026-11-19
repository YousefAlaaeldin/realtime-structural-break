"""Does the DERIVATIVE of a series (dz = z_t - z_{t-1}, d2z = dz_t - dz_{t-1}) carry break information the models
do not already have?  (oracle, break position known; owner's idea, 2026-10-16)

Same oracle setup and baseline as info_oracle.py (section 20) and longrange_oracle.py (section 34), so the numbers
are comparable:
    break series     segment = online[tau : tau + L]
    no-break series  segment = online[s : s + L], s uniform over the valid starts   (the placebo)
    every statistic is "segment vs the series' own history"; L = 50, 100, 300

Two families, kept apart on purpose:
  q_  QUADRATIC derivative statistics. Each is a fixed linear mix of the autocovariances g0, g1, g2 of z
      (var dz = 2g0 - 2g1, var d2z = 6g0 - 8g1 + 2g2, cov(dz_t, dz_t-1) = 2g1 - g0 - g2), which the baseline and the
      streaming kernel already track (z2, lag1prod, prod2, vg1..vg16). Expected incremental value: about zero.
      This family checks that claim.
        q_lvz      log var(z) segment / history              ("var x"; the baseline's b_lv is on the AR residual)
        q_lvd2     log var(d2z) ratio                        ("d2x")
        q_acd1     lag-1 autocorrelation of dz, change
      (var dz itself is the baseline's b_lvd.)
  d_  NON-QUADRATIC derivative statistics, absent from the models. For a Gaussian series each of these is again a
      function of the autocorrelations (e.g. mean|dz| = sqrt(2/pi) sd(dz)), so they can only add information by
      (a) being robust to the spikes of heavy-tailed series (less estimator noise) or (b) catching non-Gaussian
      changes in how the series moves.
        d_mad1     log mean|dz| ratio                        (robust roughness)
        d_mad2     log mean|d2z| ratio                       (robust "d2x")
        d_tail95   share of |dz| above the history's 95th percentile of |dz|, minus 0.05
        d_tail99   same at the 99th percentile, minus 0.01   ("|dx| tail rate", scale set by the history)
        d_tp       turning-point rate: share of t with sign(dz_t) != sign(dz_t-1), change (i.i.d. noise: 2/3)
        d_rrob     log( mean|dz| / mean|z - median z| ), change   (scale-free robust roughness, a robust 1 - rho1)
        d_ks       KS distance between the dz of the segment and the dz of the history

Reported per L, on all series, on the VARIANCE-MATCHED slice (|b_lv| < 0.1, the weak slice), and per HISTORY
GROUP (owner's request): lag-1 autocorrelation |rho1| < 0.3 / 0.3-0.6 / > 0.6 (74% / 20% / 6% of series) and
excess kurtosis < 3 / >= 3. The extra column |rho1| > 0.9 (near unit root, about 1% of series: 45/56 segments at
L = 50, 21/42 at L = 300) is shown for the plotted example only; it is too small to judge.
  1. single-statistic AUC, break vs placebo (magnitude |d| for the signed differences; KS is already a magnitude)
  2. INCREMENTAL AUC of a small LightGBM, baseline vs baseline + q_ / + d_ / + both, 3 x 5-fold, paired over the
     15 folds; the per-group numbers are the same out-of-fold predictions scored inside each group.

KILL RULE (written before the full run; same bar as section 34):
  PASS only if, at some L, + d_ gains on all series > +0.0066 AND > 3 SE, or on the variance-matched slice
  >= +0.005 AND > 2 SE. A group-only gain (>= +0.01 and > 3 SE inside one history group with >= 200 segments per
  class) is reported as a GROUP CANDIDATE, not a pass: with 6 groups x 3 L tested, one such hit can be luck.
  + q_ is expected to be about 0; a clear q_ gain would mean the baseline misses a quadratic combination.
  Expectation (stated before running): low. Section 35 found |log roughness ratio| (0.585) and |log variance
  ratio| (0.583) about equally separable; if anything shows, it should be d_mad1 / d_tail in the kurtosis >= 3 group.
  Caveat (as in section 20): one segment starting at the break is not the streaming model's scan, and the small
  oracle model loses AUC when noisy columns are added, so a negative increment partly measures overfitting.

CONTROL MODE (`control` as the third argument; written 2026-10-16 after the first run, HANDOFF 46.1, BEFORE running it).
The first run passed on the variance-matched slice at L = 300 (+0.0113 +- 0.0040) and the kurtosis >= 3 group gained
+0.0343 +- 0.0051 (15/15). But this oracle's baseline has no robust LEVEL statistics, while production has many. So:
  r_  robust level / residual statistics that production already computes (kernel streams absz, abse10, tail1 / 1p5 /
      2 / 3 / 4, tail_e10, u, usq, centre, abs_abs1; `--dist` Anderson-Darling and W1 on e). Production's u from
      sbrt/volstate.py (GARCH-scaled residual) is not reproduced here, so r_ is a lower bound on what production has.
        r_mabsz, r_mabse        log mean|z| / mean|e| ratio      (mean|e| is the direct rival of mean|dz|)
        r_tz1 .. r_tz4          share of |z| > 1, 1.5, 2, 3, 4, change;   r_te2  share of |e| > 2, change
        r_u, r_u2               mean and mean square of the history-ECDF rank of z (minus 0.5), change
        r_centre                share of |z| < 0.5, change;   r_absabs1  log mean |z_t||z_t-1| ratio
        r_ad_e, r_w1_e          Anderson-Darling / Wasserstein-1 of e vs the history, on the 19 history-quantile edges
  Higher derivatives (owner's question: is there more in d3, d4, ...?), k = 3, 4, 6, 8:
        hq_lv{k}                log var(d^k z) ratio (quadratic; the oracle baseline has only ac1 / ac2 / ac5, while the
                                kernel has lags up to 10-16, so a small hq_ gain here may already be in production)
        hd_mad{k}, hd_t95_{k}   log mean|d^k z| ratio; share of |d^k z| above the history's 95th percentile, minus 0.05
      Expectation: nothing beyond d_. Each difference multiplies the power spectrum by 4 sin^2(w/2), so d^k for k >= 2
      looks at ever narrower slices of the same fastest wiggles, with noise growing as C(2k, k) (6, 20, 70, 252, ...).
  Extra group column: var-match AND kurtosis < 3 (is the var-matched pass just the heavy-tail effect again?).
  Extra incremental lines: + r_ over base;  + d_ over base + r_;  + hd_ over base + r_ + d_;  + hq_ over base.
  The baseline numbers must reproduce the first run exactly (new statistics use no random numbers).
  CONTROL RULES:
    d_ SURVIVES (-> build streaming |dz| features and gate them in cv.py) only if, over base + r_, at some L:
      kurt >= 3 gains >= +0.01 and > 3 SE (>= 200 segments per class), or var-match >= +0.005 and > 2 SE.
      Otherwise CLOSE: the gain was robustness that production already has.
    hd_ ADDS (-> include orders >= 3 in that build) only if, over base + r_ + d_, at some L: all > +0.0066 and > 3 SE,
      or var-match >= +0.005 and > 2 SE, or kurt >= 3 >= +0.01 and > 3 SE.

Run:  python diagnostics/deriv_oracle.py [n_series] [threads] [control]   (default all 10,000 series, 8 processes)
Out:  cv_out/deriv_oracle.txt, or cv_out/deriv_oracle_control.txt in control mode
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
LENS = (50, 100, 300)
VAR_SLICE = 0.1
MIN_CLASS = 20          # fewer segments of either class in a group -> AUC not reported


def _tp(d):
    return float(np.mean(d[1:] * d[:-1] < 0))


def _rrob(z):
    return float(np.log(np.mean(np.abs(np.diff(z))) / max(np.mean(np.abs(z - np.median(z))), 1e-12)))


HO = (3, 4, 6, 8)                     # higher derivative orders (control mode)
TZ = ((1.0, "1"), (1.5, "15"), (2.0, "2"), (3.0, "3"), (4.0, "4"))
PROBS = np.array([0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99,
                  0.995, 0.999])      # same edges as sbrt/disttest.py


def _ad_w1(seg, edges, p, dp, dx, m):
    """Binned Anderson-Darling and Wasserstein-1 of seg against the history ECDF p on fixed edges (as disttest)."""
    n = len(seg)
    G = (seg[:, None] <= edges[None, :]).mean(0)
    f = m / (n + m)
    ad = n * f * float(np.sum((G - p) ** 2 / np.maximum(p * (1 - p), 1e-9) * dp))
    return ad, float(np.sum(np.abs(G - p) * dx))


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

    dh = np.diff(zh)
    d2h = np.diff(dh)
    adh = np.abs(dh)
    H = dict(ac={k: _ac(zh, k) for k in (1, 2, 5)}, eac1=_ac(eh, 1), kurt=kurtosis(eh), skew=skew(eh),
             tail3=float(np.mean(np.abs(eh) > 3)), vc1=_corr(np.abs(eh[:-1]), np.abs(eh[1:])),
             lvd=np.log(dh.var()),
             # derivative history constants
             lvz=np.log(zh.var()), lvd2=np.log(d2h.var()), acd1=_ac(dh, 1),
             mad1=np.log(adh.mean()), mad2=np.log(np.abs(d2h).mean()),
             q95=np.quantile(adh, 0.95), q99=np.quantile(adh, 0.99), tp=_tp(dh), rrob=_rrob(zh))
    ctx = dict(ctx_kurt=kurtosis(zh), ctx_rho1=_ac(zh, 1), ctx_logn=np.log(nh))
    # control-mode history constants: robust level statistics and higher derivatives
    azh, aeh = np.abs(zh), np.abs(eh)
    zsort = np.sort(zh)
    uh = (np.searchsorted(zsort, zh) + 0.5) / (nh + 1.0) - 0.5
    edges = np.quantile(eh, PROBS)
    pe = (eh[:, None] <= edges[None, :]).mean(0)
    RH = dict(mabsz=np.log(azh.mean()), mabse=np.log(aeh.mean()), te2=float(np.mean(aeh > 2)),
              tz={nm: float(np.mean(azh > c)) for c, nm in TZ}, u=float(uh.mean()), u2=float((uh * uh).mean()),
              centre=float(np.mean(azh < 0.5)), absabs1=np.log(np.mean(azh[1:] * azh[:-1])),
              edges=edges, pe=pe, dp=np.gradient(pe), dx=np.gradient(edges))
    HK = {}
    for k in HO:
        dk = np.diff(zh, n=k)
        adk = np.abs(dk)
        HK[k] = (np.log(dk.var()), np.log(adk.mean()), np.quantile(adk, 0.95))
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
        zs, es = zo[st:st + L], eo[st:st + L]
        r = dict(ctx)
        # baseline: identical to info_oracle.py / longrange_oracle.py
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
        # derivative of the segment
        ds = np.diff(zs)
        d2s = np.diff(ds)
        ads = np.abs(ds)
        r["q_lvz"] = float(np.log(zs.var()) - H["lvz"])
        r["q_lvd2"] = float(np.log(d2s.var()) - H["lvd2"])
        r["q_acd1"] = _ac(ds, 1) - H["acd1"]
        r["d_mad1"] = float(np.log(ads.mean()) - H["mad1"])
        r["d_mad2"] = float(np.log(np.abs(d2s).mean()) - H["mad2"])
        r["d_tail95"] = float(np.mean(ads > H["q95"]) - 0.05)
        r["d_tail99"] = float(np.mean(ads > H["q99"]) - 0.01)
        r["d_tp"] = _tp(ds) - H["tp"]
        r["d_rrob"] = _rrob(zs) - H["rrob"]
        r["d_ks"] = float(ks_2samp(ds, dh).statistic)
        # control: robust level statistics production already has
        azs, aes = np.abs(zs), np.abs(es)
        us = (np.searchsorted(zsort, zs) + 0.5) / (nh + 1.0) - 0.5
        r["r_mabsz"] = float(np.log(azs.mean()) - RH["mabsz"])
        r["r_mabse"] = float(np.log(aes.mean()) - RH["mabse"])
        for c, nm in TZ:
            r[f"r_tz{nm}"] = float(np.mean(azs > c)) - RH["tz"][nm]
        r["r_te2"] = float(np.mean(aes > 2)) - RH["te2"]
        r["r_u"] = float(us.mean()) - RH["u"]
        r["r_u2"] = float((us * us).mean()) - RH["u2"]
        r["r_centre"] = float(np.mean(azs < 0.5)) - RH["centre"]
        r["r_absabs1"] = float(np.log(max(np.mean(azs[1:] * azs[:-1]), 1e-12)) - RH["absabs1"])
        r["r_ad_e"], r["r_w1_e"] = _ad_w1(es, RH["edges"], RH["pe"], RH["dp"], RH["dx"], len(eh))
        # control: higher derivatives
        for k in HO:
            dk = np.diff(zs, n=k)
            adk = np.abs(dk)
            lvh, madh, q95h = HK[k]
            r[f"hq_lv{k}"] = float(np.log(dk.var()) - lvh)
            r[f"hd_mad{k}"] = float(np.log(adk.mean()) - madh)
            r[f"hd_t95_{k}"] = float(np.mean(adk > q95h) - 0.05)
        out.append((L, lab, r))
    return out


MAGNITUDES = ("b_ks", "d_ks", "r_ad_e", "r_w1_e")   # already >= 0; every other statistic is a signed difference


def main():
    from joblib import Parallel, delayed
    from sklearn.model_selection import StratifiedKFold
    import lightgbm as lgb

    from sbrt.data import load_train

    n_series = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    threads = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    control = len(sys.argv) > 3 and sys.argv[3] == "control"
    os.makedirs("cv_out", exist_ok=True)
    log = open(os.path.join("cv_out", "deriv_oracle_control.txt" if control else "deriv_oracle.txt"), "w",
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

    verdict, ctl_verdict = [], []
    for L in LENS:
        rows = [(lab, r) for rr in res for (LL, lab, r) in rr if LL == L]
        if len(rows) < 100:
            say(f"\nL = {L}: only {len(rows)} segments, skipped")
            continue
        y = np.array([lab for lab, _ in rows])
        names = list(rows[0][1].keys())
        X = np.nan_to_num(np.array([[r[n] for n in names] for _, r in rows], dtype=np.float64))
        col = {n: j for j, n in enumerate(names)}
        rho, ku = np.abs(X[:, col["ctx_rho1"]]), X[:, col["ctx_kurt"]]
        groups = [("all", np.ones(len(y), bool)), ("var-match", np.abs(X[:, col["b_lv"]]) < VAR_SLICE),
                  ("|r1|<.3", rho < 0.3), ("|r1|.3-.6", (rho >= 0.3) & (rho <= 0.6)), ("|r1|>.6", rho > 0.6),
                  ("|r1|>.9", rho > 0.9),
                  ("kurt<3", ku < 3), ("kurt>=3", ku >= 3)]
        if control:
            groups.append(("vm&kurt<3", (np.abs(X[:, col["b_lv"]]) < VAR_SLICE) & (ku < 3)))
        gi = {g: k for k, (g, _) in enumerate(groups)}
        base = [n for n in names if n.startswith(("b_", "ctx_"))]
        bnames = [n for n in base if n.startswith("b_")]
        qf = [n for n in names if n.startswith("q_")]
        df = [n for n in names if n.startswith("d_")]
        rf = [n for n in names if n.startswith("r_")]
        hqf = [n for n in names if n.startswith("hq_")]
        hdf = [n for n in names if n.startswith("hd_")]
        head = "".join(f"{g:>11s}" for g, _ in groups)
        say(f"\n{'=' * 100}\nL = {L}: {int(y.sum())} break segments, {int((1 - y).sum())} placebo segments")
        say(f"  segments per group (break / placebo):  " +
            "   ".join(f"{g} {int(y[m].sum())}/{int((1 - y[m]).sum())}" for g, m in groups))

        R = np.column_stack([rankdata(X[:, col[n]]) for n in bnames])
        R = (R - R.mean(0)) / (R.std(0) + 1e-12)

        def g_auc(x, m):
            n1 = int(y[m].sum())
            return _auc(x[m], y[m]) if min(n1, int(m.sum()) - n1) >= MIN_CLASS else np.nan

        fmt = lambda a: f"{a:11.4f}" if np.isfinite(a) else f"{'-':>11s}"
        say(f"\n  single-statistic AUC, break vs placebo (|d| for signed differences)")
        say(f"    {'':12s}{head}   max |rank corr| w/ baseline")
        blocks = [("baseline", ["b_lv", "b_lvd", "b_ac1", "b_ac2", "b_kurt", "b_tail3", "b_ks"]),
                  ("q_ quadratic (expected already covered)", qf), ("d_ non-quadratic (new)", df)]
        if control:
            blocks += [("r_ robust level stats production already has (control)", rf),
                       ("hq_ higher-order quadratic", hqf), ("hd_ higher-order non-quadratic", hdf)]
        for grp, cols in blocks:
            say(f"  -- {grp}")
            for n in cols:
                x = X[:, col[n]]
                x = x if n in MAGNITUDES else np.abs(x)
                line = f"    {n:12s}" + "".join(fmt(g_auc(x, m)) for _, m in groups)
                if not n.startswith("b_"):
                    rr = rankdata(x)
                    rr = (rr - rr.mean()) / (rr.std() + 1e-12)
                    cc = np.abs(R.T @ rr) / len(rr)
                    j = int(np.argmax(cc))
                    line += f"   {cc[j]:.2f} {bnames[j]}"
                say(line)

        params = dict(objective="binary", learning_rate=0.05, num_leaves=15, min_data_in_leaf=50,
                      feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1,
                      deterministic=True, force_row_wise=True, num_threads=threads, seed=0)

        def cv_auc(cols):
            """(15 folds, n_groups) AUC; NaN where a fold has fewer than 10 of a class inside the group."""
            Xc = X[:, [col[n] for n in cols]]
            out = []
            for seed in (0, 1, 2):
                for tr, va in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xc, y):
                    p = lgb.train(params, lgb.Dataset(Xc[tr], y[tr]), 300).predict(Xc[va])
                    row = []
                    for _, m in groups:
                        mv, yv = m[va], y[va]
                        n1 = int(yv[mv].sum())
                        row.append(_auc(p[mv], yv[mv]) if min(n1, int(mv.sum()) - n1) >= 10 else np.nan)
                    out.append(row)
            return np.array(out)

        b = cv_auc(base)
        say(f"\n  incremental AUC (LightGBM, 3 x 5-fold; paired vs baseline: mean gain, +- SE, folds > 0)")
        bm = [b[np.isfinite(b[:, k]), k] for k in range(len(groups))]
        say(f"    {'baseline':14s}" + "".join(f"{v.mean():>11.4f}" if len(v) else f"{'-':>11s}" for v in bm) +
            f"      ({len(base)} stats)")
        def report(label, d):
            """Print mean gain / SE / folds > 0 per group for a (15, n_groups) paired difference; return mu, se."""
            ok = np.isfinite(d)
            mu = np.array([d[ok[:, k], k].mean() if ok[:, k].any() else np.nan for k in range(len(groups))])
            se = np.array([d[ok[:, k], k].std(ddof=1) / np.sqrt(ok[:, k].sum()) if ok[:, k].sum() > 1 else np.nan
                           for k in range(len(groups))])
            pos = [f"{int((d[ok[:, k], k] > 0).sum())}/{int(ok[:, k].sum())}" for k in range(len(groups))]
            say(f"    {label:14s}" + "".join(f"{m:>+11.4f}" if np.isfinite(m) else f"{'-':>11s}" for m in mu))
            say(f"    {'  +- SE':14s}" + "".join(f"{s:>11.4f}" if np.isfinite(s) else f"{'-':>11s}" for s in se))
            say(f"    {'  folds > 0':14s}" + "".join(f"{p:>11s}" for p in pos))
            return mu, se

        big = lambda g: min(int(y[groups[gi[g]][1]].sum()), int((1 - y[groups[gi[g]][1]]).sum())) >= 200
        L_res = {}
        for fam, cols in (("+ q_", qf), ("+ d_", df), ("+ q_ + d_", qf + df)):
            L_res[fam] = report(fam, cv_auc(base + cols) - b)
        mu, se = L_res["+ d_"]
        p_all = mu[0] > 0.0066 and mu[0] > 3 * se[0]
        p_vm = mu[1] >= 0.005 and mu[1] > 2 * se[1]
        cand = [g for k, (g, m) in enumerate(groups) if k >= 2 and np.isfinite(mu[k]) and mu[k] >= 0.01
                and mu[k] > 3 * se[k] and big(g)]
        verdict.append((L, p_all, p_vm, cand))

        if control:
            br = cv_auc(base + rf)
            brd = cv_auc(base + rf + df)
            brdh = cv_auc(base + rf + df + hdf)
            bhq = cv_auc(base + hqf)
            say(f"\n  CONTROL (paired gains; the reference of each line is named in it)")
            report("+ r_ | base", br - b)
            mu_d, se_d = report("+ d_ | base+r_", brd - br)
            mu_h, se_h = report("+ hd_ | +r_+d_", brdh - brd)
            report("+ hq_ | base", bhq - b)
            k3, vm = gi["kurt>=3"], gi["var-match"]
            d_surv = bool((mu_d[k3] >= 0.01 and mu_d[k3] > 3 * se_d[k3] and big("kurt>=3")) or
                          (mu_d[vm] >= 0.005 and mu_d[vm] > 2 * se_d[vm]))
            h_adds = bool((mu_h[0] > 0.0066 and mu_h[0] > 3 * se_h[0]) or (mu_h[vm] >= 0.005 and mu_h[vm] > 2 * se_h[vm])
                          or (mu_h[k3] >= 0.01 and mu_h[k3] > 3 * se_h[k3] and big("kurt>=3")))
            ctl_verdict.append((L, d_surv, h_adds))

    say("\nVERDICT for d_ (kill rule in the docstring: all > +0.0066 and > 3 SE, or var-matched >= +0.005 and > 2 SE, "
        "at some L; group-only hits are candidates, not passes)")
    for L, pa, pv, cand in verdict:
        say(f"  L = {L}: all {'PASS' if pa else 'fail'}   var-matched {'PASS' if pv else 'fail'}   "
            f"group candidates: {', '.join(cand) if cand else 'none'}")
    ok = any(pa or pv for _, pa, pv, _ in verdict)
    anyc = any(c for *_, c in verdict)
    say(f"  => {'PASS: build streaming d_ features (cv.py) and gate them' if ok else 'FAIL on the main rule'}"
        f"{'; group candidate(s) above: confirm with a second seed / 8 folds before building' if anyc and not ok else ''}")
    if control:
        say("\nCONTROL VERDICT (rules in the docstring)")
        for L, ds_, ha in ctl_verdict:
            say(f"  L = {L}: d_ over base + r_ {'SURVIVES' if ds_ else 'gone'}   "
                f"hd_ over base + r_ + d_ {'ADDS' if ha else 'adds nothing'}")
        surv = any(ds_ for _, ds_, _ in ctl_verdict)
        adds = any(ha for *_, ha in ctl_verdict)
        say(f"  => d_: {'SURVIVES: build streaming |dz| features and gate them in cv.py' if surv else 'CLOSE: the gain was robustness production already has'}")
        say(f"  => higher orders: {'ADD: include d3+ in that build' if adds else 'nothing beyond dz / d2z'}")
    if n_series:
        say("  (subset run: too few series to read; use the full run)")
    log.close()


if __name__ == "__main__":
    main()
