"""Which NEW kinds of information separate breaks, especially the breaks that leave the variance unchanged?

Oracle setting (break position known, like sim_stats.py and null_spread.py): for every series, one segment of L
online points is compared with the series' history.
    break series     segment = online[tau : tau + L]                 (needs L post-break points)
    no-break series  segment = online[s : s + L], s uniform over the valid starts (breaks are ~uniform too)
All statistics are "segment vs history" differences. Residuals e = history-fitted AR(10) residuals (as the kernel).

Candidate families (HANDOFF section 19 follow-up, items 1, 2, 4):
  ord_   ordinal patterns: how often each rank order of 3 or 4 consecutive points occurs (6 or 24 patterns),
         chi-square divergence from the history's frequencies, and the permutation-entropy difference; on the raw
         series z and on the residuals e. Scale-free: blind to variance changes by construction.
  nl_    nonlinear volatility dependence, all scale-free correlations: leverage corr(e_{t-k}, e_t^2) (k=1, sum 1-5),
         volatility clustering corr(|e_t|, |e_{t-k}|) at k = 2, 5, 10, 20 and summed over 1-10, time
         irreversibility mean(x_t^2 x_{t-k} - x_t x_{t-k}^2)/sd^3 (k = 1, 2) on e and on z.
  mmd_   kernel two-sample test (unbiased MMD^2, Gaussian kernel) between segment and history samples of: single e
         (marginal, for comparison), pairs (e_{t-1}, e_t), triples, and pairs after dividing each sample by its own
         sd (removes the variance difference: pure shape/dependence).
  mp_    matrix profile / FLUSS idea (Gharghabi et al. 2017; Fuchs 2024 mFLUSS): windows of m = 16 or 32 points of
         the series z; each segment window's nearest neighbour (Euclidean) among a history window pool and the other
         segment windows (exclusion zone m). zn = every window rescaled to mean 0 / sd 1 (FLUSS, shape only),
         raw = no rescaling (keeps the variance). Two statistics: share = share of segment windows whose nearest
         neighbour is another segment window minus its no-break expectation (FLUSS: arcs that do not cross the
         boundary); novel = log(median distance of segment windows to their nearest HISTORY window / median
         history-to-history nearest-neighbour distance) (matrix-profile AB-join: "unlike anything seen before").
Baseline b_ (what the current models already see, as oracle statistics): log variance ratio of e and of the
differences, lag-1/2/5 autocorrelation change of z, residual lag-1 autocorrelation change, kurtosis and skew change,
3-sigma tail rate change, standardised mean, lag-1 volatility clustering change, KS distance of e; plus history
context (kurtosis, lag-1 autocorrelation, log length).

Reports, per segment length L = 50, 100, 300:
  1. single-statistic AUC (break vs no-break) on all series and on the VARIANCE-MATCHED slice (both classes with
     |log variance ratio of e| < 0.1: the population where variance says nothing, i.e. our weak slice), using the
     statistic's magnitude |d| for two-sided ones;
  2. the largest |rank correlation| with any baseline statistic;
  3. the INCREMENTAL value: 5-fold cross-validated AUC of a small LightGBM on baseline vs baseline + family
     (3 fold seeds, paired), evaluated on all series and on the variance-matched slice.

Run:  python diagnostics/info_oracle.py [n_series] [threads] [families]   (default all 10,000 series, 8 processes,
      families ord,nl,mmd,mp; e.g. "mp" to rank only the matrix-profile family against the baseline)
Out:  cv_out/info_oracle.txt
"""
import itertools
import os
import sys
import time

import numpy as np
from scipy.signal import lfilter
from scipy.stats import kurtosis, ks_2samp, rankdata, skew

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
os.chdir(HERE)

AR = 10
LENS = (50, 100, 300)
NH_MMD = 800
VAR_SLICE = 0.1


# ---------------------------------------------------------------------------------------------
def _ac(x, k):
    x = x - x.mean()
    d = float(np.dot(x, x))
    return float(np.dot(x[:-k], x[k:]) / d) if d > 0 else 0.0


def _corr(a, b):
    sa, sb = a.std(), b.std()
    if sa <= 0 or sb <= 0:
        return 0.0
    return float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb))


PERMS = {m: np.array(sorted({int((np.array(p) * (m ** np.arange(m))).sum()) for p in itertools.permutations(range(m))}))
         for m in (3, 4)}


def _ord_freq(x, m):
    W = np.lib.stride_tricks.sliding_window_view(x, m)
    r = W.argsort(axis=1, kind="stable").argsort(axis=1, kind="stable")
    code = (r * (m ** np.arange(m))).sum(1)
    return np.bincount(code, minlength=m ** m)[PERMS[m]].astype(np.float64)


def _ord_stats(seg, q, m):
    c = _ord_freq(seg, m)
    n = c.sum()
    chi = float(np.sum((c - n * q) ** 2 / (n * q)))
    p = c / n
    pe = -np.sum(p[p > 0] * np.log(p[p > 0])) / np.log(len(q))
    pe_h = -np.sum(q * np.log(q)) / np.log(len(q))
    return chi, float(pe - pe_h)


def _nl(x):
    """Scale-free nonlinear-dependence statistics of one sample of residuals x (and z for irreversibility)."""
    out = {}
    x2, ax = x * x, np.abs(x)
    out["lev1"] = _corr(x[:-1], x2[1:])
    out["levsum"] = sum(_corr(x[:-k], x2[k:]) for k in range(1, 6))
    for k in (2, 5, 10, 20):
        out[f"vc{k}"] = _corr(ax[:-k], ax[k:])
    out["vcsum"] = sum(_corr(ax[:-k], ax[k:]) for k in range(1, 11))
    s = x.std() + 1e-12
    y = x / s
    for k in (1, 2):
        out[f"irr{k}"] = float(np.mean(y[k:] ** 2 * y[:-k] - y[k:] * y[:-k] ** 2))
    return out


def _irr(x, k):
    y = (x - x.mean()) / (x.std() + 1e-12)
    return float(np.mean(y[k:] ** 2 * y[:-k] - y[k:] * y[:-k] ** 2))


def _emb(x, d):
    return np.column_stack([x[i: len(x) - d + 1 + i] for i in range(d)])


def _kmean(A, B, d, same):
    D = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * A @ B.T
    K = np.exp(-np.maximum(D, 0) / (2.0 * d))
    if same:
        n = len(A)
        return float((K.sum() - np.trace(K)) / (n * (n - 1)))
    return float(K.mean())


def _windows(x, m, norm):
    W = np.lib.stride_tricks.sliding_window_view(x, m).astype(np.float64)
    if norm:
        W = (W - W.mean(1, keepdims=True)) / np.maximum(W.std(1, keepdims=True), 1e-8)
    return W


def _nn(Q, qpos, P, ppos, excl):
    """Nearest-neighbour distance of every row of Q among the rows of P (pairs closer than excl positions skipped)."""
    D = (Q * Q).sum(1)[:, None] + (P * P).sum(1)[None, :] - 2.0 * Q @ P.T
    if excl:
        D[np.abs(qpos[:, None] - ppos[None, :]) < excl] = np.inf
    return np.sqrt(np.maximum(D.min(1), 0.0))


MP_M = (16, 32)
MP_POOL = 1200


def series_rows(i, h, o, t, seed):
    """All statistics for every segment length this series supports. Returns a list of (L, label, dict)."""
    rng = np.random.default_rng(seed)
    h = np.asarray(h, np.float64)
    o = np.asarray(o, np.float64)
    mu, sd = h.mean(), h.std() + 1e-12
    zh, zo = (h - mu) / sd, (o - mu) / sd
    nh = len(h)
    Xl = np.column_stack([zh[AR - 1 - k: nh - 1 - k] for k in range(AR)])
    coef = np.linalg.lstsq(Xl, zh[AR:], rcond=None)[0]
    s = np.r_[zh, zo]
    e = lfilter(np.r_[1.0, -coef], [1.0], s)
    se = max(float(e[2 * AR:nh].std()), 1e-8)
    e = np.clip(e / se, -10, 10)
    eh, eo = e[2 * AR:nh], e[nh:]

    # history-side constants (shared by all L)
    H = {}
    for m in (3, 4):
        H[f"qz{m}"] = (_ord_freq(zh, m) + 0.5) / (_ord_freq(zh, m).sum() + 0.5 * len(PERMS[m]))
        H[f"qe{m}"] = (_ord_freq(eh, m) + 0.5) / (_ord_freq(eh, m).sum() + 0.5 * len(PERMS[m]))
    H["nl"] = _nl(eh)
    H["irrz1"], H["irrz2"] = _irr(zh, 1), _irr(zh, 2)
    H["ac"] = {k: _ac(zh, k) for k in (1, 2, 5)}
    H["eac1"] = _ac(eh, 1)
    H["kurt"], H["skew"] = kurtosis(eh), skew(eh)
    H["tail3"] = float(np.mean(np.abs(eh) > 3))
    H["vc1"] = _corr(np.abs(eh[:-1]), np.abs(eh[1:]))
    H["lvd"] = np.log(np.diff(zh).var())
    hs = eh.std()
    mm = {}
    for nm, d in (("e1", 1), ("pair", 2), ("trip", 3)):
        E = _emb(eh, d)
        pick = rng.choice(len(E), min(NH_MMD, len(E)), replace=False)
        mm[nm] = E[pick]
        mm[nm + "_kk"] = _kmean(mm[nm], mm[nm], d, True)
    Es = _emb(eh / hs, 2)
    mm["pss"] = Es[rng.choice(len(Es), min(NH_MMD, len(Es)), replace=False)]
    mm["pss_kk"] = _kmean(mm["pss"], mm["pss"], 2, True)
    ctx = dict(ctx_kurt=kurtosis(zh), ctx_rho1=_ac(zh, 1), ctx_logn=np.log(nh))
    mp = {}
    for m in MP_M:
        for norm in (True, False):
            P = _windows(zh, m, norm)
            ppos = np.arange(len(P))
            stride = max(1, len(P) // MP_POOL)
            P, ppos = P[::stride], ppos[::stride]
            ref = rng.choice(len(P), min(300, len(P)), replace=False)
            med_hh = float(np.median(_nn(P[ref], ppos[ref], P, ppos, m)))
            mp[(m, norm)] = (P, ppos, max(med_hh, 1e-8))

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
        # baseline
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
        # 1. ordinal patterns
        for m in (3, 4):
            for v, x in (("z", zs), ("e", es)):
                chi, dpe = _ord_stats(x, H[f"q{v}{m}"], m)
                r[f"ord_chi{m}_{v}"] = chi
                r[f"ord_pe{m}_{v}"] = dpe
        # 2. nonlinear volatility dependence
        nl = _nl(es)
        for k, v in nl.items():
            r[f"nl_{k}"] = v - H["nl"][k]
        r["nl_irrz1"] = _irr(zs, 1) - H["irrz1"]
        r["nl_irrz2"] = _irr(zs, 2) - H["irrz2"]
        # 4. MMD
        for nm, d in (("e1", 1), ("pair", 2), ("trip", 3)):
            S_ = _emb(es, d)
            r[f"mmd_{nm}"] = _kmean(S_, S_, d, True) + mm[nm + "_kk"] - 2 * _kmean(S_, mm[nm], d, False)
        S_ = _emb(es / (es.std() + 1e-12), 2)
        r["mmd_pair_ss"] = _kmean(S_, S_, 2, True) + mm["pss_kk"] - 2 * _kmean(S_, mm["pss"], 2, False)
        # matrix profile / FLUSS
        for m in MP_M:
            for norm in (True, False):
                P, ppos, med_hh = mp[(m, norm)]
                Q = _windows(zs, m, norm)
                qpos = np.arange(len(Q))
                d_hist = _nn(Q, qpos, P, ppos, 0)
                if len(Q) > m:
                    d_seg = _nn(Q, qpos, Q, qpos, m)
                    n_cand = np.array([len(Q) - min(len(Q), k + m) + max(0, k - m + 1) for k in range(len(Q))])
                    expect = float(np.mean(n_cand / (n_cand + len(P))))
                    share = float(np.mean(d_seg < d_hist)) - expect
                else:
                    share = 0.0
                tag = f"{'zn' if norm else 'raw'}{m}"
                r[f"mp_{tag}_share"] = share
                r[f"mp_{tag}_novel"] = float(np.log(np.median(d_hist) / med_hh))
        out.append((L, lab, r))
    return out


# ---------------------------------------------------------------------------------------------
def _auc(s, y):
    r = rankdata(s)
    n1 = y.sum()
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


SIGNED = ("ord_chi", "mmd_", "mp_")   # already magnitudes (>= 0); everything else is a signed difference


def main():
    from joblib import Parallel, delayed
    from sklearn.model_selection import StratifiedKFold
    import lightgbm as lgb

    from sbrt.data import load_train

    n_series = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    threads = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    os.makedirs("cv_out", exist_ok=True)
    log = open(os.path.join("cv_out", "info_oracle.txt"), "w", encoding="utf-8")

    def say(s=""):
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    S, tau = load_train()
    if n_series:
        S, tau = S[:n_series], tau[:n_series]
    t0 = time.time()
    res = Parallel(n_jobs=threads, backend="loky")(
        delayed(series_rows)(i, s[1], s[2], int(tau[i]), 1000 + i) for i, s in enumerate(S))
    say(f"statistics for {len(S)} series in {time.time() - t0:.0f}s")

    for L in LENS:
        rows = [(lab, r) for rr in res for (LL, lab, r) in rr if LL == L]
        y = np.array([lab for lab, _ in rows])
        names = list(rows[0][1].keys())
        X = np.array([[r[n] for n in names] for _, r in rows], dtype=np.float64)
        X = np.nan_to_num(X)
        col = {n: j for j, n in enumerate(names)}
        vm = np.abs(X[:, col["b_lv"]]) < VAR_SLICE
        base = [n for n in names if n.startswith(("b_", "ctx_"))]
        fams_all = {"ord": [n for n in names if n.startswith("ord_")], "nl": [n for n in names if n.startswith("nl_")],
                "mmd": [n for n in names if n.startswith("mmd_")], "mp": [n for n in names if n.startswith("mp_")]}
        want = (sys.argv[3].split(",") if len(sys.argv) > 3 else list(fams_all))
        fams = {k: v for k, v in fams_all.items() if k in want}
        say(f"\n{'=' * 100}\nL = {L}: {int(y.sum())} break segments, {int((1 - y).sum())} no-break segments; "
            f"variance-matched slice (|log var ratio| < {VAR_SLICE}): {int(y[vm].sum())} / {int((1 - y[vm]).sum())}")

        R = np.column_stack([rankdata(X[:, col[n]]) for n in base if n.startswith("b_")])
        R = (R - R.mean(0)) / (R.std(0) + 1e-12)
        bnames = [n for n in base if n.startswith("b_")]

        def mag(n):
            x = X[:, col[n]]
            return x if n.startswith(SIGNED) else np.abs(x)

        say(f"\n  single-statistic AUC (magnitude |d| for signed differences)      all    var-matched   "
            f"max |rank corr| with a baseline stat")
        for fam, cols in [("baseline", bnames)] + list(fams.items()):
            say(f"  -- {fam}")
            for n in cols:
                x = mag(n)
                a, av = _auc(x, y), _auc(x[vm], y[vm])
                if fam == "baseline":
                    say(f"    {n:18s} {a:.4f}   {av:.4f}")
                else:
                    rr = rankdata(x)
                    rr = (rr - rr.mean()) / (rr.std() + 1e-12)
                    cc = np.abs(R.T @ rr) / len(rr)
                    j = int(np.argmax(cc))
                    say(f"    {n:18s} {a:.4f}   {av:.4f}      {cc[j]:.2f} {bnames[j]}")

        # incremental value: baseline vs baseline + family, small LightGBM, 3 x 5-fold, paired per fold
        params = dict(objective="binary", learning_rate=0.05, num_leaves=15, min_data_in_leaf=50,
                      feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1,
                      deterministic=True, force_row_wise=True, num_threads=threads, seed=0)

        def cv_auc(cols):
            Xc = X[:, [col[n] for n in cols]]
            out = []
            for seed in (0, 1, 2):
                oof = np.zeros(len(y))
                for tr, va in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xc, y):
                    m = lgb.train(params, lgb.Dataset(Xc[tr], y[tr]), 300)
                    oof[va] = m.predict(Xc[va])
                    out.append((_auc(oof[va], y[va]), _auc(oof[va][vm[va]], y[va][vm[va]])))
            return np.array(out)

        b = cv_auc(base)
        say(f"\n  incremental value (LightGBM, 3 x 5-fold CV; paired difference vs baseline, mean +- se over 15 folds)")
        say(f"    baseline ({len(base)} stats):            all {b[:, 0].mean():.4f}   var-matched {b[:, 1].mean():.4f}")
        for fam, cols in list(fams.items()) + [("all", sum(fams.values(), []))]:
            c = cv_auc(base + cols)
            d = c - b
            se = d.std(0, ddof=1) / np.sqrt(len(d))
            say(f"    + {fam:10s} ({len(cols):2d} stats):     all {c[:, 0].mean():.4f} ({d[:, 0].mean():+.4f} +- {se[0]:.4f}, "
                f"{int((d[:, 0] > 0).sum())}/15)   var-matched {c[:, 1].mean():.4f} ({d[:, 1].mean():+.4f} +- {se[1]:.4f}, "
                f"{int((d[:, 1] > 0).sum())}/15)")
    say("\nReading it: a family is worth building as streaming features if its INCREMENTAL gain is clearly positive "
        "(several standard errors), above all on the variance-matched slice. Single-statistic AUCs alone are not "
        "enough (the KS/AD lesson: low pairwise correlation, little added value).")
    log.close()


if __name__ == "__main__":
    main()
