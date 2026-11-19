"""EDA: at the breaks that leave the variance unchanged, what DOES change? (owner's question, 2026-10-17)

Oracle setting (break position known), like info_oracle.py, but built to be LOOKED AT: an HTML report with figures.

Design
  cut       break series: tau (first post-break online point). No-break series: a PLACEBO cut drawn uniformly over
            the online starts that leave L points (breaks are ~uniform too). Every comparison below is break vs
            placebo, so anything that also happens at an arbitrary point of a no-break series cancels.
  windows   pre = the PRE points right before the cut (default 1,000: history + any online part before the cut),
            post = EXACTLY L points from the cut (L = 100 and 300 by default; a series needs L post points).
            A fixed L matters: with variable post lengths the variance filter below would keep shorter (noisier)
            break windows than placebo windows, and every noisy statistic would look like a signal.
  residual  e = AR(10) residual with coefficients fitted on the history (as the kernel's e10).
  slices    all       every usable segment
            var       |log var(post)/var(pre)| < BAND for the raw series x (default 0.1): "variance unchanged"
            var+      the same AND for the AR residual e AND for the differences dx (variance, innovation variance
                      and roughness all unchanged)
  per statistic: delta = post minus pre (log ratio for scales, pre-sd units for location, distances for two-sample
            tests). Breaks can move a statistic either way, so the main score is the two-sided "magnitude AUC":
            AUC of |delta - placebo median| (break vs placebo); distances are used as they are. 0.5 = no
            information. The chance level for the BEST of ~60 statistics comes from label permutations.

Report sections per L (figures are PNGs embedded in one HTML file):
  1 selection  how the variance filter picks the slice (and a sanity check: in the slice the variance itself
               must score ~0.5)
  2 forest     magnitude AUC of every statistic on the slice, with bootstrap 95% intervals and the chance band
  3 quantiles  how the shape of the distribution moves (own-sd standardised quantiles), signed and magnitude
  4 ACF        autocorrelation change by lag for x, e and |e| (volatility clustering), signed and magnitude
  5 spectrum   where the power sits (share per frequency bin)
  6 event      rolling statistics aligned on the cut: does something switch at the break?
  7 top stats  break vs placebo distributions of the 6 best statistics
  8 omnibus    is a slice break unusual in ANY aspect? share above the placebos' 95th percentile of max robust z
  9 gallery    random slice breaks (and the most unusual ones) next to placebos: series, histograms, ACF

Earlier result to compare with (cv_out/info_oracle.txt, pre = whole history, slice on e): at L = 300 the lag-1
autocorrelation change was the best statistic on the variance-matched slice (b_eac1 0.592, b_ac1 0.585), then
roughness (0.56), kurtosis / skew / tails (~0.55); at L = 100 everything sat at 0.51-0.54. If this script does not
roughly reproduce that, suspect the script first.

Run (CPU only):
    python diagnostics/eda_varmatched.py                     # all 10,000 series, L = 100 and 300
    python diagnostics/eda_varmatched.py --n 1500 --boot 50 --perm 20     # quick look
Out:  eda_out/varmatched/report.html (open in a browser), stats_L<L>.csv (one row per statistic),
      series_L<L>.csv (one row per series: cut, slice flags, every delta, max robust z), summary.txt
"""
import argparse
import base64
import html
import io
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
os.chdir(HERE)

import numpy as np  # noqa: E402
from numpy.lib.stride_tricks import sliding_window_view  # noqa: E402
from scipy.signal import lfilter  # noqa: E402
from scipy.stats import ks_2samp, rankdata, wasserstein_distance  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sbrt.data import load_train  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=0, help="use the first n series (0 = all)")
ap.add_argument("--lens", default="100,300", help="post-window lengths L")
ap.add_argument("--pre", type=int, default=1000, help="points before the cut used as the reference")
ap.add_argument("--band", type=float, default=0.1, help="|log variance ratio| below this = variance unchanged")
ap.add_argument("--boot", type=int, default=200, help="bootstrap resamples for the AUC intervals")
ap.add_argument("--perm", type=int, default=100, help="label permutations for the chance band")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default=os.path.join("eda_out", "varmatched"))
args = ap.parse_args()

AR = 10
LENS = [int(v) for v in args.lens.split(",")]
PRE = args.pre
EV_W, EV_STEP, EV_LO = 50, 10, -300          # event study: 50-point windows every 10 points from -300
QLEV = np.linspace(0.02, 0.98, 49)
NACF = 20
SPEC_BINS = np.linspace(0.0, 0.5, 21)

# colours: dataviz reference palette, categorical slots 1-2 (validated pair) + neutral inks
C_BRK, C_PLA = "#eb6834", "#2a78d6"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10, "axes.titlecolor": INK,
    "legend.frameon": False, "lines.linewidth": 2,
})

# ---------------------------------------------------------------------------------------------- statistics
# name -> (family, kind, description). kind: lr = log ratio post/pre, std = (post - pre) / pre sd, d = post - pre,
# dist = two-sample distance (already >= 0)
STATS = {}


def _s(name, fam, kind, desc):
    STATS[name] = (fam, kind, desc)


for _n, _d in [("var_x", "variance of x"), ("var_e", "variance of the AR(10) residual e (innovation variance)"),
               ("var_dx", "variance of the differences dx (roughness)"), ("mad_x", "median absolute deviation of x"),
               ("iqr_x", "interquartile range of x")]:
    _s(_n, "scale", "lr", _d)
_s("mean_x", "location", "std", "mean of x, in pre-window sd units")
_s("median_x", "location", "std", "median of x, in pre-window sd units")
for _n, _d in [("skew_x", "skewness of x"), ("kurt_x", "excess kurtosis of x"), ("skew_e", "skewness of e"),
               ("kurt_e", "excess kurtosis of e")]:
    _s(_n, "shape", "d", _d)
QSTAT = [2, 5, 10, 25, 50, 75, 90, 95, 98]
for _q in QSTAT:
    _s(f"q{_q:02d}", "shape", "d", f"{_q}% quantile of x standardised by its own mean and sd")
_s("tail2", "tails", "d", "share of |x| > 2 own sd")
_s("tail3", "tails", "d", "share of |x| > 3 own sd")
_s("tailasym2", "tails", "d", "share above +2 sd minus share below -2 sd")
_s("centre", "tails", "d", "share of |x| < 0.25 own sd (peakedness)")
_s("tail3_e", "tails", "d", "share of |e| > 3 own sd")
for _k in (1, 2, 3, 5, 10):
    _s(f"acf_x{_k}", "dependence", "d", f"autocorrelation of x at lag {_k}")
_s("acfsum_x", "dependence", "d", "sum of the autocorrelations of x at lags 1-10 (persistence)")
_s("acf_e1", "dependence", "d", "lag-1 autocorrelation of e (does the history's AR still whiten?)")
_s("acf_e2", "dependence", "d", "lag-2 autocorrelation of e")
_s("lbq_e", "dependence", "d", "sum of squared autocorrelations of e at lags 1-10 (Ljung-Box style)")
_s("wander_x", "dependence", "d", "log variance of 10-point block means x 10 / variance (slow mean wander)")
for _k in (1, 2, 5):
    _s(f"acfabs_e{_k}", "vol-dependence", "d", f"autocorrelation of |e| at lag {_k} (volatility clustering)")
_s("acfabssum_e", "vol-dependence", "d", "sum of the |e| autocorrelations at lags 1-10")
_s("acfsq_e1", "vol-dependence", "d", "lag-1 autocorrelation of e^2")
_s("lev_e", "vol-dependence", "d", "leverage: corr(e[t-1], e[t]^2)")
_s("volvol_e", "vol-dependence", "d", "log coefficient of variation of the rolling 20-point variance of e")
for _b, (_lo, _hi) in enumerate([(0, .05), (.05, .15), (.15, .30), (.30, .5)]):
    _s(f"spec{_b}", "spectrum", "d", f"share of the power of x at frequencies {_lo}-{_hi} cycles/step")
_s("signpers_x", "nonlinear", "d", "share of consecutive points on the same side of the median (x)")
_s("signpers_e", "nonlinear", "d", "share of consecutive points on the same side of the median (e)")
_s("turn_x", "nonlinear", "d", "share of turning points (local max/min)")
_s("irrev_dx", "nonlinear", "d", "skewness of dx (time irreversibility)")
_s("pe3_x", "nonlinear", "d", "permutation entropy of order 3 (normalised)")
_s("rep_x", "number format", "d", "share of points equal to the previous one")
_s("uniq_x", "number format", "d", "share of distinct values (length-dependent; the placebo cancels that)")
_s("ks_x", "distance", "dist", "Kolmogorov-Smirnov distance, x post vs pre")
_s("ks_xs", "distance", "dist", "KS distance after standardising each window by its own mean and sd (shape only)")
_s("w1_xs", "distance", "dist", "Wasserstein distance, own-standardised windows (shape only)")
_s("ks_es", "distance", "dist", "KS distance of own-standardised e (shape of the innovations)")
_s("ord3_chi2", "distance", "dist", "chi-square distance between the order-3 ordinal-pattern frequencies")
NAMES = list(STATS)
FAMILIES = ["scale", "location", "shape", "tails", "dependence", "vol-dependence", "spectrum", "nonlinear",
            "number format", "distance"]
MATCHED = {"var_x", "var_e", "var_dx"}  # used to define the slices: left out of the omnibus


def _acf(x, kmax):
    x = x - x.mean()
    d = float((x * x).sum())
    if d <= 0:
        return np.zeros(kmax)
    return np.array([float((x[k:] * x[:-k]).sum()) / d for k in range(1, kmax + 1)])


def _corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def _moments(x):
    x = x - x.mean()
    m2 = (x * x).mean()
    if m2 <= 0:
        return 0.0, 0.0
    return float((x ** 3).mean() / m2 ** 1.5), float((x ** 4).mean() / m2 ** 2 - 3)


def _ord3(x):
    w = sliding_window_view(x, 3)
    code = np.argsort(w, axis=1, kind="stable") @ np.array([9, 3, 1])
    f = np.array([(code == c).sum() for c in (5, 7, 11, 15, 19, 21)], dtype=float)  # the 6 permutations
    return f / max(f.sum(), 1.0)


def _spec(xs):
    p = np.abs(np.fft.rfft(xs))[1:] ** 2
    f = np.fft.rfftfreq(len(xs))[1:]
    p = p / max(p.sum(), 1e-300)
    return f, p


def seg_feats(x, e):
    """Per-window statistics (before taking post minus pre) + profiles for the figures."""
    S = {}
    sd = x.std()
    xs = (x - x.mean()) / sd if sd > 0 else x * 0
    dx = np.diff(x)
    e = e[np.isfinite(e)]
    esd = e.std()
    es = (e - e.mean()) / esd if esd > 0 else e * 0
    S["var_x"], S["var_e"], S["var_dx"] = x.var(), e.var(), dx.var()
    S["mad_x"] = np.median(np.abs(x - np.median(x)))
    q25, q75 = np.percentile(x, [25, 75])
    S["iqr_x"] = q75 - q25
    S["mean_x"], S["median_x"] = x.mean(), np.median(x)
    S["skew_x"], S["kurt_x"] = _moments(x)
    S["skew_e"], S["kurt_e"] = _moments(e)
    qs = np.percentile(xs, QSTAT)
    for q, v in zip(QSTAT, qs):
        S[f"q{q:02d}"] = v
    a = np.abs(xs)
    S["tail2"], S["tail3"] = (a > 2).mean(), (a > 3).mean()
    S["tailasym2"] = (xs > 2).mean() - (xs < -2).mean()
    S["centre"] = (a < 0.25).mean()
    S["tail3_e"] = (np.abs(es) > 3).mean()
    ax = _acf(x, NACF)
    for k in (1, 2, 3, 5, 10):
        S[f"acf_x{k}"] = ax[k - 1]
    S["acfsum_x"] = ax[:10].sum()
    ae = _acf(e, NACF)
    S["acf_e1"], S["acf_e2"] = ae[0], ae[1]
    S["lbq_e"] = (ae[:10] ** 2).sum()
    nb = len(x) // 10
    bm = x[:nb * 10].reshape(nb, 10).mean(1)
    S["wander_x"] = np.log(max(bm.var() * 10, 1e-12) / max(x.var(), 1e-12))
    aa = _acf(np.abs(e), NACF)
    S["acfabs_e1"], S["acfabs_e2"], S["acfabs_e5"] = aa[0], aa[1], aa[4]
    S["acfabssum_e"] = aa[:10].sum()
    S["acfsq_e1"] = _acf(e * e, 1)[0]
    S["lev_e"] = _corr(e[:-1], e[1:] ** 2)
    rv = sliding_window_view(e, 20).var(axis=1)
    S["volvol_e"] = np.log(max(rv.std() / max(rv.mean(), 1e-12), 1e-6))
    f, p = _spec(xs)
    for b, (lo, hi) in enumerate([(0, .05), (.05, .15), (.15, .30), (.30, .51)]):
        S[f"spec{b}"] = p[(f >= lo) & (f < hi)].sum()
    m = np.median(x)
    sx = np.sign(x - m)
    S["signpers_x"] = (sx[1:] == sx[:-1]).mean()
    se_ = np.sign(e - np.median(e))
    S["signpers_e"] = (se_[1:] == se_[:-1]).mean()
    S["turn_x"] = (((x[1:-1] > x[:-2]) & (x[1:-1] > x[2:])) | ((x[1:-1] < x[:-2]) & (x[1:-1] < x[2:]))).mean()
    S["irrev_dx"] = _moments(dx)[0]
    o3 = _ord3(x)
    nz = o3[o3 > 0]
    S["pe3_x"] = float(-(nz * np.log(nz)).sum() / np.log(6))
    S["rep_x"] = (x[1:] == x[:-1]).mean()
    S["uniq_x"] = len(np.unique(x)) / len(x)
    prof = dict(q=np.quantile(xs, QLEV), acf_x=ax, acf_e=ae, acf_abse=aa,
                spec=np.array([p[(f >= lo) & (f < hi)].sum() for lo, hi in zip(SPEC_BINS[:-1],
                                                                                SPEC_BINS[1:] + 1e-9 * (SPEC_BINS[1:] == .5))]))
    return S, prof, (xs, es, o3)


def deltas(Sa, Sb, ra, rb, pre_x, post_x):
    """post (b) vs pre (a) per statistic."""
    out = np.full(len(NAMES), np.nan)
    sd = pre_x.std()
    for j, n in enumerate(NAMES):
        kind = STATS[n][1]
        if kind == "lr":
            out[j] = np.log(max(Sb[n], 1e-12) / max(Sa[n], 1e-12))
        elif kind == "std":
            out[j] = (Sb[n] - Sa[n]) / sd if sd > 0 else 0.0
        elif kind == "d":
            out[j] = Sb[n] - Sa[n]
    xa, ea, oa = ra
    xb, eb, ob = rb
    for n, v in (("ks_x", ks_2samp(pre_x, post_x).statistic), ("ks_xs", ks_2samp(xa, xb).statistic), ("w1_xs", wasserstein_distance(xa, xb)),
                 ("ks_es", ks_2samp(ea, eb).statistic),
                 ("ord3_chi2", (len(xb) - 2) * float(((ob - oa) ** 2 / np.maximum(oa, 1e-3)).sum()))):
        out[NAMES.index(n)] = v
    return out


def event_rows(x, e, c, L, mu, sd, var_pre, med):
    """Rolling statistics in EV_W-point windows starting at offsets EV_LO .. L - EV_W from the cut."""
    seg = x[c + EV_LO: c + L]
    sege = e[c + EV_LO: c + L]
    W = sliding_window_view(seg, EV_W)[::EV_STEP]
    WE = sliding_window_view(sege, EV_W)[::EV_STEP]
    dm = W - W.mean(1, keepdims=True)
    v = (dm * dm).mean(1)
    lv = np.log(np.maximum(v, 1e-12) / var_pre)
    ac1 = (dm[:, 1:] * dm[:, :-1]).sum(1) / np.maximum((dm * dm).sum(1), 1e-12)
    ku = (dm ** 4).mean(1) / np.maximum(v, 1e-12) ** 2 - 3
    tl = (np.abs((W - mu) / sd) > 2).mean(1)
    a = np.abs(WE)
    am = a - a.mean(1, keepdims=True)
    aac = (am[:, 1:] * am[:, :-1]).sum(1) / np.maximum((am * am).sum(1), 1e-12)
    s = np.sign(W - med)
    sp = (s[:, 1:] == s[:, :-1]).mean(1)
    return np.stack([lv, ac1, ku, tl, aac, sp], 1)


EV_NAMES = ["log variance vs pre window", "lag-1 autocorrelation of x", "excess kurtosis of x",
            "share |x| > 2 pre sd", "lag-1 autocorrelation of |e|", "sign persistence about the pre median"]


# ---------------------------------------------------------------------------------------------- AUC helpers
def auc(s, y):
    r = rankdata(s)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def magnitude(D, y):
    """two-sided score per statistic: |d - placebo median|; distances as they are."""
    M = np.empty_like(D)
    for j, n in enumerate(NAMES):
        if STATS[n][1] == "dist":
            M[:, j] = D[:, j]
        else:
            M[:, j] = np.abs(D[:, j] - np.median(D[y == 0, j]))
    return M


def fig_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def band(ax, xv, A, color, label):
    m = np.nanmean(A, 0)
    n = np.sum(np.isfinite(A), 0)
    se = np.nanstd(A, 0) / np.sqrt(np.maximum(n, 1))
    ax.fill_between(xv, m - 1.96 * se, m + 1.96 * se, color=color, alpha=0.18, lw=0)
    ax.plot(xv, m, color=color, label=label)


# ---------------------------------------------------------------------------------------------- main
def main():
    t0 = time.time()
    os.makedirs(args.out, exist_ok=True)
    S, tau = load_train()
    if args.n:
        S, tau = S[:args.n], tau[:args.n]
    N = len(S)
    log = []

    def say(s=""):
        print(s, flush=True)
        log.append(s)

    say(f"EDA variance-matched breaks: {N} series, L = {LENS}, pre = {PRE}, band = {args.band}")
    # per-series residuals with history-fitted AR(10)
    X, E, NH, HCTX = [], [], [], []
    for _, h, o in S:
        x = np.r_[h, o]
        A = np.column_stack([np.ones(len(h) - AR)] + [h[AR - k: len(h) - k] for k in range(1, AR + 1)])
        beta = np.linalg.lstsq(A, h[AR:], rcond=None)[0]
        e = lfilter(np.r_[1.0, -beta[1:]], [1.0], x) - beta[0]
        e[:AR] = np.nan
        X.append(x)
        E.append(e)
        NH.append(len(h))
        HCTX.append((_acf(h, 1)[0], _moments(h)[1]))
    say(f"AR(10) residuals in {time.time() - t0:.0f}s")

    sections = []
    for L in LENS:
        rng = np.random.default_rng(args.seed + L)
        rows, ids, ys, cuts, P = [], [], [], [], {k: [] for k in ("q", "acf_x", "acf_e", "acf_abse", "spec")}
        EV, keep_raw = [], []
        tl = time.time()
        for i, (sid, h, o) in enumerate(S):
            t = int(tau[i])
            if t >= 0:
                if len(o) - t < L:
                    continue
                cut, y = t, 1
            else:
                if len(o) < L:
                    continue
                cut, y = int(rng.integers(0, len(o) - L + 1)), 0
            x, e, c = X[i], E[i], NH[i] + cut
            pre, post = x[c - PRE: c], x[c: c + L]
            Sa, pa, ra = seg_feats(pre, e[c - PRE: c])
            Sb, pb, rb = seg_feats(post, e[c: c + L])
            rows.append(deltas(Sa, Sb, ra, rb, pre, post))
            for k in P:
                P[k].append(pb[k] - pa[k])
            EV.append(event_rows(x, e, c, L, pre.mean(), pre.std(), pre.var(), np.median(pre)))
            ids.append(sid)
            ys.append(y)
            cuts.append((i, cut))
        D = np.array(rows)
        bad = [NAMES[j] for j in range(len(NAMES)) if not np.isfinite(D[:, j]).all()]
        assert not bad, f"non-finite deltas in {bad}"
        y = np.array(ys)
        P = {k: np.array(v) for k, v in P.items()}
        EVA = np.array(EV)
        lvx, lve, lvd = (D[:, NAMES.index(n)] for n in ("var_x", "var_e", "var_dx"))
        slices = {"all": np.ones(len(y), bool), "var": np.abs(lvx) < args.band}
        slices["var+"] = slices["var"] & (np.abs(lve) < args.band) & (np.abs(lvd) < args.band)
        say(f"\n=== L = {L}: {int(y.sum())} break / {int((1 - y).sum())} placebo segments "
            f"({time.time() - tl:.0f}s)")
        for sn, m in slices.items():
            say(f"  slice {sn:4s}: {int(y[m].sum()):5d} break / {int((1 - y[m]).sum()):5d} placebo  "
                f"(break share kept {y[m].sum() / max(y.sum(), 1):.1%}, placebo {(1 - y[m]).sum() / max((1 - y).sum(), 1):.1%})")

        # AUC tables
        res = {}
        for sn, m in slices.items():
            Mg = magnitude(D[m], y[m])
            res[sn] = dict(mag=np.array([auc(Mg[:, j], y[m]) for j in range(len(NAMES))]),
                           sgn=np.array([auc(D[m][:, j], y[m]) for j in range(len(NAMES))]))
        m = slices["var"]
        Dm, ym = D[m], y[m]
        Mg = magnitude(Dm, ym)
        brng = np.random.default_rng(args.seed + 1000 + L)
        i1, i0 = np.flatnonzero(ym == 1), np.flatnonzero(ym == 0)
        boots = np.empty((args.boot, len(NAMES)))
        for b in range(args.boot):
            ib = np.r_[brng.choice(i1, len(i1)), brng.choice(i0, len(i0))]
            boots[b] = [auc(Mg[ib, j], ym[ib]) for j in range(len(NAMES))]
        lo, hi = np.nanpercentile(boots, [2.5, 97.5], axis=0)
        pmax = np.empty(args.perm)
        for b in range(args.perm):
            yp = brng.permutation(ym)
            Mp = magnitude(Dm, yp)
            pmax[b] = np.nanmax(np.abs(np.array([auc(Mp[:, j], yp) for j in range(len(NAMES))]) - 0.5))
        chance = float(np.percentile(pmax, 95)) if args.perm else 0.03
        say(f"  chance level (95th pct of the best |AUC - 0.5| over {len(NAMES)} stats, {args.perm} label "
            f"permutations): 0.5 +- {chance:.4f}")
        j_lv = NAMES.index("var_x")
        say(f"  sanity: var_x magnitude AUC in the var slice {res['var']['mag'][j_lv]:.4f} (should be ~0.5)")

        # omnibus: robust z of every non-matching statistic vs the slice's placebos
        cols, zs = [], []
        for j, n in enumerate(NAMES):
            if n in MATCHED:
                continue
            p = Mg[ym == 0, j]
            med = np.median(p)
            sc = 1.4826 * np.median(np.abs(p - med))
            if sc <= 1e-9 * (1 + abs(med)):
                sc = p.std()
            if sc <= 1e-9 * (1 + abs(med)):   # (almost) constant among placebos, e.g. rep_x: no usable scale
                continue
            cols.append(j)
            zs.append((Mg[:, j] - med) / sc)
        Z = np.column_stack(zs)
        maxz = Z.max(1)
        topstat = np.array([NAMES[cols[k]] for k in Z.argmax(1)])
        thr = np.percentile(maxz[ym == 0], 95)
        frac = (maxz[ym == 1] > thr).mean()
        say(f"  omnibus: {frac:.1%} of slice breaks exceed the placebos' 95th percentile of max robust z "
            f"(5% = nothing unusual); AUC of max z {auc(maxz, ym):.4f}")
        order = np.argsort(-res["var"]["mag"])
        say(f"  top statistics on the var slice (magnitude AUC [95% CI], signed AUC, all-series AUC):")
        for j in order[:12]:
            flag = "  above chance" if res["var"]["mag"][j] - 0.5 > chance else ""
            say(f"    {NAMES[j]:12s} {res['var']['mag'][j]:.4f} [{lo[j]:.3f}, {hi[j]:.3f}]  signed "
                f"{res['var']['sgn'][j]:.4f}  all {res['all']['mag'][j]:.4f}{flag}")

        # CSV outputs
        with open(os.path.join(args.out, f"stats_L{L}.csv"), "w") as f:
            f.write("stat,family,auc_mag_var,ci_lo,ci_hi,auc_signed_var,auc_mag_all,auc_mag_varplus,"
                    "median_delta_break_var,median_delta_placebo_var,description\n")
            for j in order:
                n = NAMES[j]
                f.write(f"{n},{STATS[n][0]},{res['var']['mag'][j]:.4f},{lo[j]:.4f},{hi[j]:.4f},"
                        f"{res['var']['sgn'][j]:.4f},{res['all']['mag'][j]:.4f},{res['var+']['mag'][j]:.4f},"
                        f"{np.median(Dm[ym == 1, j]):.5g},{np.median(Dm[ym == 0, j]):.5g},\"{STATS[n][2]}\"\n")
        mz_full = np.full(len(y), np.nan)
        mz_full[m] = maxz
        ts_full = np.full(len(y), "", dtype=object)
        ts_full[m] = topstat
        with open(os.path.join(args.out, f"series_L{L}.csv"), "w") as f:
            f.write("id,is_break,online_cut,slice_var,slice_varplus,max_robust_z,most_unusual," + ",".join(NAMES) + "\n")
            for r in range(len(y)):
                f.write(f"{ids[r]},{y[r]},{cuts[r][1]},{int(slices['var'][r])},{int(slices['var+'][r])},"
                        f"{mz_full[r]:.3f},{ts_full[r]}," + ",".join(f"{v:.5g}" for v in D[r]) + "\n")

        # ---------------------------------------------------------------- figures
        figs = []
        # 1 selection
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.2))
        bins = np.linspace(-1.5, 1.5, 61)
        for yy, col, lab in ((0, C_PLA, "placebo cut (no break)"), (1, C_BRK, "break")):
            ax[0].hist(np.clip(lvx[y == yy], -1.5, 1.5), bins, density=True, histtype="step", color=col, lw=2, label=lab)
        ax[0].axvspan(-args.band, args.band, color=MUTED, alpha=0.15, lw=0)
        ax[0].set_xlabel("log variance ratio, post / pre (x)")
        ax[0].set_ylabel("density")
        ax[0].set_title("Variance change at the cut; shaded = the 'variance unchanged' slice")
        ax[0].legend(loc="upper left")
        names_ = list(slices)
        nb_ = [int(y[slices[s]].sum()) for s in names_]
        np_ = [int((1 - y[slices[s]]).sum()) for s in names_]
        xx = np.arange(len(names_))
        ax[1].bar(xx - 0.2, np_, 0.38, color=C_PLA, label="placebo")
        ax[1].bar(xx + 0.2, nb_, 0.38, color=C_BRK, label="break")
        for k in range(len(names_)):
            ax[1].text(xx[k] - 0.2, np_[k], f"{np_[k]}", ha="center", va="bottom", color=INK2, fontsize=8)
            ax[1].text(xx[k] + 0.2, nb_[k], f"{nb_[k]}", ha="center", va="bottom", color=INK2, fontsize=8)
        ax[1].set_xticks(xx, ["all", "variance\nunchanged", "variance, innovation\nvariance, roughness\nunchanged"])
        ax[1].set_ylabel("segments")
        ax[1].set_title("Segments per slice")
        ax[1].legend()
        figs.append(("1. Which series are in the slice",
                     "Left: how much the variance moved at the cut. Breaks have a wider spread than placebos (that "
                     "is what the models already use). The shaded band is the slice studied below. Right: how many "
                     "segments survive each slice. Everything below compares breaks with placebos INSIDE the "
                     "variance-unchanged slice.", fig_b64(fig)))

        # 2 forest
        fig, ax = plt.subplots(figsize=(9, 0.19 * len(NAMES) + 1.2))
        yy_ = np.arange(len(order))[::-1]
        ax.axvspan(0.5 - chance, 0.5 + chance, color=MUTED, alpha=0.15, lw=0, label="chance band for the best of all stats")
        ax.axvline(0.5, color=MUTED, lw=1)
        mv = res["var"]["mag"][order]
        ax.hlines(yy_, lo[order], hi[order], color=C_PLA, lw=2)
        ax.plot(mv, yy_, "o", color=C_PLA, ms=5, label="variance-unchanged slice (95% bootstrap interval)")
        ax.plot(res["all"]["mag"][order], yy_, "o", mfc="none", mec=INK2, ms=5, mew=1.2, label="all segments")
        ax.set_yticks(yy_, [f"{NAMES[j]}  ({STATS[NAMES[j]][0]})" for j in order], fontsize=7.5)
        ax.set_xlabel("magnitude AUC, break vs placebo (0.5 = no information)")
        ax.set_ylim(-1, len(order))
        ax.legend(loc="lower right", fontsize=8)
        ax.set_title(f"Which statistics still separate breaks when the variance did not move (L = {L})")
        figs.append(("2. What separates breaks from placebos on the slice",
                     "Each row is one statistic (post window minus the 1,000 points before the cut). Filled = "
                     "the variance-unchanged slice, hollow = all segments. Points inside the grey band are "
                     "consistent with chance, even for the best of all these statistics. Scale rows (var_x and "
                     "friends) are near 0.5 on the slice by construction.", fig_b64(fig)))

        # 3 quantiles, 4 ACF, 5 spectrum: signed and magnitude profiles
        def profile_fig(blocks, xlab, title):
            fig, ax = plt.subplots(len(blocks), 2, figsize=(11, 2.8 * len(blocks)), squeeze=False)
            for r, (key, xv, lab) in enumerate(blocks):
                A = P[key][m]
                med0 = np.median(A[ym == 0], 0)
                for yy, col, nm in ((0, C_PLA, "placebo"), (1, C_BRK, "break")):
                    band(ax[r, 0], xv, A[ym == yy], col, nm)
                    band(ax[r, 1], xv, np.abs(A[ym == yy] - med0), col, nm)
                ax[r, 0].axhline(0, color=MUTED, lw=1)
                ax[r, 0].set_title(f"{lab}: mean change (signed)")
                ax[r, 1].set_title(f"{lab}: mean size of the change |change - placebo median|")
                for c_ in (0, 1):
                    ax[r, c_].set_xlabel(xlab)
                ax[r, 0].legend(fontsize=8)
            fig.suptitle(title, color=INK, fontsize=10)
            fig.tight_layout()
            return fig_b64(fig)

        figs.append(("3. Distribution shape",
                     "Quantiles of each window standardised by its own mean and sd, post minus pre. Variance is "
                     "fixed on this slice, so any gap here is a change of SHAPE (tails vs shoulders vs centre). "
                     "Left: average signed shift (changes in opposite directions cancel). Right: average size of "
                     "the shift: if the orange line sits above the blue one, breaks reshape the distribution.",
                     profile_fig([("q", QLEV, "quantile shift")], "quantile level",
                                 f"Quantile profile, variance-unchanged slice (L = {L})")))
        lags = np.arange(1, NACF + 1)
        figs.append(("4. Dependence: autocorrelation by lag",
                     "Change of the autocorrelation function, post minus pre, for the raw series x, the AR(10) "
                     "residual e (structure the history's AR model no longer removes) and |e| (volatility "
                     "clustering). Right column = size of the change.",
                     profile_fig([("acf_x", lags, "ACF of x"), ("acf_e", lags, "ACF of e"),
                                  ("acf_abse", lags, "ACF of |e|")], "lag",
                                 f"Autocorrelation change, variance-unchanged slice (L = {L})")))
        fc = (SPEC_BINS[:-1] + SPEC_BINS[1:]) / 2
        figs.append(("5. Spectrum",
                     "Share of the series' power per frequency bin, post minus pre (low frequency = slow wander, "
                     "0.5 = alternating every step). A rise on the left with a fall on the right = smoother, more "
                     "persistent motion at the same variance.",
                     profile_fig([("spec", fc, "power share")], "frequency (cycles per step)",
                                 f"Spectrum change, variance-unchanged slice (L = {L})")))

        # 6 event study
        offs = np.arange(EV_LO, L - EV_W + 1, EV_STEP) + EV_W / 2
        fig, ax = plt.subplots(2, 3, figsize=(12, 5.6))
        EVm = EVA[m]
        for k in range(6):
            a_ = ax.flat[k]
            for yy, col, nm in ((0, C_PLA, "placebo"), (1, C_BRK, "break")):
                band(a_, offs, EVm[ym == yy][:, :, k], col, nm)
            a_.axvline(0, color=INK2, lw=1, ls="--")
            a_.set_title(EV_NAMES[k])
            a_.set_xlabel("window centre, points from the cut")
        ax.flat[0].legend(fontsize=8)
        fig.suptitle(f"Rolling {EV_W}-point statistics aligned on the cut, variance-unchanged slice (L = {L})",
                     color=INK, fontsize=10)
        fig.tight_layout()
        figs.append(("6. Event study: does anything switch at the break?",
                     f"{EV_W}-point windows slid across the cut (dashed line). Windows centred left of "
                     f"-{EV_W // 2} are fully before it, right of +{EV_W // 2} fully after. A step in the orange "
                     "line at 0 that the blue line does not show is a change caused by the break. Averages hide "
                     "changes that go both ways; the forest plot and section 3-4 magnitudes catch those.",
                     fig_b64(fig)))

        # 7 top statistics distributions
        top6 = [j for j in order if NAMES[j] not in MATCHED][:6]
        fig, ax = plt.subplots(2, 3, figsize=(12, 5.4))
        for k, j in enumerate(top6):
            a_ = ax.flat[k]
            v = Dm[:, j]
            lo_, hi_ = np.percentile(v, [1, 99])
            bins = np.linspace(lo_, hi_ if hi_ > lo_ else lo_ + 1, 41)
            for yy, col, nm in ((0, C_PLA, "placebo"), (1, C_BRK, "break")):
                a_.hist(np.clip(v[ym == yy], lo_, hi_), bins, density=True, histtype="step", color=col, lw=2, label=nm)
            a_.set_title(f"{NAMES[j]}  (AUC {res['var']['mag'][j]:.3f})")
            a_.set_xlabel("post minus pre" if STATS[NAMES[j]][1] != "dist" else "distance")
        ax.flat[0].legend(fontsize=8)
        fig.tight_layout()
        figs.append(("7. The six best statistics, break vs placebo",
                     "Distributions on the slice. Wider orange than blue = breaks move the statistic in both "
                     "directions; a shifted orange = a consistent direction.", fig_b64(fig)))

        # 8 omnibus
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
        hb = np.linspace(min(maxz.min(), 0), min(np.percentile(maxz, 99), 4 * thr), 50)
        for yy, col, nm in ((0, C_PLA, "placebo"), (1, C_BRK, "break")):
            ax[0].hist(np.clip(maxz[ym == yy], hb[0], hb[-1]), hb, density=True, histtype="step", color=col, lw=2, label=nm)
        ax[0].axvline(thr, color=INK2, ls="--", lw=1)
        ax[0].text(thr, ax[0].get_ylim()[1] * 0.9, f" placebo 95th pct\n {frac:.1%} of breaks above (5% = chance)",
                   color=INK2, fontsize=8, va="top")
        ax[0].set_xlabel(f"max robust z over {len(cols)} statistics (right edge = clipped tail)")
        ax[0].legend(loc="center right")
        ax[0].set_title("Is a break unusual in ANY aspect?")
        un = topstat[(ym == 1) & (maxz > thr)]
        if len(un):
            fam_counts = {}
            for n in un:
                fam_counts[STATS[n][0]] = fam_counts.get(STATS[n][0], 0) + 1
            pl = topstat[(ym == 0) & (maxz > thr)]
            fam_p = {}
            for n in pl:
                fam_p[STATS[n][0]] = fam_p.get(STATS[n][0], 0) + 1
            fams = [f_ for f_ in FAMILIES if f_ in fam_counts or f_ in fam_p]
            xx = np.arange(len(fams))
            ax[1].barh(xx + 0.2, [fam_counts.get(f_, 0) / max(len(un), 1) for f_ in fams], 0.38, color=C_BRK, label="unusual breaks")
            ax[1].barh(xx - 0.2, [fam_p.get(f_, 0) / max(len(pl), 1) for f_ in fams], 0.38, color=C_PLA, label="unusual placebos")
            ax[1].set_yticks(xx, fams)
            ax[1].set_xlabel("share of the unusual segments whose most extreme statistic is in this family")
            ax[1].legend(fontsize=8)
            ax[1].set_title("In what way are the unusual ones unusual?")
        fig.tight_layout()
        figs.append(("8. Omnibus: how many slice breaks show ANY detectable change?",
                     "For every segment, each statistic's size of change is turned into a robust z against the "
                     "slice's placebos, and the largest is kept. By construction 5% of placebos exceed their 95th "
                     "percentile. The excess of breaks above 5% is the share of variance-unchanged breaks that "
                     "leave some visible trace (at this window length, break position known). Right: which family "
                     "carried the most extreme statistic, for breaks vs for the false-alarm placebos.",
                     fig_b64(fig)))

        # 9 gallery
        grng = np.random.default_rng(args.seed + 7 + L)
        idx_m = np.flatnonzero(m)
        brk_r = grng.choice(idx_m[ym == 1], min(6, int(ym.sum())), replace=False)
        brk_u = idx_m[ym == 1][np.argsort(-maxz[ym == 1])[:4]]
        pla_r = grng.choice(idx_m[ym == 0], min(4, int((1 - ym).sum())), replace=False)
        pick = [(r, "random break") for r in brk_r] + [(r, "most unusual break") for r in brk_u] + \
               [(r, "random placebo") for r in pla_r]
        pos_in_slice = {r: k for k, r in enumerate(idx_m)}
        fig, ax = plt.subplots(len(pick), 3, figsize=(13, 1.9 * len(pick)), gridspec_kw=dict(width_ratios=[3.2, 1, 1]))
        for row, (r, kind) in enumerate(pick):
            i, cut = cuts[r]
            x, c = X[i], NH[i] + cut
            col = C_BRK if y[r] else C_PLA
            tt = np.arange(-300, L)
            ax[row, 0].plot(tt[:300], x[c - 300: c], color=MUTED, lw=0.8)
            ax[row, 0].plot(tt[300:], x[c: c + L], color=col, lw=0.8)
            ax[row, 0].axvline(0, color=INK2, ls="--", lw=1)
            k = pos_in_slice[r]
            ax[row, 0].set_title(f"{kind}  id {ids[r]}  log var ratio {lvx[r]:+.2f}  most unusual: {topstat[k]} "
                                 f"(z {maxz[k]:.1f})", fontsize=8, loc="left")
            pre, post = x[c - PRE: c], x[c: c + L]
            hb = np.linspace(-4, 4, 33)
            ax[row, 1].hist(np.clip((pre - pre.mean()) / pre.std(), -4, 4), hb, density=True, histtype="step", color=MUTED, lw=1.5, label="pre")
            ax[row, 1].hist(np.clip((post - post.mean()) / post.std(), -4, 4), hb, density=True, histtype="step", color=col, lw=1.5, label="post")
            ax[row, 2].plot(np.arange(1, 11), _acf(pre, 10), "o-", color=MUTED, ms=3, lw=1, label="pre")
            ax[row, 2].plot(np.arange(1, 11), _acf(post, 10), "o-", color=col, ms=3, lw=1, label="post")
            ax[row, 2].axhline(0, color=MUTED, lw=0.6)
            if row == 0:
                ax[row, 1].set_title("own-standardised histogram", fontsize=8)
                ax[row, 2].set_title("ACF lags 1-10", fontsize=8)
                ax[row, 1].legend(fontsize=7)
        ax[-1, 0].set_xlabel("points from the cut (grey = before, colour = after)")
        fig.tight_layout()
        figs.append(("9. Gallery",
                     "Slice members drawn at random, the four most unusual breaks, and random placebos for "
                     "comparison. Can you tell the breaks from the placebos by eye? Every series of the slice is "
                     f"listed in series_L{L}.csv (slice_var = 1) with its deltas and its most unusual statistic.",
                     fig_b64(fig)))

        # table for the report
        trs = []
        for j in order[:25]:
            n = NAMES[j]
            above = res["var"]["mag"][j] - 0.5 > chance
            trs.append(f"<tr{' class=hit' if above else ''}><td>{n}</td><td>{STATS[n][0]}</td>"
                       f"<td>{res['var']['mag'][j]:.3f} <span class=ci>[{lo[j]:.3f}, {hi[j]:.3f}]</span></td>"
                       f"<td>{res['var']['sgn'][j]:.3f}</td><td>{res['var+']['mag'][j]:.3f}</td>"
                       f"<td>{res['all']['mag'][j]:.3f}</td><td>{html.escape(STATS[n][2])}</td></tr>")
        hits = [NAMES[j] for j in order if res["var"]["mag"][j] - 0.5 > chance and NAMES[j] not in MATCHED]
        summ = (f"<p><b>{int(ym.sum())}</b> breaks and <b>{int((1 - ym).sum())}</b> placebos have an unchanged variance "
                f"(|log ratio| &lt; {args.band}). Sanity: the variance itself scores {res['var']['mag'][j_lv]:.3f} "
                f"on the slice (should be ~0.5). Chance band for the best statistic: 0.5 &plusmn; {chance:.3f}.</p>"
                f"<p><b>Above chance on the slice:</b> {', '.join(hits[:12]) if hits else 'nothing'}"
                f"{' ...' if len(hits) > 12 else ''}.</p>"
                f"<p><b>Omnibus:</b> {frac:.1%} of slice breaks are more unusual than 95% of slice placebos "
                f"(5% = no trace at all); AUC of the max robust z {auc(maxz, ym):.3f}.</p>")
        table = ("<table><tr><th>statistic</th><th>family</th><th>AUC |&Delta;| var slice [95% CI]</th>"
                 "<th>signed AUC</th><th>AUC |&Delta;| var+ slice</th><th>AUC |&Delta;| all</th><th>meaning</th></tr>"
                 + "".join(trs) + "</table>")
        sections.append((L, summ, table, figs))

    # ---------------------------------------------------------------- HTML
    css = ("body{background:#fcfcfb;color:#0b0b0b;font:14px/1.5 system-ui,Segoe UI,sans-serif;max-width:1150px;"
           "margin:0 auto;padding:16px}h1{font-size:22px}h2{font-size:18px;margin-top:36px;border-top:1px solid #e4e3df;"
           "padding-top:16px}h3{font-size:15px;margin-bottom:2px}p.cap{color:#52514e;margin-top:2px}"
           "img{max-width:100%;height:auto}table{border-collapse:collapse;font-size:12px;width:100%}"
           "td,th{border-bottom:1px solid #e4e3df;padding:3px 6px;text-align:left;vertical-align:top}"
           "tr.hit td:first-child{font-weight:600;color:#c24a1c}.ci{color:#8a8984}nav a{margin-right:12px}"
           ".note{background:#f0efec;padding:8px 12px;border-radius:6px}")
    parts = [f"<!doctype html><html><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
             f"<title>Variance-matched breaks</title><style>{css}</style></head><body>",
             "<h1>At breaks that leave the variance unchanged, what changes?</h1>",
             f"<p class=note>Oracle EDA: break position known. Pre window = the {PRE} points before the cut, post "
             f"window = exactly L points after it. Placebo cuts in no-break series are the control: anything that "
             f"also happens at an arbitrary point cancels. Magnitude AUC = how well the SIZE of a change separates "
             f"breaks from placebos, whatever its direction. {N} series; generated {time.strftime('%Y-%m-%d %H:%M')} "
             f"by diagnostics/eda_varmatched.py.</p>",
             "<nav>" + "".join(f"<a href='#L{L}'>L = {L}</a>" for L, *_ in sections) + "</nav>"]
    for L, summ, table, figs in sections:
        parts.append(f"<h2 id=L{L}>Post window L = {L} points</h2>{summ}<h3>Top 25 statistics (bold = above the "
                     f"chance band)</h3>{table}")
        for title, cap, b64 in figs:
            parts.append(f"<h3>{html.escape(title)}</h3><p class=cap>{html.escape(cap)}</p>"
                         f"<img alt='{html.escape(title)}' src='data:image/png;base64,{b64}'>")
    parts.append("</body></html>")
    rep = os.path.join(args.out, "report.html")
    with open(rep, "w", encoding="utf-8") as f:
        f.write("".join(parts))
    say(f"\nwrote {rep} and the CSVs in {args.out} ({time.time() - t0:.0f}s total)")
    with open(os.path.join(args.out, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log) + "\n")


if __name__ == "__main__":
    main()
