"""Simulation: which statistic catches which kind of structural break?

Run:    python sim_stats.py
Faster: python sim_stats.py --trials 300 --trials-b 200
Writes: sim_out/summary.txt, sim_out/*.csv, sim_out/*.png

What it does
  Part A  Build synthetic series (AR(1) base process, history z-scored like the competition data),
          inject one kind of break at a time, and measure how well each statistic separates
          break from no-break when computed on the first L points after the break.
  Part B  Compare *detectors* built on the same statistic over time: running mean over the whole
          online segment, last-30 / last-100 windows, and the scan (max over all start points).
          This is scored with the competition metric (TS-AUC), with the break at a random time.
  Part C  A gallery of one example series per break type, to see what each break looks like.
"""
import argparse
import os
import sys
import time

import numpy as np
from scipy.signal import lfilter
from scipy.stats import rankdata, skew, kurtosis

# ---------------------------------------------------------------------------------------------
# settings (edit freely)
N_HIST = 2000        # history length
N_ON = 400           # online length used in Part B / C
BASE_PHI = 0.3       # AR(1) coefficient of the base process (unit marginal variance)
BURN = 100
AR = 10              # order of the AR residual statistic

parser = argparse.ArgumentParser()
parser.add_argument("--trials", type=int, default=800, help="trials per break type in Part A")
parser.add_argument("--trials-b", type=int, default=500, help="series per family in Part B")
parser.add_argument("--Ls", type=str, default="30,100,300", help="post-break lengths for Part A (max 300)")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", type=str, default="sim_out")
args = parser.parse_args()
LS = [int(v) for v in args.Ls.split(",")]
assert max(LS) <= 300, "Part A builds 300 post-break points; use --Ls with values <= 300"

os.makedirs(args.out, exist_ok=True)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None
    print("matplotlib not found -> tables only (pip install matplotlib for figures)")


class Tee:
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")

    def write(self, s):
        sys.__stdout__.write(s)
        self.f.write(s)

    def flush(self):
        sys.__stdout__.flush()
        self.f.flush()


sys.stdout = Tee(os.path.join(args.out, "summary.txt"))
rng = np.random.default_rng(args.seed)


def section(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78, flush=True)


def savefig(name):
    if plt is None:
        return
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, name), dpi=110)
    plt.close()
    print(f"[saved {args.out}/{name}]")


# ---------------------------------------------------------------------------------------------
# process generators
def innov(kind, n, r):
    if kind == "gauss":
        return r.standard_normal(n)
    if kind == "t3":  # heavy tails, unit variance
        return r.standard_t(3, n) / np.sqrt(3.0)
    if kind == "skew":  # centred exponential: unit variance, skewness 2
        return r.exponential(1.0, n) - 1.0
    raise ValueError(kind)


def garch_innov(n, r, a=0.3, b=0.6):
    """GARCH(1,1) innovations with unit unconditional variance (volatility clustering)."""
    w = 1.0 - a - b
    z = r.standard_normal(n)
    e = np.empty(n)
    s2 = 1.0
    for t in range(n):
        e[t] = np.sqrt(s2) * z[t]
        s2 = w + a * e[t] ** 2 + b * s2
    return e


def gen_process(n, r, phi=BASE_PHI, kind="gauss", garch=False):
    """AR(1) with unit marginal variance driven by the chosen innovations."""
    m = n + BURN
    e = garch_innov(m, r) if garch else innov(kind, m, r)
    x = lfilter([np.sqrt(1.0 - phi ** 2)], [1.0, -phi], e)
    return x[BURN:]


BREAKS = {
    "mean +0.5 sd": lambda n, r: gen_process(n, r) + 0.5,
    "variance x1.5": lambda n, r: 1.5 * gen_process(n, r),
    "variance x0.67": lambda n, r: 0.67 * gen_process(n, r),
    "dependence up (phi .3 -> .7)": lambda n, r: gen_process(n, r, phi=0.7),
    "dependence down (phi .3 -> -.3)": lambda n, r: gen_process(n, r, phi=-0.3),
    "heavy tails (t3 noise)": lambda n, r: gen_process(n, r, kind="t3"),
    "skewed noise": lambda n, r: gen_process(n, r, kind="skew"),
    "vol clustering (GARCH)": lambda n, r: gen_process(n, r, garch=True),
}


def make_series(brk, tau, r, n_on=N_ON):
    """History + online segment, z-scored with the history's own mean/std (like the real data).
    brk=None -> no break. Break (if any) starts at online index tau."""
    pre = gen_process(N_HIST + tau, r)
    hist, pre_on = pre[:N_HIST], pre[N_HIST:]
    m, s = hist.mean(), hist.std()
    post = BREAKS[brk](n_on - tau, r) if brk else gen_process(n_on - tau, r)
    online = np.r_[pre_on, post]
    return (hist - m) / s, (online - m) / s


# ---------------------------------------------------------------------------------------------
# Part A statistics
STAT_NAMES = ["mean", "log variance", "mean |x|", "roughness (diff var)", "tail share >2",
              "autocorr lag1", "autocorr lag2", "skew", "excess kurtosis", "vol clustering",
              "AR(10) resid var", "decile chi2"]
# 1 = report (segment - history); 0 = report the segment value only (history value is 0 / by construction)
REL = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], dtype=float)


def make_ctx(h):
    n = len(h)
    Xl = np.column_stack([h[AR - 1 - i: n - 1 - i] for i in range(AR)])
    coef = np.linalg.lstsq(Xl, h[AR:], rcond=None)[0]
    edges = np.quantile(h, np.arange(1, 10) / 10.0)
    return coef, edges


def _ac(x, k):
    xc = x - x.mean()
    return float(np.dot(xc[:-k], xc[k:]) / max(np.dot(xc, xc), 1e-12))


def compute_stats(x, ctx):
    coef, edges = ctx
    n = len(x)
    ax = np.abs(x)
    d = np.diff(x)
    Xl = np.column_stack([x[AR - 1 - i: n - 1 - i] for i in range(AR)])
    e = x[AR:] - Xl @ coef
    cnt = np.bincount(np.searchsorted(edges, x), minlength=10)
    chi = ((cnt - n / 10.0) ** 2).sum() / (n / 10.0)
    a = ax - ax.mean()
    return np.array([
        x.mean() * np.sqrt(n),
        np.log(np.mean(x * x) + 1e-12),
        ax.mean(),
        np.log(d.var() + 1e-12),
        np.mean(ax > 2.0),
        _ac(x, 1),
        _ac(x, 2),
        skew(x),
        kurtosis(x),
        float(np.dot(a[:-1], a[1:]) / max(np.dot(a, a), 1e-12)),
        np.log(np.mean(e * e) + 1e-12),
        (chi - 9.0) / np.sqrt(18.0),
    ])


def auc(a, b):
    r = rankdata(np.r_[a, b])
    return (r[: len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))


def part_a():
    section("PART A. Which statistic separates which break?  (break position known)")
    print(f"AR(1) base process phi={BASE_PHI}, history {N_HIST} points, {args.trials} trials per break type.")
    print("Each cell: two-sided AUC of the statistic's change (segment - history) for break vs no-break,")
    print("computed on the first L points after the break. 0.50 = useless, 1.00 = perfect.")
    print("(+) = the statistic goes UP after the break, (-) = it goes DOWN.\n")
    names = ["no break"] + list(BREAKS)
    E = {nm: np.zeros((args.trials, len(STAT_NAMES), len(LS))) for nm in names}
    t0 = time.time()
    for nm in names:
        brk = None if nm == "no break" else nm
        for i in range(args.trials):
            hist, seg = make_series(brk, 0, rng, n_on=300)
            ctx = make_ctx(hist)
            hs = compute_stats(hist, ctx)
            for li, L in enumerate(LS):
                E[nm][i, :, li] = compute_stats(seg[:L], ctx) - REL * hs
        print(f"  simulated {nm:34s} ({time.time() - t0:5.0f}s)", flush=True)

    for li, L in enumerate(LS):
        null = E["no break"][:, :, li]
        med = np.median(null, axis=0)
        A = np.zeros((len(BREAKS), len(STAT_NAMES)))
        D = np.zeros_like(A)
        for bi, nm in enumerate(BREAKS):
            be = E[nm][:, :, li]
            for si in range(len(STAT_NAMES)):
                A[bi, si] = auc(np.abs(be[:, si] - med[si]), np.abs(null[:, si] - med[si]))
                D[bi, si] = np.sign(np.median(be[:, si]) - med[si])
        print(f"\n--- L = {L} points after the break ---")
        w = 14
        print(" " * 33 + "".join(f"{s[:w - 1]:>{w}s}" for s in STAT_NAMES))
        for bi, nm in enumerate(BREAKS):
            cells = "".join(f"{A[bi, si]:>{w - 3}.2f}({'+' if D[bi, si] > 0 else '-'}) " for si in range(len(STAT_NAMES)))
            best = STAT_NAMES[int(np.argmax(A[bi]))]
            print(f"{nm:33s}" + cells + f"  best: {best}")
        np.savetxt(os.path.join(args.out, f"partA_auc_L{L}.csv"), A, delimiter=",", header=",".join(STAT_NAMES), comments="")
        if plt is not None:
            fig, ax = plt.subplots(figsize=(13, 4.6))
            im = ax.imshow(A, cmap="RdYlGn", vmin=0.5, vmax=1.0, aspect="auto")
            ax.set_xticks(range(len(STAT_NAMES)))
            ax.set_xticklabels(STAT_NAMES, rotation=35, ha="right")
            ax.set_yticks(range(len(BREAKS)))
            ax.set_yticklabels(list(BREAKS))
            for bi in range(A.shape[0]):
                for si in range(A.shape[1]):
                    ax.text(si, bi, f"{A[bi, si]:.2f}", ha="center", va="center", fontsize=8)
            ax.set_title(f"AUC of each statistic for each break type, L = {L} points after the break")
            fig.colorbar(im, ax=ax, fraction=0.025)
            savefig(f"A_heatmap_L{L}.png")


# ---------------------------------------------------------------------------------------------
# Part B: detectors over time, scored with TS-AUC
FAMILIES = [
    ("variance (g = x^2)", lambda f: f[1:] ** 2, "variance x1.5"),
    ("roughness (g = (x_t - x_t-1)^2)", lambda f: np.diff(f) ** 2, "dependence up (phi .3 -> .7)"),
    ("tail share (g = 1{|x| > 2})", lambda f: (np.abs(f[1:]) > 2).astype(float), "heavy tails (t3 noise)"),
    ("mean (g = x)", lambda f: f[1:], "mean +0.5 sd"),
]
DETECTORS = ["full online mean", "window 30", "window 100", "scan (max over start j)"]


def lrv_batch(g):
    c = g - g.mean()
    n = len(c)
    b = max(5, min(50, n // 20))
    nb = n // b
    bm = c[: nb * b].reshape(nb, b).sum(axis=1) / np.sqrt(b)
    return max(float(bm.var()), 0.25 * float(c.var()), 1e-8)


def detector_scores(g_on, g_hist):
    """Two-sided standardised scores for the 4 detectors at every online step."""
    c = g_hist.mean()
    lrv = lrv_batch(g_hist)
    C = np.r_[0.0, np.cumsum(g_on - c)]
    n_on = len(g_on)
    n = np.arange(1, n_on + 1)
    out = {}
    out["full online mean"] = np.abs(C[n]) / np.sqrt(n * lrv)
    for w, nm in ((30, "window 30"), (100, "window 100")):
        we = np.minimum(w, n)
        out[nm] = np.abs(C[n] - C[n - we]) / np.sqrt(we * lrv)
    t = np.arange(n_on)[:, None]
    j = np.arange(n_on)[None, :]
    L = t + 1 - j
    val = np.abs(C[t + 1] - C[j]) / np.sqrt(np.maximum(L, 1) * lrv)
    out["scan (max over start j)"] = np.where(L >= 8, val, 0.0).max(axis=1)
    return out


def ts_auc_mat(S, Y):
    num = den = 0.0
    curve = np.full(S.shape[1], np.nan)
    for t in range(S.shape[1]):
        y = Y[:, t]
        npos = int(y.sum())
        nneg = len(y) - npos
        if npos < 3 or nneg < 3:
            continue
        r = rankdata(S[:, t])
        a = (r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)
        curve[t] = a
        num += npos * nneg * a
        den += npos * nneg
    return (num / den if den else 0.5), curve


def part_b():
    section("PART B. Detector design over time (TS-AUC, break at a random time in the online segment)")
    print(f"{args.trials_b} series per family, half with a break at tau ~ Uniform(0, 250), online length {N_ON}.")
    print("Same statistic, four ways to turn it into a detector. Higher TS-AUC is better.\n")
    n = args.trials_b
    results = {}
    for fname, gfun, brk in FAMILIES:
        S = {d: np.zeros((n, N_ON)) for d in DETECTORS}
        Y = np.zeros((n, N_ON), dtype=int)
        for i in range(n):
            is_b = i % 2 == 0
            tau = int(rng.integers(0, 250)) if is_b else -1
            hist, online = make_series(brk if is_b else None, max(tau, 0), rng)
            g_h = gfun(hist)
            g_on = gfun(np.r_[hist[-1], online])
            sc = detector_scores(g_on, g_h)
            for d in DETECTORS:
                S[d][i] = sc[d]
            if is_b:
                Y[i, tau:] = 1
        results[fname] = {}
        for d in DETECTORS:
            tsa, curve = ts_auc_mat(S[d], Y)
            results[fname][d] = (tsa, curve)
        print(f"  done {fname}", flush=True)

    print("\nTS-AUC by family and detector (family is paired with the break it is designed for):")
    print("%-34s %-26s" % ("family", "break") + "".join(f"{d:>26s}" for d in DETECTORS))
    for fname, _, brk in FAMILIES:
        print("%-34s %-26s" % (fname, brk[:25]) + "".join(f"{results[fname][d][0]:>26.3f}" for d in DETECTORS))
    if plt is not None:
        fig, axes = plt.subplots(1, len(FAMILIES), figsize=(5 * len(FAMILIES), 3.8), sharey=True)
        k = 15
        for ax, (fname, _, brk) in zip(axes, FAMILIES):
            for d in DETECTORS:
                tsa, curve = results[fname][d]
                cc = np.where(np.isnan(curve), 0.5, curve)
                sm = np.convolve(cc, np.ones(k) / k, mode="same")
                ax.plot(sm, label=f"{d} ({tsa:.3f})")
            ax.axhline(0.5, color="k", lw=0.6, ls=":")
            ax.set_title(f"{fname}\nbreak: {brk}", fontsize=9)
            ax.set_xlabel("online step t")
            ax.legend(fontsize=7)
        axes[0].set_ylabel("AUC across series at step t (smoothed)")
        savefig("B_detectors_over_time.png")


# ---------------------------------------------------------------------------------------------
def part_c():
    section("PART C. Gallery: one example of every break (break at online step 150)")
    if plt is None:
        print("(plots skipped)")
        return
    names = ["no break"] + list(BREAKS)
    fig, axes = plt.subplots(3, 3, figsize=(16, 9))
    for ax, nm in zip(axes.ravel(), names):
        hist, online = make_series(None if nm == "no break" else nm, 150, rng)
        v = np.r_[hist[-150:], online]
        x = np.arange(-150, N_ON)
        ax.plot(x, v, lw=0.7)
        ra = np.convolve(np.abs(v), np.ones(30) / 30, mode="same")
        ax.plot(x, ra, color="tab:orange", lw=1.3)
        ax.axvline(0, color="k", ls="--", lw=0.8)
        if nm != "no break":
            ax.axvline(150, color="r", lw=1.2)
        ax.set_title(nm, fontsize=10)
    fig.suptitle("last 150 history points | dashed = online start | red = break | orange = rolling mean |x| (30)")
    savefig("C_gallery.png")


def guide():
    section("HOW TO READ THIS")
    print("""Expected pattern if the maths holds (check against Part A):
  mean shift            -> 'mean' wins; nothing else reacts (and mean shifts were NOT common in the real data)
  variance x1.5 / x0.67 -> 'log variance' and 'mean |x|' react; 'roughness' also reacts (amplitude changes the differences)
  dependence up / down  -> 'autocorr', 'roughness' and 'AR(10) resid var' react; 'log variance' does NOT (variance is preserved)
  heavy tails           -> 'tail share', 'excess kurtosis', 'decile chi2' react; mean/variance barely do
  skewed noise          -> 'skew' (and 'decile chi2') react
  vol clustering        -> 'vol clustering' reacts, and 'excess kurtosis' (GARCH has fat tails)
Check against Part B:
  full online mean is diluted by the pre-break points when the break comes late;
  short windows are fast but noisy; the scan usually matches or beats the best fixed window without having to guess it.
Things to try: edit the effect sizes in BREAKS (e.g. variance x1.2) or BASE_PHI (try 0.8, a persistent series) and re-run.""")


if __name__ == "__main__":
    t_start = time.time()
    part_c()
    part_a()
    part_b()
    guide()
    print(f"\nDone in {time.time() - t_start:.0f}s. Outputs are in {args.out}/")
