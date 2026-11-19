"""EDA for the ADIA Lab Structural Break Challenge (Real-Time Edition).

Run:   python eda.py
Needs: the data/ folder from `crunch setup-notebook`. Writes figures + summary.txt to eda_out/.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.stats import rankdata, skew, kurtosis, ks_2samp

from sbrt.data import load_train, load_test_reduced

OUT = "eda_out"
os.makedirs(OUT, exist_ok=True)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # plots are optional
    plt = None
    print("matplotlib not found -> tables only (pip install matplotlib for figures)")


class Tee:
    """Print to the terminal and to eda_out/summary.txt."""

    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")

    def write(self, s):
        sys.__stdout__.write(s)
        self.f.write(s)

    def flush(self):
        sys.__stdout__.flush()
        self.f.flush()


sys.stdout = Tee(os.path.join(OUT, "summary.txt"))


def section(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def q(a, qs=(0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)):
    return {f"{p:g}": round(float(v), 3) for p, v in zip(qs, np.quantile(a, qs))}


def auc(a, b):
    """P(a > b) via ranks."""
    r = rankdata(np.r_[a, b])
    return (r[: len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))


def savefig(name):
    if plt is None:
        return
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, name), dpi=110)
    plt.close()
    print(f"[saved {OUT}/{name}]")


# ---------------------------------------------------------------------------------------------
S, tau = load_train()
n_series = len(S)
hist_len = np.array([len(s[1]) for s in S])
onl_len = np.array([len(s[2]) for s in S])
has_brk = tau >= 0
rel_tau = np.where(has_brk, tau / onl_len, np.nan)

# ---------------------------------------------------------------------------------------------
section("1. SIZE AND SHAPE")
print("series:", n_series, "| rows:", int((hist_len + onl_len).sum()))
print("history length  :", q(hist_len))
print("online length   :", q(onl_len))
print("share with break:", round(float(has_brk.mean()), 3))
print("online length < 50 :", round(float((onl_len < 50).mean()), 3),
      "| < 100:", round(float((onl_len < 100).mean()), 3), "| >= 500:", round(float((onl_len >= 500).mean()), 3))
print("history z-scored? mean/std of history, quantiles ->")
print("   mean:", q([s[1].mean() for s in S]))
print("   std :", q([s[1].std() for s in S]))
print("online mean/std of NO-break series (should be ~0/~1 if no change):")
nb_mean = np.array([s[2].mean() for s, b in zip(S, has_brk) if not b])
nb_std = np.array([s[2].std() for s, b in zip(S, has_brk) if not b])
print("   mean:", q(nb_mean))
print("   std :", q(nb_std))
print("unique-value ratio (1.0 = continuous, small = discrete/ties):", q([len(np.unique(s[1])) / len(s[1]) for s in S[:2000]]))

if plt is not None:
    fig, ax = plt.subplots(1, 4, figsize=(17, 3.4))
    ax[0].hist(hist_len, 50); ax[0].set_title("history length")
    ax[1].hist(onl_len, 50); ax[1].set_title("online length")
    ax[2].hist(tau[has_brk], 50); ax[2].set_title("tau_index (online step of break)")
    ax[3].hist(rel_tau[has_brk], 40); ax[3].set_title("tau / online length")
    savefig("01_lengths_and_tau.png")

# ---------------------------------------------------------------------------------------------
section("2. WHERE THE METRIC IS DECIDED (TS-AUC weights per online step)")
T = int(onl_len.max())
alive = np.array([(onl_len > t).sum() for t in range(T)])
pos = np.array([((onl_len > t) & has_brk & (tau <= t)).sum() for t in range(T)])
neg = alive - pos
w = pos.astype(float) * neg
cw = np.cumsum(w) / w.sum()
print("share of total TS-AUC weight from steps < 25 / 50 / 100 / 200 / 400 / 700:",
      [round(float(cw[min(k, T - 1)]), 3) for k in (25, 50, 100, 200, 400, 700)])
print("series alive at step 0 / 100 / 500 / 900:", [int(alive[k]) for k in (0, 100, 500, 900) if k < T])
print("positive share at step 0 / 50 / 200 / 500 / 900:",
      [round(float(pos[k] / max(alive[k], 1)), 3) for k in (0, 50, 200, 500, 900) if k < T])
print("-> early steps have almost no positives; most weight sits where both classes are plentiful.")
if plt is not None:
    fig, ax = plt.subplots(1, 3, figsize=(16, 3.4))
    ax[0].plot(alive); ax[0].set_title("series still alive at step t")
    ax[1].plot(pos / np.maximum(alive, 1)); ax[1].set_title("share positive at step t")
    ax[2].plot(w / w.sum()); ax[2].set_title("TS-AUC weight of step t (npos*nneg, normalised)")
    savefig("02_step_weights.png")

# ---------------------------------------------------------------------------------------------
section("3. IS THERE SIGNAL IN THE HISTORY ALONE? (should be none: history is break-free)")


def hist_stats(h):
    return dict(rho1=np.corrcoef(h[:-1], h[1:])[0, 1], skew=skew(h), kurt=kurtosis(h),
                tail2=(np.abs(h) > 2).mean(), n=len(h))


hs = [hist_stats(s[1]) for s in S]
for key in ("rho1", "skew", "kurt", "tail2", "n"):
    a = np.array([d[key] for d in hs])
    print(f"{key:6s} all: {q(a)}")
    print(f"       AUC(break vs no-break) = {auc(a[has_brk], a[~has_brk]):.3f}   KS p = {ks_2samp(a[has_brk], a[~has_brk]).pvalue:.3f}")
print("-> an AUC far from 0.5 here would mean history leaks the label (would be a data-generation artifact).")
rho = np.array([d["rho1"] for d in hs])
print("share of series with |lag-1 autocorr| > 0.3:", round(float((np.abs(rho) > 0.3).mean()), 3),
      "| heavy tails (excess kurtosis > 3):", round(float((np.array([d['kurt'] for d in hs]) > 3).mean()), 3))
if plt is not None:
    fig, ax = plt.subplots(1, 4, figsize=(17, 3.4))
    for a_, key in zip(ax, ("rho1", "skew", "kurt", "tail2")):
        v = np.array([d[key] for d in hs])
        lo, hi = np.quantile(v, [0.01, 0.99])
        a_.hist(v[has_brk], 50, range=(lo, hi), alpha=0.5, density=True, label="break")
        a_.hist(v[~has_brk], 50, range=(lo, hi), alpha=0.5, density=True, label="no break")
        a_.set_title("history " + key)
    ax[0].legend()
    savefig("03_history_stats.png")

# ---------------------------------------------------------------------------------------------
section("4. WHAT DO BREAKS LOOK LIKE? (effect sizes, break series vs matched no-break null)")
SEG = 60  # compare 60 points after the break (or from online start for no-break series)
rng = np.random.default_rng(0)


def seg_effects(h, seg):
    d = {}
    d["mean_shift"] = seg.mean()  # history mean is 0 and std 1
    d["log_sd_ratio"] = np.log(seg.std() + 1e-9)
    d["ac1_change"] = np.corrcoef(seg[:-1], seg[1:])[0, 1] - np.corrcoef(h[:-1], h[1:])[0, 1]
    d["abs_level"] = np.abs(seg).mean() - np.abs(h).mean()
    d["log_diff_var"] = np.log(np.diff(seg).var() + 1e-9) - np.log(np.diff(h).var() + 1e-9)
    return d


post, null = [], []
for (i, h, o), t in zip(S, tau):
    if t >= 0 and len(o) - t >= SEG:
        post.append(seg_effects(h, o[t:t + SEG]))
    elif t < 0 and len(o) >= SEG:
        st = rng.integers(0, len(o) - SEG + 1)
        null.append(seg_effects(h, o[st:st + SEG]))
print(f"break segments: {len(post)}, no-break (null) segments: {len(null)}, segment length: {SEG}")
print("%-14s %-34s %-34s %s" % ("effect", "break: 5%/50%/95%", "null: 5%/50%/95%", "AUC(|break|>|null|)"))
for key in post[0]:
    a = np.array([d[key] for d in post]); b = np.array([d[key] for d in null])
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    med = np.median(b)
    print("%-14s %-34s %-34s %.3f" % (key, np.round(np.quantile(a, [.05, .5, .95]), 3), np.round(np.quantile(b, [.05, .5, .95]), 3),
                                      auc(np.abs(a - med), np.abs(b - med))))
print("-> an AUC near 0.5 means that effect alone cannot separate break from no-break; mixed break types dilute each statistic.")
if plt is not None:
    keys = list(post[0])
    fig, ax = plt.subplots(1, len(keys), figsize=(4 * len(keys), 3.2))
    for a_, key in zip(ax, keys):
        a = np.array([d[key] for d in post]); b = np.array([d[key] for d in null])
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        lo, hi = np.quantile(np.r_[a, b], [0.01, 0.99])
        a_.hist(a, 50, range=(lo, hi), alpha=0.5, density=True, label="post-break")
        a_.hist(b, 50, range=(lo, hi), alpha=0.5, density=True, label="no-break")
        a_.set_title(key)
    ax[0].legend()
    savefig("04_break_effect_sizes.png")

# ---------------------------------------------------------------------------------------------
section("5. ORACLE DETECTABILITY (break position known, L points after it) -- upper bound per statistic")


def stat_vec(h, seg):
    z = seg
    c = lambda x, k: np.corrcoef(x[:-k], x[k:])[0, 1]
    return {
        "mean (|.|)": z.mean() * np.sqrt(len(z)),
        "log var": np.log(z.var() + 1e-9),
        "mean |z|": np.abs(z).mean() - np.abs(h).mean(),
        "ac1": c(z, 1) - c(h, 1),
        "ac2": c(z, 2) - c(h, 2),
        "skew": skew(z) - skew(h),
        "kurt": kurtosis(z) - kurtosis(h),
        "tail>2": (np.abs(z) > 2).mean() - (np.abs(h) > 2).mean(),
        "vol clustering": c(np.abs(z), 1) - c(np.abs(h), 1),
        "log var(diff)": np.log(np.diff(z).var() + 1e-9) - np.log(np.diff(h).var() + 1e-9),
    }


table = {}
for L in (30, 100, 300):
    bk, nbk = [], []
    for (i, h, o), t in zip(S, tau):
        if t >= 0 and len(o) - t >= L:
            bk.append(stat_vec(h, o[t:t + L]))
        elif t < 0 and len(o) >= L:
            nbk.append(stat_vec(h, o[:L]))
    for key in bk[0]:
        a = np.array([d[key] for d in bk]); b = np.array([d[key] for d in nbk])
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        med = np.median(b)
        table.setdefault(key, {})[L] = auc(np.abs(a - med), np.abs(b - med))
print("%-16s %8s %8s %8s   (two-sided AUC, break vs no-break segment of L points)" % ("statistic", "L=30", "L=100", "L=300"))
for key, row in sorted(table.items(), key=lambda kv: -max(kv[1].values())):
    print("%-16s %8.3f %8.3f %8.3f" % (key, row[30], row[100], row[300]))
print("-> TS-AUC on the leaderboard tops out near 0.72, so single-statistic ceilings in the 0.55-0.60 range here are expected.")

# ---------------------------------------------------------------------------------------------
section("6. EXAMPLE SERIES")
if plt is not None:
    pick_b = rng.choice(np.flatnonzero(has_brk & (onl_len > 300) & (tau > 60)), 6, replace=False)
    pick_n = rng.choice(np.flatnonzero(~has_brk & (onl_len > 300)), 6, replace=False)
    fig, axes = plt.subplots(6, 2, figsize=(15, 14))
    for r in range(6):
        for c, (pk, ttl) in enumerate(((pick_b, "BREAK"), (pick_n, "no break"))):
            i = pk[r]
            h, o = S[i][1], S[i][2]
            tail = h[-250:]
            x = np.arange(-len(tail), len(o))
            v = np.r_[tail, o]
            ax = axes[r, c]
            ax.plot(x, v, lw=0.7)
            rs = np.convolve(np.abs(v), np.ones(30) / 30, mode="same")
            ax.plot(x, rs, lw=1.2, color="tab:orange", alpha=0.8)
            ax.axvline(0, color="k", ls="--", lw=0.8)
            if tau[i] >= 0:
                ax.axvline(tau[i], color="r", lw=1.2)
            ax.set_title(f"{ttl}  id={S[i][0]}  tau={tau[i] if tau[i] >= 0 else None}", fontsize=9)
    fig.suptitle("last 250 history points | dashed = online start | red = break | orange = rolling mean |x| (30)")
    savefig("06_examples.png")
else:
    print("(plots skipped)")

# ---------------------------------------------------------------------------------------------
section("7. LOCAL REDUCED TEST SET (100 series) vs TRAIN")
test_series, y_test = load_test_reduced()
tl = np.array([len(s[2]) for s in test_series])
th = np.array([len(s[1]) for s in test_series])
print("history length train vs test medians:", int(np.median(hist_len)), int(np.median(th)))
print("online  length train vs test medians:", int(np.median(onl_len)), int(np.median(tl)))
print("positive row share (test, 100 series):", round(float(y_test.mean()), 3),
      "| train:", round(float(sum(max(0, l - t) for l, t in zip(onl_len[has_brk], tau[has_brk])) / onl_len.sum()), 3))
print("-> 100 series is far too noisy for model selection; use cross-validation on the 10,000 training series.")

print("\nDone. Figures and summary.txt are in", OUT)
