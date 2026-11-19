"""Plot a training series next to its derivatives (repeated first differences), to build intuition.

For each series one PNG on a shared time axis (0 = first online point):
  * x_t, the signal (history is z-scored, so online values are in history units)
  * dx_t = x_t - x_{t-1}, and with --order K also d2x = d(dx), d3x = d(d2x), ... up to dKx
  * trailing rolling std (window W) of every order k, divided by sqrt(C(2k, k)) (1, 2, 6, 20, 70, ...):
    the variance of the k-th difference of white noise is C(2k, k) times its variance, so for white noise all
    lines sit at the same level. They separate when the series is autocorrelated, and a gap that moves at the
    break means the dependence changed. Each difference multiplies the power spectrum by 4 sin^2(w/2), so high
    orders see only the fastest wiggles of the series (they look like +-+- alternation).
Only the last --hist points of the history are drawn so the break is not squashed.

Usage (seconds; reads Data/X_train.parquet):
    python viz/plot_derivative.py                              # curated set of 7 (5 break, 2 no-break)
    python viz/plot_derivative.py --ids 0 17 4242              # any series ids
    python viz/plot_derivative.py --random 6                   # random series
    python viz/plot_derivative.py --ids 1725 831 --order 4     # x, dx, d2x, d3x, d4x (max order 7)
Outputs: viz/derivative/series_<id>.png (order 1) or series_<id>_d<K>.png
"""
import argparse
import os
from math import comb

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "Data")
OUT = os.path.join(ROOT, "viz", "derivative")

RATIO_W = 200   # before/after window around tau for the variance ratios
# one colour per derivative order, same in every panel (categorical slots in fixed order)
ORDER_C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MAX_ORDER = len(ORDER_C) - 1
INK, INK2, GRID, HIST_BG = "#0b0b0b", "#52514e", "#e4e3df", "#f1f0ec"


def ac1(x):
    x = x - x.mean()
    d = (x * x).sum()
    return float((x[1:] * x[:-1]).sum() / d) if d > 0 else 0.0


def kurt(x):
    x = x - x.mean()
    v = (x * x).mean()
    return float((x ** 4).mean() / v ** 2 - 3) if v > 0 else 0.0


def rolling_std(x, w):
    """Trailing std over the last w points (NaN until w points exist)."""
    c1 = np.r_[0.0, np.cumsum(x, dtype=np.float64)]
    c2 = np.r_[0.0, np.cumsum(x.astype(np.float64) ** 2)]
    out = np.full(len(x), np.nan)
    if len(x) >= w:
        m = (c1[w:] - c1[:-w]) / w
        v = (c2[w:] - c2[:-w]) / w - m * m
        out[w - 1:] = np.sqrt(np.maximum(v, 0.0))
    return out


def diff_k(x, k):
    """k-th difference aligned with x: element t uses x[t-k .. t]; the first k entries are NaN."""
    return np.r_[np.full(k, np.nan), np.diff(x, n=k)] if k else x.astype(np.float64)


def log_ratio_k(x, g, k, w=RATIO_W):
    """log var after / var before of the k-th difference, in a +-w window around global index g."""
    a = np.diff(x[max(g - w, k) - k:g], n=k)
    b = np.diff(x[g - k:g + w], n=k)
    return float(np.log(b.var() / a.var()))


def load():
    X = pd.read_parquet(os.path.join(DATA, "X_train.parquet")).reset_index()
    X = X.sort_values(["id", "time"])
    yi = pd.read_parquet(os.path.join(DATA, "y_train_index.parquet"))
    ids = X["id"].to_numpy()
    vals = X["value"].to_numpy(np.float64)
    per = X["period"].to_numpy()
    starts = np.r_[0, np.flatnonzero(np.diff(ids)) + 1]
    ends = np.r_[starts[1:], len(ids)]
    series = {}
    for s, e in zip(starts, ends):
        i = int(ids[s])
        h = int((per[s:e] == 1).sum())          # period 1 = history (checked on series 0 below)
        series[i] = (vals[s:e], h, int(yi.at[i, "tau_index"]))
    v0, h0, t0 = series[0]
    assert h0 + t0 == int(yi.at[0, "tau"]), "history/online split does not match tau"
    return series


def curated(series):
    rows = []
    for i, (v, h, t) in series.items():
        hs = v[:h]
        r = dict(id=i, hlen=h, olen=len(v) - h, tau=t, ac1=ac1(hs), hkurt=kurt(hs))
        if t >= 0 and len(v) - h - t >= 150:
            r["lr_x"], r["lr_dx"] = log_ratio_k(v, h + t, 0), log_ratio_k(v, h + t, 1)
        rows.append(r)
    m = pd.DataFrame(rows).set_index("id")
    brk = m[m.tau >= 0].dropna(subset=["lr_dx"])
    plain = brk[(brk.ac1.abs() < 0.3) & (brk.hkurt < 3)]

    def at_quantile(df, col, q):
        target = df[col].quantile(q)
        return int((df[col] - target).abs().idxmin())

    unit = brk[brk.ac1 > 0.9].assign(a=lambda d: d.lr_dx.abs())
    heavy = brk[brk.hkurt > 5].assign(a=lambda d: d.lr_dx.abs())
    pick = [
        (at_quantile(plain, "lr_dx", 0.98), "break: derivative variance goes UP (98th pct)"),
        (at_quantile(plain, "lr_dx", 0.02), "break: derivative variance goes DOWN (2nd pct)"),
        (at_quantile(unit, "a", 0.90), "break: near unit root (history lag-1 AC > 0.9)"),
        (at_quantile(heavy, "a", 0.90), "break: heavy-tailed history (kurtosis > 5)"),
        (at_quantile(brk.assign(a=brk.lr_dx.abs()), "a", 0.50), "break: TYPICAL (median |change|)"),
    ]
    nob = m[m.tau < 0]
    for ref, cond, label in ((pick[0][0], (nob.ac1.abs() < 0.3) & (nob.hkurt < 3), "no break (control, plain)"),
                             (pick[2][0], nob.ac1 > 0.9, "no break (control, near unit root)")):
        c = nob[cond]
        pick.append((int((c.olen - m.at[ref, "olen"]).abs().idxmin()), label))
    return pick


def order_name(k):
    return {0: "x", 1: "dx"}.get(k, f"d{k}x")


def plot(i, label, series, hist_show, w, order=1):
    v, h, t = series[i]
    x_rel = np.arange(len(v)) - h                       # 0 = first online point
    D = [diff_k(v, k) for k in range(order + 1)]
    norm = [np.sqrt(comb(2 * k, k)) for k in range(order + 1)]
    S = [np.r_[np.full(k, np.nan), rolling_std(D[k][k:], w)] / norm[k] for k in range(order + 1)]
    lo = max(0, h - hist_show)
    sl = slice(lo, len(v))
    xr = x_rel[sl]

    n = order + 2
    height = 1.3 + 2.3 * (order + 1) + 2.1
    fig, ax = plt.subplots(n, 1, figsize=(13, height), sharex=True,
                           gridspec_kw=dict(height_ratios=[1] * (order + 1) + [0.9], hspace=0.3))
    fig.subplots_adjust(top=1 - 1.25 / height)
    fig.patch.set_facecolor("#fcfcfb")
    sub = f"history {h} pts (showing last {h - lo}), online {len(v) - h} pts, history lag-1 AC {ac1(v[:h]):+.2f}, kurtosis {kurt(v[:h]):.1f}"
    if t >= 0 and len(v) - h - t >= 20:
        ww = min(RATIO_W, len(v) - h - t)
        sub += f"\nvariance after/before break (+-{ww} pts):  " + ",   ".join(
            f"{order_name(k)} {np.exp(log_ratio_k(v, h + t, k, ww)):.2f}x" for k in range(order + 1))
    fig.suptitle(f"Series {i}  -  {label}", x=0.06, y=0.995, ha="left", fontsize=13, color=INK, weight="bold")
    fig.text(0.06, 1 - 0.55 / height, sub, ha="left", va="top", fontsize=9.5, color=INK2)

    names = ["x (signal)", "dx = x[t] - x[t-1]"] + [
        f"{order_name(k)} = {order_name(k - 1)}[t] - {order_name(k - 1)}[t-1]" for k in range(2, order + 1)]
    panels = [(ax[k], [(D[k][sl], ORDER_C[k], names[k])], order_name(k)) for k in range(order + 1)]
    panels.append((ax[-1], [(S[k][sl], ORDER_C[k], f"rolling std of {order_name(k)}" +
                             (f" / sqrt({comb(2 * k, k)})" if k else "") + (f" ({w} pts)" if k == 0 else ""))
                            for k in range(order + 1)], "rolling std"))
    for a, lines, ylab in panels:
        a.set_facecolor("#fcfcfb")
        a.axvspan(xr[0], -0.5, color=HIST_BG, zorder=0, lw=0)
        for y, c, name in lines:
            lw = 0.6 if a is not ax[-1] else 1.5
            a.plot(xr, y, color=c, lw=lw, label=name, zorder=2)
        if t >= 0:
            a.axvline(t, color=INK, lw=1.2, ls=(0, (4, 3)), zorder=3)
        a.set_ylabel(ylab, color=INK2, fontsize=10)
        a.grid(axis="y", color=GRID, lw=0.6)
        a.tick_params(colors=INK2, labelsize=9, length=0)
        for s_ in ("top", "right", "left"):
            a.spines[s_].set_visible(False)
        a.spines["bottom"].set_color(GRID)
        a.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4, borderaxespad=0.2,
                 fontsize=8.5, frameon=False, labelcolor=INK2)
    # symmetric derivative limits from a robust quantile so one spike does not flatten the panel
    for k in range(1, order + 1):
        q = np.nanquantile(np.abs(D[k][sl]), 0.999)
        ax[k].set_ylim(-1.15 * q, 1.15 * q)
    ax[-1].set_ylim(bottom=0)
    bot, top = ax[0].get_ylim()
    ax[0].text(xr[0] + 5, bot, "history (break-free)", va="bottom", fontsize=8.5, color=INK2)
    if t >= 0:
        ax[0].text(t + 6, top, f"break at online step {t}", va="top", fontsize=8.5, color=INK)
    ax[-1].set_xlabel("online step (0 = first online point; shaded = history)", color=INK2, fontsize=9.5)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"series_{i}.png" if order == 1 else f"series_{i}_d{order}.png")
    fig.savefig(path, dpi=110, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", type=int, nargs="*")
    ap.add_argument("--random", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hist", type=int, default=500, help="history points to show before online start")
    ap.add_argument("--w", type=int, default=50, help="rolling-std window")
    ap.add_argument("--order", type=int, default=1, help=f"highest derivative order (1-{MAX_ORDER})")
    a = ap.parse_args()
    if not 1 <= a.order <= MAX_ORDER:
        ap.error(f"--order must be 1..{MAX_ORDER}")

    print("reading parquet ...")
    series = load()
    if a.ids:
        pick = [(i, "break" if series[i][2] >= 0 else "no break") for i in a.ids]
    elif a.random:
        rng = np.random.default_rng(a.seed)
        pick = [(int(i), "break" if series[int(i)][2] >= 0 else "no break")
                for i in rng.choice(list(series), a.random, replace=False)]
    else:
        pick = curated(series)
    for i, label in pick:
        print(f"  {plot(i, label, series, a.hist, a.w, a.order)}   ({label})")


if __name__ == "__main__":
    main()
