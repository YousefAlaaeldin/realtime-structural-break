"""Covariate-shift weights for the 2025-edition training series (owner's request 2026-10-18; HANDOFF section 19).

WHY. The learning curve says we are data-limited (doubling the data ~ +0.015 for the trees, HANDOFF section 3). The
only extra labelled data is the 2025 edition (~10,000 series), and it failed as-is (trees -0.0057) and source-balanced
(-0.0006) because it is a different population: heavier tails, less autocorrelation, stronger volatility clustering,
29% breaks. The standard fix for "extra data from a shifted source" is IMPORTANCE WEIGHTING: weight each 2025 series
by the density ratio p_2026(x) / p_2025(x) of its normal behaviour x, so the weighted 2025 set looks like our
population. Untried here (section 8: only plain, balanced and a never-run --ext-rel 0.5).

WHAT x IS. Shape statistics of the LAST 1,000 HISTORY POINTS only (the break-free part, so the weights cannot see
the label; and a fixed window, because 2025 histories are capped at ~2,400 points by its period-0 length while ours
run 1,000-5,000: history length is our cut geometry, not behaviour, and would otherwise dominate the classifier).
14 statistics: sd, skew, kurtosis, tail shares |z| > 2 / 3 / 4, lag 1 / 2 / 5 autocorrelation, autocorrelation of
z^2 at lags 1 / 5 / 20 (volatility clustering), sd of the log variance of 10 blocks of 100 (variance wandering, the
main no-break nuisance), and the AR(5) one-step residual variance ratio. Computed on the history after the same
z-scoring the series already has.

HOW. A gradient-boosted classifier (sklearn HistGradientBoosting, shallow) tells 2025 cuts from Real-Time originals;
CROSS-FITTED (5 folds over the pooled series), so no series is weighted by a model that saw it. Weight of a 2025
series = (n25 / n26) * p / (1 - p), p = P(Real-Time | x), clipped at --clip, then scaled to mean 1 over the 2025
series. cv.py multiplies a 2025 row's weight by its series' weight (`--ext-weight <file>`), on top of --ext-rel and
--ext-balance (which still fixes the 29% vs 50% break-rate difference per step).
Label-free: the classifier sees histories of all 10,000 Real-Time originals (also the CV validation folds' histories,
which carries no label information; production would use the same training histories).

DECISION RULE (written before any run):
  GATE (this script, no cv.py run):
    - domain AUC < 0.60: the populations barely differ, weighting = the plain run that already failed -> CLOSED.
    - effective sample size of the weighted 2025 set ESS = (sum w)^2 / sum w^2 < 1,000 series (10% of our data): too
      little Real-Time-like data to matter -> CLOSED.
    - also printed: weighted 2025 means of the statistics vs Real-Time (the weighting should close most of the gap)
      and the weighted 2025 break rate.
  If the gate passes:
    1. cv.py trees seed 0 with --ext 1 --ext-balance --ext-weight vs the current best seed-0 trees. Early stop: folds
       0 and 1 both <= 0 -> CLOSED.
    2. 5-fold paired mean > 0: seed 1, the seed-average comparison (diagnostics/oof_paired.py), then the trees-only
       blend (blend.py, trees pair replaced; same 3 members).
    3. ADOPT if the blend gains >= +0.0005, >= 4/5 folds, > 2 SE (section 5). Then the GRU/SVM with the same data.
  Inference cost: none (the cloud never trains; the 2025 data only enters the local training).
EXPECTATION (written before any run): the ESS decides the ceiling. ESS ~2,000 is ~+20% data, ~0.26 doublings,
i.e. trees roughly +0.002-0.004 if the weighted data were as good as ours, then shrunk ~2x into the blend; realistic
blend gain +0.000 to +0.002.

Run:   python diagnostics/ext_weights.py            -> data/ext2025/ext_weights_s0_h1000.npz + diagnostics/ext_weights.txt
       python diagnostics/ext_weights.py --smoke    (600 series each side, nothing written)
"""
import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from sbrt import external as XT  # noqa: E402
from sbrt.data import load_train  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ext-dir", default=os.path.join(ROOT, "data", "ext2025"))
ap.add_argument("--ext", type=int, default=1, help="cuts per 2025 series: must match cv.py --ext")
ap.add_argument("--ext-seed", type=int, default=0)
ap.add_argument("--ext-hmin", type=int, default=1000)
ap.add_argument("--clip", type=float, default=10.0, help="largest weight before normalising to mean 1")
ap.add_argument("--window", type=int, default=1000, help="history points the statistics use (the last ones)")
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--out", default="")
args = ap.parse_args()
STAT_NAMES = ["sd", "skew", "kurt", "tail2", "tail3", "tail4", "ac1", "ac2", "ac5", "ac2_1", "ac2_5", "ac2_20",
              "sd_logvar100", "ar5_resvar"]


def acf(x, k):
    return float(np.dot(x[:-k], x[k:]) / max(np.dot(x, x), 1e-300))


def stats(h, W):
    z = np.asarray(h[-W:], dtype=np.float64)
    sd = z.std()
    c = (z - z.mean()) / max(sd, 1e-12)
    q = c * c - 1.0
    blocks = c[len(c) % 100:].reshape(-1, 100)
    lv = np.log(np.maximum(blocks.var(1), 1e-12))
    # AR(5) one-step residual variance / variance (least squares on the window)
    p = 5
    Xl = np.column_stack([c[p - j - 1:len(c) - j - 1] for j in range(p)])
    yv = c[p:]
    beta, *_ = np.linalg.lstsq(Xl, yv, rcond=None)
    rv = float(np.mean((yv - Xl @ beta) ** 2))
    a = np.abs(c)
    return [sd, float(np.mean(c ** 3)), float(np.mean(c ** 4) - 3), float(np.mean(a > 2)), float(np.mean(a > 3)),
            float(np.mean(a > 4)), acf(c, 1), acf(c, 2), acf(c, 5), acf(q, 1), acf(q, 5), acf(q, 20),
            float(lv.std()), rv]


class Tee:
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8") if path else None

    def write(self, s):
        sys.__stdout__.write(s)
        if self.f:
            self.f.write(s)
            self.f.flush()

    def flush(self):
        sys.__stdout__.flush()


def main():
    wfile = os.path.join(args.ext_dir, f"ext_weights_e{args.ext}_s{args.ext_seed}_h{args.ext_hmin}.npz")
    sys.stdout = Tee("" if args.smoke else (args.out or os.path.join(ROOT, "diagnostics", "ext_weights.txt")))
    t0 = time.time()
    print(f"settings: {vars(args)}")
    S, tau = load_train()
    # exactly the cuts cv.py --ext builds (same function, order and seed); limit=0 = all 2025 series
    V = XT.make_external(args.ext_dir, args.ext, seed=args.ext_seed, hmin=args.ext_hmin, limit=0)
    vlen = np.array([len(v[2]) for v in V])
    vtau = np.array([v[3] for v in V])
    hl25 = np.array([len(v[1]) for v in V])
    i26 = np.arange(len(S))
    i25 = np.arange(len(V))
    if args.smoke:
        rs = np.random.default_rng(0)
        i26, i25 = np.sort(rs.choice(len(S), 600, replace=False)), np.sort(rs.choice(len(V), 600, replace=False))
    W = args.window
    X26 = np.array([stats(S[i][1], W) for i in i26])
    X25 = np.array([stats(V[i][1], W) for i in i25])
    print(f"{len(i26):,} Real-Time histories, {len(i25):,} 2025 cuts (break rate {np.mean(vtau[i25] >= 0):.3f}); "
          f"history length median {np.median([len(S[i][1]) for i in i26]):.0f} vs {np.median(hl25[i25]):.0f} "
          f"(not a classifier input); statistics in {time.time() - t0:.0f}s")

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    Xa = np.vstack([X26, X25])
    ya = np.r_[np.ones(len(X26)), np.zeros(len(X25))]          # 1 = Real-Time
    p = np.zeros(len(ya))
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=0).split(Xa, ya):
        m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300, l2_regularization=1.0,
                                           random_state=0)
        m.fit(Xa[tr], ya[tr])
        p[va] = m.predict_proba(Xa[va])[:, 1]
    auc = roc_auc_score(ya, p)
    p25 = np.clip(p[len(X26):], 1e-6, 1 - 1e-6)
    raw = (len(X25) / len(X26)) * p25 / (1 - p25)
    w = np.minimum(raw, args.clip)
    w = w / w.mean()
    ess = w.sum() ** 2 / (w ** 2).sum()
    print(f"\ndomain classifier (cross-fitted): AUC {auc:.3f}  (0.5 = same population)")
    print(f"weights: clipped {int((raw > args.clip).sum())} of {len(raw):,} at {args.clip:g}; quantiles 10/50/90/99% "
          f"{np.round(np.percentile(w, [10, 50, 90, 99]), 3)}; share of weight in the top 10% of series "
          f"{np.sort(w)[::-1][:max(1, len(w) // 10)].sum() / w.sum():.2f}")
    print(f"EFFECTIVE SAMPLE SIZE of the weighted 2025 set: {ess:,.0f} of {len(w):,} series "
          f"({ess / len(X26):.1%} of the Real-Time data)")
    br = vtau[i25] >= 0
    print(f"2025 break rate: unweighted {br.mean():.3f}, weighted {np.average(br, weights=w):.3f}  (Real-Time 0.497)")
    print("\nbalance check, mean of each statistic: Real-Time | 2025 unweighted | 2025 weighted  (gap closed)")
    for j, nm in enumerate(STAT_NAMES):
        a, b, c = X26[:, j].mean(), X25[:, j].mean(), np.average(X25[:, j], weights=w)
        closed = 1 - abs(c - a) / max(abs(b - a), 1e-12)
        print(f"  {nm:13s} {a:9.4f} | {b:9.4f} | {c:9.4f}   ({closed:+.0%})")

    print("\nGATE (written before any run): domain AUC >= 0.60 AND ESS >= 1,000")
    ok = auc >= 0.60 and ess >= 1000
    if args.smoke:
        print(f"  smoke run (600 + 600 series): {'would pass' if ok else 'would fail'}; numbers at this size mean little")
    else:
        print(f"  -> {'PASS: run the cv.py commands in HANDOFF section 19' if ok else 'FAIL: CLOSED, no cv.py run'}")
        np.savez(wfile, w=w.astype(np.float64), vlen=vlen, auc=auc, ess=ess)
        print(f"saved {wfile}")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
