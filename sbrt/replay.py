"""Per-series null calibration by pseudo-online replay of the history (a CHECK; not wired into cv.py yet).

Idea: a statistic value that is alarming for one series is normal for another (autocorrelation, heavy tails and
volatility clustering change how much every statistic wanders without a break). The history is break-free, so it
can show each series' own "normal" behaviour of every statistic:

    replay r:  pseudo-history = hist[:c_r],  pseudo-online = hist[c_r : c_r + L_r]   with c_r = n_h - L_r
               (each replay ends at the history's end, the part closest to the real online segment; the
               pseudo-history must keep >= HMIN points). The exact streaming kernels are run on it, refitting
               every per-series parameter on the pseudo-history, so a replay is the real situation minus the break.

At online step k the series' null values are the replay values at step k (only replays longer than k). With few
replays per step (6 at the start, 1-2 late, none for long steps of short histories) the per-series mean and sd
are shrunk toward the population values at that step (pooled over all series' replays; unsupervised, no labels):
    m_k = (sum_r x_rk + KAPPA m_pop_k) / (R_k + KAPPA)
    s_k^2 = ((R_k - 1) s_series_k^2 + KAPPA s_pop_k^2) / (R_k - 1 + KAPPA)
Two calibrated versions of each statistic x:   diff = x - m_k     z = (x - m_k) / s_k

Check (single-feature TS-AUC, raw vs calibrated, on held-out series):  python -m sbrt.replay [offset] [n_series]
"""
import numpy as np

from sbrt import disttest as DT
from sbrt import features as F
from sbrt import volstate as VS

HMIN = 500
LENS = (50, 100, 200, 400, 700, 999)
KAPPA = 3.0
SUB = 3
NROW = (999 + SUB - 1) // SUB + 1

STAT_K = ["full_e10sq", "win100_e10sq", "glrmax_e10sq", "glrmax_abse10", "glrmax_tail3", "chi2_full"]
STAT_V = ["vol_full", "vol_scanmax", "vol_lr_full"]
STAT_D = ["dist_e_ad", "dist_e_w1", "dist_u_adscan"]
STATS = STAT_K + STAT_V + STAT_D
CK = [F.FEATURE_NAMES.index(s) for s in STAT_K]
CV = [VS.FEATURE_NAMES.index(s) for s in STAT_V]
CD = [DT.FEATURE_NAMES.index(s) for s in STAT_D]


def stat_matrix(h, o):
    """The chosen statistics at online steps 0, 3, 6, ... (rows), exactly as the streaming kernels give them."""
    return np.hstack([F.run_series(h, o)[::SUB][:, CK], VS.run_series_vol(h, o)[::SUB][:, CV],
                      DT.run_series_dist(h, o, SUB)[:, CD]])


def splits(nh):
    return [(nh - L, L) for L in LENS if nh - L >= HMIN]


def replay_sums(h):
    """Per-step sums over this series' replays: S1, S2 (NROW, n_stats) and counts N (NROW,)."""
    h = np.asarray(h, dtype=np.float64)
    S1 = np.zeros((NROW, len(STATS)))
    S2 = np.zeros((NROW, len(STATS)))
    N = np.zeros(NROW)
    for c, L in splits(len(h)):
        M = stat_matrix(h[:c], h[c:c + L])
        S1[:len(M)] += M
        S2[:len(M)] += M * M
        N[:len(M)] += 1
    return S1, S2, N


def calibrate(X, S1, S2, N, mpop, spop):
    """X: (T, n_stats) online rows; returns diff and z versions."""
    T = len(X)
    s1, s2, n = S1[:T], S2[:T], N[:T, None]
    m = (s1 + KAPPA * mpop[:T]) / (n + KAPPA)
    vs = np.where(n >= 2, (s2 - s1 * s1 / np.maximum(n, 1)) / np.maximum(n - 1, 1), 0.0)
    dof = np.maximum(n - 1, 0)
    s = np.sqrt((dof * np.maximum(vs, 0) + KAPPA * spop[:T] ** 2) / (dof + KAPPA))
    return X - m, (X - m) / np.maximum(s, 1e-9)


def _selftest(offset=5000, n_series=2000, threads=8):
    import os
    import sys
    import time
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from joblib import Parallel, delayed

    from sbrt.data import load_train
    from sbrt.metric import ts_auc

    S, tau = load_train()
    S, tau = S[offset:offset + n_series], tau[offset:offset + n_series]
    print(f"series {offset}..{offset + n_series - 1}; replays end at the history end, lengths {LENS}, "
          f"pseudo-history >= {HMIN} points, shrinkage KAPPA = {KAPPA:g}")

    # determinism + the replay really is the real kernel on a shorter history
    h = S[0][1]
    a, b = replay_sums(h), replay_sums(h)
    assert all(np.array_equal(x, y) for x, y in zip(a, b)), "replays not deterministic"
    c, L = splits(len(h))[-1]
    assert np.array_equal(stat_matrix(h[:c], h[c:c + L])[:, 0], F.run_series(h[:c], h[c:c + L])[::SUB][:, CK[0]])
    print("replays deterministic and identical to the streaming kernels: OK")

    t0 = time.time()
    X = Parallel(n_jobs=threads, backend="threading")(delayed(stat_matrix)(s[1], s[2]) for s in S)
    R = Parallel(n_jobs=threads, backend="threading")(delayed(replay_sums)(s[1]) for s in S)
    print(f"online statistics + replays for {n_series} series in {time.time() - t0:.0f}s")

    P1 = sum(r[0] for r in R)
    P2 = sum(r[1] for r in R)
    PN = np.maximum(sum(r[2] for r in R), 1)[:, None]
    mpop = P1 / PN
    spop = np.sqrt(np.maximum(P2 / PN - mpop ** 2, 1e-12))
    # steps no replay of any series reaches (should not happen): fall back to the last available row
    Ncount = sum(r[2] for r in R)
    print("average replays per series at online steps 0 / 99 / 300 / 600 / 900: "
          + " / ".join(f"{Ncount[k // SUB] / n_series:.2f}" for k in (0, 99, 300, 600, 900)))

    D, Z = [], []
    for x, (s1, s2, n) in zip(X, R):
        d, z = calibrate(x, s1, s2, n, mpop, spop)
        D.append(d)
        Z.append(z)
    X, D, Z = np.concatenate(X), np.concatenate(D), np.concatenate(Z)
    step = np.concatenate([np.arange(0, len(s[2]), SUB) for s in S])
    sid = np.repeat(np.arange(n_series), [(len(s[2]) + SUB - 1) // SUB for s in S])
    y = ((tau[sid] >= 0) & (step >= tau[sid])).astype(np.int8)
    assert len(y) == len(X)

    buckets = ((0, 100), (100, 300), (300, 700), (700, 10000))

    def aucs(x):
        return [ts_auc(x, y, step)] + [ts_auc(x[(step >= lo) & (step < hi)], y[(step >= lo) & (step < hi)],
                                              step[(step >= lo) & (step < hi)]) for lo, hi in buckets]

    m99 = (step == 99) & (tau[sid] < 0)
    print("\nsingle-feature TS-AUC: raw / calibrated diff / calibrated z     columns: all, 0-100, 100-300, 300-700, 700+")
    print("(null spread = sd over no-break series at step 99, in units of the raw statistic's sd there)")
    gains = []
    for j, nm in enumerate(STATS):
        r, d, z = aucs(X[:, j]), aucs(D[:, j]), aucs(Z[:, j])
        sd_raw = X[m99, j].std()
        print(f"  {nm:16s} raw  " + " ".join(f"{v:.4f}" for v in r))
        print(f"  {'':16s} diff " + " ".join(f"{v:.4f}" for v in d) + f"   ({d[0] - r[0]:+.4f};  null spread {D[m99, j].std() / sd_raw:.2f})")
        print(f"  {'':16s} z    " + " ".join(f"{v:.4f}" for v in z) + f"   ({z[0] - r[0]:+.4f})")
        gains.append((max(d[0], z[0]) - r[0], nm))
    print("\nbest calibrated gain per statistic: " + ", ".join(f"{n} {g:+.4f}" for g, n in sorted(gains, reverse=True)))
    print("\nGo/no-go: worth wiring into cv.py if the calibrated versions of the strongest statistics (vol_scanmax, "
          "vol_lr_full, glrmax_e10sq, dist_u_adscan) gain about +0.005 or more, ideally in 100-700 where the metric's "
          "weight is. The trees would get BOTH raw and calibrated columns.")
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    import sys
    _selftest(offset=int(sys.argv[1]) if len(sys.argv) > 1 else 5000,
              n_series=int(sys.argv[2]) if len(sys.argv) > 2 else 2000)
