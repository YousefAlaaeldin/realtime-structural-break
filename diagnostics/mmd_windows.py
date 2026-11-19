"""Which window length is best for the streaming MMD statistic? (single-feature TS-AUC, held-out series)

Same embedding and normalisation as sbrt/mmdstream.py (u2 = pairs of volatility-adjusted residuals, e2 = pairs of
AR residuals): stat_w = L |mean phi over the last L = min(w, n) online pairs - history mean|^2 / tr(history cov).
Windows 30 ... 700 and 'all' (every online point); the scan (max over all starts) is shown for reference.

Run: python diagnostics/mmd_windows.py [offset] [n_series]      (default 5000 2000)
"""
import os
import sys

import numpy as np
from joblib import Parallel, delayed

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
os.chdir(HERE)
from sbrt import features as F  # noqa: E402
from sbrt import mmdstream as MM  # noqa: E402
from sbrt import volstate as VS  # noqa: E402
from sbrt.data import load_train  # noqa: E402
from sbrt.metric import ts_auc  # noqa: E402

WINDOWS = (30, 50, 100, 150, 200, 300, 500, 700, 100000)
SUB = 3


def one(h, o):
    H = VS.fit_history(h)
    Q, MU, TR, ST, C = MM.make_mmd_params(h)
    nh = len(H["z"])
    zz = np.r_[H["z"], (np.asarray(o, float) - H["P"][F.P_MU]) / H["P"][F.P_SD]]
    Xo = np.column_stack([zz[nh - 1 - k: len(zz) - 1 - k] for k in range(MM.AR)])
    eo = np.clip((zz[nh:] - Xo @ H["coef"]) / H["se"], -10, 10)
    k = np.arange(len(eo))
    uo = eo / np.sqrt(H["L"] + (H["s2T1"] - H["L"]) * H["p"] ** k)
    vh_last = H["eh"][-1] / np.sqrt(H["s2h"][-1])
    out = {}
    for v, kk, x, prev in (("u2", 2, uo, vh_last), ("e2", 0, eo, H["eh"][-1])):
        P = MM._phi(np.column_stack([np.r_[prev, x[:-1]], x]), kk)
        Cs = np.vstack([np.zeros(MM.D), np.cumsum(P, axis=0)])
        rows = np.arange(0, len(x), SUB)
        n = rows + 1
        M = np.zeros((len(rows), len(WINDOWS)))
        for j, w in enumerate(WINDOWS):
            L = np.minimum(w, n)
            Dl = (Cs[n] - Cs[n - L]) / L[:, None] - MU[kk]
            M[:, j] = L * (Dl * Dl).sum(1) / TR[kk]
        out[v] = M
    out["scan"] = MM.run_series_mmd(h, o, SUB)[:, [MM.FEATURE_NAMES.index("mmd_u2_scan"),
                                                     MM.FEATURE_NAMES.index("mmd_e2_scan")]]
    return out


def main():
    offset = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    n_series = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    S, tau = load_train()
    S, tau = S[offset:offset + n_series], tau[offset:offset + n_series]
    # sanity: the numpy windows equal the streaming kernel's w100 / w300 / full columns
    h, o = S[0][1], S[0][2]
    r, ref = one(h, o), MM.run_series_mmd(h, o, SUB)
    for w, nm in ((100, "w100"), (300, "w300"), (100000, "full")):
        assert np.allclose(r["u2"][:, WINDOWS.index(w)], ref[:, MM.FEATURE_NAMES.index(f"mmd_u2_{nm}")], rtol=1e-6, atol=1e-8)
    print("numpy windows match the streaming kernel: OK")
    R = Parallel(n_jobs=8, backend="threading")(delayed(one)(s[1], s[2]) for s in S)
    step = np.concatenate([np.arange(0, len(s[2]), SUB) for s in S])
    sid = np.repeat(np.arange(n_series), [(len(s[2]) + SUB - 1) // SUB for s in S])
    y = ((tau[sid] >= 0) & (step >= tau[sid])).astype(np.int8)
    buckets = ((0, 100), (100, 300), (300, 700), (700, 10000))

    def row(nm, x):
        r = [ts_auc(x, y, step)] + [ts_auc(x[(step >= lo) & (step < hi)], y[(step >= lo) & (step < hi)],
                                          step[(step >= lo) & (step < hi)]) for lo, hi in buckets]
        print(f"  {nm:12s} " + "  ".join(f"{v:7.4f}" for v in r))

    print(f"series {offset}..{offset + n_series - 1}; single-feature TS-AUC      all     0-100  100-300  300-700    700+")
    for v in ("u2", "e2"):
        M = np.concatenate([q[v] for q in R])
        print(f"-- {v}")
        for j, w in enumerate(WINDOWS):
            row("all" if w >= 100000 else f"w{w}", M[:, j])
        row("scan", np.concatenate([q["scan"][:, 0 if v == "u2" else 1] for q in R]))


if __name__ == "__main__":
    main()
