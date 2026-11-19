"""Predictor bake-off: which "predict the next point" model gives the best break signal?

Run:   python predictor_bakeoff.py                 # 3000 random training series, ~5-10 min
       python predictor_bakeoff.py --n 10000       # all series (slower)
       python predictor_bakeoff.py --no-stack      # skip the (slowest) LightGBM stacking part
       python predictor_bakeoff.py --calib-split 0.8   # fit on first 80% of history, calibrate on last 20%
Output: bakeoff_out/summary.txt, bakeoff_out/*.csv

Idea. A predictor is fitted ONCE on the break-free history and then frozen. For every new online point
we get a one-step-ahead prediction error ("residual"), standardised by the history's own residual scale.
If nothing changed, the residuals look like the history's. Different summaries of the residual
("families") react to different changes:
    mean  : r             level the predictor failed to anticipate
    var   : r^2           bigger / smaller errors
    abs   : |r|           same, robust to outliers
    t99   : 1{|r|>q99}    extreme errors (q99 = the history's own 99% quantile, so rate 1% under no change)
    t999  : 1{|r|>q999}   very extreme errors (0.1% under no change)
    lag1  : r_t * r_t-1   leftover dependence between consecutive errors
    nll   : -log density of the error under a Student-t fitted to the history (the "surprise")
    pit   : |F(r) - 0.5|  where F = that Student-t CDF (spread of the probability integral transform)

Three questions are answered for every (predictor, family):
  A. ORACLE  AUC: break position known, L points after it, break vs no-break  (upper bound, L = 30/100/300)
  B. SCAN    TS-AUC: break position unknown; the scan statistic (max over all start times of the
             standardised sum of increments) scored with the competition metric
  C. STACK   TS-AUC: a small LightGBM on ALL families of one predictor (or of several), cross-validated by
             series -- this is the 'does it add value beyond what we already have' test.
"""
import argparse
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from numba import njit
from scipy.signal import lfilter
from scipy.special import gammaln, stdtr
from scipy.stats import rankdata

from sbrt.data import load_train
from sbrt.metric import ts_auc

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=3000, help="number of training series to use")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--pmax", type=int, default=30, help="largest AR order considered for BIC")
ap.add_argument("--lam", type=float, default=0.99, help="EWMA decay of the adaptive-scale predictor")
ap.add_argument("--predictors", type=str, default="raw,AR1,AR10,AR20,AR30,AR50,ARridge50,ARbic",
                help="comma list. Options: raw, AR<p> (e.g. AR20), ARridge<p> (ridge-regularised AR(p)), ARbic, "
                     "ARbic+scale, RW, EWMA0.1. 'raw' is always included. Round 1 was: "
                     "raw,AR1,AR10,ARbic,ARbic+scale,RW,EWMA0.1")
ap.add_argument("--ridge", type=float, default=0.02,
                help="ridge penalty for ARridge<p>, relative to the sample size (X'X + ridge*n*I)")
ap.add_argument("--gbm-pool", type=int, default=2000,
                help="ARGBM: number of OTHER series whose histories train the global nonlinear correction")
ap.add_argument("--gbm-lags", type=int, default=20, help="ARGBM: number of past values given to the correction model")
ap.add_argument("--gbm-rounds", type=int, default=300, help="ARGBM: boosting rounds")
ap.add_argument("--calib-split", type=float, default=1.0,
                help="1.0 = fit and calibrate on the whole history; 0.8 = fit on first 80%%, calibrate on last 20%%")
ap.add_argument("--no-stack", action="store_true")
ap.add_argument("--threads", type=int, default=8, help="LightGBM threads for the stacking part")
ap.add_argument("--out", type=str, default="bakeoff_out")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)


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


def section(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78, flush=True)


PREDICTORS = [p for p in args.predictors.split(",") if p]
if "raw" not in PREDICTORS:
    PREDICTORS.insert(0, "raw")
_orders = [int(m.group(1)) for p in PREDICTORS for m in [re.fullmatch(r"AR(?:ridge)?(\d+)", p)] if m]
PMAX_EFF = max([args.pmax] + _orders)  # largest order the Levinson recursion must reach
FAMS = ["mean", "var", "abs", "t99", "t999", "lag1", "nll", "pit"]
MINSEG = {"mean": 8, "var": 8, "abs": 8, "t99": 20, "t999": 40, "lag1": 8, "nll": 8, "pit": 8}
ABS_FLOOR = {"t999": 0.004}
LS = (30, 100, 300)
BURN = max(40, PMAX_EFF)  # skip the first points, where the zero-initialised filters give wrong residuals
chosen_p = []


# ---------------------------------------------------------------------------------------------
# predictors
def levinson(r, pmax):
    """Levinson-Durbin: AR coefficients and innovation variance for every order 0..pmax."""
    phi = np.zeros(0)
    sig = float(r[0])
    coefs, sigs = [phi], [sig]
    for k in range(1, pmax + 1):
        acc = r[k] - float(np.dot(phi, r[k - 1:0:-1]))
        kappa = acc / max(sig, 1e-12)
        new = np.empty(k)
        new[: k - 1] = phi - kappa * phi[::-1]
        new[k - 1] = kappa
        sig = sig * (1.0 - kappa ** 2)
        phi = new
        coefs.append(phi)
        sigs.append(max(sig, 1e-12))
    return coefs, np.array(sigs)


def compute_e(name, s0, fit_end):
    """One-step-ahead prediction errors for the whole series s0 (history + online), and an optional
    per-time adaptive scale. Every prediction uses only earlier points."""
    if name == "raw":
        return s0.copy(), None
    if name == "RW":
        e = np.zeros_like(s0)
        e[1:] = np.diff(s0)
        return e, None
    if name == "EWMA0.1":
        a = 0.1
        m = lfilter([a], [1.0, -(1.0 - a)], s0)
        e = np.empty_like(s0)
        e[0] = s0[0]
        e[1:] = s0[1:] - m[:-1]
        return e, None
    if name == "ARGBM":
        e_lin, _ = compute_e("AR10", s0, fit_end)
        idx, X, y, se = gbm_rows(s0, e_lin, fit_end)
        yhat = np.zeros(len(s0))
        yhat[idx] = GBM.predict(X, num_threads=args.threads)
        return e_lin - se * yhat, None
    x = s0[:fit_end]
    n = len(x)
    mr = re.fullmatch(r"ARridge(\d+)", name)
    if mr:  # ridge-regularised AR(p): shrinks the many small long-lag coefficients
        p = int(mr.group(1))
        X = np.column_stack([x[p - 1 - i: n - 1 - i] for i in range(p)])
        coef = np.linalg.solve(X.T @ X + args.ridge * (n - p) * np.eye(p), X.T @ x[p:])
        e = lfilter(np.r_[1.0, -coef], [1.0], s0)
        return e, None
    acov = np.array([np.dot(x[: n - k], x[k:]) / n for k in range(PMAX_EFF + 1)])
    coefs, sigs = levinson(acov, PMAX_EFF)
    ma = re.fullmatch(r"AR(\d+)", name)
    if ma:
        p = int(ma.group(1))
    else:  # ARbic, ARbic+scale: order by BIC among 0..pmax
        bic = n * np.log(sigs[: args.pmax + 1]) + np.arange(args.pmax + 1) * np.log(n)
        p = int(np.argmin(bic))
        chosen_p.append(p)
    e = lfilter(np.r_[1.0, -coefs[p]], [1.0], s0)
    if name == "ARbic+scale":
        lam = args.lam
        m2 = float(np.mean(e[BURN:fit_end] ** 2))
        v = lfilter([1.0 - lam], [1.0, -lam], e ** 2 - m2) + m2
        scale = np.sqrt(np.maximum(np.r_[v[0], v[:-1]], 1e-12))  # uses errors up to t-1 only
        return e, scale
    return e, None


# ---------------------------------------------------------------------------------------------
# ARGBM: AR(10) forecast + a global nonlinear correction learned across OTHER series' histories
GBM = None


def set_gbm(model):
    global GBM
    GBM = model


def gbm_rows(s0, e_lin, fit_end):
    """Feature rows for every position t >= lags: the last `lags` values, the AR(10) forecast, a local
    volatility and three series descriptors. Target: the AR(10) error in units of its history std."""
    L = args.gbm_lags
    n = len(s0)
    se = max(float(e_lin[BURN:fit_end].std()), 1e-8)
    idx = np.arange(L, n)
    lags = np.stack([s0[idx - k] for k in range(1, L + 1)], axis=1)
    pred = (s0 - e_lin)[idx]
    h = s0[:fit_end]
    rho1 = float(np.dot(h[:-1], h[1:]) / max(float(np.dot(h, h)), 1e-12))
    kurt = float(np.mean(h ** 4) / max(float(np.mean(h ** 2)) ** 2, 1e-12) - 3.0)
    ctx = np.tile([rho1, kurt, se], (len(idx), 1))
    X = np.column_stack([lags, pred, lags.std(axis=1), ctx]).astype(np.float32)
    y = np.clip(e_lin[idx] / se, -8.0, 8.0).astype(np.float32)
    return idx, X, y, se


def train_gbm(S, pool):
    import lightgbm as lgb
    r = np.random.default_rng(args.seed + 1)
    Xs, ys, gs = [], [], []
    for j, i in enumerate(pool):
        h = S[i][1]
        s0 = h - h.mean()
        e_lin, _ = compute_e("AR10", s0, len(s0))
        idx, X, y, se = gbm_rows(s0, e_lin, len(s0))
        ok = np.flatnonzero(idx >= BURN)
        pick = r.choice(ok, min(400, len(ok)), replace=False)
        Xs.append(X[pick])
        ys.append(y[pick])
        gs.append(np.full(len(pick), j))
    X, y, g = np.concatenate(Xs), np.concatenate(ys), np.concatenate(gs)
    va = (g % 10) == 0
    P = dict(objective="regression", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200, feature_fraction=0.8,
             bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, seed=args.seed, deterministic=True,
             force_row_wise=True, num_threads=args.threads, verbose=-1)
    m = lgb.train(P, lgb.Dataset(X[~va], y[~va]), args.gbm_rounds)
    yh = m.predict(X[va], num_threads=args.threads)
    r2 = 1.0 - float(np.sum((y[va] - yh) ** 2) / np.sum(y[va] ** 2))
    print(f"\nARGBM diagnostic: trained on {int((~va).sum()):,} rows from {int(len(pool) * 0.9)} other series' histories.")
    print(f"  share of the AR(10) error variance the nonlinear correction explains on {int(len(pool) * 0.1)} held-out series: "
          f"{100 * r2:.2f}%   (0% = AR(10) already captures everything predictable)", flush=True)
    return m


# ---------------------------------------------------------------------------------------------
# increments per family
def increments(r_all, nh, cs):
    rc = r_all[cs:nh]
    absr = np.abs(r_all)
    q99, q999 = np.quantile(np.abs(rc), [0.99, 0.999])
    kx = float(np.mean(rc ** 4) / max(np.mean(rc ** 2) ** 2, 1e-12) - 3.0)
    nu = 4.0 + 6.0 / kx if kx > 0.05 else 200.0
    nu = min(max(nu, 2.5), 200.0)
    st = np.sqrt((nu - 2.0) / nu)
    x = r_all / st
    logpdf = (gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(nu * np.pi)
              - (nu + 1) / 2 * np.log1p(x * x / nu) - np.log(st))
    lag = np.zeros_like(r_all)
    lag[1:] = r_all[1:] * r_all[:-1]
    return {
        "mean": r_all, "var": r_all ** 2, "abs": absr,
        "t99": (absr > q99).astype(np.float64), "t999": (absr > q999).astype(np.float64),
        "lag1": lag, "nll": -logpdf, "pit": np.abs(stdtr(nu, x) - 0.5),
    }


def lrv_batch(c, abs_floor):
    n = len(c)
    b = max(5, min(50, n // 20))
    nb = n // b
    bm = c[: nb * b].reshape(nb, b).sum(axis=1) / np.sqrt(b)
    return max(float(bm.var()), 0.25 * float(c.var()), abs_floor)


@njit(cache=True)
def scan_scores(gc, invsd, minseg):
    """For every online step n: max over start j of |sum(gc[j:n])|/sqrt(n-j) (two-sided) and of the
    signed sum (upward), each times invsd. gc is the centred increment series."""
    T = len(gc)
    C = np.zeros(T + 1)
    for i in range(T):
        C[i + 1] = C[i] + gc[i]
    out_abs = np.zeros(T)
    out_up = np.zeros(T)
    for t in range(T):
        n = t + 1
        best_a = 0.0
        best_u = 0.0
        for j in range(0, n - minseg + 1):
            s = (C[n] - C[j]) / np.sqrt(n - j * 1.0)
            if abs(s) > best_a:
                best_a = abs(s)
            if s > best_u:
                best_u = s
        out_abs[t] = best_a * invsd
        out_up[t] = best_u * invsd
    return out_abs, out_up


# ---------------------------------------------------------------------------------------------
# evaluation helpers
def auc_two(a, b):
    r = rankdata(np.r_[a, b])
    return (r[: len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))


def ts_auc_matrix(M, Y, alive):
    num = den = 0.0
    for t in range(M.shape[1]):
        idx = alive[:, t]
        y = Y[idx, t]
        npos = int(y.sum())
        nneg = len(y) - npos
        if npos < 5 or nneg < 5:
            continue
        r = rankdata(M[idx, t])
        num += npos * nneg * (r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)
        den += npos * nneg
    return num / den if den else 0.5


def step_weights(y, step):
    T = int(step.max()) + 1
    npos = np.bincount(step, weights=y, minlength=T)
    n = np.bincount(step, minlength=T).astype(float)
    nneg = n - npos
    w = np.where(y == 1, nneg[step] / 2.0, npos[step] / 2.0)
    return w / max(w.mean(), 1e-12)


def stack_eval(blocks, names, y, step, sid):
    import lightgbm as lgb
    from sklearn.model_selection import GroupKFold
    X = np.hstack([blocks[nm] for nm in names] + [np.log1p(step)[:, None]]).astype(np.float32)
    P = dict(objective="binary", learning_rate=0.1, num_leaves=15, min_data_in_leaf=200, feature_fraction=0.8,
             bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, seed=0, deterministic=True,
             force_row_wise=True, num_threads=args.threads, verbose=-1)
    oof = np.zeros(len(y))
    for tr, va in GroupKFold(3).split(X, y, sid):
        m = lgb.train(P, lgb.Dataset(X[tr], y[tr], weight=step_weights(y[tr], step[tr])), 150)
        oof[va] = m.predict(X[va], raw_score=True)
    return ts_auc(oof, y, step)


# ---------------------------------------------------------------------------------------------
def main():
    t_start = time.time()
    S, tau_all = load_train()
    rng = np.random.default_rng(args.seed)
    idx = np.sort(rng.choice(len(S), min(args.n, len(S)), replace=False))
    hist = [S[i][1] for i in idx]
    onl = [S[i][2] for i in idx]
    tau = tau_all[idx]
    n = len(idx)
    lens = np.array([len(o) for o in onl])
    Tmax = int(lens.max())
    alive = np.arange(Tmax)[None, :] < lens[:, None]
    Y = np.zeros((n, Tmax), dtype=np.int8)
    for i in range(n):
        if tau[i] >= 0:
            Y[i, tau[i]: lens[i]] = 1

    section("SETUP")
    print(f"{n} series ({int((tau >= 0).sum())} with a break) | calib-split {args.calib_split} | "
          f"predictors {PREDICTORS}")
    print("Evaluation: A = oracle AUC (break position known), B = scan TS-AUC (position unknown), C = stacked LightGBM.")

    if "ARGBM" in PREDICTORS:
        rest = np.setdiff1d(np.arange(len(S)), idx)
        if len(rest) < 200:
            sys.exit("ARGBM needs series that are NOT in the evaluation set to train on: use a smaller --n "
                     f"(only {len(rest)} left over)")
        pool = rng.choice(rest, min(args.gbm_pool, len(rest)), replace=False)
        t0 = time.time()
        set_gbm(train_gbm(S, pool))
        print(f"  (ARGBM correction model trained in {time.time() - t0:.0f}s)", flush=True)

    steps5 = [np.arange(0, L, 5) for L in lens]
    sid_rows = np.concatenate([np.full(len(s5), i) for i, s5 in enumerate(steps5)])
    step_rows = np.concatenate(steps5)
    y_rows = np.concatenate([Y[i, s5] for i, s5 in enumerate(steps5)]).astype(np.int8)

    oracle = {}   # (pred, fam) -> {L: auc}
    scan = {}     # (pred, fam) -> (abs ts-auc, up ts-auc)
    blocks = {}   # pred -> (rows, 2*len(FAMS)) feature matrix for stacking

    for pname in PREDICTORS:
        t0 = time.time()
        absM = {f: np.full((n, Tmax), np.nan, dtype=np.float32) for f in FAMS}
        upM = {f: np.full((n, Tmax), np.nan, dtype=np.float32) for f in FAMS}
        orac = {f: {L: ([], []) for L in LS} for f in FAMS}
        rows = []
        for i in range(n):
            h, o, t = hist[i], onl[i], int(tau[i])
            s = np.r_[h, o]
            nh = len(h)
            fit_end = nh if args.calib_split >= 1.0 else int(nh * args.calib_split)
            cs = BURN if args.calib_split >= 1.0 else fit_end
            s0 = s - s[:fit_end].mean()
            e_all, scale_all = compute_e(pname, s0, fit_end)
            if scale_all is None:
                scale_all = max(float(e_all[cs:nh].std()), 1e-8)
            r_all = np.clip(np.nan_to_num(e_all / scale_all), -50.0, 50.0)
            incs = increments(r_all, nh, cs)
            feat = []
            for f in FAMS:
                g = incs[f]
                gc = g[cs:nh]
                c = float(gc.mean())
                lrv = lrv_batch(gc - c, ABS_FLOOR.get(f, 1e-8))
                invsd = 1.0 / np.sqrt(lrv)
                gon = g[nh:] - c
                a, u = scan_scores(gon, invsd, MINSEG[f])
                absM[f][i, : len(o)] = a
                upM[f][i, : len(o)] = u
                feat.append(a[steps5[i]])
                feat.append(u[steps5[i]])
                for L in LS:
                    if t >= 0 and len(o) - t >= L:
                        orac[f][L][0].append(abs(gon[t: t + L].sum()) * invsd / np.sqrt(L))
                    elif t < 0 and len(o) >= L:
                        orac[f][L][1].append(abs(gon[:L].sum()) * invsd / np.sqrt(L))
            rows.append(np.array(feat).T)
        blocks[pname] = np.concatenate(rows).astype(np.float32)
        for f in FAMS:
            scan[(pname, f)] = (ts_auc_matrix(absM[f], Y, alive), ts_auc_matrix(upM[f], Y, alive))
            oracle[(pname, f)] = {L: auc_two(np.array(orac[f][L][0]), np.array(orac[f][L][1])) for L in LS}
        extra = f" | median chosen AR order {int(np.median(chosen_p))}" if pname.startswith("ARbic") and chosen_p else ""
        print(f"  {pname:12s} done in {time.time() - t0:5.0f}s{extra}", flush=True)

    # ---------------- tables
    def matrix(title, getter):
        print(f"\n{title}")
        print(f"{'predictor':13s}" + "".join(f"{f:>8s}" for f in FAMS))
        for p in PREDICTORS:
            print(f"{p:13s}" + "".join(f"{getter(p, f):8.3f}" for f in FAMS))

    section("A. ORACLE AUC (break position known; two-sided; 0.5 = no signal)")
    for L in LS:
        matrix(f"L = {L} points after the break", lambda p, f, L=L: oracle[(p, f)][L])

    section("B. SCAN TS-AUC (break position unknown; competition metric; two-sided max over start times)")
    matrix("scan |.|  (two-sided)", lambda p, f: scan[(p, f)][0])
    matrix("scan up   (one-sided: only increases of the increment series)", lambda p, f: scan[(p, f)][1])
    allrows = sorted(((max(v), k) for k, v in scan.items()), reverse=True)
    print("\nTop 15 single (predictor, family) detectors by scan TS-AUC (best of two-sided / one-sided):")
    for sc, (p, f) in allrows[:15]:
        print(f"  {sc:.4f}  {p:12s} {f:6s}  (two-sided {scan[(p, f)][0]:.4f}, up {scan[(p, f)][1]:.4f})")
    best_by_pred = {p: max(max(scan[(p, f)]) for f in FAMS) for p in PREDICTORS}
    print("\nBest single family per predictor:", {p: round(v, 4) for p, v in best_by_pred.items()})

    with open(os.path.join(args.out, "single_detectors.csv"), "w", encoding="utf-8") as fh:
        fh.write("predictor,family,scan_abs,scan_up,oracle30,oracle100,oracle300\n")
        for p in PREDICTORS:
            for f in FAMS:
                fh.write(f"{p},{f},{scan[(p, f)][0]:.5f},{scan[(p, f)][1]:.5f}," +
                         ",".join(f"{oracle[(p, f)][L]:.5f}" for L in LS) + "\n")
    print(f"[saved {args.out}/single_detectors.csv]")

    if not args.no_stack:
        section("C. STACKED LightGBM TS-AUC (3-fold CV by series; all 8 families x {two-sided, up} per predictor)")
        res = {}
        for p in PREDICTORS:
            res[p] = stack_eval(blocks, [p], y_rows, step_rows, sid_rows)
            print(f"  {p:12s} alone        {res[p]:.4f}", flush=True)
        base = res["raw"]
        print(f"\nIncremental value over the 'raw' detectors (raw alone = {base:.4f}):")
        for p in PREDICTORS:
            if p == "raw":
                continue
            sc = stack_eval(blocks, ["raw", p], y_rows, step_rows, sid_rows)
            print(f"  raw + {p:12s} {sc:.4f}   ({sc - base:+.4f})", flush=True)
        if "AR10" in PREDICTORS:
            print(f"\nDo longer / regularised memories add anything beyond AR10?  (AR10 alone = {res['AR10']:.4f})")
            for p in PREDICTORS:
                if p in ("raw", "AR10"):
                    continue
                sc = stack_eval(blocks, ["AR10", p], y_rows, step_rows, sid_rows)
                print(f"  AR10 + {p:12s} {sc:.4f}   ({sc - res['AR10']:+.4f})", flush=True)
        sc_all = stack_eval(blocks, PREDICTORS, y_rows, step_rows, sid_rows)
        print(f"\n  ALL predictors together     {sc_all:.4f}   ({sc_all - base:+.4f} vs raw alone)")
        print("\n(With only ~%d series these stacking numbers are a ranking tool, not a leaderboard estimate.)" % n)

    section("HOW TO READ THIS")
    print("""- 'raw' is the no-prediction baseline (the standardised series itself): it is what our current
  mean/variance/tail features already do. A predictor is only worth adding if it beats raw in B or adds value in C.
- A. tells you which residual families carry signal when the break position is known; B tells you how much
  survives when the position is unknown (that is the real metric); C tells you whether the signal is NEW.
- ARbic+scale adapts its error scale to recent data: if it scores worse than ARbic on the 'var'/'abs'
  families, that confirms adapting hides variance breaks (the frozen predictor is the right design).
- Compare 'up' with two-sided: if one-sided wins, increases are where the signal is.
- Try --calib-split 0.8 to see whether fitting and calibrating on the same history flatters the predictors.""")
    print(f"\nDone in {time.time() - t_start:.0f}s. Outputs in {args.out}/")


if __name__ == "__main__":
    main()
