"""Gate for the break transplant (sbrt/breakgen.py): are transplanted breaks "matched" to real ones? No CV run.

Three sections (all CPU, numba kernels; `--sections` picks them):

precheck  Is a break's change predictable from its history? Per real donor (post-break >= 300 points) and per
          placebo donor (no-break series, fake break), six components of the whitened post-break stream over its
          first 300 points: mean, log var, skew, excess kurtosis, lag-1 autocorrelation, lag-1 autocorrelation of
          |w|. Reports (a) the SIGNAL SHARE of each component, 1 - var(placebo) / var(real): how much of a donor's
          measured change is real change rather than sampling noise; (b) 5-fold CV R^2 of LightGBM from the 8
          history descriptors (NormalModel.desc) to each component, on real donors and on placebo (control ~0).
calib     The synth_calibrate.py oracle (break position known, 10 single statistics of the L points after the
          break vs no-break segments, two-sided AUC, L = 30 / 100 / 300) for: REAL breaks, transplants at each
          --lams value, PLACEBO transplants (no-break donors: noise only), and the old hand-made synth.py set
          (no-mean types, scale 0.5) as the reference. Distance = sum over L of |average AUC - real average AUC|.
disc      (REPLACED by train-on-synthetic / test-on-real: see the fourth amendment below; this text is the
          original design) Fingerprint check. A LightGBM classifier tells REAL break segments from SYNTHETIC ones, on
          the production streaming features (270-column kernel + 13 vol + 10 bocpd) read at online step t + L - 1, L = 30 / 100 /
          300; geometry columns (ctx_logn, t_log) dropped; 5-fold StratifiedGroupKFold by parent series; AUC. Arms: transplant at --disc-lam, old synth.py (reference),
          placebo (no break at all: real breaks vs none, a check that the test has power), and a
          real-vs-real control (random split of the real breaks: ~0.5).

DECISION RULE (written 2026-10-20, before any run):
  P. precheck: match_k > 0 (matched donors) is worth a CV arm only if some component reaches R^2 >= 0.05 on real
     donors while the placebo R^2 stays <= 0.01; otherwise donors are drawn uniformly (match_k = 0).
  C. calib PASS for a lam if (i) distance < 0.0162 (the old synth's best, cv_out/synth_calibration_nomean_full.txt)
     AND (ii) |average AUC - real| <= 0.005 at EACH of L = 30, 100, 300 (the old set failed on shape: too clear
     early, too faint at L = 300). The best passing lam becomes --disc-lam.
  D. disc PASS if the transplant's AUC <= 0.55 at every L, the real-vs-real control is within 0.5 +- 0.02 and the
     placebo arm is >= 0.58 at L = 300 (the test has power: placebo vs real ~ how detectable real
     breaks are, so it is near 0.5 at small L by nature).
  Amended 2026-10-20 after the 400-series smoke run, before any full run: geometry columns dropped from D and
  history lengths drawn from the real distribution (the smoke placebo scored 0.69 at L = 30 on geometry alone);
  StratifiedGroupKFold (the grouped control was 0.32-0.54 at smoke size); lam grid widened down to 0.25 (smoke:
  lam 1 far too visible, 0.633 vs real 0.555 at L = 300). Second smoke (600 series): placebo donors alone gave
  0.604 at L = 300 (> real), driven by mean and ac1 -> breakgen VERSION 2 (smooth change map, no location,
  dependence shrunk by --lam-dep 0.5 x lam). Thresholds unchanged.
  Third amendment (same day, before any full run): breakgen VERSION 3 adds empirical-Bayes deconvolution; arms
  'transplant deconv' (lam 1, posterior draws) and its control 'placebo deconv' (should be ~0.5). The best passing
  transplant set of C (any arm) goes to D.
  Fourth amendment (same day, before any full run): the fingerprint discriminator has a floor of ~0.60 for ANY
  re-cut data (real no-break series vs re-cut no-break variants, no break anywhere: AUC 0.597, mostly vol_lvh1000),
  so it cannot judge break fidelity. D is now TRAIN-ON-SYNTHETIC / TEST-ON-REAL (TSTR): LightGBM break-vs-no-break
  on the same features at step t + L - 1 (no-break rows at a uniform fake t, so both classes share the geometry),
  trained on synthetic series whose parents are outside fold 0, tested on all real fold-0 series; compared with
  training on the real series outside fold 0 (TRTR) and on real + synthetic. Rule D: mean TSTR >= old synth's
  + 0.01, TSTR >= TRTR - 0.02 at every L, and real + synthetic >= TRTR on the mean.
  Fifth amendment (same day, after the advisor review, before any full run; thresholds unchanged): (1) rule C's
  reference is now real breaks in RE-CUT geometry (augment.make_variants, exact labels, scored against re-cut
  nulls like every synthetic set), since re-cut nulls differ from real nulls; the original-series row is printed
  for information. (2) D's real training set is all ~8,000 real series outside fold 0, a paired series-bootstrap SE
  of TSTR(arm) - TSTR(old synth) is printed, and if the old synth set (failed in CV) still shows real + it >=
  TRTR, D is demoted to INFORMATIONAL (the proxy cannot reject a known loser). D never overrides C.
  C and D both pass -> wire `--bgen` into cv.py (same plumbing as --synth) and run the standing pass rule (paired
  trees, early stop on folds 0 and 1). Either fails -> report which statistics / steps mismatch; one retry with a
  changed lam or min_post is allowed, then CLOSE.

Run (owner):  python diagnostics/breakgen_gate.py                      # all sections, ~40-70 min on 12 threads (CPU)
Smoke:        python diagnostics/breakgen_gate.py --n-series 400 --lams 1.0 --out cv_out/breakgen_gate_smoke.txt
Output:       cv_out/breakgen_gate.txt
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import kurtosis, rankdata, skew

from sbrt import bocpd as BO
from sbrt import breakgen as BG
from sbrt import features as F
from sbrt import volstate as VS
from sbrt.data import load_train
from sbrt.synth import fold_ids, make_synthetic

ap = argparse.ArgumentParser()
ap.add_argument("--sections", default="precheck,calib,disc")
ap.add_argument("--n-series", type=int, default=3000, help="parents for the synthetic sets (calib, disc)")
ap.add_argument("--per-series", type=int, default=3)
ap.add_argument("--lams", type=float, nargs="+", default=[0.25, 0.4, 0.55, 0.7, 0.85, 1.0])
ap.add_argument("--disc-lam", type=float, default=None, help="lam for the disc section (default: best calib lam)")
ap.add_argument("--min-post", type=int, default=100, help="donors need this many post-break points")
ap.add_argument("--match-k", type=int, default=0)
ap.add_argument("--lam-dep", type=float, default=0.5, help="dependence shrink as a FRACTION of lam (ac1 signal share ~0.36)")
ap.add_argument("--keep-mean", action="store_true")
ap.add_argument("--threads", type=int, default=12)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default=os.path.join("cv_out", "breakgen_gate.txt"))
args = ap.parse_args()
SECTIONS = set(x.strip() for x in args.sections.split(","))


class Tee:
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")

    def write(self, s):
        sys.__stdout__.write(s)
        self.f.write(s)

    def flush(self):
        sys.__stdout__.flush()
        self.f.flush()


os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
sys.stdout = Tee(args.out)
print(f"breakgen gate  {time.strftime('%Y-%m-%d %H:%M')}  args: {vars(args)}")

S, tau = load_train()
tau = np.asarray(tau)
FOLDS = fold_ids(tau)
LS = (30, 100, 300)
T0 = time.time()


def stamp(msg):
    print(f"[{time.time() - T0:7.1f} s] {msg}")


# =============================================================================================
# precheck
def components(w):
    w = w[:300]
    c = lambda x: float(np.corrcoef(x[:-1], x[1:])[0, 1])
    return [w.mean(), np.log(w.var() + 1e-9), skew(w), kurtosis(w), c(w), c(np.abs(w))]


COMP = ["mean", "log var", "skew", "kurt", "ac1", "ac1 |w|"]


def cv_r2(X, Y, seed=0):
    import lightgbm as lgb
    from sklearn.model_selection import KFold
    pred = np.zeros_like(Y)
    for tr, va in KFold(5, shuffle=True, random_state=seed).split(X):
        m = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.03, num_leaves=15, min_child_samples=50,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=seed)
        m.fit(X[tr], Y[tr])
        pred[va] = m.predict(X[va])
    return 1.0 - ((Y - pred) ** 2).mean() / Y.var()


if "precheck" in SECTIONS:
    print("\n=== PRECHECK: is a break's change predictable from its history? ===")
    idx = np.arange(len(S))
    real = Parallel(n_jobs=args.threads, backend="threading")(
        delayed(BG.donor_pool)(S, tau, [i], "break", 300) for i in idx)
    plac = Parallel(n_jobs=args.threads, backend="threading")(
        delayed(BG.donor_pool)(S, tau, [i], "placebo", 300, seed=args.seed) for i in idx)
    real = [c for r in real for c in r]
    plac = [c for r in plac for c in r]
    stamp(f"{len(real)} real donors, {len(plac)} placebo donors (>= 300 post points)")

    def comp_of(ch):
        # rebuild the time-ordered stream from the source series (Change keeps only the sorted values)
        _, h, o = S[ch.src]
        t = int(tau[ch.src])
        if t < 0:
            t = int(np.random.default_rng([args.seed, 11, ch.src]).integers(0, len(o) - 300 + 1))
        M = BG.NormalModel(np.r_[h, o[:t]])
        return components(M.forward(o[t:t + 300]))

    Cr = np.array(Parallel(n_jobs=args.threads, backend="threading")(delayed(comp_of)(c) for c in real))
    Cp = np.array(Parallel(n_jobs=args.threads, backend="threading")(delayed(comp_of)(c) for c in plac))
    Dr = np.array([c.desc for c in real])
    Dp = np.array([c.desc for c in plac])
    print("%-10s %10s %10s %12s %12s %12s %12s" % ("component", "sd real", "sd placebo", "signal share",
                                                   "robust share", "R2 real", "R2 placebo"))
    rsd = lambda x: float(np.subtract(*np.percentile(x, [75, 25]))) / 1.349
    for k, nm in enumerate(COMP):
        sr, sp = Cr[:, k].std(), Cp[:, k].std()
        share = 1.0 - sp ** 2 / sr ** 2
        r2r, r2p = cv_r2(Dr, Cr[:, k]), cv_r2(Dp, Cp[:, k])
        rshare = 1.0 - rsd(Cp[:, k]) ** 2 / rsd(Cr[:, k]) ** 2      # IQR-based: not driven by a few huge breaks
        print("%-10s %10.4f %10.4f %12.3f %12.3f %12.4f %12.4f" % (nm, sr, sp, share, rshare, r2r, r2p))
    print("Rule P: matched donors only if some R2 real >= 0.05 with R2 placebo <= 0.01.")
    stamp("precheck done")


# =============================================================================================
# calib (oracle of synth_calibrate.py, copied so this script does not run that one)
def auc(a, b):
    r = rankdata(np.r_[a, b])
    return (r[: len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))


def stat_vec(h, z):
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


def oracle(items):
    items = list(items)
    table = {}
    for L in LS:
        bk, nbk = [], []
        for h, o, t in items:
            mu, sd = h.mean(), h.std() + 1e-12
            hz, oz = (h - mu) / sd, (o - mu) / sd
            if t >= 0 and len(o) - t >= L:
                bk.append(stat_vec(hz, oz[t:t + L]))
            elif t < 0 and len(o) >= L:
                nbk.append(stat_vec(hz, oz[:L]))
        if len(bk) < 30 or len(nbk) < 30:
            continue
        for key in bk[0]:
            a = np.array([d[key] for d in bk])
            b = np.array([d[key] for d in nbk])
            a, b = a[np.isfinite(a)], b[np.isfinite(b)]
            med = np.median(b)
            table.setdefault(key, {})[L] = auc(np.abs(a - med), np.abs(b - med))
    return table


def summary(table):
    return {L: float(np.mean([row[L] for row in table.values() if L in row])) for L in LS}


sub = slice(0, args.n_series)
Ssub, tsub, fsub = S[sub], tau[sub], FOLDS[sub]
POOLS = {}


def pools(mode):
    if mode not in POOLS:
        POOLS[mode] = {k: BG.donor_pool(Ssub, tsub, np.flatnonzero(fsub == k), mode, args.min_post, seed=args.seed)
                       for k in range(5)}
        stamp(f"{mode} donor pools: {[len(v) for v in POOLS[mode].values()]}")
    return POOLS[mode]


def gen(lam, mode="break", deconv=False):
    """deconv: posterior draws of the donors' true change (sbrt.breakgen.Deconv), lam_dep = 1 (the deconvolution
    already removes the noise). For mode 'placebo' the noise of fold k comes from fold k+1's placebo donors."""
    chunks = np.array_split(np.arange(len(Ssub)), args.threads)
    P = pools(mode)
    D = None
    if deconv:
        PP = pools("placebo")
        if mode == "break":
            D = BG.fit_deconv(P, PP)
        else:
            D = {k: BG.Deconv([c.comp() for c in P[k]], [c.comp() for c in PP[(k + 1) % 5]]) for k in P}
        for k in (0,):
            stamp(f"deconv ({mode}) fold {k}: noise sd {np.round(D[k].sd, 3)}  true-change sd {np.round(D[k].prior_sd(), 3)}")

    def run(ix):
        V = BG.make_breakgen([Ssub[i] for i in ix], tsub[ix], args.per_series, seed=args.seed, lam=lam,
                             folds=fsub[ix], min_post=args.min_post, match_k=args.match_k, donors=mode, pools=P,
                             lam_dep=(1.0 if deconv else args.lam_dep) * lam, keep_mean=args.keep_mean,
                             deconv=D, ids=ix, hist_pool=HPOOL)
        return [(int(ix[v[0]]),) + tuple(v[1:]) for v in V]

    out = Parallel(n_jobs=args.threads, backend="threading")(delayed(run)(ix) for ix in chunks if len(ix))
    return [v for o in out for v in o]


HPOOL = [len(s[1]) for s in Ssub]


NOMEAN = ("variance", "dependence", "tails", "skew", "volclust", "splice")
SETS = {}
best_lam = args.disc_lam
best_name = None if args.disc_lam is None else f"transplant lam {args.disc_lam:g}"

if "calib" in SECTIONS or "disc" in SECTIONS:
    stamp("building synthetic sets")
    for lam in sorted(set(args.lams + ([args.disc_lam] if args.disc_lam is not None else []))):
        SETS[f"transplant lam {lam:g}"] = gen(lam)
    SETS["placebo lam 1"] = gen(1.0, "placebo")
    SETS["transplant deconv"] = gen(1.0, "break", deconv=True)
    SETS["placebo deconv"] = gen(1.0, "placebo", deconv=True)
    SETS["old synth nomean 0.5"] = make_synthetic(Ssub, tsub, args.per_series, seed=0, scale=0.5, p_zero=0.25,
                                                  folds=fsub, types=NOMEAN)
    stamp("synthetic sets: " + ", ".join(f"{k} {len(v):,}" for k, v in SETS.items()))

if "calib" in SECTIONS:
    print("\n=== CALIB: oracle AUC of 10 single statistics, break position known ===")
    real = oracle((s[1], s[2], int(t)) for s, t in zip(S, tau))
    rs0 = summary(real)
    # REFERENCE for rule C: real breaks in RE-CUT geometry (augment.make_variants: exact labels, every value real),
    # scored against re-cut no-break segments like the synthetic sets (re-cut nulls differ from real nulls)
    from sbrt.augment import make_variants
    RC = make_variants(Ssub, tsub, args.per_series, seed=args.seed)
    recut = oracle((v[2], v[3], int(v[4])) for v in RC)
    rs = summary(recut)
    print(f"reference: real re-cuts {len(RC):,} (augment.make_variants, {args.per_series} per series)")
    res = {}
    for name, V in SETS.items():
        tab = oracle((v[3], v[4], v[5]) for v in V)
        res[name] = (tab, summary(tab))
        print(f"\n{name} ({len(V):,} pseudo-series)")
        print("%-16s %8s %8s %8s      %s" % ("statistic", "L=30", "L=100", "L=300", "real"))
        for key, row in sorted(tab.items(), key=lambda kv: -max(kv[1].values())):
            print("%-16s %8.3f %8.3f %8.3f      %.3f %.3f %.3f" % (key, row[30], row[100], row[300],
                                                                 real[key][30], real[key][100], real[key][300]))
    print("\nAVERAGE oracle AUC (rule C: distance < 0.0162 and |gap| <= 0.005 at each L)")
    print("%-24s %8s %8s %8s %10s %6s" % ("set", "L=30", "L=100", "L=300", "distance", "PASS"))
    print("%-24s %8.3f %8.3f %8.3f" % ("real (original series)", rs0[30], rs0[100], rs0[300]))
    print("%-24s %8.3f %8.3f %8.3f   <- reference of rule C" % ("real re-cuts", rs[30], rs[100], rs[300]))
    bestd = 1e9
    for name, (_, s) in res.items():
        d = sum(abs(s[L] - rs[L]) for L in LS)
        ok = d < 0.0162 and all(abs(s[L] - rs[L]) <= 0.005 for L in LS)
        print("%-24s %8.3f %8.3f %8.3f %10.4f %6s" % (name, s[30], s[100], s[300], d, "yes" if ok else "no"))
        if name.startswith("transplant") and ok and d < bestd and args.disc_lam is None:
            bestd, best_name = d, name
    print(f"best passing transplant set: {best_name}")
    stamp("calib done")


# =============================================================================================
# disc
def feat_rows(h, o, t):
    """Production streaming features (kernel + vol + bocpd) at online steps t + L - 1 for every L that fits."""
    end = min(len(o), t + max(LS))
    oo = o[:end]
    X = np.hstack([F.run_series(h, oo), VS.run_series_vol(h, oo), BO.run_series_bocpd(h, oo)])
    X = X[:, KEEP]
    return {L: X[t + L - 1] for L in LS if t + L - 1 < end}


# geometry columns are dropped: history length and online step say nothing about the break's SHAPE
_ALL = list(F.FEATURE_NAMES) + list(VS.FEATURE_NAMES) + list(BO.FEATURE_NAMES)
KEEP = np.array([i for i, n in enumerate(_ALL) if n not in ("ctx_logn", "t_log")])


def disc_auc(A, B, groups_a, groups_b, seed=0):
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedGroupKFold
    X = np.vstack([A, B])
    y = np.r_[np.zeros(len(A)), np.ones(len(B))]
    g = np.r_[groups_a, groups_b]
    p = np.zeros(len(y))
    for tr, va in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, g):
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=40,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.5, verbose=-1, random_state=seed)
        m.fit(X[tr], y[tr])
        p[va] = m.predict_proba(X[va])[:, 1]
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, p)


if "disc" in SECTIONS:
    print("\n=== DISC (TSTR): train break-vs-no-break on SYNTHETIC series, test on held-out REAL series ===")
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score

    TEST_FOLD = 0
    test_idx = np.flatnonzero(FOLDS == TEST_FOLD)          # all fold-0 series (their labels never generate data)

    def fake(i, n, L):
        return int(np.random.default_rng([args.seed, 23, int(i), n, L]).integers(0, n - L + 1))

    def one(g, h, o, t):
        if t >= 0:
            return feat_rows(h, o, t), 1
        # no-break rows: a uniform fake t per L, so both classes have the same geometry
        return {L: feat_rows(h, o, fake(g, len(o), L))[L] for L in LS if len(o) >= L}, 0

    def rows(items):
        """items (group, hist, online, tau) -> {L: (X, y, group)}: break rows at tau + L - 1 (label 1), no-break
        rows at fake t + L - 1 (label 0)."""
        R = Parallel(n_jobs=args.threads, backend="threading")(delayed(one)(*it) for it in items)
        out = {L: ([], [], []) for L in LS}
        for (g, _, _, _), (r, y) in zip(items, R):
            for L, x in r.items():
                out[L][0].append(x)
                out[L][1].append(y)
                out[L][2].append(g)
        return {L: tuple(np.array(v) for v in out[L]) for L in LS}

    PRED = {}

    def fit_auc(tr, te, key=None):
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=40,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.5, verbose=-1,
                               random_state=args.seed)
        m.fit(tr[0], tr[1])
        p = m.predict_proba(te[0])[:, 1]
        if key is not None:
            PRED[key] = p
        return roc_auc_score(te[1], p)

    def paired_se(ka, kb, n_boot=200):
        """Series-bootstrap SE of mean-over-L AUC(a) - AUC(b) on the shared test set."""
        rng = np.random.default_rng(args.seed)
        ug = np.unique(TE[LS[0]][2])
        idx = {L: {g: np.flatnonzero(TE[L][2] == g) for g in ug} for L in LS}
        d = []
        for _ in range(n_boot):
            gs = rng.choice(ug, len(ug))
            v = []
            for L in LS:
                ii = np.concatenate([idx[L][g] for g in gs if len(idx[L][g])])
                y = TE[L][1][ii]
                if y.min() == y.max():
                    continue
                v.append(roc_auc_score(y, PRED[(ka, L)][ii]) - roc_auc_score(y, PRED[(kb, L)][ii]))
            d.append(np.mean(v))
        return float(np.std(d))

    TE = rows([(int(i), S[i][1], S[i][2], int(tau[i])) for i in test_idx])
    # real training set = ALL real series outside the test fold (~8,000: production-sized, so 'real + synthetic'
    # is not flattered by a tiny real set)
    TR = rows([(int(i), S[i][1], S[i][2], int(tau[i])) for i in range(len(S)) if FOLDS[i] != TEST_FOLD])
    stamp(f"test rows (real fold {TEST_FOLD}): {[len(TE[L][1]) for L in LS]}; real training rows: "
          f"{[len(TR[L][1]) for L in LS]}")
    trtr = [fit_auc(TR[L], TE[L], ("TRTR", L)) for L in LS]
    se = [float(np.sqrt(a * (1 - a) / min((TE[L][1] == 1).sum(), (TE[L][1] == 0).sum()))) for L, a in zip(LS, trtr)]
    print("%-30s %8s %8s %8s %8s" % ("training set", "L=30", "L=100", "L=300", "mean"))
    print("%-30s %8.3f %8.3f %8.3f %8.3f   (rough SE per L %.3f / %.3f / %.3f; arms share the test set)"
          % ("REAL (TRTR)", *trtr, np.mean(trtr), *se))
    res = {}
    for name, V in SETS.items():
        SY_ = rows([(int(v[0]), v[3], v[4], int(v[5])) for v in V if fsub[v[0]] != TEST_FOLD])
        tstr = [fit_auc(SY_[L], TE[L], (name, L)) for L in LS]
        aug = [fit_auc((np.vstack([TR[L][0], SY_[L][0]]), np.r_[TR[L][1], SY_[L][1]]), TE[L], (name + "+R", L))
               for L in LS]
        res[name] = (tstr, aug)
        print("%-30s %8.3f %8.3f %8.3f %8.3f" % (name + " (TSTR)", *tstr, np.mean(tstr)))
        print("%-30s %8.3f %8.3f %8.3f %8.3f" % ("   real + it", *aug, np.mean(aug)))
    ref = np.mean(res["old synth nomean 0.5"][0])
    print("\nRule D (TSTR): a transplant set passes if its mean TSTR >= old synth's + 0.01, its TSTR >= TRTR - 0.02 at")
    print("every L, and 'real + it' >= TRTR on the mean. Placebo sets are controls (TSTR ~0.5 expected).")
    old_aug = np.mean(res["old synth nomean 0.5"][1])
    proxy_ok = old_aug < np.mean(trtr)
    print(f"proxy check: old synth (failed in CV) real + it {old_aug:.4f} vs TRTR {np.mean(trtr):.4f} -> "
          + ("the 'real + it' proxy separates it: D is a gate" if proxy_ok else
             "the proxy does NOT reject the known loser: D is INFORMATIONAL only, C and the cv.py run decide"))
    for name, (tstr, aug) in res.items():
        if name.startswith("transplant"):
            ok = (np.mean(tstr) >= ref + 0.01 and all(t >= r - 0.02 for t, r in zip(tstr, trtr))
                  and np.mean(aug) >= np.mean(trtr))
            dse = paired_se(name, "old synth nomean 0.5")
            print(f"  {name:28s} TSTR - old synth {np.mean(tstr) - ref:+.4f} +- {dse:.4f} (paired series bootstrap)"
                  f"   {'PASS' if ok else 'no'}{'' if proxy_ok else ' (informational)'}")
    stamp("disc done")

print(f"\ndone in {time.time() - T0:.0f} s")
