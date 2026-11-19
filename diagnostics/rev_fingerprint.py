"""Fingerprint check for time-reversed training series (`cv.py --aug-rev`, sbrt/augment.make_reversed). No CV run.

Question: can a classifier tell a REAL series from its time-reversed copy, on the production streaming features?
If yes much more easily than it tells a real series from a boundary re-cut (already adopted, `--aug 2`), the trees
would learn the reversal artefact instead of breaks, so `--aug-rev` is not worth a CV run.

Arms (LightGBM, 5-fold StratifiedGroupKFold grouped by parent series, so a series and its copy never sit on
opposite sides; features = 270 kernel + 13 vol + 10 bocpd, geometry columns ctx_logn / t_log dropped; one row per
series per online step k = 30 / 100 / 300 (rows at step k - 1, when the online part is long enough)):
  control      real vs real, random half split of the same series pool (should be ~0.50)
  re-cut       real vs a boundary re-cut of it (make_variants, 1 per series): the floor every adopted re-cut has
  reversed     real vs its reversed copy at the same boundary (make_reversed shift 0): the candidate
  rev+re-cut   real vs a re-cut of the reversed copy (make_reversed, the shift != 0 variant)
Run separately on NO-BREAK series (pure normal behaviour) and on BREAK series (rows are only post-break rows here:
step k - 1 after tau, i.e. tau + k - 1, so the post-break regime played backwards is what is compared).

DECISION RULE (written 2026-10-20, before any run):
  GO to the paired cv.py trees run if, on the no-break series AND on the break series, reversed <= re-cut + 0.02 at
  every k and the control is within 0.50 +- 0.03 (the test is sane). Otherwise the reversal leaves a fingerprint:
  report the top features of the reversed arm and CLOSE (an owner override would still be the paired cv.py run).

Run (owner):  python diagnostics/rev_fingerprint.py                          # ~10-25 min CPU, 12 threads
Smoke:        python diagnostics/rev_fingerprint.py --n-series 600 --out cv_out/rev_fingerprint_smoke.txt
Output:       cv_out/rev_fingerprint.txt

--mode flip (HANDOFF section 40, 2026-10-21): the same test for SIGN-REFLECTED copies (sbrt/augment.make_flipped,
(-history, -online), same geometry and tau). Arms: control, re-cut, flipped (shift 0), flip+re-cut.
DECISION RULE (written 2026-10-21, before any run; same thresholds as the reversal rule):
  GO to a paired cv.py --aug-flip trees run if, on the no-break series AND on the break series, flipped <= re-cut + 0.02
  at every k and the control is within 0.50 +- 0.03. Otherwise CLOSE and report the top features of the flipped arm.
Run (owner):  python diagnostics/rev_fingerprint.py --mode flip              # -> cv_out/flip_fingerprint.txt
Smoke:        python diagnostics/rev_fingerprint.py --mode flip --n-series 600 --out cv_out/flip_fingerprint_smoke.txt
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from joblib import Parallel, delayed

from sbrt import augment as A
from sbrt import bocpd as BO
from sbrt import features as F
from sbrt import volstate as VS
from sbrt.data import load_train

ap = argparse.ArgumentParser()
ap.add_argument("--n-series", type=int, default=0, help="0 = all 10,000")
ap.add_argument("--threads", type=int, default=12)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--mode", choices=("rev", "flip"), default="rev",
                help="rev = time reversal (section 37); flip = sign reflection (section 40)")
ap.add_argument("--out", default=None, help="default cv_out/rev_fingerprint.txt or cv_out/flip_fingerprint.txt")
args = ap.parse_args()
if args.out is None:
    args.out = os.path.join("cv_out", f"{args.mode}_fingerprint.txt")
CAND, COMBO = ("reversed", "rev+re-cut") if args.mode == "rev" else ("flipped", "flip+re-cut")


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
print(f"{'reversal' if args.mode == 'rev' else 'sign-flip'} fingerprint check  {time.strftime('%Y-%m-%d %H:%M')}  args: {vars(args)}")
T0 = time.time()


def stamp(msg):
    print(f"[{time.time() - T0:7.1f} s] {msg}", flush=True)


KS = (30, 100, 300)
_ALL = list(F.FEATURE_NAMES) + list(VS.FEATURE_NAMES) + list(BO.FEATURE_NAMES)
KEEP = np.array([i for i, n in enumerate(_ALL) if n not in ("ctx_logn", "t_log")])
NAMES = [_ALL[i] for i in KEEP]

S, tau = load_train()
tau = np.asarray(tau)
if args.n_series:
    S, tau = S[:args.n_series], tau[:args.n_series]


def feat_rows(h, o, t0):
    """Production streaming features at online steps t0 + k - 1 for every k that fits ({k: row})."""
    end = min(len(o), t0 + max(KS))
    oo = o[:end]
    X = np.hstack([F.run_series(h, oo), VS.run_series_vol(h, oo), BO.run_series_bocpd(h, oo)])[:, KEEP]
    return {k: X[t0 + k - 1] for k in KS if t0 + k - 1 < end}


def rows(items):
    """items: (parent, hist, online, tau) -> {k: (X, parent)}. Rows at step k - 1 (no break) or tau + k - 1."""
    R = Parallel(n_jobs=args.threads, backend="threading")(
        delayed(feat_rows)(h, o, max(t, 0)) for _, h, o, t in items)
    out = {k: ([], []) for k in KS}
    for (p, _, _, _), r in zip(items, R):
        for k, x in r.items():
            out[k][0].append(x)
            out[k][1].append(p)
    return {k: (np.array(out[k][0]), np.array(out[k][1])) for k in KS}


def disc(Ra, Rb, seed=0):
    """AUC of real (a) vs variant (b), out of fold, grouped by parent; plus mean gain importance."""
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold
    X = np.vstack([Ra[0], Rb[0]])
    y = np.r_[np.zeros(len(Ra[0])), np.ones(len(Rb[0]))]
    g = np.r_[Ra[1], Rb[1]]
    p = np.zeros(len(y))
    imp = np.zeros(X.shape[1])
    for tr, va in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, g):
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=40,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.5, verbose=-1,
                               random_state=seed, importance_type="gain", n_jobs=args.threads)
        m.fit(X[tr], y[tr])
        p[va] = m.predict_proba(X[va])[:, 1]
        imp += m.feature_importances_
    return roc_auc_score(y, p), imp / max(imp.sum(), 1e-12)


def pick(R, idx_set):
    return {k: (R[k][0][np.isin(R[k][1], idx_set)], R[k][1][np.isin(R[k][1], idx_set)]) for k in KS}


VERDICT = {}
for label, sel in (("NO-BREAK", tau < 0), ("BREAK", tau >= 0)):
    idx = np.flatnonzero(sel)
    sub = [S[i] for i in idx]
    st = tau[idx]
    print(f"\n=== {label} series: {len(idx):,} ===")
    real = [(int(i), S[i][1], S[i][2], int(tau[i])) for i in idx]
    rc = A.make_variants(sub, st, 1, seed=args.seed)
    recut = [(int(idx[v[0]]), v[2], v[3], v[4]) for v in rc]
    make = A.make_reversed if args.mode == "rev" else A.make_flipped
    rv = make(sub, st, 2, seed=args.seed)
    rev = [(int(idx[v[0]]), v[2], v[3], v[4]) for v in rv if v[1] == 0]
    revcut = [(int(idx[v[0]]), v[2], v[3], v[4]) for v in rv if v[1] != 0]
    stamp(f"variants: re-cut {len(recut):,}, {CAND} {len(rev):,}, {COMBO} {len(revcut):,}")
    R_real = rows(real)
    stamp("real features done")
    R = {"re-cut": rows(recut), CAND: rows(rev), COMBO: rows(revcut)}
    stamp("variant features done")
    half = np.random.default_rng([args.seed, 5]).permutation(idx)
    A_, B_ = half[: len(half) // 2], half[len(half) // 2:]
    res = {}
    print(f"{'arm':12s} " + " ".join(f"   k={k:<4d}" for k in KS) + "   (AUC real vs arm; rows per class)")
    for arm in ("control", "re-cut", CAND, COMBO):
        aucs, imps = [], None
        cells = []
        for k in KS:
            if arm == "control":
                a, b = pick(R_real, A_)[k], pick(R_real, B_)[k]
            else:
                a, b = R_real[k], R[arm][k]
            if len(a[0]) < 50 or len(b[0]) < 50:
                aucs.append(np.nan)
                cells.append("     -    ")
                continue
            auc, imp = disc(a, b, seed=args.seed)
            aucs.append(auc)
            imps = imp if imps is None else imps + imp
            cells.append(f"{auc:.3f} ({min(len(a[0]), len(b[0])):>5d})")
        res[arm] = (np.array(aucs), imps)
        print(f"{arm:12s} " + " ".join(cells))
        if arm in ("re-cut", CAND) and imps is not None:
            top = np.argsort(imps)[::-1][:6]
            print("             top features: " + ", ".join(f"{NAMES[j]} {imps[j] / len(KS):.2f}" for j in top))
    ctl, rc_a, rv_a = res["control"][0], res["re-cut"][0], res[CAND][0]
    ok_ctl = bool(np.all(np.abs(ctl[~np.isnan(ctl)] - 0.5) <= 0.03))
    m = ~np.isnan(rc_a) & ~np.isnan(rv_a)
    ok_rev = bool(np.all(rv_a[m] <= rc_a[m] + 0.02))
    VERDICT[label] = (ok_ctl, ok_rev)
    print(f"{label}: control sane {ok_ctl}; {CAND} <= re-cut + 0.02 at every k {ok_rev} "
          f"({CAND} - re-cut: {np.round(rv_a - rc_a, 3)})")

go = all(c and r for c, r in VERDICT.values())
flag = "--aug-rev" if args.mode == "rev" else "--aug-flip"
print(f"\nDECISION: {'GO -> paired cv.py ' + flag + ' trees run' if go else 'FAIL -> CLOSE (' + CAND + ' copies leave a fingerprint, or the test is not sane)'}")
stamp("done")
