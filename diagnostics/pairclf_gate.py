"""Single-feature gate for the pair-classifier scan features (sbrt/pairclf.py, HANDOFF section 38).

Train the window-vs-history classifier on the ORIGINAL series 0 .. n_train-1 (default 0-4999), run the scan on the
held-out series offset .. offset+n_eval-1 (default 5000-6999) at every 3rd online step (cv.py's rows), and report:
  1. held-out PAIR AUC by window length (the classifier's own task, weighted like training)
  2. single-feature TS-AUC of every pc_ column: all, steps 0-100 / 100-300 / 300-700 / 700+, and the
     variance-unchanged slice (no-break series + breaks whose whitened post-break variance is 0.85-1.15; the same
     definition as sbrt/nbm.py), next to the references bo_w_logodds (~0.60), bo_wu_lvr and nb_logodds (~0.62;
     the NBM gate model data/nbm_v1.pt through the numba path, CPU, prior fitted on series 0 .. n_train-1)
  3. redundancy: max |Spearman rank correlation| of each pc_ column with (a) the 270 kernel features
     (data/feat_270.npz) and (b) the bo_ / vol_ / nb_ features, and which feature it is
The bo_ / vol_ / kernel columns are sliced from the cv.py caches (data/bo1_orig.npz, data/vol2_orig.npz,
data/feat_270.npz; rows = every 3rd online step, series in order), so they are the exact values the trees see.

What the gate can and cannot say (HANDOFF 33.7 / 35.1): it answers "is the family working, and is it a copy?".
It does NOT decide adoption, and single-feature strength does not predict what the trees gain from a complementary
column (bo_wu_lvr: about 0.56 here, 6th by gain in the trees; nb_lvr, the top tree feature, about 0.58). So the
paired cv.py run follows unless the family is BROKEN or a COPY. DECISION RULE (written before any full run):
  STOP (record as rejected at the gate) only if
      (a) the held-out pair AUC at L = 256 is < 0.55 (the classifier learned nothing), or
      (b) every pc_ column is < 0.53 overall, or
      (c) every pc_ column has max |rank corr| >= 0.92 with some existing feature (kernel, bo_, vol_, nb_).
  GO to the paired cv.py run otherwise.
  INSTRUMENT: the strength part of the rule ((b): >= 0.53 overall) is also applied to nb_lvr and bo_wu_lvr, both
  useful in the trees. If it would stop either of them, the gate prints NOT TRUSTED and the paired run goes ahead.
  Reported only (no decision): columns >= 0.585 overall, or >= max(0.52, nb_logodds + 0.015) on the var-same slice.
  (Amended twice before any full run: a first version required +0.010 over nb_logodds on var-same, which passes on
  noise because nb sits near 0.50 there; a second required >= 0.585 overall, which would have stopped bo_wu_lvr.)

Run (CPU; about 3-5 min, my estimate):  python diagnostics/pairclf_gate.py
Smoke:                                  python diagnostics/pairclf_gate.py --n-train 600 --n-eval 300 --no-nbm
-> cv_out/pairclf_gate.txt (smoke: cv_out/pairclf_gate_smoke.txt)
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sbrt import bocpd as BO  # noqa: E402
from sbrt import features as F  # noqa: E402
from sbrt import pairclf as PC  # noqa: E402
from sbrt import volstate as VS  # noqa: E402
from sbrt.data import load_train  # noqa: E402
from sbrt.metric import ts_auc  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n-train", type=int, default=5000)
ap.add_argument("--offset", type=int, default=5000)
ap.add_argument("--n-eval", type=int, default=2000)
ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--k", type=int, default=PC.K_PER)
ap.add_argument("--rounds", type=int, default=PC.ROUNDS)
ap.add_argument("--leaves", type=int, default=PC.LEAVES)
ap.add_argument("--no-nbm", action="store_true", help="skip the NBM reference (smoke runs)")
ap.add_argument("--corr-rows", type=int, default=150000)
args = ap.parse_args()
assert args.n_train <= args.offset, "training series must not overlap the evaluation series"
SUB = 3
smoke = args.n_train < 5000 or args.n_eval < 2000
os.makedirs("cv_out", exist_ok=True)
log = open(os.path.join("cv_out", "pairclf_gate_smoke.txt" if smoke else "pairclf_gate.txt"), "w", encoding="utf-8")


def say(s=""):
    print(s, flush=True)
    log.write(s + "\n")
    log.flush()


S_all, tau_all = load_train()
say(f"settings: {vars(args)}")
t00 = time.time()

# 1. classifier on the training originals; held-out pair AUC
tr = np.arange(args.n_train)
ev = np.arange(args.offset, args.offset + args.n_eval)
t0 = time.time()
P = PC.build_pairs([(S_all[i][1], S_all[i][2]) for i in tr], tau_all[tr], tr, args.k, 0, args.threads)
PH = PC.build_pairs([(S_all[i][1], S_all[i][2]) for i in ev], tau_all[ev], ev, args.k, 0, args.threads)
say(f"pairs: {len(P['y']):,} training rows from {len(tr):,} series, {len(PH['y']):,} held-out rows ({time.time() - t0:.0f}s)")
t0 = time.time()
b = PC.train_classifier(P, rounds=args.rounds, leaves=args.leaves)
M = PC.export_model(b)
say(f"classifier: {args.rounds} rounds, {args.leaves} leaves in {time.time() - t0:.0f}s")
pair_auc = PC.pair_auc_by_L(M, PH, np.arange(len(PH["y"])))
say("held-out pair AUC by window length: " + "  ".join(f"L{L} {v:.4f}" for L, v in zip(PC.LS, pair_auc)))
imp = b.feature_importance("gain")
imp = imp / imp.sum()
say("classifier inputs by gain: " + ", ".join(f"{PC.INPUT_NAMES[j]} {100 * imp[j]:.1f}%" for j in np.argsort(-imp)[:15]))
del P, PH

# 2. scan features on the evaluation series
S, tau = [S_all[i] for i in ev], tau_all[ev]
t0 = time.time()
PCF = np.concatenate(PC.features_parallel([(s[1], s[2]) for s in S], M, SUB, args.threads))
npts = sum(len(s[2]) for s in S)
say(f"pc features for {len(S):,} series ({npts:,} points) in {time.time() - t0:.0f}s on {args.threads} threads")
step = np.concatenate([np.arange(0, len(s[2]), SUB) for s in S])
sid = np.repeat(np.arange(len(S)), [(len(s[2]) + SUB - 1) // SUB for s in S])
y = ((tau[sid] >= 0) & (step >= tau[sid])).astype(np.int8)
assert len(PCF) == len(y) and np.isfinite(PCF).all()

# references, sliced from the cv.py caches (exact rows the trees see)
rows_all = np.array([(len(s[2]) + SUB - 1) // SUB for s in S_all])
r0 = int(rows_all[:args.offset].sum())
r1 = r0 + int(rows_all[ev].sum())
REF, REF_NAMES = [], []
bo = np.load(os.path.join("data", f"bo{BO.VERSION}_orig.npz"))["B"]
assert len(bo) == rows_all.sum(), "bo cache layout"
REF.append(bo[r0:r1])
REF_NAMES += list(BO.FEATURE_NAMES)
del bo
vo = np.load(os.path.join("data", f"vol{VS.VERSION}_orig.npz"))["V"]
REF.append(vo[r0:r1])
REF_NAMES += list(VS.FEATURE_NAMES)
del vo
if not args.no_nbm:
    from sbrt import nbm as NBM
    t0 = time.time()
    Mn = NBM.load_model()
    sf = NBM.sums_parallel([(s[1], s[2]) for s in S_all[:args.n_train]], Mn, args.threads)
    prior, nfit = NBM.fit_prior_w([NBM.w_of(q) for q in sf], tau_all[:args.n_train])
    del sf
    se = NBM.sums_parallel([(s[1], s[2]) for s in S], Mn, args.threads)
    NB = np.concatenate([NBM.features_from_sums(q, prior, SUB) for q in se])
    del se
    REF.append(NB)
    REF_NAMES += list(NBM.FEATURE_NAMES)
    say(f"NBM reference (data/nbm_v1.pt, numba, prior from {nfit} breaks) in {time.time() - t0:.0f}s")
REF = np.column_stack(REF)
assert len(REF) == len(y)
t0 = time.time()
K = np.load(os.path.join("data", f"feat_{F.N_FEAT}.npz"))
assert np.array_equal(K["step"][r0:r1], step) and np.array_equal(K["sid"][r0:r1], sid + args.offset), "kernel layout"
KX = K["X"][r0:r1]
del K
say(f"kernel features sliced from data/feat_{F.N_FEAT}.npz in {time.time() - t0:.0f}s")

# variance-unchanged slice (as sbrt/nbm.py)
vr = np.full(len(S), np.nan)
for i, (s, t) in enumerate(zip(S, tau)):
    if t >= 0 and len(s[2]) - t >= 30:
        vr[i] = BO.whitened_online(s[1], s[2])[0][t:t + 300].var()
same = (vr > 0.85) & (vr < 1.15)
in_slice = (tau[sid] < 0) | same[sid]
buckets = ((0, 100), (100, 300), (300, 700), (700, 10000))


def aucs(x):
    r = [ts_auc(x, y, step)]
    for lo, hi in buckets:
        m = (step >= lo) & (step < hi)
        r.append(ts_auc(x[m], y[m], step[m]))
    r.append(ts_auc(x[in_slice], y[in_slice], step[in_slice]))
    return r


from scipy.stats import rankdata  # noqa: E402
rng = np.random.default_rng(0)
sel = np.sort(rng.choice(len(y), min(len(y), args.corr_rows), replace=False))


def zr(A):
    R = np.column_stack([rankdata(A[:, j]) for j in range(A.shape[1])])
    return (R - R.mean(0)) / (R.std(0) + 1e-12)


RK, RR = zr(KX[sel]), zr(REF[sel])
say(f"\nsingle-feature TS-AUC on series {args.offset}-{args.offset + args.n_eval - 1} "
    f"({int(same.sum())} variance-unchanged breaks)")
say(f"  {'feature':16s}   all     0-100  100-300  300-700    700+   var-same   max|rho| kernel / bo,vol,nb")
res = {}
for nm in ("bo_w_logodds", "bo_wu_logodds", "bo_wu_lvr", "nb_logodds", "nb_lvr"):
    if nm in REF_NAMES:
        res[nm] = aucs(REF[:, REF_NAMES.index(nm)])
        say(f"  {nm:16s} " + "  ".join(f"{v:7.4f}" for v in res[nm]))
nb_vs = res.get("nb_logodds", [np.nan] * 6)[5]
strong_cols, alive, not_copy = [], [], []
for c, nm in enumerate(PC.FEATURE_NAMES):
    r = aucs(PCF[:, c])
    z = rankdata(PCF[sel, c])
    z = (z - z.mean()) / (z.std() + 1e-12)
    ck = np.abs(RK.T @ z) / len(z)
    cr = np.abs(RR.T @ z) / len(z)
    jk, jr = int(np.nanargmax(ck)), int(np.nanargmax(cr))
    mx = max(ck[jk], cr[jr])
    say(f"  {nm:16s} " + "  ".join(f"{v:7.4f}" for v in r)
        + f"   {ck[jk]:.2f} {F.FEATURE_NAMES[jk]} / {cr[jr]:.2f} {REF_NAMES[jr]}")
    if r[0] >= 0.53:
        alive.append(nm)
    if mx < 0.92:
        not_copy.append(nm)
    if r[0] >= 0.585 or (np.isfinite(nb_vs) and r[5] >= max(0.52, nb_vs + 0.015)):
        strong_cols.append(nm)

broken = pair_auc[-1] < 0.55
stop = broken or not alive or not not_copy
inst = [(nm, res[nm][0]) for nm in ("nb_lvr", "bo_wu_lvr") if nm in res]
trusted = all(v >= 0.53 for _, v in inst)
say(f"\nreported only: columns >= 0.585 overall or >= max(0.52, nb_logodds + 0.015) on var-same: "
    f"{', '.join(strong_cols) if strong_cols else 'none'}")
say("INSTRUMENT: the strength rule (>= 0.53 overall) on known-useful tree features: "
    + ", ".join(f"{nm} {v:.4f}" for nm, v in inst) + (" -> OK" if trusted else " -> NOT TRUSTED (paired run goes ahead)"))
verdict = "STOP" if (stop and trusted) else "GO"
say(f"DECISION (rule in the docstring; STOP only if broken or a copy): {verdict}  [pair AUC L256 "
    f"{pair_auc[-1]:.4f} (stop < 0.55); columns >= 0.53: {len(alive)}/{PC.N_PC}; columns not a copy (< 0.92): "
    f"{len(not_copy)}/{PC.N_PC}]" + ("   (SMOKE SIZE: not evidence)" if smoke else ""))
say(f"total {time.time() - t00:.0f}s")
