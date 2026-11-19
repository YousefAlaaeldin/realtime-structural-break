"""Per-step LightGBM on streaming features; 5-fold CV grouped by series; scored with TS-AUC.

Run:    python cv.py                      # full 5 folds
        python cv.py --folds 1            # quick first look
        python cv.py --rebuild            # recompute the feature cache
Output: cv_out/summary.txt, cv_out/importance.csv, cv_out/oof.npz

Design notes
  * Features are computed for every online step, then every 3rd step is kept (SUB) to save memory;
    TS-AUC is evaluated on those steps, which is an unbiased estimate of the all-steps value.
  * Folds are split by SERIES (rows of one series are near-copies), stratified by has-break.
  * --no-weights turns off the default TS-AUC-matched weighting: at each step t a positive gets weight
    n_neg(t)/2 and a negative n_pos(t)/2, so every step is class-balanced and counts in proportion to
    the metric's own weight n_pos*n_neg. This stops the model from just learning the time base rate.
  * The boosting round is chosen by TS-AUC on the validation fold (grid every --grid rounds), and the
    reported headline number is the best shared grid point on the AVERAGE curve (not per-fold best).
  * --aug N adds N boundary re-cut variants of every TRAINING series (sbrt/augment.py: exact labels,
    real values only). Folds are still split on the 10,000 original series, a variant only trains when
    its parent series is in the training fold, and validation uses original rows only, so the OOF file
    has the same layout as before and --compare gives a paired per-fold difference. Step weights take
    their per-step class targets from the original training rows (augmentation cannot shift what the
    metric weights); --aug-rel sets how much an augmented row counts relative to an original one.

Augmentation run (paired against the 270-feature baseline):
        python cv.py --min-leaf 2000 --leaves 31 --lr 0.03 --rounds 800 --threads 16 --aug 2 --compare cv_out/oof_loc.npz --tag aug2
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.stats import kurtosis
import lightgbm as lgb
from joblib import Parallel, delayed
from numba import njit, prange
from sklearn.model_selection import StratifiedKFold

from sbrt.data import load_train
from sbrt import augment as A
from sbrt import evidence as EV
from sbrt import external as XT
from sbrt import synth as SY
from sbrt import breakgen as BGN
from sbrt import volstate as VS
from sbrt import disttest as DT
from sbrt import mmdstream as MM
from sbrt import bocpd as BO
from sbrt import bocpd2 as B2
from sbrt import msvol as MSV
from sbrt import lratio as LR
from sbrt import nbm as NBM
from sbrt import nbdist as NBD
from sbrt import nbscore as NSC
from sbrt import pairclf as PC
from sbrt import pshape as PS
from sbrt import shape as SH
from sbrt import riocpd as RIO
from sbrt import sensdet as SD
from sbrt import embfeat as EMB
from sbrt import eqfore as EQ
from sbrt import features as F
from sbrt.metric import ts_auc

ap = argparse.ArgumentParser()
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--rounds", type=int, default=600)
ap.add_argument("--grid", type=int, default=50)
ap.add_argument("--lr", type=float, default=0.05)
ap.add_argument("--leaves", type=int, default=63)
ap.add_argument("--min-leaf", type=int, default=200)
ap.add_argument("--ff", type=float, default=0.5, help="feature_fraction")
ap.add_argument("--no-weights", action="store_true")
ap.add_argument("--rebuild", action="store_true")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--dart", type=float, default=0.0,
                help="XGBoost DART booster with this rate_drop (0 = plain gbtree; HANDOFF 45). Dropout acts in training "
                     "only; prediction uses every tree with its final weight. The round curve is the first k trees "
                     "with their FINAL weights (not the model as it was at round k)")
ap.add_argument("--dart-skip", type=float, default=0.5, help="--dart: skip_drop (probability of no dropout in a round)")
ap.add_argument("--lpo", type=int, default=-1,
                help="leave-pair-out teacher for outer fold K (gru_cv.py --distill, HANDOFF 44): for each inner fold i != K, "
                     "train on the 3 folds not in {K, i} and score fold i, with fold K's NBM / hpca columns. The saved OOF "
                     "covers the 8,000 training series of outer fold K (fold K rows stay NaN) and never saw fold K's labels")
ap.add_argument("--drop", type=str, default="",
                help="comma-separated feature-name prefixes to remove, e.g. ctx_ or ctx_,win10_ ; "
                     "a leading ~ means 'name contains', e.g. ~tail5")
ap.add_argument("--tag", type=str, default="", help="suffix for the output files (keeps ablation runs apart)")
ap.add_argument("--gpu", action="store_true", help="train with XGBoost on the CUDA GPU (shorthand for --backend xgb --device cuda)")
ap.add_argument("--backend", choices=["lgb", "xgb", "svm", "cat"], default="lgb",
                help="svm = least-squares SVM with an RBF kernel approximated by random Fourier features, solved in "
                     "closed form on the GPU (see svm_fold); its 'rounds' grid is a grid of regularisation strengths. "
                     "cat = CatBoost (symmetric trees by default; --device cuda for the GPU; HANDOFF section 26)")
ap.add_argument("--cat-depth", type=int, default=6, help="cat: tree depth (symmetric: 2^depth leaves)")
ap.add_argument("--cat-grow", choices=["SymmetricTree", "Depthwise", "Lossguide"], default="SymmetricTree",
                help="cat: grow policy; Lossguide uses --leaves and --min-leaf like the other tree backends")
ap.add_argument("--cat-ordered", action="store_true", help="cat: ordered boosting (slower; default Plain)")
ap.add_argument("--cat-border", type=int, default=254, help="cat: histogram bins per feature")
ap.add_argument("--lambda", dest="lam", type=float, default=10.0, help="xgb: reg_lambda (L2 on leaf values; HANDOFF section 27)")
ap.add_argument("--monotone", action="store_true",
                help="xgb: the score may only RISE with the one-sided evidence columns (BOCPD/NBM/evidence log-odds and "
                     "log Bayes factors, CUSUM/SR, every dist_ distance); all other columns free (HANDOFF section 27)")
ap.add_argument("--base-margin", type=str, default="",
                help="xgb: boost on the residual of this column (e.g. nb_logodds): per fold a weighted logistic fit of y "
                     "on [s, s*log1p(t), log1p(t)] on the training rows becomes every row's starting logit (HANDOFF section 27)")
ap.add_argument("--svm-dim", type=int, default=8192, help="random Fourier features (kernel approximation size)")
ap.add_argument("--svm-gmul", type=float, default=1.0, help="RBF bandwidth: gamma = gmul / number of input columns")
ap.add_argument("--svm-top", type=int, default=100,
                help="input columns: the top-K of --svm-imp (plus t_log and all ctx_ columns); 0 = all columns")
ap.add_argument("--svm-imp", type=str, default=os.path.join("cv_out", "importance_aug2_evc_vol_df.csv"))
ap.add_argument("--svm-lams", type=str, default="1e-5,1e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1,3,10,30,100",
                help="regularisation grid, in units of the mean eigenvalue of the weighted feature Gram matrix")
ap.add_argument("--svm-chunk", type=int, default=32768, help="rows per GPU chunk")
ap.add_argument("--nbm-hpca", type=int, default=0,
                help="append K columns nbh_pc0.. = the NBM GRU hidden state after each online point projected on its first K "
                     "principal components (per fold; same features and cache as gru_cv.py --nbm-hpca, HANDOFF 24); "
                     "needs --nbm; the SVM always takes them as inputs")
ap.add_argument("--svm-rank", type=float, default=0.0,
                help="share of the SVM loss that is PAIRWISE within each online step (positive vs negative rows of the "
                     "same step, pair weights matching TS-AUC); 0 = pointwise LS-SVM as before, 1 = pure per-step ranking; "
                     "still solved in closed form")
ap.add_argument("--svm-ard", choices=["off", "family", "column"], default="off",
                help="learn one RBF width per input family (ctx, t, nb, vol, bo, ev, dist, win, scan, full) or per column "
                     "on an inner 80/20 series split of the training fold before the closed-form solve")
ap.add_argument("--svm-ard-dim", type=int, default=2048, help="random features while learning the widths")
ap.add_argument("--svm-ard-rows", type=int, default=120000, help="inner-train rows (inner-validation rows: half)")
ap.add_argument("--svm-ard-steps", type=int, default=150, help="Adam steps")
ap.add_argument("--svm-ard-lr", type=float, default=0.03)
ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="xgb and cat only")
ap.add_argument("--threads", type=int, default=0, help="CPU threads to use (0 = all cores); lower it to keep the machine cool")
ap.add_argument("--aug", type=int, default=0, help="boundary re-cut variants per training series (0 = off)")
ap.add_argument("--aug-seed", type=int, default=0)
ap.add_argument("--aug-min-shift", type=int, default=20, help="smallest boundary move, in points")
ap.add_argument("--aug-trim", type=float, default=0.0,
                help="probability that a re-cut variant also drops the START of its history (new length drawn from "
                     "the original history-length distribution; labels unchanged)")
ap.add_argument("--aug-trim-only", type=int, default=0,
                help="extra variants per training series with the original boundary/online segment and a history "
                     "trimmed by at least 20%%")
ap.add_argument("--aug-early", type=int, default=0,
                help="extra re-cuts per series aimed at the first online steps (HANDOFF 43): break series get the break "
                     "in the first 100 online points, no-break series an ordinary re-cut (break share 0.5), every "
                     "online segment cut to 100 points; shifts of the --aug re-cuts excluded")
ap.add_argument("--tta", type=str, default="",
                help="test-time views, e.g. 51,150: every VALIDATION series is also scored with its boundary moved earlier by "
                     "each shift (multiples of 3; score-only sources, never trained on); the fold model's logits are averaged "
                     "over the views that exist at each step and reported paired against the plain score of the SAME model "
                     "(oof file <tag>_tta.npz). Trees only (lgb / xgb, binary)")
ap.add_argument("--aug-rev", type=int, default=0,
                help="time-reversed variants per training series (sbrt/augment.make_reversed): the series reversed "
                     "regime by regime at its own boundary, plus N - 1 re-cuts of it; exact labels; 0 = off")
ap.add_argument("--aug-rev-rel", type=float, default=None,
                help="weight of a reversed row relative to an original row (default: --aug-rel)")
ap.add_argument("--aug-rel", type=float, default=1.0,
                help="weight of an augmented row relative to an original row (e.g. 0.33 to count the variants of a series like one extra series)")
ap.add_argument("--compare", type=str, default="",
                help="OOF file of a baseline run (e.g. cv_out/oof_loc.npz): prints a paired per-fold difference")
ap.add_argument("--n-series", type=int, default=0, help="debug: use only the first N series")
ap.add_argument("--synth", type=int, default=0, help="synthetic-break pseudo-series per training series (sbrt/synth.py; 0 = off)")
ap.add_argument("--synth-scale", type=float, default=0.5, help="size scale of synthetic breaks (calibrated with synth_calibrate.py)")
ap.add_argument("--synth-pzero", type=float, default=0.25, help="share of synthetic breaks with size 0 (invisible)")
ap.add_argument("--synth-seed", type=int, default=0)
ap.add_argument("--synth-rel", type=float, default=1.0, help="weight of a synthetic row relative to an original row")
ap.add_argument("--synth-types", type=str, default="all",
                help="comma list of break types to inject (variance,mean,dependence,tails,skew,volclust,splice) or all")
ap.add_argument("--bgen", type=int, default=0,
                help="break-transplant pseudo-series per training series (sbrt/breakgen.py, HANDOFF 36; 0 = off)")
ap.add_argument("--bgen-mode", type=str, default="deconv", choices=["deconv", "lam"],
                help="deconv = posterior draws of the donors' true change (lam 1); lam = raw donor change shrunk by --bgen-lam")
ap.add_argument("--bgen-lam", type=float, default=1.0)
ap.add_argument("--bgen-min-post", type=int, default=100, help="donors need this many post-break points")
ap.add_argument("--bgen-seed", type=int, default=0)
ap.add_argument("--bgen-rel", type=float, default=1.0, help="weight of a transplant row relative to an original row")
ap.add_argument("--ext", type=int, default=0,
                help="cuts per 2025 offline-edition series added as extra REAL training series (sbrt/external.py; "
                     "0 = off; run ext2025.py once first)")
ap.add_argument("--ext-dir", type=str, default=os.path.join("data", "ext2025"))
ap.add_argument("--ext-seed", type=int, default=0)
ap.add_argument("--ext-hmin", type=int, default=1000, help="shortest history allowed when cutting a 2025 series")
ap.add_argument("--ext-rel", type=float, default=1.0, help="weight of a 2025 row relative to an original row")
ap.add_argument("--ext-weight", type=str, default="",
                help="npz from diagnostics/ext_weights.py: per-2025-series covariate-shift weights (density ratio Real-Time / 2025 of the history shape), multiplied into every row of that series (HANDOFF 19)")
ap.add_argument("--ext-balance", action="store_true",
                help="re-weight 2025 NEGATIVE rows per step so the 2025 source holds the same share of the positive "
                     "and of the negative weight: being a 2025-like series (short history, heavy tails, 29%% breaks) "
                     "then says nothing about the label")
ap.add_argument("--ev", choices=["none", "compact", "full"], default="none",
                help="append evidence features (sbrt/evidence.py): compact = 6 model-averaged ones, full = 48")
ap.add_argument("--vol", action="store_true",
                help="append the 13 volatility-state features (sbrt/volstate.py): online variance measured against a "
                     "frozen forecast from the end of the history (GARCH start, decaying to the slow end-of-history level)")
ap.add_argument("--dist", choices=["none", "compact", "full"], default="none",
                help="append whole-distribution tests online vs history (sbrt/disttest.py: KS, Anderson-Darling, "
                     "Cramer-von Mises, Wasserstein-1): compact = 8 (AD and W1 on the residuals), full = 24")
ap.add_argument("--bocpd", action="store_true",
                help="append 10 Bayesian online change-point posterior features (sbrt/bocpd.py): exact one-break "
                     "posterior with the generator's hazard and an empirical-Bayes Normal-Inverse-Gamma prior")
ap.add_argument("--bocpd2", action="store_true",
                help="append 12 richer BOCPD features (sbrt/bocpd2.py): mean/variance/AR(1)-change model and a tail-"
                     "frequency stream")
ap.add_argument("--ms", action="store_true",
                help="append 13 Markov-switching features (sbrt/msvol.py, HANDOFF section 45): a 1-3 regime Student-t "
                     "variance model fitted on the history and frozen; its filter is the no-break model of a one-break "
                     "posterior, plus surprise and regime-occupancy scans")
ap.add_argument("--nbm", action="store_true",
                help="append 6 learned normal-behaviour features (sbrt/nbm.py): a GRU pretrained PER FOLD on that fold's "
                     "original training histories (no labels), its one-step prediction errors through the BOCPD "
                     "posterior with a per-fold empirical-Bayes prior")
ap.add_argument("--nbm-dist", action="store_true",
                help="append 8 distribution tests (KS / AD / CvM / W1, last 100, scans) of the NBM's whitened "
                     "conditional-PIT stream vs its history (sbrt/nbdist.py, HANDOFF section 41); needs --nbm")
ap.add_argument("--nbm-score", action="store_true",
                help="append 9 score-test features of the fold NBM's Student-t prediction (location / log-scale / "
                     "nu scores and their interactions with 8 hidden-state PCs, whitened by the series' history "
                     "scores; CUSUMs and LM statistics, last 100, scans; sbrt/nbscore.py, HANDOFF section 43); needs --nbm")
ap.add_argument("--shape", action="store_true",
                help="append 12 break-shape / location-persistence / detector-agreement / symbol-transition features (sbrt/shape.py, HANDOFF section 20)")
ap.add_argument("--xsec", action="store_true",
                help="per-step cross-sectional normalisation of the features (HANDOFF section 22): every column except ctx_* and t_log becomes (x - median) / (IQR / 1.349) of the fold's TRAINING rows in the same online-step bin (41 log-spaced bins); validation rows use the training tables (what production would ship). TS-AUC ranks series within a step; this makes a tree split mean the same thing at every step")
ap.add_argument("--rio", action="store_true",
                help="append 10 single-series Rio-CPD features: geodesic distances between lag-covariance matrices of the online part and the history, and a history-calibrated CUSUM (sbrt/riocpd.py, HANDOFF section 21)")
ap.add_argument("--sens", choices=["none", "ev", "bo", "both"], default="none",
                help="append small-break detector variants (sbrt/sensdet.py, HANDOFF section 34): ev = 6 evidence columns "
                     "on 14 small alternatives (ev2_), bo = 6 BOCPD columns with a narrow post-break prior (bn_)")
ap.add_argument("--pshape", action="store_true",
                help="append the SHAPE of the break-location posterior (sbrt/pshape.py, HANDOFF section 31): entropy, "
                     "KL to the prior, sd of log time-since-break; 6 bo_ columns always, 3 nb_ columns with --nbm")
ap.add_argument("--femb", action="store_true",
                help="append 18 embedding-drift columns from a contrastive MLP on 15 window statistics (sbrt/embfeat.py, "
                     "HANDOFF section 33), trained PER FOLD on that fold's original training histories")
ap.add_argument("--emb", action="store_true",
                help="append 18 embedding-drift columns from a contrastive 1-D CNN on the raw residual windows "
                     "(sbrt/embfeat.py), trained PER FOLD (GPU recommended)")
ap.add_argument("--eqf", action="store_true",
                help="append 2 frozen variance-forecast columns (sbrt/eqfore.py, HANDOFF sections 39 and 42), vol_lr_full's form with a better forecast: eq_f2_lr (per-series multi-scale variance equation) and eq_f3l_lr (ridge across series, trained PER FOLD on pseudo-splits inside the fold's original training histories)")
ap.add_argument("--femb-steps", type=int, default=2000, help="training steps of each --femb MLP")
ap.add_argument("--emb-steps", type=int, default=3000, help="training steps of each --emb CNN")
ap.add_argument("--emb-width", type=int, default=64,
                help="filters per --emb CNN layer (embedding_sweep.py, HANDOFF 33.6: 128 was the only setting that helped)")
ap.add_argument("--nbm-epochs", type=int, default=12)
ap.add_argument("--nbm-hidden", type=int, default=64, help="NBM GRU width (pick with diagnostics/nbm_sweep.py)")
ap.add_argument("--nbm-lr", type=float, default=2e-3)
ap.add_argument("--nbm-batch", type=int, default=48)
ap.add_argument("--nbm-wd", type=float, default=1e-4)
ap.add_argument("--nbm-loss-from", type=int, default=50)
ap.add_argument("--nbm-multi", type=int, default=1,
                help="NBM pretrained with heads for 1..M steps ahead, only the 1-step head kept (HANDOFF 41)")
ap.add_argument("--pc", action="store_true",
                help="append 11 pair-classifier scan features (sbrt/pairclf.py, HANDOFF section 38): a classifier of "
                     "'is this whole online window post-break?' (window vs history statistics), trained PER FOLD on "
                     "that fold's original training series (cross-fitted for the training rows), run on suffix windows "
                     "and accumulated over break times; columns placed after the --nbm columns")
ap.add_argument("--pc-per", type=int, default=PC.K_PER, help="--pc: windows per class per series per window length")
ap.add_argument("--pc-rounds", type=int, default=PC.ROUNDS, help="--pc: boosting rounds of each pair classifier")
ap.add_argument("--pc-leaves", type=int, default=PC.LEAVES, help="--pc: leaves per tree of each pair classifier")
ap.add_argument("--pc-inner", type=int, default=4, help="--pc: inner folds for the cross-fitted training rows")
ap.add_argument("--lratio", action="store_true",
                help="append 5 learned-ratio features (sbrt/lratio.py; build the caches first with "
                     "python -m sbrt.lratio build --aug 2)")
ap.add_argument("--near-k", type=int, default=0,
                help="training only: down-weight rows within K points of the break (linear ramp; 0 = off)")
ap.add_argument("--near-mode", choices=["after", "both"], default="after",
                help="after: positives in the first K points after the break; both: also negatives in the K points "
                     "before it")
ap.add_argument("--near-floor", type=float, default=0.0, help="smallest weight multiplier inside the window")
ap.add_argument("--detw", type=float, default=0.0,
                help="detectability weights (HANDOFF section 31): every TRAINING break series (originals and re-cuts) "
                     "gets a change size s = max over 5 axes of |post-break change| / threshold (log var 0.1, log "
                     "diff-var 0.1, rho1 0.03, kurtosis 0.5, mean shift 2 SE), measured on the whole post-break "
                     "part with the known break position (>= 30 post points, else no change); its POSITIVE rows get "
                     "the multiplier min(1, max(detw, s / 2)), then every step's positive weight is rescaled back to "
                     "its old total (class balance unchanged). Validation rows untouched. 0 = off")
ap.add_argument("--mmd", action="store_true",
                help="append 12 streaming MMD features (sbrt/mmdstream.py): kernel two-sample test of consecutive "
                     "residual pairs/triples, online vs history")
ap.add_argument("--objective", choices=["binary", "rank"], default="binary",
                help="rank = XGBoost pairwise ranking with the online step as the query group (needs --gpu or --backend xgb)")
ap.add_argument("--rank-pairs", type=int, default=8, help="sampled pairs per row for --objective rank")
ap.add_argument("--rank-mcw", type=float, default=-1.0,
                help="min_child_weight for --objective rank (default: same mapping as binary)")
ap.add_argument("--prune-keep", type=int, default=0,
                help="in-fold feature pruning (0 = off): per fold, fit on all columns (pass 1), rank the columns by the "
                     "total gain of pass 1's first --prune-rounds trees (training rows only), refit on the top K. Pass 1 "
                     "is also scored on validation: a determinism check against --compare and the paired difference at "
                     "round --prune-rounds (production's round count)")
ap.add_argument("--prune-rounds", type=int, default=200,
                help="--prune-keep: pass-1 trees whose gain ranks the columns, and the round of the reported paired check")
args = ap.parse_args()
if args.gpu:
    args.backend, args.device = "xgb", "cuda"
if args.dart > 0 and (args.backend != "xgb" or args.objective != "binary" or args.base_margin):
    sys.exit("--dart needs --backend xgb (the default with --gpu), --objective binary and no --base-margin")
if args.objective == "rank" and args.backend != "xgb":
    sys.exit("--objective rank needs XGBoost: add --gpu (or --backend xgb)")
if (args.monotone or args.base_margin) and (args.backend != "xgb" or args.objective != "binary" or args.prune_keep):
    sys.exit("--monotone / --base-margin need XGBoost (--gpu) with --objective binary and no --prune-keep")
if args.prune_keep and (args.backend in ("svm", "cat") or args.objective != "binary"):
    sys.exit("--prune-keep needs the tree backends (lgb/xgb) with --objective binary")
TTA_SHIFTS = [int(v) for v in args.tta.split(",") if v.strip()]
if TTA_SHIFTS:
    if args.backend not in ("lgb", "xgb") or args.objective != "binary" or args.prune_keep or args.base_margin or args.xsec:
        sys.exit("--tta needs the lgb / xgb trees with --objective binary and no --prune-keep / --base-margin / --xsec")
    if args.pc or args.emb or args.femb or args.eqf or args.lratio:
        sys.exit("--tta: --pc / --emb / --femb / --eqf / --lratio fit per-fold models on the extra sources; not supported")
    if any(m <= 0 or m % 3 for m in TTA_SHIFTS) or len(set(TTA_SHIFTS)) != len(TTA_SHIFTS):
        sys.exit("--tta shifts must be distinct positive multiples of 3 (rows are kept every 3rd step)")
TAG = ("_" + args.tag) if args.tag else ""
THREADS = args.threads if args.threads > 0 else (os.cpu_count() or 1)
LGB_PARAMS = dict(objective="binary", learning_rate=args.lr, num_leaves=args.leaves,
                  min_data_in_leaf=args.min_leaf, feature_fraction=args.ff, bagging_fraction=0.8,
                  bagging_freq=1, lambda_l2=10.0, seed=args.seed, deterministic=True,
                  force_row_wise=True, num_threads=THREADS, verbose=-1)

SUB = 3
CACHE = os.path.join("data", f"feat_{F.N_FEAT}.npz")  # one cache per feature-set size
OUT = "cv_out"
os.makedirs(OUT, exist_ok=True)


class Tee:
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")

    def write(self, s):
        sys.__stdout__.write(s)
        self.f.write(s)

    def flush(self):
        sys.__stdout__.flush()
        self.f.flush()


sys.stdout = Tee(os.path.join(OUT, f"summary{TAG}.txt"))


def _feat(h, o):
    return F.run_series(h, o)[::SUB].astype(np.float32)


def build():
    S, tau = load_train()
    if os.path.exists(CACHE) and not args.rebuild:
        z = np.load(CACHE)
        if z["X"].shape[1] == F.N_FEAT:
            return z["X"], z["y"], z["sid"], z["step"], tau
        print("cache has a different feature count -> rebuilding")
    t0 = time.time()
    mats = Parallel(n_jobs=THREADS, backend="threading")(delayed(_feat)(s[1], s[2]) for s in S)
    print(f"features built in {time.time() - t0:.0f}s")
    X = np.concatenate(mats)
    counts = np.array([len(m) for m in mats])
    sid = np.repeat(np.arange(len(S)), counts)
    step = np.concatenate([np.arange(0, len(s[2]), SUB) for s in S])
    ti = tau[sid]
    y = ((ti >= 0) & (step >= ti)).astype(np.int8)
    np.savez(CACHE, X=X, y=y, sid=sid, step=step)
    return X, y, sid, step, tau


def _ev_feat(h, o):
    return EV.run_series_ev(h, o)[::SUB].astype(np.float32)


def ev_matrix(items, cache):
    """Evidence features (all 48 columns) of a list of (hist, online), cached when `cache` is a path."""
    if cache and os.path.exists(cache) and not args.rebuild:
        z = np.load(cache)
        if z["E"].shape[1] == EV.N_EV and int(z["n"]) == len(items):
            return z["E"]
    t0 = time.time()
    mats = Parallel(n_jobs=THREADS, backend="threading")(delayed(_ev_feat)(h, o) for h, o in items)
    E = np.concatenate(mats)
    print(f"evidence features: {len(mats):,} series in {time.time() - t0:.0f}s")
    if cache:
        np.savez(cache, E=E, n=len(items))
    return E


def _vol_feat(h, o):
    return VS.run_series_vol(h, o)[::SUB].astype(np.float32)


def vol_matrix(items, cache):
    """Volatility-state features of a list of (hist, online), cached when `cache` is a path."""
    if cache and os.path.exists(cache) and not args.rebuild:
        z = np.load(cache)
        if z["V"].shape[1] == VS.N_VOL and int(z["n"]) == len(items):
            return z["V"]
    t0 = time.time()
    mats = Parallel(n_jobs=THREADS, backend="threading")(delayed(_vol_feat)(h, o) for h, o in items)
    V = np.concatenate(mats)
    print(f"volatility-state features: {len(mats):,} series in {time.time() - t0:.0f}s")
    if cache:
        np.savez(cache, V=V, n=len(items))
    return V


def _dist_feat(h, o):
    return DT.run_series_dist(h, o, SUB).astype(np.float32)   # scan evaluated only on the kept rows


def dist_matrix(items, cache):
    """Distribution-test features (all 24 columns) of a list of (hist, online), cached when `cache` is a path."""
    if cache and os.path.exists(cache) and not args.rebuild:
        z = np.load(cache)
        if z["D"].shape[1] == DT.N_DIST and int(z["n"]) == len(items):
            return z["D"]
    t0 = time.time()
    mats = Parallel(n_jobs=THREADS, backend="threading")(delayed(_dist_feat)(h, o) for h, o in items)
    D = np.concatenate(mats)
    print(f"distribution-test features: {len(mats):,} series in {time.time() - t0:.0f}s")
    if cache:
        np.savez(cache, D=D, n=len(items))
    return D


def _bo_feat(h, o):
    return BO.run_series_bocpd(h, o, SUB).astype(np.float32)   # posterior evaluated only on the kept rows


def bo_matrix(items, cache):
    """BOCPD features (10 columns) of a list of (hist, online), cached when `cache` is a path."""
    if cache and os.path.exists(cache) and not args.rebuild:
        z = np.load(cache)
        if z["B"].shape[1] == BO.N_BO and int(z["n"]) == len(items):
            return z["B"]
    t0 = time.time()
    mats = Parallel(n_jobs=THREADS, backend="threading")(delayed(_bo_feat)(h, o) for h, o in items)
    B = np.concatenate(mats)
    print(f"BOCPD features: {len(mats):,} series in {time.time() - t0:.0f}s")
    if cache:
        np.savez(cache, B=B, n=len(items))
    return B


def _sh_feat(h, o):
    return SH.run_series_shape(h, o, SUB).astype(np.float32)


def sh_matrix(items, cache):
    """--shape: sbrt/shape.py columns (12), cached when `cache` is a path."""
    if cache and os.path.exists(cache) and not args.rebuild:
        z = np.load(cache)
        if z["B"].shape[1] == SH.N_SH and int(z["n"]) == len(items):
            return z["B"]
    t0 = time.time()
    mats = Parallel(n_jobs=THREADS, backend="threading")(delayed(_sh_feat)(h, o) for h, o in items)
    B = np.concatenate(mats)
    print(f"shape features: {len(mats):,} series in {time.time() - t0:.0f}s")
    if cache:
        np.savez(cache, B=B, n=len(items))
    return B


def _rio_feat(h, o):
    return RIO.run_series_rio(h, o, SUB).astype(np.float32)


def rio_matrix(items, cache):
    """--rio: sbrt/riocpd.py columns (10), cached when `cache` is a path."""
    if cache and os.path.exists(cache) and not args.rebuild:
        z = np.load(cache)
        if z["B"].shape[1] == RIO.N_RIO and int(z["n"]) == len(items):
            return z["B"]
    t0 = time.time()
    mats = Parallel(n_jobs=THREADS, backend="threading")(delayed(_rio_feat)(h, o) for h, o in items)
    B = np.concatenate(mats)
    print(f"Rio-CPD features: {len(mats):,} series in {time.time() - t0:.0f}s")
    if cache:
        np.savez(cache, B=B, n=len(items))
    return B


def _sd_feat(h, o):
    return SD.run_series_sens(h, o, SUB, args.sens).astype(np.float32)


def sd_matrix(items, cache):
    """--sens: sbrt/sensdet.py columns, cached when `cache` is a path."""
    nc = len(SD.names(args.sens))
    if cache and os.path.exists(cache) and not args.rebuild:
        z = np.load(cache)
        if z["B"].shape[1] == nc and int(z["n"]) == len(items):
            return z["B"]
    t0 = time.time()
    mats = Parallel(n_jobs=THREADS, backend="threading")(delayed(_sd_feat)(h, o) for h, o in items)
    B = np.concatenate(mats)
    print(f"small-break detector features: {len(mats):,} series in {time.time() - t0:.0f}s")
    if cache:
        np.savez(cache, B=B, n=len(items))
    return B


def _ps_feat(h, o):
    return PS.bo_shape(h, o, SUB).astype(np.float32)


def ps_matrix(items, cache):
    """--pshape: posterior-shape columns of the bocpd.py streams (6), cached when `cache` is a path."""
    if cache and os.path.exists(cache) and not args.rebuild:
        z = np.load(cache)
        if z["B"].shape[1] == len(PS.BO_NAMES) and int(z["n"]) == len(items):
            return z["B"]
    t0 = time.time()
    mats = Parallel(n_jobs=THREADS, backend="threading")(delayed(_ps_feat)(h, o) for h, o in items)
    B = np.concatenate(mats)
    print(f"posterior-shape features: {len(mats):,} series in {time.time() - t0:.0f}s")
    if cache:
        np.savez(cache, B=B, n=len(items))
    return B


def _bo2_feat(h, o):
    return B2.run_series_bocpd2(h, o, SUB).astype(np.float32)


def bo2_matrix(items, cache):
    """Richer BOCPD features (12 columns) of a list of (hist, online), cached when `cache` is a path."""
    if cache and os.path.exists(cache) and not args.rebuild:
        z = np.load(cache)
        if z["B"].shape[1] == B2.N_BO2 and int(z["n"]) == len(items):
            return z["B"]
    t0 = time.time()
    mats = Parallel(n_jobs=THREADS, backend="threading")(delayed(_bo2_feat)(h, o) for h, o in items)
    B = np.concatenate(mats)
    print(f"BOCPD2 features: {len(mats):,} series in {time.time() - t0:.0f}s")
    if cache:
        np.savez(cache, B=B, n=len(items))
    return B


def _ms_feat(h, o):
    return MSV.run_series_ms(h, o, SUB).astype(np.float32)


def ms_matrix(items, cache):
    """Markov-switching features (13 columns) of a list of (hist, online), cached when `cache` is a path."""
    if cache and os.path.exists(cache) and not args.rebuild:
        z = np.load(cache)
        if z["B"].shape[1] == MSV.N_MS and int(z["n"]) == len(items):
            return z["B"]
    t0 = time.time()
    mats = Parallel(n_jobs=THREADS, backend="threading")(delayed(_ms_feat)(h, o) for h, o in items)
    B = np.concatenate(mats)
    print(f"Markov-switching features: {len(mats):,} series in {time.time() - t0:.0f}s")
    if cache:
        np.savez(cache, B=B, n=len(items))
    return B


def lr_matrix(stub, n_items, n_rows=None):
    """Learned-ratio features from the cache written by  python -m sbrt.lratio build  (cross-fitted there)."""
    path = os.path.join("data", f"lr{LR.VERSION}_{stub}.npz")
    if not os.path.exists(path):
        sys.exit(f"{path} missing: run  python -m sbrt.lratio build --aug {args.aug} --threads {THREADS}  first")
    z = np.load(path)
    if int(z["n"]) != n_items:
        sys.exit(f"{path} was built for {int(z['n'])} series, need {n_items}: rebuild it")
    return z["L"] if n_rows is None else z["L"][:n_rows]


def near_mult(y, step, ti):
    """--near-k weight multipliers (training rows only): linear ramp inside K points of the break."""
    K, fl = args.near_k, args.near_floor
    m = np.ones(len(y))
    d = step - ti
    brk = ti >= 0
    a = brk & (y == 1) & (d < K)
    m[a] = fl + (1 - fl) * (d[a] + 1) / (K + 1)
    if args.near_mode == "both":
        b = brk & (y == 0) & (-d <= K)
        m[b] = fl + (1 - fl) * (-d[b]) / (K + 1)
    return m


def det_size(h, o, t):
    """--detw: change size of one break series (np.inf = not measured: no break or < 30 post-break points)."""
    if t < 0 or len(o) - t < 30:
        return np.inf
    z = np.asarray(o[t:], dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    ac = lambda x: np.corrcoef(x[:-1], x[1:])[0, 1]
    dlv = np.log(z.var() / h.var())
    dabs = np.log(np.diff(z).var() / np.diff(h).var())
    dr1 = ac(z) - ac(h)
    dku = kurtosis(z) - kurtosis(h)
    dmean = abs(z.mean() - h.mean()) / h.std() * np.sqrt(len(z))
    v = max(abs(dlv) / 0.1, abs(dabs) / 0.1, abs(dr1) / 0.03, abs(dku) / 0.5, dmean / 2.0)
    return v if np.isfinite(v) else np.inf


def det_mult_of(sizes):
    return np.minimum(1.0, np.maximum(args.detw, sizes / 2.0))


def det_apply(w, y, step, dm, f):
    """--detw: multiply positive rows by dm, then rescale each step's positives back to their old total weight."""
    pos = y == 1
    T = int(step.max()) + 1
    before = np.bincount(step[pos], weights=w[pos], minlength=T)
    w = w.copy()
    w[pos] *= dm[pos]
    after = np.bincount(step[pos], weights=w[pos], minlength=T)
    k = np.where(after > 0, before / np.maximum(after, 1e-300), 1.0)
    w[pos] *= k[step[pos]]
    low = pos & (dm < 1)
    print(f"fold {f}: --detw {args.detw:g}: {100 * low.sum() / max(pos.sum(), 1):.1f}% of positive training rows "
          f"down-weighted (mean multiplier {dm[low].mean() if low.any() else 1:.2f}); positive weight per step kept", flush=True)
    return w / max(w.mean(), 1e-12)


def _mmd_feat(h, o):
    return MM.run_series_mmd(h, o, SUB).astype(np.float32)   # scan evaluated only on the kept rows


def mmd_matrix(items, cache):
    """MMD features (12 columns) of a list of (hist, online), cached when `cache` is a path."""
    if cache and os.path.exists(cache) and not args.rebuild:
        z = np.load(cache)
        if z["M"].shape[1] == MM.N_MMD and int(z["n"]) == len(items):
            return z["M"]
    t0 = time.time()
    mats = Parallel(n_jobs=THREADS, backend="threading")(delayed(_mmd_feat)(h, o) for h, o in items)
    M = np.concatenate(mats)
    print(f"MMD features: {len(mats):,} series in {time.time() - t0:.0f}s")
    if cache:
        np.savez(cache, M=M, n=len(items))
    return M


def ev_cols():
    return list(range(EV.N_COMPACT)) if args.ev == "compact" else list(range(EV.N_EV))


def synth_types():
    if args.synth_types == "all":
        return SY.TYPES
    t = tuple(x.strip() for x in args.synth_types.split(",") if x.strip())
    bad = [x for x in t if x not in SY.TYPES]
    if bad or not t:
        sys.exit(f"--synth-types: unknown {bad}; choose from {SY.TYPES}")
    return t


def synth_stub():
    return "" if args.synth_types == "all" else "_t" + "-".join(x[:3] for x in synth_types())


def make_bgen(S, tau):
    """--bgen variants (sbrt/breakgen.py), generated in parallel chunks; independent of the chunking (ids)."""
    t0 = time.time()
    tau = np.asarray(tau)
    folds = SY.fold_ids(tau)
    ks = np.unique(folds)
    pool = lambda k, mode: BGN.donor_pool(S, tau, np.flatnonzero(folds == k), mode, args.bgen_min_post,
                                          seed=args.bgen_seed)
    P = dict(zip(ks, Parallel(n_jobs=THREADS, backend="threading")(delayed(pool)(k, "break") for k in ks)))
    D, lam, lam_dep = None, args.bgen_lam, 0.5 * args.bgen_lam
    if args.bgen_mode == "deconv":
        PP = dict(zip(ks, Parallel(n_jobs=THREADS, backend="threading")(delayed(pool)(k, "placebo") for k in ks)))
        D = BGN.fit_deconv(P, PP)
        lam, lam_dep = 1.0, 1.0
    hp = [len(q[1]) for q in S]
    chunks = [c for c in np.array_split(np.arange(len(S)), 4 * THREADS) if len(c)]

    def run(ix):
        V = BGN.make_breakgen([S[i] for i in ix], tau[ix], args.bgen, seed=args.bgen_seed, lam=lam, folds=folds[ix],
                              min_post=args.bgen_min_post, pools=P, hist_pool=hp, lam_dep=lam_dep, deconv=D, ids=ix)
        return [(int(ix[v[0]]),) + tuple(v[1:]) for v in V]

    V = [v for o in Parallel(n_jobs=THREADS, backend="threading")(delayed(run)(ix) for ix in chunks) for v in o]
    print(f"transplant variants: {len(V):,} ({sum(v[5] >= 0 for v in V):,} with a break; donors per fold "
          f"{[len(P[k]) for k in ks]}; mode {args.bgen_mode}) in {time.time() - t0:.0f}s")
    return V


def extra_sources(S, tau):
    """Sources of extra TRAINING series: [(name, variants as (parent, hist, online, tau), cache stub, rel weight)]."""
    out = []
    aug_stub = f"aug{args.aug}_s{args.aug_seed}_m{args.aug_min_shift}" + (f"_tr{args.aug_trim:g}_x{args.aug_trim_only}" if (args.aug_trim > 0 or args.aug_trim_only > 0) else "")
    V_aug = []
    if args.aug > 0:
        V = A.make_variants(S, tau, args.aug, seed=args.aug_seed, min_shift=args.aug_min_shift,
                            trim_p=args.aug_trim, n_trim_only=args.aug_trim_only)
        V_aug = V
        hv = np.array([len(v[2]) for v in V])
        print(f"re-cut variants: {len(V):,}; history length quantiles 10/50/90% originals "
              f"{np.percentile([len(q[1]) for q in S], [10, 50, 90]).astype(int)} variants "
              f"{np.percentile(hv, [10, 50, 90]).astype(int)}")
        out.append(("re-cut", [(v[0], v[2], v[3], v[4]) for v in V], aug_stub, args.aug_rel))
    if args.aug_early > 0:   # HANDOFF 43: break in the first A.EARLY online points, online cut to A.EARLY points
        used = {}
        for v in V_aug:
            used.setdefault(v[0], set()).add(v[1])
        VE = A.make_early_variants(S, tau, args.aug_early, used, seed=args.aug_seed, min_shift=args.aug_min_shift)
        print(f"early re-cut variants: {len(VE):,} ({sum(v[4] >= 0 for v in VE):,} with the break in the first "
              f"{A.EARLY} online points; online segments cut to {A.EARLY} points)")
        out.append(("early re-cut", [(v[0], v[2], v[3], v[4]) for v in VE],
                    f"early{args.aug_early}_e{A.EARLY}_s{args.aug_seed}_m{args.aug_min_shift}_x{aug_stub}", args.aug_rel))
    if args.aug_rev > 0:   # time-reversed series, each regime reversed on its own (exact labels)
        VR = A.make_reversed(S, tau, args.aug_rev, seed=args.aug_seed, min_shift=args.aug_min_shift)
        print(f"reversed variants: {len(VR):,} ({sum(v[1] == 0 for v in VR):,} at the original boundary, "
              f"{sum(v[4] >= 0 for v in VR):,} with a break)")
        out.append(("reversed", [(v[0], v[2], v[3], v[4]) for v in VR],
                    f"rev{args.aug_rev}v{A.REV_VERSION}_s{args.aug_seed}_m{args.aug_min_shift}",
                    args.aug_rel if args.aug_rev_rel is None else args.aug_rev_rel))
    if args.synth > 0:
        V = SY.make_synthetic(S, tau, args.synth, seed=args.synth_seed, scale=args.synth_scale,
                              p_zero=args.synth_pzero, folds=SY.fold_ids(tau), types=synth_types())
        out.append(("synthetic", [(v[0], v[3], v[4], v[5]) for v in V],
                    f"synth{args.synth}_v1_s{args.synth_seed}_sc{args.synth_scale:g}_pz{args.synth_pzero:g}{synth_stub()}",
                    args.synth_rel))
    if args.bgen > 0:   # HANDOFF 36: donors, placebo noise and the deconvolution all come from the PARENT's own fold
        if args.folds != 5:
            sys.exit("--bgen needs --folds 5 (donor pools use the 5-fold split of synth.fold_ids)")
        V = make_bgen(S, tau)
        stub = (f"bgen{args.bgen}_v{BGN.VERSION}_{args.bgen_mode}" + ("" if args.bgen_mode == "deconv" else f"{args.bgen_lam:g}")
                + f"_mp{args.bgen_min_post}_s{args.bgen_seed}")
        out.append(("transplant", [(v[0], v[3], v[4], v[5]) for v in V], stub, args.bgen_rel))
    if args.ext > 0:  # parent = the Real-Time twin found by ext2025.py, or -1 (trains in every fold)
        V = XT.make_external(args.ext_dir, args.ext, seed=args.ext_seed, hmin=args.ext_hmin, limit=args.n_series)
        out.append(("2025", V, f"ext{args.ext}_s{args.ext_seed}_h{args.ext_hmin}", args.ext_rel))
    return out


def tta_sources(S, tau):
    """--tta: score-only view sources [(name, variants as (parent, hist, online, tau), cache stub, rel, shift)]."""
    out = []
    for m in TTA_SHIFTS:
        V = A.make_views(S, tau, m)
        print(f"test-time view +{m}: {len(V):,} of {len(S):,} series (history >= {A.H_MIN} after the shift)")
        out.append((f"view+{m}", V, f"view{m}v1", 0.0, m))
    return out


def build_extra(name, V, stub, rel):
    """Features of one source of extra training series (cached). Returns a dict of per-row arrays."""
    use_cache = args.n_series == 0
    vlen = np.array([len(v[2]) for v in V])
    vtau = np.array([v[3] for v in V])
    cache = os.path.join("data", f"feat_{F.N_FEAT}_{stub}.npz")
    XA = None
    if use_cache and os.path.exists(cache) and not args.rebuild:
        z = np.load(cache)
        if z["X"].shape[1] == F.N_FEAT and np.array_equal(z["vlen"], vlen):
            XA = z["X"]
            print(f"{name}: loaded {len(V):,} series from {cache}")
        else:
            print(f"{name}: cache does not match the current settings -> rebuilding")
    if XA is None:
        t0 = time.time()
        mats = Parallel(n_jobs=THREADS, backend="threading")(delayed(_feat)(v[1], v[2]) for v in V)
        XA = np.concatenate(mats)
        del mats
        print(f"{name}: {len(V):,} series, {len(XA):,} rows, features built in {time.time() - t0:.0f}s")
        if use_cache:
            np.savez(cache, X=XA, vlen=vlen)
    counts = (vlen + SUB - 1) // SUB
    parent = np.repeat(np.array([v[0] for v in V]), counts)
    stepA = np.concatenate([np.arange(0, L, SUB) for L in vlen])
    ti = np.repeat(vtau, counts)
    yA = ((ti >= 0) & (stepA >= ti)).astype(np.int8)
    assert len(stepA) == len(XA)
    E = None
    if args.ev != "none":
        E = ev_matrix([(v[1], v[2]) for v in V], os.path.join("data", f"ev{EV.VERSION}_{stub}.npz") if use_cache else None)
    VV = None
    if args.vol:
        VV = vol_matrix([(v[1], v[2]) for v in V], os.path.join("data", f"vol{VS.VERSION}_{stub}.npz") if use_cache else None)
    DD = None
    if args.dist != "none":
        DD = dist_matrix([(v[1], v[2]) for v in V], os.path.join("data", f"dist{DT.VERSION}_{stub}.npz") if use_cache else None)
    BOx = None
    if args.bocpd:
        BOx = bo_matrix([(v[1], v[2]) for v in V], os.path.join("data", f"bo{BO.VERSION}_{stub}.npz") if use_cache else None)
    SHx = None
    if args.shape:
        SHx = sh_matrix([(v[1], v[2]) for v in V], os.path.join("data", f"sh{SH.VERSION}_{stub}.npz") if use_cache else None)
    RIx = None
    if args.rio:
        RIx = rio_matrix([(v[1], v[2]) for v in V], os.path.join("data", f"rio{RIO.VERSION}_{stub}.npz") if use_cache else None)
    SDx = None
    if args.sens != "none":
        SDx = sd_matrix([(v[1], v[2]) for v in V], os.path.join("data", f"sens{SD.VERSION}{args.sens}_{stub}.npz") if use_cache else None)
    PSx = None
    if args.pshape:
        PSx = ps_matrix([(v[1], v[2]) for v in V], os.path.join("data", f"ps{PS.VERSION}_{stub}.npz") if use_cache else None)
    MMx = None
    if args.mmd:
        MMx = mmd_matrix([(v[1], v[2]) for v in V], os.path.join("data", f"mmd{MM.VERSION}_{stub}.npz") if use_cache else None)
    BO2x = None
    if args.bocpd2:
        BO2x = bo2_matrix([(v[1], v[2]) for v in V], os.path.join("data", f"bo2v{B2.VERSION}_{stub}.npz") if use_cache else None)
    MSx = None
    if args.ms:
        MSx = ms_matrix([(v[1], v[2]) for v in V], os.path.join("data", f"ms{MSV.VERSION}_{stub}.npz") if use_cache else None)
    LRx = None
    if args.lratio:
        if not use_cache or name != "re-cut":
            sys.exit("--lratio supports the full data set and re-cut variants only")
        LRx = lr_matrix(stub, len(V))
    return dict(name=name, X=XA, E=E, VV=VV, DD=DD, MM=MMx, BO=BOx, BO2=BO2x, LR=LRx, PS=PSx, MS=MSx, SH=SHx, RI=RIx, SD=SDx, y=yA, parent=parent,
                step=stepA, ti=ti, rel=rel, vlen=vlen, vtau=vtau, items=[(v[1], v[2]) for v in V], stub=stub,
                vparent=np.array([v[0] for v in V], dtype=np.int64))


NBM_STATE = {}   # fold -> NBM state dict trained in this run (nbm_fold)


def nbm_fold(f, tr_s, S, tau, parts):
    """--nbm: fold-f features for every original series and every extra source (cached per fold).
    The GRU is pretrained on the histories of the fold's ORIGINAL training series only (10% of them held out for
    early stopping); the empirical-Bayes prior is fitted on the fold's original training breaks."""
    use_cache = args.n_series == 0
    cfg = f"e{args.nbm_epochs}"   # default settings keep the original cache names
    if (args.nbm_hidden, args.nbm_lr, args.nbm_batch, args.nbm_wd, args.nbm_loss_from) != (64, 2e-3, 48, 1e-4, 50):
        cfg += f"_h{args.nbm_hidden}_lr{args.nbm_lr:g}_b{args.nbm_batch}_wd{args.nbm_wd:g}_lf{args.nbm_loss_from}"
    if args.nbm_multi != 1:   # new name only when used, so the existing caches keep theirs
        cfg += f"_m{args.nbm_multi}"
    wpath = os.path.join("data", f"nbm_v{NBM.VERSION}_cv_f{f}_{cfg}.pt")
    stubs = ["orig"] + [P["stub"] for P in parts]
    paths = [os.path.join("data", f"nbm{NBM.VERSION}_f{f}_{cfg}_{st}.npz") for st in stubs]
    ppaths = [os.path.join("data", f"nbps{PS.VERSION}_f{f}_{cfg}_{st}.npz") for st in stubs]
    dpaths = [os.path.join("data", f"nbd{NBD.VERSION}_f{f}_{cfg}_{st}.npz") for st in stubs]
    counts = [len(S)] + [len(P["items"]) for P in parts]
    need = paths + (ppaths if args.pshape else []) + (dpaths if args.nbm_dist else [])
    if use_cache and not args.rebuild and all(os.path.exists(q) for q in need):
        mats = [np.load(q) for q in paths]
        pm = [np.load(q) for q in ppaths] if args.pshape else []
        dm = [np.load(q) for q in dpaths] if args.nbm_dist else []
        if all(int(z["n"]) == c for zz in (mats, pm, dm) for z, c in zip(zz, counts)):
            print(f"fold {f}: NBM features loaded from cache", flush=True)
            return ([z["B"] for z in mats], ([z["B"] for z in pm] if args.pshape else None),
                    ([z["B"] for z in dm] if args.nbm_dist else None))
    import torch
    t0 = time.time()
    if use_cache and os.path.exists(wpath) and not args.rebuild:
        state = torch.load(wpath, map_location="cpu")
    else:
        rs = np.random.default_rng(1000 + f)
        trp = rs.permutation(tr_s)
        nva = max(1, len(trp) // 10)
        state = NBM.train_model([S[i][1] for i in trp[nva:]], [S[i][1] for i in trp[:nva]],
                                epochs=args.nbm_epochs, seed=f, verbose=False, hidden=args.nbm_hidden,
                                lr=args.nbm_lr, batch=args.nbm_batch, wd=args.nbm_wd, loss_from=args.nbm_loss_from,
                                multi=args.nbm_multi)
        if use_cache:
            torch.save(state, wpath)
    NBM_STATE[f] = state              # reused by --nbm-score (smoke runs have no weight file on disk)
    M = NBM.model_arrays(state)
    t1 = time.time()
    so = NBM.sums_parallel([(q[1], q[2]) for q in S], M, THREADS)
    prior, nfit = NBM.fit_prior_w([NBM.w_of(so[i]) for i in tr_s], tau[tr_s])
    out = [np.concatenate([NBM.features_from_sums(q, prior, SUB) for q in so]).astype(np.float32)]
    pout = [np.concatenate([PS.nb_shape(q, prior, SUB) for q in so]).astype(np.float32)] if args.pshape else None
    dout = [nbd_block([len(q[1]) for q in S], so)] if args.nbm_dist else None
    del so
    for P in parts:
        sp = NBM.sums_parallel(P["items"], M, THREADS)
        out.append(np.concatenate([NBM.features_from_sums(q, prior, SUB) for q in sp]).astype(np.float32))
        if args.pshape:
            pout.append(np.concatenate([PS.nb_shape(q, prior, SUB) for q in sp]).astype(np.float32))
        if args.nbm_dist:
            dout.append(nbd_block([len(it[0]) for it in P["items"]], sp))
        del sp
    print(f"fold {f}: NBM pretrained in {t1 - t0:.0f}s; prior from {nfit} training breaks (kappa0 {prior[0]:.0f}, "
          f"alpha0 {prior[1]:.2f}, beta0 {prior[2]:.2f}); features for {sum(counts):,} series in {time.time() - t1:.0f}s",
          flush=True)
    if use_cache:
        for B, q, c in zip(out, paths, counts):
            np.savez(q, B=B, n=c)
        if args.pshape:
            for B, q, c in zip(pout, ppaths, counts):
                np.savez(q, B=B, n=c)
        if args.nbm_dist:
            for B, q, c in zip(dout, dpaths, counts):
                np.savez(q, B=B, n=c)
    return out, pout, dout


def nbd_block(hist_lens, sums):
    """--nbm-dist: the 8 nbd_ columns of every series (rows at online steps 0, SUB, ...), from the NBM sums."""
    from joblib import Parallel, delayed
    R = Parallel(n_jobs=THREADS, backend="threading")(
        delayed(NBD.from_sums)(nh, q, SUB) for nh, q in zip(hist_lens, sums))
    return np.concatenate(R).astype(np.float32)


def hpca_fold(f, tr_s, S, parts, step_o):
    """--nbm-hpca: fold-f NBM hidden-state PCA columns (one matrix per source: originals, then each extra source), at
    the rows' online steps 0, SUB, 2 SUB, ... Same construction as gru_cv.py nbm_hidden_fold (PCA on the online parts
    of every 10th original training series of the fold, first K components, standardised, clipped +-6); its per-fold
    cache (full step resolution) is read when it exists, otherwise built here and saved under the same name."""
    import torch
    K = args.nbm_hpca
    use_cache = args.n_series == 0
    cfg = f"e{args.nbm_epochs}"   # = nbm_fold / gru_cv.py nbm_cfg()
    if (args.nbm_hidden, args.nbm_lr, args.nbm_batch, args.nbm_wd, args.nbm_loss_from) != (64, 2e-3, 48, 1e-4, 50):
        cfg += f"_h{args.nbm_hidden}_lr{args.nbm_lr:g}_b{args.nbm_batch}_wd{args.nbm_wd:g}_lf{args.nbm_loss_from}"
    if args.nbm_multi != 1:
        cfg += f"_m{args.nbm_multi}"
    stub = "".join("_" + P["stub"] for P in parts)
    cache = os.path.join("data", f"gru_nbmh{K}v1_f{f}_{cfg}{stub}.npz")
    srcs = [[(q[1], q[2]) for q in S]] + [P["items"] for P in parts]
    count = sum(len(it) for it in srcs)
    out = None
    if use_cache and not args.rebuild and os.path.exists(cache):
        z = np.load(cache)
        if int(z["n"]) == count:
            flat, off = z["flat"], z["off"]
            out = [flat[off[i]:off[i + 1]] for i in range(len(off) - 1)]
            print(f"fold {f}: NBM hidden-state PCA loaded from {cache}", flush=True)
    if out is None:
        t0 = time.time()
        wpath = os.path.join("data", f"nbm_v{NBM.VERSION}_cv_f{f}_{cfg}.pt")
        if not os.path.exists(wpath):
            sys.exit(f"--nbm-hpca: no NBM weights for fold {f} ({wpath}); run with --nbm and the cache on first")
        M = NBM.model_arrays(torch.load(wpath, map_location="cpu"))
        tr_o = np.sort(tr_s)
        samp = Parallel(n_jobs=THREADS, backend="threading")(
            delayed(NBM.run_hidden)(S[i][1], S[i][2], M) for i in tr_o[::10])
        Hs = np.concatenate(samp)
        del samp
        mean = Hs.mean(0)
        _, sv, Vt = np.linalg.svd(Hs - mean, full_matrices=False)
        comp = Vt[:K]
        sd = ((Hs - mean) @ comp.T).std(0) + 1e-6
        ev = (sv ** 2) / (sv ** 2).sum()
        del Hs
        print(f"fold {f}: NBM hidden-state PCA from {len(tr_o[::10])} training series: first {K} components explain "
              f"{100 * ev[:K].sum():.1f}% of the variance", flush=True)
        items = [it for its in srcs for it in its]
        out = Parallel(n_jobs=THREADS, backend="threading")(
            delayed(NBM.run_hidden)(h, o, M, mean, comp) for h, o in items)
        out = [np.clip(o / sd, -6, 6).astype(np.float32) for o in out]
        print(f"fold {f}: NBM hidden-state PCA for {count:,} series in {time.time() - t0:.0f}s", flush=True)
        if use_cache:
            np.savez(cache, flat=np.concatenate(out), off=np.r_[0, np.cumsum([len(x) for x in out])], n=count)
    res, o = [], 0
    for k, its in enumerate(srcs):
        B = np.concatenate([out[o + i][::SUB] for i in range(len(its))])
        o += len(its)
        want = step_o if k == 0 else parts[k - 1]["step"]
        exp = np.concatenate([np.arange(0, len(x[1]), SUB) for x in its])
        if len(B) != len(want) or not np.array_equal(exp, want):
            sys.exit(f"--nbm-hpca: row layout mismatch in source {k} ({len(B)} vs {len(want)} rows)")
        res.append(B)
    return res


def nsc_fold(f, tr_s, S, parts, step_o):
    """--nbm-score: fold-f score-test columns (one matrix per source: originals, then each extra source), at the rows'
    online steps 0, SUB, ... The NBM is the fold's (as nbm_fold); the 8-PC basis is fitted like hpca_fold (online parts
    of every 10th original training series). Cached per fold."""
    import torch
    use_cache = args.n_series == 0
    cfg = f"e{args.nbm_epochs}"   # = nbm_fold
    if (args.nbm_hidden, args.nbm_lr, args.nbm_batch, args.nbm_wd, args.nbm_loss_from) != (64, 2e-3, 48, 1e-4, 50):
        cfg += f"_h{args.nbm_hidden}_lr{args.nbm_lr:g}_b{args.nbm_batch}_wd{args.nbm_wd:g}_lf{args.nbm_loss_from}"
    if args.nbm_multi != 1:
        cfg += f"_m{args.nbm_multi}"
    stub = "".join("_" + P["stub"] for P in parts)
    cache = os.path.join("data", f"nbsc{NSC.VERSION}_f{f}_{cfg}{stub}.npz")
    srcs = [[(q[1], q[2]) for q in S]] + [P["items"] for P in parts]
    count = sum(len(it) for it in srcs)
    if use_cache and not args.rebuild and os.path.exists(cache):
        z = np.load(cache)
        if int(z["n"]) == count:
            print(f"fold {f}: NBM score-test features loaded from {cache}", flush=True)
            flat, off = z["flat"], z["off"]
            res = [flat[off[k]:off[k + 1]] for k in range(len(srcs))]
            return res
    t0 = time.time()
    wpath = os.path.join("data", f"nbm_v{NBM.VERSION}_cv_f{f}_{cfg}.pt")
    if f in NBM_STATE:
        state = NBM_STATE[f]
    elif os.path.exists(wpath):
        state = torch.load(wpath, map_location="cpu")
    else:
        sys.exit(f"--nbm-score: no NBM weights for fold {f} ({wpath})")
    M = NBM.model_arrays(state)
    NU = NSC.nu_consts(state)
    tr_o = np.sort(tr_s)
    samp = Parallel(n_jobs=THREADS, backend="threading")(
        delayed(NBM.run_hidden)(S[i][1], S[i][2], M) for i in tr_o[::10])
    B = NSC.basis_of(np.concatenate(samp))
    del samp
    items = [it for its in srcs for it in its]
    R = Parallel(n_jobs=THREADS, backend="threading")(
        delayed(NSC.run_series)(h, o, M, B, NU, SUB) for h, o in items)
    res, o = [], 0
    for its in srcs:
        res.append(np.concatenate(R[o:o + len(its)]).astype(np.float32))
        o += len(its)
    del R
    print(f"fold {f}: NBM score-test features for {count:,} series in {time.time() - t0:.0f}s (nu {NU[0]:.2f})",
          flush=True)
    if use_cache:
        np.savez(cache, flat=np.concatenate(res), off=np.r_[0, np.cumsum([len(b) for b in res])], n=count)
    for k, its in enumerate(srcs):
        want = step_o if k == 0 else parts[k - 1]["step"]
        if len(res[k]) != len(want):
            sys.exit(f"--nbm-score: row layout mismatch in source {k} ({len(res[k])} vs {len(want)} rows)")
    return res


PC_PAIRS = None   # --pc training pairs of all original series (model-independent; built once per run)


def pc_fold(f, tr_s, S, tau, parts):
    """--pc: fold-f pair-classifier scan features for every original series and every extra source (cached per fold).
    The classifiers are trained on the fold's ORIGINAL training series only (pairs from re-cuts would be near-copies
    of their parents' windows): training originals get cross-fitted features (inner --pc-inner folds), validation
    originals the model trained on all of them, an extra series its parent's inner model (a parent < 0 never trains a
    classifier -> the full model; a validation parent -> zeros, those rows never train in this fold)."""
    global PC_PAIRS
    use_cache = args.n_series == 0
    cfg = f"k{args.pc_per}_r{args.pc_rounds}_l{args.pc_leaves}_i{args.pc_inner}"
    stubs = ["orig"] + [P["stub"] for P in parts]
    paths = [os.path.join("data", f"pc{PC.VERSION}_f{f}_{cfg}_{st}.npz") for st in stubs]
    counts = [len(S)] + [len(P["items"]) for P in parts]
    if use_cache and not args.rebuild and all(os.path.exists(q) for q in paths):
        mats = [np.load(q) for q in paths]
        if all(int(z["n"]) == c for z, c in zip(mats, counts)):
            print(f"fold {f}: pc features loaded from cache", flush=True)
            return [z["B"] for z in mats]
    if PC_PAIRS is None:
        t0 = time.time()
        PC_PAIRS = PC.build_pairs([(q[1], q[2]) for q in S], tau, np.arange(len(S)), args.pc_per, 0, THREADS)
        print(f"pc: {len(PC_PAIRS['y']):,} training pairs from {len(S):,} original series in {time.time() - t0:.0f}s",
              flush=True)
    out, _ = PC.fold_features(S, tau, tr_s, [(P["items"], P["vparent"]) for P in parts], k=args.pc_per,
                              rounds=args.pc_rounds, leaves=args.pc_leaves, inner=args.pc_inner, fold=f, sub=SUB,
                              threads=THREADS, say=lambda m: print(f"fold {f}: {m}", flush=True), pairs=PC_PAIRS)
    if use_cache:
        for B, q, c in zip(out, paths, counts):
            np.savez(q, B=B, n=c)
    return out


def step_weights(y, step):
    """Class-balance every step and weight it like the metric: pos -> nneg/2, neg -> npos/2."""
    T = int(step.max()) + 1
    npos = np.bincount(step, weights=y, minlength=T)
    n = np.bincount(step, minlength=T).astype(float)
    nneg = n - npos
    w = np.where(y == 1, nneg[step] / 2.0, npos[step] / 2.0)
    return w / max(w.mean(), 1e-12)


# ---- model backend: LightGBM (CPU) or XGBoost (CPU or CUDA GPU) -----------------------------
MONO = None   # --monotone: constraint string for XGBoost, set in main() once the column names are known


def monotone_vector(names):
    """+1 for one-sided evidence columns (larger = more evidence of a break at the same step), 0 otherwise.
    Two-sided columns (log variance ratios, shifts, time since break, posterior type shares) stay free."""
    up = {"nb_logodds", "ev_mix_logodds", "ev_logbf"}
    vec = []
    for nm in names:
        one_sided = (nm in up or nm.startswith(("dist_", "ev_cusum_", "ev_logsr_"))
                     or (nm.startswith("bo_") and nm.endswith(("_logodds", "_logbf"))))
        vec.append(1 if one_sided else 0)
    return vec


def base_margin_fold(f, Xtr, ytr, w, str_, Xva, sva, names):
    """--base-margin: per-fold starting logit from one evidence column s. Weighted logistic regression (the training
    row weights, so steps count like in the metric) of y on [s, s*log1p(t), log1p(t)] on a training subsample; the
    slope in s may change smoothly with the step. Returns (training margins, validation margins)."""
    from sklearn.linear_model import LogisticRegression
    t0 = time.time()
    j = names.index(args.base_margin)
    rng = np.random.default_rng(3000 + f)
    idx = np.sort(rng.choice(len(ytr), min(len(ytr), 500000), replace=False))
    med = float(np.nanmedian(Xtr[idx, j]))

    def design(s, st):
        s = np.nan_to_num(s.astype(np.float64), nan=med)
        lt = np.log1p(st.astype(np.float64))
        return np.column_stack([s, s * lt, lt])

    D = design(Xtr[idx, j], str_[idx])
    mu, sd = D.mean(0), np.maximum(D.std(0), 1e-12)
    clf = LogisticRegression(C=1e6, max_iter=2000).fit((D - mu) / sd, ytr[idx], sample_weight=w[idx])
    b = clf.coef_[0] / sd

    def margin(s, st):
        return ((design(s, st) - mu) / sd @ clf.coef_[0] + clf.intercept_[0]).astype(np.float32)

    lts = np.log1p(np.array([0, 10, 100, 300, 1000]))
    slope = b[0] + b[1] * lts
    print(f"fold {f}: --base-margin {args.base_margin}: slope in s at steps 0/10/100/300/1000 "
          + " ".join(f"{v:+.3f}" for v in slope)
          + ("" if (slope > 0).all() else "  WARNING: slope <= 0 somewhere (reverses that step's ranking)")
          + f" | {time.time() - t0:.0f}s", flush=True)
    return margin(Xtr[:, j], str_), margin(Xva[:, j], sva)


def fit(Xtr, ytr, w, bm=None):
    if args.backend == "xgb":
        import xgboost as xgb
        dtr = xgb.QuantileDMatrix(Xtr, label=ytr, weight=w, base_margin=bm, max_bin=255)
        p = dict(objective="binary:logistic", tree_method="hist", device=args.device,
                 learning_rate=args.lr, grow_policy="lossguide", max_leaves=args.leaves, max_depth=0,
                 min_child_weight=max(1.0, 0.15 * args.min_leaf),  # ~ min_data_in_leaf for p(1-p)~0.15
                 subsample=0.8, colsample_bytree=args.ff, reg_lambda=args.lam, max_bin=255,
                 seed=args.seed, nthread=THREADS, verbosity=1)
        if MONO is not None:
            p["monotone_constraints"] = MONO
        if args.dart > 0:
            p.update(booster="dart", rate_drop=args.dart, skip_drop=args.dart_skip, sample_type="uniform",
                     normalize_type="tree")
        try:
            return xgb.train(p, dtr, args.rounds)
        except xgb.core.XGBoostError as e:
            sys.exit(f"XGBoost failed on device={args.device}: {e}\n"
                     "  - try: pip install -U xgboost   (a recent build is needed for the RTX 50-series)\n"
                     "  - or run without --gpu to use LightGBM on the CPU")
    if args.backend == "cat":
        from catboost import CatBoostClassifier, Pool
        gpu = args.device == "cuda"
        p = dict(loss_function="Logloss", iterations=args.rounds, learning_rate=args.lr, depth=args.cat_depth,
                 grow_policy=args.cat_grow, l2_leaf_reg=10.0, border_count=args.cat_border,
                 bootstrap_type="Bernoulli", subsample=0.8, boosting_type="Ordered" if args.cat_ordered else "Plain",
                 random_seed=args.seed, thread_count=THREADS, task_type="GPU" if gpu else "CPU",
                 allow_writing_files=False, verbose=max(args.rounds // 10, 1))
        if args.cat_grow == "Lossguide":
            p.update(max_leaves=args.leaves, depth=16)
        if args.cat_grow != "SymmetricTree":
            p["min_data_in_leaf"] = args.min_leaf
        if not gpu:
            p["rsm"] = args.ff          # per-split feature sampling: CPU only for Logloss
        if gpu and "torch" in sys.modules:   # free the NBM's cached VRAM before CatBoost reserves its share
            sys.modules["torch"].cuda.empty_cache()
        m = CatBoostClassifier(**p)
        m.fit(Pool(Xtr, ytr, weight=w))
        return m
    return lgb.train(LGB_PARAMS, lgb.Dataset(Xtr, ytr, weight=w), args.rounds)


def fit_rank(Xtr, ytr, str_, is_orig):
    """XGBoost pairwise ranking (RankNet loss: the all-pairs form of LambdaRank, without NDCG's top-of-list
    weighting, which does not match AUC) with the online step as the query: only rows of the SAME step are
    compared, exactly like TS-AUC. Pairs are sampled (--rank-pairs per row). XGBoost ranking weights are per query
    group: each step gets npos*nneg/n from the ORIGINAL training rows, so steps count like in the metric."""
    import xgboost as xgb
    order = np.argsort(str_, kind="stable")
    Xs = Xtr[order]
    ys, qs, io = ytr[order], str_[order], is_orig[order]
    T = int(qs.max()) + 1
    npos = np.bincount(qs[io], weights=ys[io], minlength=T)
    n_o = np.bincount(qs[io], minlength=T).astype(float)
    groups, counts = np.unique(qs, return_counts=True)
    gw = (npos * (n_o - npos))[groups] / counts
    gw = gw / max(gw.mean(), 1e-12)
    dtr = xgb.QuantileDMatrix(Xs, label=ys, qid=qs, weight=gw, max_bin=255)
    del Xs
    mcw = args.rank_mcw if args.rank_mcw >= 0 else max(1.0, 0.15 * args.min_leaf)
    p = dict(objective="rank:pairwise", lambdarank_pair_method="mean", lambdarank_num_pair_per_sample=args.rank_pairs,
             tree_method="hist", device=args.device, learning_rate=args.lr, grow_policy="lossguide",
             max_leaves=args.leaves, max_depth=0, min_child_weight=mcw, subsample=0.8, colsample_bytree=args.ff,
             reg_lambda=10.0, max_bin=255, seed=args.seed, nthread=THREADS, verbosity=1)
    return xgb.train(p, dtr, args.rounds)


XSEC_EDGES = np.unique(np.round(np.geomspace(1, 1000, 41)).astype(np.int64)) - 1   # bin starts (online steps)
XSEC_SAMPLE = 50000


@njit(parallel=True, cache=True)
def _xsec_apply(X, b, cols, med, sc):
    for i in prange(X.shape[0]):
        bi = b[i]
        for k in range(len(cols)):
            c = cols[k]
            X[i, c] = (X[i, c] - med[bi, k]) / sc[bi, k]


def xsec_fold(f, Xtr, st_tr, Xva, st_va, names):
    """--xsec: per-step-bin robust standardisation, statistics from the training rows; Xtr in place, returns Xva."""
    t0 = time.time()
    cols = np.array([j for j, nm in enumerate(names) if not (nm.startswith("ctx_") or nm == "t_log")], dtype=np.int64)
    nb = len(XSEC_EDGES)
    btr = (np.searchsorted(XSEC_EDGES, st_tr, "right") - 1).astype(np.int64)
    bva = (np.searchsorted(XSEC_EDGES, st_va, "right") - 1).astype(np.int64)
    med = np.zeros((nb, len(cols)), dtype=np.float32)
    sc = np.ones((nb, len(cols)), dtype=np.float32)
    rng = np.random.default_rng(1000 + f)
    have = np.zeros(nb, bool)
    n_fb = 0
    for b in range(nb):
        idx = np.flatnonzero(btr == b)
        if len(idx) == 0:
            continue
        if len(idx) > XSEC_SAMPLE:
            idx = np.sort(rng.choice(idx, XSEC_SAMPLE, replace=False))
        sub = Xtr[idx][:, cols].astype(np.float64)
        q1, q2, q3 = np.percentile(sub, [25, 50, 75], axis=0)
        s_ = (q3 - q1) / 1.349
        bad = s_ <= 1e-6
        if bad.any():
            n_fb += int(bad.sum())
            sd = sub.std(0)
            s_ = np.where(bad, np.where(sd > 1e-6, sd, 1.0), s_)
        med[b], sc[b], have[b] = q2, s_, True
    for b in range(nb):                          # bins without training rows borrow the nearest earlier bin
        if not have[b]:
            prev = np.flatnonzero(have[:b])
            src = prev[-1] if len(prev) else np.flatnonzero(have)[0]
            med[b], sc[b] = med[src], sc[src]
    _xsec_apply(Xtr, btr, cols, med, sc)
    Xva = np.ascontiguousarray(Xva, dtype=Xtr.dtype)
    _xsec_apply(Xva, bva, cols, med, sc)
    print(f"fold {f}: --xsec: {len(cols)} of {len(names)} columns standardised per step bin ({int(have.sum())} of {nb} "
          f"bins with training rows; {n_fb} bin-columns fell back from IQR to sd) in {time.time() - t0:.0f}s", flush=True)
    return Xva


def valid_obj(Xva):
    if args.backend == "xgb":
        import xgboost as xgb
        return xgb.DMatrix(Xva, nthread=THREADS)
    if args.backend == "cat":
        from catboost import Pool
        return Pool(Xva)
    return Xva


def predict_chunk(m, v, prev, k):
    """Raw scores of trees prev..k. (XGBoost adds the same constant base score to every chunk,
    which shifts all rows equally and so cannot change any per-step AUC.)"""
    if args.backend == "xgb":
        return m.predict(v, iteration_range=(prev, k), output_margin=True)
    if args.backend == "cat":   # ntree_start > 0 leaves out the constant bias: a shift, no AUC change
        return m.predict(v, prediction_type="RawFormulaVal", ntree_start=prev, ntree_end=k, thread_count=THREADS)
    return m.predict(v, start_iteration=prev, num_iteration=k - prev, raw_score=True)


def gain(m, n):
    if args.backend == "xgb":
        out = np.zeros(n)
        for key, val in m.get_score(importance_type="total_gain").items():
            out[int(key[1:])] = val
        return out
    if args.backend == "cat":   # loss-free importance (no data needed); a share, like gain
        return np.asarray(m.get_feature_importance(type="PredictionValuesChange"), dtype=float)
    return m.feature_importance("gain")


def curve_of(m, vobj, yva, sva, grid):
    """Validation TS-AUC at every grid round (the same chunked scoring as the fold loop)."""
    raw = np.zeros(len(yva))
    prev, out = 0, []
    for k in grid:
        raw = raw + predict_chunk(m, vobj, prev, k)
        prev = k
        out.append(ts_auc(raw, yva, sva))
    return np.array(out)


def prune_columns(m, n):
    """--prune-keep: sorted indices of the top K columns by the total gain of pass 1's first --prune-rounds trees
    (the late trees fit mostly after the validation peak, so their gain would favour noise columns)."""
    r = min(args.prune_rounds, args.rounds)
    g = gain(m[:r], n) if args.backend == "xgb" else m.feature_importance("gain", iteration=r)
    keep = np.sort(np.argsort(-g, kind="stable")[:args.prune_keep])
    return keep, g


def svm_columns(names):
    """Indices of the SVM's input columns: t_log, all ctx_ columns, and the top-K others by tree importance."""
    if args.svm_top <= 0:
        return list(range(len(names)))
    if not os.path.exists(args.svm_imp):
        sys.exit(f"--svm-top needs {args.svm_imp} (an importance file from cv.py); or use --svm-top 0")
    with open(args.svm_imp, encoding="utf-8") as fh:
        next(fh)
        ranked = [line.split(",")[0] for line in fh]
    keep = [nm for nm in names if nm == "t_log" or nm.startswith("ctx_")]
    n_fixed = len(keep)
    for nm in ranked:
        if len(keep) >= args.svm_top + n_fixed:   # 5 ctx_ + t_log are always in
            break
        if nm in names and nm not in keep:
            keep.append(nm)
    keep += [nm for nm in names if nm.startswith("nbh_") and nm not in keep]   # --nbm-hpca: always in
    return [names.index(nm) for nm in keep]


def _svm_prep(Xc, mu, sd):
    Z = np.sign(Xc) * np.log1p(np.abs(Xc))
    return np.clip((Z - mu) / sd, -6.0, 6.0).astype(np.float32)


def _svm_family(nm):
    """--svm-ard family: the kernel-width group of an SVM input column."""
    if nm == "t_log":
        return "t"
    if nm.startswith("win") and "_" in nm:
        return "win"
    if nm.startswith(("glr", "scan")):
        return "scan"
    if nm.startswith(("full_", "chi2")):
        return "full"
    if nm.startswith(("vol_", "ewma")):
        return "vol"
    return nm.split("_")[0]


def _pair_weights(wt, ytr, st, T):
    """--svm-rank: per-row pair weights for the per-step pairwise loss.
    Pair (i positive, j negative, same step t) gets weight a_i b_j. a, b are the row weights rescaled so that a step's
    pairs together weigh sqrt(Wpos_t) * sqrt(Wneg_t), where W = the metric-matched row weights per step and class:
    with uniform rows that is the metric's own n_pos x n_neg, and re-cut rows keep their relative share.
    Returns pw (a or b per row), per-step totals At (positives) and Bt (negatives), and the diagonal of the pairwise
    loss d_i = Bt a_i (positives) or At b_j (negatives)."""
    pos = ytr == 1
    pw = wt.astype(np.float64)
    Wp = np.bincount(st[pos], weights=pw[pos], minlength=T)
    Wn = np.bincount(st[~pos], weights=pw[~pos], minlength=T)
    pw = pw / np.sqrt(np.maximum(np.where(pos, Wp[st], Wn[st]), 1e-12))
    At = np.bincount(st[pos], weights=pw[pos], minlength=T)
    Bt = np.bincount(st[~pos], weights=pw[~pos], minlength=T)
    d = np.where(pos, Bt[st], At[st]) * pw
    return pw, At, Bt, d


def _svm_system(F, rowA, rowb, alpha, kp, pa, pb, st, At, Bt, T):
    """Normal equations (A, b) of the mixed loss (1 - alpha) * pointwise + alpha * kp * per-step pairwise, for the
    feature rows F (torch, differentiable). pa / pb = pair weights on positive / negative rows (0 elsewhere).
    Pairwise part, per step: sum_ij a_i b_j (r_i - r_j)^2 with r = (+-1) - f, i.e. r' M r with M = diag(d) - p q' - q p';
    so A gets Phi' diag(d) Phi - Sp' Sn - Sn' Sp and b gets 2 (Sp' Bt - Sn' At), Sp_t / Sn_t = sum over the step's
    positive / negative rows of a phi / b phi."""
    import torch
    A = F.T @ (F * rowA[:, None])
    b = F.T @ rowb
    if alpha > 0:
        Sp = torch.zeros(T, F.shape[1], dtype=F.dtype, device=F.device).index_add(0, st, F * pa[:, None])
        Sn = torch.zeros(T, F.shape[1], dtype=F.dtype, device=F.device).index_add(0, st, F * pb[:, None])
        C2 = Sp.T @ Sn
        A = A - alpha * kp * (C2 + C2.T)
        b = b + alpha * kp * 2.0 * (Sp.T @ Bt - Sn.T @ At)
    return A, b


def _svm_rows(wt, ytr, st, alpha, T):
    """Row coefficients of the mixed loss: rowA (Gram diagonal), rowb (pointwise right-hand side), pair weights."""
    ypm = (2.0 * ytr - 1.0)
    pos = ytr == 1
    if alpha <= 0:
        z = np.zeros(len(wt))
        return wt.astype(np.float64), wt * ypm, z, z, np.zeros(1), np.zeros(1), 1.0
    pw, At, Bt, d = _pair_weights(wt, ytr, st, T)
    kp = wt.sum() / max(float((At * Bt).sum()), 1e-12)   # pairwise total weight = pointwise total weight
    rowA = (1.0 - alpha) * wt + alpha * kp * d
    rowb = (1.0 - alpha) * wt * ypm
    return rowA, rowb, np.where(pos, pw, 0.0), np.where(pos, 0.0, pw), At, Bt, kp


def _svm_loss(f, wt, ypm, alpha, kp, pa, pb, st, At, Bt, T):
    """Value of the mixed loss for scores f (torch), normalised by the total row weight."""
    import torch
    r = ypm - f
    out = (1.0 - alpha) * (wt * r * r).sum()
    if alpha > 0:
        z = torch.zeros(T, dtype=f.dtype, device=f.device)
        sp_r = z.index_add(0, st, pa * r)
        sn_r = z.index_add(0, st, pb * r)
        sq = (Bt[st] * pa + At[st] * pb) * r * r
        out = out + alpha * kp * (sq.sum() - 2.0 * (sp_r * sn_r).sum())
    return out / wt.sum()


def svm_ard(Xtr, ytr, wt, str_, cols, names, mu, sd, gamma, fold, dev):
    """--svm-ard: learn one kernel width per input family (or per column) before the closed-form solve.
    Training series are split 80/20 by series (a new series starts where the online step does not increase); on a
    row sample a kernel ridge with --svm-ard-dim random features, log widths theta and log ridge rho is solved in
    closed form on the inner-train rows and theta, rho follow the gradient of the same loss (--svm-rank mix) on the
    inner-validation rows (Adam). Returns the per-column multipliers exp(theta) of the RBF frequencies
    (bandwidth sqrt(2 gamma) * exp(theta): theta > 0 = that family varies the kernel faster, i.e. matters more)."""
    import torch
    K, D = len(cols), args.svm_ard_dim
    grp = np.cumsum(np.r_[True, str_[1:] <= str_[:-1]])
    rng = np.random.default_rng(args.seed + 1000 * fold + 7)
    vg = rng.random(int(grp.max()) + 1) < 0.2   # inner-validation series
    iv_all, it_all = np.flatnonzero(vg[grp]), np.flatnonzero(~vg[grp])
    it = np.sort(rng.choice(it_all, min(args.svm_ard_rows, len(it_all)), replace=False))
    iv = np.sort(rng.choice(iv_all, min(args.svm_ard_rows // 2, len(iv_all)), replace=False))
    if args.svm_ard == "family":
        fams = [_svm_family(names[j]) for j in cols]
        labels = sorted(set(fams))
        gidx = np.array([labels.index(fa) for fa in fams])
    else:
        labels = [names[j] for j in cols]
        gidx = np.arange(K)
    T = int(str_.max()) + 1
    alpha = args.svm_rank

    def side(ix):
        Z = torch.as_tensor(_svm_prep(Xtr[ix][:, cols], mu, sd), device=dev)
        w_ = wt[ix].astype(np.float64)
        w_ = w_ / w_.mean()
        rowA, rowb, pa, pb, At, Bt, kp = _svm_rows(w_, ytr[ix], str_[ix], alpha, T)
        t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=dev)
        return dict(Z=Z, rowA=t(rowA), rowb=t(rowb), pa=t(pa), pb=t(pb), At=t(At), Bt=t(Bt), kp=float(kp),
                    st=torch.as_tensor(str_[ix].astype(np.int64), device=dev), w=t(w_),
                    ypm=t(2.0 * ytr[ix] - 1.0), y=ytr[ix], s=str_[ix])

    tr, va = side(it), side(iv)
    Om = torch.as_tensor(rng.standard_normal((K, D)) * np.sqrt(2.0 * gamma), dtype=torch.float32, device=dev)
    be = torch.as_tensor(rng.uniform(0, 2 * np.pi, D), dtype=torch.float32, device=dev)
    sc = float(np.sqrt(2.0 / D))
    g_t = torch.as_tensor(gidx, device=dev)
    theta = torch.zeros(len(labels), device=dev, requires_grad=True)
    rho = torch.zeros(1, device=dev, requires_grad=True)          # log ridge, in units of the mean Gram eigenvalue
    opt = torch.optim.Adam([theta, rho], lr=args.svm_ard_lr)

    def phi(S, th):
        f = torch.cos((S["Z"] * torch.exp(th)[g_t]) @ Om + be) * sc
        return torch.cat([f, torch.ones(len(f), 1, device=dev)], dim=1)

    def fit_score(th, r_):
        F = phi(tr, th)
        A, b = _svm_system(F, tr["rowA"], tr["rowb"], alpha, tr["kp"], tr["pa"], tr["pb"], tr["st"], tr["At"],
                           tr["Bt"], T)
        A, b = A.double(), b.double()
        lam = torch.exp(r_).double() * torch.diagonal(A).mean()
        c = torch.linalg.solve(A + lam * torch.eye(D + 1, dtype=A.dtype, device=dev), b)
        return phi(va, th) @ c.float()

    t0 = time.time()
    with torch.no_grad():
        f0 = fit_score(theta, rho)
    auc0 = ts_auc(f0.cpu().numpy().astype(np.float64), va["y"], va["s"])
    for k in range(args.svm_ard_steps):
        opt.zero_grad()
        fv = fit_score(theta, rho)
        loss = _svm_loss(fv, va["w"], va["ypm"], alpha, va["kp"], va["pa"], va["pb"], va["st"], va["At"], va["Bt"], T)
        loss.backward()
        opt.step()
        with torch.no_grad():
            theta.clamp_(-3.0, 3.0)
            rho.clamp_(-12.0, 6.0)
        if k % 20 == 0 or k == args.svm_ard_steps - 1:
            print(f"fold {fold}: --svm-ard step {k}: inner-validation loss {loss.item():.5f}", flush=True)
    with torch.no_grad():
        f1 = fit_score(theta, rho)
    auc1 = ts_auc(f1.cpu().numpy().astype(np.float64), va["y"], va["s"])
    th = theta.detach().cpu().numpy()
    order = np.argsort(-th)
    show = order if len(labels) <= 20 else np.r_[order[:10], order[-10:]]
    print(f"fold {fold}: --svm-ard {args.svm_ard} ({len(labels)} widths, {len(it):,} / {len(iv):,} inner rows, "
          f"{time.time() - t0:.0f}s): inner TS-AUC {auc0:.4f} -> {auc1:.4f}; ridge x{np.exp(rho.item()):.3g}; "
          f"width multipliers " + "  ".join(f"{labels[i]}:{np.exp(th[i]):.2f}" for i in show), flush=True)
    del tr, va
    torch.cuda.empty_cache()
    return np.exp(th)[gidx]


def svm_fold(Xtr, ytr, w, str_, Xva, yva, sva, cols, names, fold):
    """Least-squares SVM (kernel ridge classification, labels +-1, metric-matched row weights) with an RBF kernel
    exp(-gamma |a - b|^2) approximated by D random Fourier features phi(x) = sqrt(2/D) cos(x Omega + beta) plus a
    constant. Inputs: signed log, standardised on the training rows, clipped to +-6 (constants that would ship with
    the model). Solved in closed form: A = Phi' W Phi and b = Phi' W y are accumulated on the GPU in float32 chunks
    (summed in float64), A is eigendecomposed once on the CPU, and every regularisation value of --svm-lams then costs
    one D x D product. Deterministic for a given --seed. Returns best score, best raw scores, best grid index, curve.
    --svm-rank alpha mixes in a per-step pairwise loss (see _svm_system; still closed form); --svm-ard learns
    per-family kernel widths first (svm_ard); with both off the result is identical to the original code."""
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    K, D, ch = len(cols), args.svm_dim, args.svm_chunk
    alpha = args.svm_rank
    t0 = time.time()
    # preprocessing constants from the training rows (chunked: no second copy of the training matrix)
    s1 = np.zeros(K)
    s2 = np.zeros(K)
    for i in range(0, len(Xtr), ch):
        Z = Xtr[i:i + ch][:, cols].astype(np.float64)
        Z = np.sign(Z) * np.log1p(np.abs(Z))
        s1 += Z.sum(0)
        s2 += (Z * Z).sum(0)
    mu = s1 / len(Xtr)
    sd = np.sqrt(np.maximum(s2 / len(Xtr) - mu * mu, 0)) + 1e-6
    wt = np.ones(len(ytr), np.float32) if w is None else w.astype(np.float32)
    gamma = args.svm_gmul / K
    colmul = svm_ard(Xtr, ytr, wt, str_, cols, names, mu, sd, gamma, fold, dev) if args.svm_ard != "off" else None
    rng = np.random.default_rng(args.seed + 1000 * fold)
    Om_np = rng.standard_normal((K, D)) * np.sqrt(2.0 * gamma)
    if colmul is not None:
        Om_np = Om_np * colmul[:, None]
    Om = torch.as_tensor(Om_np, dtype=torch.float32, device=dev)
    be = torch.as_tensor(rng.uniform(0, 2 * np.pi, D), dtype=torch.float32, device=dev)
    sc = float(np.sqrt(2.0 / D))

    def phi(Xc):
        z = torch.as_tensor(_svm_prep(Xc[:, cols], mu, sd), device=dev)
        f = torch.cos(z @ Om + be) * sc
        return torch.cat([f, torch.ones(len(f), 1, device=dev)], dim=1)

    t_ard = time.time() - t0
    A = torch.zeros(D + 1, D + 1, dtype=torch.float64, device=dev)
    bv = torch.zeros(D + 1, dtype=torch.float64, device=dev)
    T = int(str_.max()) + 1
    rowA, rowb, pa, pb, At, Bt, kp = _svm_rows(wt.astype(np.float64), ytr, str_, alpha, T)
    rowA, rowb, pa, pb = (a.astype(np.float32) for a in (rowA, rowb, pa, pb))
    if alpha > 0:
        Sp = torch.zeros(T, D + 1, dtype=torch.float64, device=dev)
        Sn = torch.zeros(T, D + 1, dtype=torch.float64, device=dev)
    for i in range(0, len(Xtr), ch):
        F_ = phi(Xtr[i:i + ch])
        ra = torch.as_tensor(rowA[i:i + ch], device=dev)
        A += (F_.T @ (F_ * ra[:, None])).double()
        bv += (F_.T @ torch.as_tensor(rowb[i:i + ch], device=dev)).double()
        if alpha > 0:
            st = torch.as_tensor(str_[i:i + ch].astype(np.int64), device=dev)
            for S_, p_ in ((Sp, pa), (Sn, pb)):
                pc = p_[i:i + ch]
                m_ = torch.as_tensor(np.flatnonzero(pc > 0), device=dev)
                if len(m_):
                    tmp = torch.zeros(T, D + 1, dtype=torch.float32, device=dev)
                    tmp.index_add_(0, st[m_], F_[m_] * torch.as_tensor(pc, device=dev)[m_][:, None])
                    S_ += tmp.double()
    if alpha > 0:
        At_t = torch.as_tensor(At, dtype=torch.float64, device=dev)
        Bt_t = torch.as_tensor(Bt, dtype=torch.float64, device=dev)
        C2 = Sp.T @ Sn
        A -= alpha * kp * (C2 + C2.T)
        bv += alpha * kp * 2.0 * (Sp.T @ Bt_t - Sn.T @ At_t)
        del Sp, Sn, C2
    t_acc = time.time() - t0 - t_ard
    A, bv = A.cpu().numpy(), bv.cpu().numpy()
    ev, V = np.linalg.eigh(A)
    ev = np.maximum(ev, 0.0)
    Vb = V.T @ bv
    lams = [float(v) for v in args.svm_lams.split(",")]
    base = float(ev.mean())
    C = np.stack([V @ (Vb / (ev + lm * base)) for lm in lams], axis=1)   # (D+1, n_lams)
    t_solve = time.time() - t0 - t_ard - t_acc
    Ct = torch.as_tensor(C, dtype=torch.float32, device=dev)
    S = np.concatenate([(phi(Xva[i:i + ch]) @ Ct).cpu().numpy() for i in range(0, len(Xva), ch)])
    curve = [ts_auc(S[:, j], yva, sva) for j in range(len(lams))]
    j = int(np.argmax(curve))
    print(f"fold {fold}: SVM {K} columns, D={D}, gamma={gamma:.4g}, rank mix {alpha:g}, ard {args.svm_ard}: "
          f"Gram accumulation {t_acc:.0f}s, eigen-solve {t_solve:.0f}s; TS-AUC by lambda: "
          + "  ".join(f"{lm:g}:{c:.4f}" for lm, c in zip(lams, curve)), flush=True)
    if fold == 0:   # what one online point would cost at inference (numpy, one row at a time)
        Om_n, be_n, c_n = Om.cpu().numpy(), be.cpu().numpy(), C[:, j].astype(np.float32)
        rows = Xva[:2000]
        t1 = time.time()
        for r in rows:
            z = _svm_prep(r[cols][None, :], mu, sd)[0]
            float(np.cos(z @ Om_n + be_n) @ c_n[:D] * sc + c_n[D])
        us = 1e6 * (time.time() - t1) / len(rows)
        print(f"fold 0: SVM inference cost {us:.0f} microseconds per online point (numpy, single row) -> about "
              f"{us * 8e6 / 6e7:.0f} min for the ~8M cloud steps (on top of the feature kernels)", flush=True)
    return curve[j], S[:, j].astype(np.float64), j, curve


def emb_fold(kind, f, tr_s, S, tau, parts):
    """--femb / --emb: fold-f embedding-drift columns for every original series and every extra source (cached)."""
    use_cache = args.n_series == 0
    steps = args.femb_steps if kind == "femb" else args.emb_steps
    stubs = ["orig"] + [P["stub"] for P in parts]
    cfg = f"st{steps}" + (f"_w{args.emb_width}" if kind == "emb" and args.emb_width != 64 else "")
    paths = [os.path.join("data", f"{kind}{EMB.VERSION}_f{f}_{cfg}_{st}.npz") for st in stubs]
    counts = [len(S)] + [len(P["items"]) for P in parts]
    if use_cache and not args.rebuild and all(os.path.exists(q) for q in paths):
        mats = [np.load(q) for q in paths]
        if all(int(z["n"]) == c for z, c in zip(mats, counts)):
            print(f"fold {f}: {kind} features loaded from cache", flush=True)
            return [z["B"] for z in mats]
    print(f"fold {f}: {kind} -- training and featurising ...", flush=True)
    out = EMB.fold_features(kind, S, tau, tr_s, [P["items"] for P in parts], steps, THREADS,
                            say=lambda m: print(f"fold {f}: {m}", flush=True), width=args.emb_width)
    if use_cache:
        for B, q, c in zip(out, paths, counts):
            np.savez(q, B=B, n=c)
    return out


EQ_BASE = None   # --eqf fold-independent pieces (per source, and the originals' pseudo-split rows); built once per run


def _eq_parts(h, o):
    return EQ.series_parts(h, o, SUB)


def eq_source(items, cache):
    """--eqf fold-independent pieces of a list of (hist, online): running log mean q, eq_f2_lr, F3l inputs (cached)."""
    if cache and os.path.exists(cache) and not args.rebuild:
        z = np.load(cache)
        if int(z["n"]) == len(items):
            return dict(LQ=z["LQ"], F2=z["F2"], XH=z["XH"], counts=z["counts"])
    t0 = time.time()
    res = Parallel(n_jobs=THREADS, backend="threading")(delayed(_eq_parts)(h, o) for h, o in items)
    D = dict(LQ=np.concatenate([r[0] for r in res]), F2=np.concatenate([r[1] for r in res]),
             XH=np.array([r[2] for r in res]), counts=np.array([len(r[0]) for r in res]))
    print(f"equation forecasts: {len(items):,} series in {time.time() - t0:.0f}s", flush=True)
    if cache:
        np.savez(cache, n=len(items), **D)
    return D


def eq_pseudo(S, cache):
    """--eqf F3l training rows: pseudo-splits inside every original HISTORY (no online data, no labels; cached)."""
    if cache and os.path.exists(cache) and not args.rebuild:
        z = np.load(cache)
        if int(z["n"]) == len(S):
            return z["XP"], z["YP"], z["owner"]
    t0 = time.time()
    res = Parallel(n_jobs=THREADS, backend="threading")(delayed(EQ.pseudo_rows)(q[1]) for q in S)
    XP = np.concatenate([r[0] for r in res])
    YP = np.concatenate([r[1] for r in res])
    owner = np.repeat(np.arange(len(S)), [len(r[0]) for r in res])
    print(f"equation forecasts: {len(XP):,} pseudo-splits in {len(S):,} histories in {time.time() - t0:.0f}s", flush=True)
    if cache:
        np.savez(cache, n=len(S), XP=XP, YP=YP, owner=owner)
    return XP, YP, owner


def eq_fold(f, tr_s, S, parts):
    """--eqf: fold-f columns (eq_f2_lr, eq_f3l_lr) for the originals and every extra source. eq_f2_lr does not
    depend on the fold; the F3l ridge is trained on the pseudo-split rows of the fold's ORIGINAL training histories."""
    global EQ_BASE
    if EQ_BASE is None:
        use_cache = args.n_series == 0
        stubs = ["orig"] + [P["stub"] for P in parts]
        items = [[(q[1], q[2]) for q in S]] + [P["items"] for P in parts]
        src = [eq_source(it, os.path.join("data", f"eqf{EQ.VERSION}_{st}.npz") if use_cache else None)
               for it, st in zip(items, stubs)]
        EQ_BASE = (src, eq_pseudo(S, os.path.join("data", f"eqfps{EQ.VERSION}_orig.npz") if use_cache else None))
    src, (XP, YP, owner) = EQ_BASE
    mk = np.isin(owner, tr_s)
    models = EQ.fit_f3l(XP[mk], YP[mk])
    out = [np.column_stack([D["F2"], EQ.f3l_column(D["LQ"], D["counts"], EQ.predict_f3l(models, D["XH"]), SUB)])
           .astype(np.float32) for D in src]
    bad = sum(int((~np.isfinite(B)).sum()) for B in out)
    print(f"fold {f}: F3l ridge trained on {int(mk.sum()):,} pseudo-splits of {len(tr_s):,} training histories"
          + (f"; WARNING {bad} non-finite values" if bad else ""), flush=True)
    return out


TTA_W0 = (1.0, 2.0)   # weight of the plain score in the view average (1 = equal weights; 2 = plain counts double)


def tta_fold(f, m, k, Xva_, va, va_s, parts, y, step, sid, oof_tta, res):
    """--tta: score fold f's validation series with the fold model at its best round k, plain and averaged over the
    test-time views. Every logit is ONE predict call over rounds 0..k (XGBoost adds its constant base score once per
    call, so plain and view logits share one offset and averaging over different numbers of views stays fair).
    A view of shift sh at view step v scores the original step v - sh; rows are matched by (series, step)."""
    t0 = time.time()
    va_idx = np.flatnonzero(va)
    plain = predict_chunk(m, valid_obj(Xva_), 0, k).astype(np.float64)
    vsum = np.zeros(len(va_idx))
    vcnt = np.zeros(len(va_idx))
    pos = np.full(len(y), -1, dtype=np.int64)
    pos[va_idx] = np.arange(len(va_idx))
    row0 = np.searchsorted(sid, np.arange(int(sid.max()) + 2))   # first row of each series (rows sorted by sid, step)
    for P in parts:
        if not P["score_only"]:
            continue
        sh = P["shift"]
        q = np.flatnonzero(np.isin(P["parent"], va_s) & (P["step"] >= sh))
        par, ostep = P["parent"][q], P["step"][q] - sh
        r = row0[par] + ostep // SUB
        ok = r < row0[par + 1]
        q, par, ostep, r = q[ok], par[ok], ostep[ok], r[ok]
        assert np.array_equal(sid[r], par) and np.array_equal(step[r], ostep), "--tta: view row alignment"
        j = pos[r]
        assert (j >= 0).all()
        vsum[j] += predict_chunk(m, valid_obj(P["X"][q]), 0, k)
        vcnt[j] += 1
    yv, sv = y[va], step[va]
    a_plain = ts_auc(plain, yv, sv)
    line = {"fold": f, "plain": a_plain, "cover": float((vcnt > 0).mean()), "nview": float(vcnt.mean())}
    for w0 in TTA_W0:
        avg = (w0 * plain + vsum) / (w0 + vcnt)
        oof_tta[w0][va_idx] = avg
        line[w0] = ts_auc(avg, yv, sv)
    res.append(line)
    print(f"fold {f}: --tta {args.tta} at round {k}: plain {a_plain:.4f} | "
          + " | ".join(f"plain x{w0:g} + views {line[w0]:.4f} ({line[w0] - a_plain:+.4f})" for w0 in TTA_W0)
          + f" | rows with >= 1 view {100 * line['cover']:.1f}%, mean views {line['nview']:.2f} ({time.time() - t0:.0f}s)",
          flush=True)


def tta_report(res, oof_tta, y, step, sid):
    print(f"\nTEST-TIME VIEWS --tta {args.tta} (same fold models, each fold's best round; paired vs the plain score):")
    for w0 in TTA_W0:
        d = np.array([r[w0] - r["plain"] for r in res])
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
        print(f"  plain weight {w0:g}: {d.mean():+.4f} +- {se:.4f} (SE), {int((d > 0).sum())}/{len(d)} folds better; "
              "per fold " + " ".join(f"{v:+.4f}" for v in d))
    done = ~np.isnan(oof_tta[TTA_W0[0]])
    for lo, hi in ((0, 100), (100, 300), (300, 600), (600, 10000)):
        mk = done & (step >= lo) & (step < hi)
        if mk.sum() > 0:
            print(f"  steps [{lo:4d}, {hi:5d}): "
                  + "  ".join(f"plain x{w0:g} {ts_auc(oof_tta[w0][mk], y[mk], step[mk]):.4f}" for w0 in TTA_W0))
    for w0 in TTA_W0:
        path = os.path.join(OUT, f"oof{TAG}_tta" + ("" if w0 == 1.0 else f"_w{w0:g}") + ".npz")
        np.savez(path, oof=oof_tta[w0], y=y, step=step, sid=sid)
        print(f"  saved {path}")


def main():
    X, y, sid, step, tau = build()
    if args.n_series:
        mk = sid < args.n_series
        X, y, sid, step, tau = X[mk], y[mk], sid[mk], step[mk], tau[:args.n_series]
    drop = [p for p in args.drop.split(",") if p]
    dropped = lambda nm: any((p[1:] in nm) if p.startswith("~") else nm.startswith(p) for p in drop)  # "~txt" = contains
    keep = [j for j, nm in enumerate(F.FEATURE_NAMES) if not dropped(nm)]
    vcols = [j for j, nm in enumerate(VS.FEATURE_NAMES) if not dropped(nm)]   # --drop also applies to --vol columns
    dcols = [j for j in (DT.COMPACT_COLS if args.dist == "compact" else range(DT.N_DIST))
             if not dropped(DT.FEATURE_NAMES[j])]                              # ... and to --dist columns
    mcols = [j for j, nm in enumerate(MM.FEATURE_NAMES) if not dropped(nm)]  # ... and to --mmd columns
    bcols = [j for j, nm in enumerate(BO.FEATURE_NAMES) if not dropped(nm)]  # ... and to --bocpd columns
    b2cols = [j for j, nm in enumerate(B2.FEATURE_NAMES) if not dropped(nm)]
    mscols = [j for j, nm in enumerate(MSV.FEATURE_NAMES) if not dropped(nm)]
    lcols = [j for j, nm in enumerate(LR.FEATURE_NAMES) if not dropped(nm)]
    ncols = [j for j, nm in enumerate(NBM.FEATURE_NAMES) if not dropped(nm)]
    pcols = [j for j, nm in enumerate(PS.BO_NAMES) if not dropped(nm)]
    shcols = [j for j, nm in enumerate(SH.FEATURE_NAMES) if not dropped(nm)]
    ricols = [j for j, nm in enumerate(RIO.FEATURE_NAMES) if not dropped(nm)]
    sdcols = [j for j, nm in enumerate(SD.names(args.sens)) if not dropped(nm)]
    pncols = [j for j, nm in enumerate(PS.NB_NAMES) if not dropped(nm)]
    nbdcols = [j for j, nm in enumerate(NBD.FEATURE_NAMES) if not dropped(nm)]
    nsccols = [j for j, nm in enumerate(NSC.FEATURE_NAMES) if not dropped(nm)]
    pccols = [j for j, nm in enumerate(PC.FEATURE_NAMES) if not dropped(nm)]
    eqcols = [j for j, nm in enumerate(EQ.FEATURE_NAMES) if not dropped(nm)]
    names = [F.FEATURE_NAMES[j] for j in keep]
    if len(keep) < len(F.FEATURE_NAMES):
        X = X[:, keep]
        print(f"dropped {len(F.FEATURE_NAMES) - len(keep)} features with prefixes {drop}; {len(keep)} remain")
    n_series = int(sid.max()) + 1
    S = load_train()[0][:n_series]
    if args.ev != "none":
        Eo = ev_matrix([(q[1], q[2]) for q in S],
                       os.path.join("data", f"ev{EV.VERSION}_orig.npz") if args.n_series == 0 else None)
        X = np.hstack([X, Eo[:, ev_cols()]])
        names = names + [EV.FEATURE_NAMES[j] for j in ev_cols()]
        del Eo
        print(f"evidence features ({args.ev}): {len(ev_cols())} columns appended")
    if args.vol:
        Vo = vol_matrix([(q[1], q[2]) for q in S],
                        os.path.join("data", f"vol{VS.VERSION}_orig.npz") if args.n_series == 0 else None)
        X = np.hstack([X, Vo[:, vcols]])
        names = names + [VS.FEATURE_NAMES[j] for j in vcols]
        del Vo
        print(f"volatility-state features: {len(vcols)} columns appended")
    if args.dist != "none":
        Do = dist_matrix([(q[1], q[2]) for q in S],
                         os.path.join("data", f"dist{DT.VERSION}_orig.npz") if args.n_series == 0 else None)
        X = np.hstack([X, Do[:, dcols]])
        names = names + [DT.FEATURE_NAMES[j] for j in dcols]
        del Do
        print(f"distribution-test features ({args.dist}): {len(dcols)} columns appended")
    if args.mmd:
        Mo = mmd_matrix([(q[1], q[2]) for q in S],
                        os.path.join("data", f"mmd{MM.VERSION}_orig.npz") if args.n_series == 0 else None)
        X = np.hstack([X, Mo[:, mcols]])
        names = names + [MM.FEATURE_NAMES[j] for j in mcols]
        del Mo
        print(f"MMD features: {len(mcols)} columns appended")
    if args.bocpd:
        Bo = bo_matrix([(q[1], q[2]) for q in S],
                       os.path.join("data", f"bo{BO.VERSION}_orig.npz") if args.n_series == 0 else None)
        X = np.hstack([X, Bo[:, bcols]])
        names = names + [BO.FEATURE_NAMES[j] for j in bcols]
        del Bo
        print(f"BOCPD features: {len(bcols)} columns appended")
    if args.pshape:
        Po = ps_matrix([(q[1], q[2]) for q in S],
                       os.path.join("data", f"ps{PS.VERSION}_orig.npz") if args.n_series == 0 else None)
        X = np.hstack([X, Po[:, pcols]])
        names = names + [PS.BO_NAMES[j] for j in pcols]
        del Po
        print(f"posterior-shape features: {len(pcols)} columns appended")
    if args.shape:
        SHo = sh_matrix([(q[1], q[2]) for q in S],
                        os.path.join("data", f"sh{SH.VERSION}_orig.npz") if args.n_series == 0 else None)
        X = np.hstack([X, SHo[:, shcols]])
        names = names + [SH.FEATURE_NAMES[j] for j in shcols]
        del SHo
        print(f"shape features: {len(shcols)} columns appended")
    if args.rio:
        RIo = rio_matrix([(q[1], q[2]) for q in S],
                         os.path.join("data", f"rio{RIO.VERSION}_orig.npz") if args.n_series == 0 else None)
        X = np.hstack([X, RIo[:, ricols]])
        names = names + [RIO.FEATURE_NAMES[j] for j in ricols]
        del RIo
        print(f"Rio-CPD features: {len(ricols)} columns appended")
    if args.sens != "none":
        SDo = sd_matrix([(q[1], q[2]) for q in S],
                        os.path.join("data", f"sens{SD.VERSION}{args.sens}_orig.npz") if args.n_series == 0 else None)
        X = np.hstack([X, SDo[:, sdcols]])
        names = names + [SD.names(args.sens)[j] for j in sdcols]
        del SDo
        print(f"small-break detector features ({args.sens}): {len(sdcols)} columns appended")
    if args.bocpd2:
        B2o = bo2_matrix([(q[1], q[2]) for q in S],
                         os.path.join("data", f"bo2v{B2.VERSION}_orig.npz") if args.n_series == 0 else None)
        X = np.hstack([X, B2o[:, b2cols]])
        names = names + [B2.FEATURE_NAMES[j] for j in b2cols]
        del B2o
        print(f"BOCPD2 features: {len(b2cols)} columns appended")
    if args.lratio:
        Lo = lr_matrix("orig", 10000, len(y))
        X = np.hstack([X, Lo[:, lcols]])
        names = names + [LR.FEATURE_NAMES[j] for j in lcols]
        del Lo
        print(f"learned-ratio features: {len(lcols)} columns appended")
    if args.ms:
        MSo = ms_matrix([(q[1], q[2]) for q in S],
                        os.path.join("data", f"ms{MSV.VERSION}_orig.npz") if args.n_series == 0 else None)
        X = np.hstack([X, MSo[:, mscols]])
        names = names + [MSV.FEATURE_NAMES[j] for j in mscols]
        del MSo
        print(f"Markov-switching features: {len(mscols)} columns appended")
    nb_pos = []
    if args.nbm:   # placeholder columns, filled per fold (the NBM and its prior depend on the fold)
        nb_pos = list(range(X.shape[1], X.shape[1] + len(ncols)))
        X = np.hstack([X, np.zeros((len(X), len(ncols)), dtype=X.dtype)])
        names = names + [NBM.FEATURE_NAMES[j] for j in ncols]
        print(f"NBM features: {len(ncols)} columns appended (computed per fold)")
    nps_pos = []
    if args.nbm and args.pshape:
        nps_pos = list(range(X.shape[1], X.shape[1] + len(pncols)))
        X = np.hstack([X, np.zeros((len(X), len(pncols)), dtype=X.dtype)])
        names = names + [PS.NB_NAMES[j] for j in pncols]
        print(f"NBM posterior-shape features: {len(pncols)} columns appended (computed per fold)")
    nbd_pos = []
    if args.nbm_dist:   # placeholder columns, filled per fold (the NBM depends on the fold)
        if not args.nbm:
            sys.exit("--nbm-dist needs --nbm")
        nbd_pos = list(range(X.shape[1], X.shape[1] + len(nbdcols)))
        X = np.hstack([X, np.zeros((len(X), len(nbdcols)), dtype=X.dtype)])
        names = names + [NBD.FEATURE_NAMES[j] for j in nbdcols]
        print(f"NBM distribution-test features: {len(nbdcols)} columns appended (computed per fold)")
    pc_pos = []
    if args.pc and pccols:   # placeholder columns, filled per fold (the pair classifiers depend on the fold)
        pc_pos = list(range(X.shape[1], X.shape[1] + len(pccols)))
        assert not nb_pos or min(pc_pos) > max(nb_pos + nps_pos + nbd_pos), "pc columns must come after the nb columns"
        X = np.hstack([X, np.zeros((len(X), len(pccols)), dtype=X.dtype)])
        names = names + [PC.FEATURE_NAMES[j] for j in pccols]
        print(f"pair-classifier scan features: {len(pccols)} columns appended (computed per fold)")
    emb_pos = {}
    for kind in ("femb", "emb"):   # placeholder columns, filled per fold (the nets and the prior depend on the fold)
        if getattr(args, kind):
            ecols = [j for j, nm in enumerate(EMB.names(kind)) if not dropped(nm)]
            emb_pos[kind] = (ecols, list(range(X.shape[1], X.shape[1] + len(ecols))))
            X = np.hstack([X, np.zeros((len(X), len(ecols)), dtype=X.dtype)])
            names = names + [EMB.names(kind)[j] for j in ecols]
            print(f"{kind} features: {len(ecols)} columns appended (computed per fold)")
    eq_pos = []
    if args.eqf and eqcols:   # placeholder columns, filled per fold (the F3l ridge depends on the fold)
        eq_pos = list(range(X.shape[1], X.shape[1] + len(eqcols)))
        X = np.hstack([X, np.zeros((len(X), len(eqcols)), dtype=X.dtype)])
        names = names + [EQ.FEATURE_NAMES[j] for j in eqcols]
        print(f"equation-forecast features: {len(eqcols)} columns appended (computed per fold)")
    hp_pos = []
    if args.nbm_hpca:   # placeholder columns, filled per fold (the NBM depends on the fold)
        if not args.nbm:
            sys.exit("--nbm-hpca needs --nbm")
        hp_pos = list(range(X.shape[1], X.shape[1] + args.nbm_hpca))
        X = np.hstack([X, np.zeros((len(X), args.nbm_hpca), dtype=X.dtype)])
        names = names + [f"nbh_pc{k}" for k in range(args.nbm_hpca)]
        print(f"NBM hidden-state PCA features: {args.nbm_hpca} columns appended (computed per fold)")
    nsc_pos = []
    if args.nbm_score and nsccols:   # placeholder columns, filled per fold (the NBM depends on the fold)
        if not args.nbm:
            sys.exit("--nbm-score needs --nbm")
        nsc_pos = list(range(X.shape[1], X.shape[1] + len(nsccols)))
        X = np.hstack([X, np.zeros((len(X), len(nsccols)), dtype=X.dtype)])
        names = names + [NSC.FEATURE_NAMES[j] for j in nsccols]
        print(f"NBM score-test features: {len(nsccols)} columns appended (computed per fold)")
    print(f"rows {X.shape[0]:,} x features {X.shape[1]} | positive rate {y.mean():.3f} | series {n_series}")
    print(f"settings: {vars(args)}")
    if args.monotone:
        global MONO
        mv = monotone_vector(names)
        MONO = "(" + ",".join(str(v) for v in mv) + ")"
        up = [nm for nm, v in zip(names, mv) if v]
        print(f"--monotone: {len(up)} columns constrained to raise the score: {', '.join(up)}")
        if nb_pos and any(names[j] == "nb_logodds" for j in nb_pos):
            print("  (nb_ columns are filled per fold; their direction is checked by the fold's training fit only)")
    if args.base_margin and args.base_margin not in names:
        sys.exit(f"--base-margin column {args.base_margin} is not in the feature set")
    parts = []
    orig_no = np.array([len(q[2]) for q in S])
    srcs = [(n_, V_, s_, r_, None) for n_, V_, s_, r_ in extra_sources(S, tau)] + tta_sources(S, tau)
    for name, V, stub, rel, shift in srcs:
        P = build_extra(name, V, stub, rel)
        P["shift"], P["score_only"] = shift, shift is not None
        XA = P["X"]
        if len(keep) < len(F.FEATURE_NAMES):
            XA = XA[:, keep]
        if P["E"] is not None:
            XA = np.hstack([XA, P["E"][:, ev_cols()]])
        if P["VV"] is not None:
            XA = np.hstack([XA, P["VV"][:, vcols]])
        if P["DD"] is not None:
            XA = np.hstack([XA, P["DD"][:, dcols]])
        if P["MM"] is not None:
            XA = np.hstack([XA, P["MM"][:, mcols]])
        if P["BO"] is not None:
            XA = np.hstack([XA, P["BO"][:, bcols]])
        if P["PS"] is not None:
            XA = np.hstack([XA, P["PS"][:, pcols]])
        if P["SH"] is not None:
            XA = np.hstack([XA, P["SH"][:, shcols]])
        if P["RI"] is not None:
            XA = np.hstack([XA, P["RI"][:, ricols]])
        if P["SD"] is not None:
            XA = np.hstack([XA, P["SD"][:, sdcols]])
        if P["BO2"] is not None:
            XA = np.hstack([XA, P["BO2"][:, b2cols]])
        if P["LR"] is not None:
            XA = np.hstack([XA, P["LR"][:, lcols]])
        if P["MS"] is not None:
            XA = np.hstack([XA, P["MS"][:, mscols]])
        if args.nbm:
            XA = np.hstack([XA, np.zeros((len(XA), len(ncols)), dtype=XA.dtype)])
        if args.nbm and args.pshape:
            XA = np.hstack([XA, np.zeros((len(XA), len(pncols)), dtype=XA.dtype)])
        if nbd_pos:
            XA = np.hstack([XA, np.zeros((len(XA), len(nbdcols)), dtype=XA.dtype)])
        if pc_pos:
            XA = np.hstack([XA, np.zeros((len(XA), len(pccols)), dtype=XA.dtype)])
        for kind, (ecols, _) in emb_pos.items():
            XA = np.hstack([XA, np.zeros((len(XA), len(ecols)), dtype=XA.dtype)])
        if eq_pos:
            XA = np.hstack([XA, np.zeros((len(XA), len(eqcols)), dtype=XA.dtype)])
        if hp_pos:
            XA = np.hstack([XA, np.zeros((len(XA), len(hp_pos)), dtype=XA.dtype)])
        if nsc_pos:
            XA = np.hstack([XA, np.zeros((len(XA), len(nsc_pos)), dtype=XA.dtype)])
        P["X"], P["E"], P["VV"], P["DD"], P["MM"], P["BO"], P["BO2"], P["LR"] = XA, None, None, None, None, None, None, None
        P["PS"] = None
        P["MS"] = None
        P["SH"] = None
        P["RI"] = None
        P["SD"] = None
        print(f"distribution check, original series vs {name} series:")
        A.describe(orig_no, tau, P["vlen"], P["vtau"])
        if name == "2025" and args.ext_weight:
            zw = np.load(args.ext_weight)
            n_v = len(P["vlen"])   # --n-series limits the cuts to a prefix of the same deterministic list
            if not np.array_equal(zw["vlen"][:n_v], P["vlen"]):
                sys.exit(f"--ext-weight {args.ext_weight} does not belong to these 2025 cuts (--ext/--ext-seed/--ext-hmin)")
            wv = zw["w"][:n_v]
            P["roww"] = np.repeat(wv, (P["vlen"] + SUB - 1) // SUB)
            print(f"2025: covariate-shift weights from {args.ext_weight} (ESS {float(zw['ess']):,.0f}); series weight quantiles 10/50/90% {np.round(np.percentile(wv, [10, 50, 90]), 3)}")
        parts.append(P)
    DETM = None
    if args.detw > 0:
        t0 = time.time()
        sz_o = np.array(Parallel(n_jobs=THREADS, backend="threading")(
            delayed(det_size)(S[i][1], S[i][2], int(tau[i])) for i in range(n_series)))
        DETM = det_mult_of(sz_o)
        for P in parts:
            sz = np.array(Parallel(n_jobs=THREADS, backend="threading")(
                delayed(det_size)(h, o, int(t)) for (h, o), t in zip(P["items"], P["vtau"])))
            P["detm"] = np.repeat(det_mult_of(sz), (P["vlen"] + SUB - 1) // SUB)
            assert len(P["detm"]) == len(P["y"]), "--detw: re-cut row layout mismatch"
            print(f"--detw: {P['name']}: {(sz < 2).mean():.1%} of variants with multiplier < 1")
        bs = np.isfinite(sz_o)
        print(f"--detw {args.detw:g}: original break series measured {bs.sum():,}; change size < 1 ('quiet' on all 5 axes) "
              f"{(sz_o[bs] < 1).mean():.1%}, < 2 (multiplier < 1) {(sz_o[bs] < 2).mean():.1%}; median size "
              f"{np.median(sz_o[bs]):.1f} ({time.time() - t0:.0f}s)", flush=True)
    if args.objective == "rank" and any(P["rel"] != 1.0 for P in parts):
        print("NOTE: --aug-rel / --synth-rel have no effect with --objective rank (XGBoost ranking weights are per step)")
    base_oof = None
    if args.compare:
        c = np.load(args.compare)
        cm = c["sid"] < n_series
        if not (np.array_equal(c["sid"][cm], sid) and np.array_equal(c["step"][cm], step)):
            sys.exit(f"--compare file {args.compare} has a different row layout")
        base_oof = c["oof"][cm]
    diffs = []
    has_break = (tau >= 0).astype(int)
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    print(f"backend: {args.backend}" + (f" on {args.device}" if args.backend in ("xgb", "cat") else "")
          + f" | objective: {args.objective} | CPU threads: {THREADS}")
    grid = list(range(args.grid, args.rounds + 1, args.grid))
    svm_cols = None
    if args.backend == "svm":
        if args.objective != "binary":
            sys.exit("--backend svm supports --objective binary only")
        svm_cols = svm_columns(names)
        grid = [float(v) for v in args.svm_lams.split(",")]   # 'rounds' = regularisation strengths
        print(f"SVM input columns ({len(svm_cols)}): {[names[j] for j in svm_cols][:12]} ...")
    curves = []
    oof = np.full(len(y), np.nan)
    oof_tta = {w0: np.full(len(y), np.nan) for w0 in TTA_W0}
    tta_res = []
    imp_total = np.zeros(len(names))
    if args.prune_keep:
        if args.prune_keep >= len(names):
            sys.exit(f"--prune-keep {args.prune_keep} keeps every one of the {len(names)} columns")
        i_pr = max(int(np.searchsorted(grid, args.prune_rounds, side="right")) - 1, 0)   # grid round <= --prune-rounds
        p1_curves, p1_vs_base, kept = [], [], []
        print(f"--prune-keep {args.prune_keep}: per fold, pass 1 on all {len(names)} columns, then a refit on the top "
              f"{args.prune_keep} by the gain of pass 1's first {min(args.prune_rounds, args.rounds)} trees; paired "
              f"check at round {grid[i_pr]}", flush=True)

    splits = list(skf.split(np.arange(n_series), has_break))
    if args.lpo >= 0:   # (fold index, feature fold, feature-fold training set, training set, scored set)
        K = args.lpo
        plan = [(i, K, splits[K][0], np.setdiff1d(splits[K][0], splits[i][1]), splits[i][1]) for i in range(5) if i != K]
        print(f"--lpo {K}: 4 fits, each on 6,000 series (folds outside {{{K}, i}}), scoring fold i; fold {K} untouched")
    else:
        plan = [(f, f, tr_s, tr_s, va_s) for f, (tr_s, va_s) in enumerate(splits)][: args.folds]
    for f, fk, ftr_s, tr_s, va_s in plan:
        t0 = time.time()
        tr = np.isin(sid, tr_s)
        va = np.isin(sid, va_s)
        keep = None
        if args.nbm:
            NBx, NPx, NDx = nbm_fold(fk, ftr_s, S, tau, parts)
            X[:, nb_pos] = NBx[0][:, ncols]
            for P, B in zip(parts, NBx[1:]):
                P["X"][:, nb_pos] = B[:, ncols]
            if NPx is not None:
                X[:, nps_pos] = NPx[0][:, pncols]
                for P, B in zip(parts, NPx[1:]):
                    P["X"][:, nps_pos] = B[:, pncols]
            if NDx is not None:
                X[:, nbd_pos] = NDx[0][:, nbdcols]
                for P, B in zip(parts, NDx[1:]):
                    P["X"][:, nbd_pos] = B[:, nbdcols]
            del NBx, NPx, NDx
        if hp_pos:
            HPx = hpca_fold(fk, ftr_s, S, parts, step)
            X[:, hp_pos] = HPx[0]
            for P, B in zip(parts, HPx[1:]):
                P["X"][:, hp_pos] = B
            del HPx
        if nsc_pos:
            NSx = nsc_fold(fk, ftr_s, S, parts, step)
            X[:, nsc_pos] = NSx[0][:, nsccols]
            for P, B in zip(parts, NSx[1:]):
                P["X"][:, nsc_pos] = B[:, nsccols]
            del NSx
        if pc_pos:
            PCx = pc_fold(fk, ftr_s, S, tau, parts)
            X[:, pc_pos] = PCx[0][:, pccols]
            for P, B in zip(parts, PCx[1:]):
                P["X"][:, pc_pos] = B[:, pccols]
            del PCx
        for kind, (ecols, pos) in emb_pos.items():
            EBx = emb_fold(kind, fk, ftr_s, S, tau, parts)
            X[:, pos] = EBx[0][:, ecols]
            for P, B in zip(parts, EBx[1:]):
                P["X"][:, pos] = B[:, ecols]
            del EBx
        if eq_pos:
            EQx = eq_fold(fk, ftr_s, S, parts)
            assert len(EQx[0]) == len(X) and all(len(B) == len(P["X"]) for P, B in zip(parts, EQx[1:]))
            X[:, eq_pos] = EQx[0][:, eqcols]
            for P, B in zip(parts, EQx[1:]):
                P["X"][:, eq_pos] = B[:, eqcols]
            del EQx
        # originals + extra series whose PARENT is a training series, in one preallocated matrix
        ti_o = np.flatnonzero(tr)
        sel = [np.empty(0, dtype=np.int64) if P["score_only"] else np.flatnonzero(np.isin(P["parent"], tr_s) | (P["parent"] < 0))
               for P in parts]   # --tta views never train
        n_o = len(ti_o)
        Xtr = np.empty((n_o + sum(len(q) for q in sel), X.shape[1]), dtype=np.float32)
        np.take(X, ti_o, axis=0, out=Xtr[:n_o], mode="clip")  # 'clip' writes in place ('raise' buffers a copy)
        ys, ss, rels, tis, o = [y[ti_o]], [step[ti_o]], [np.ones(n_o)], [tau[sid[ti_o]]], n_o
        for P, q in zip(parts, sel):
            np.take(P["X"], q, axis=0, out=Xtr[o:o + len(q)], mode="clip")
            o += len(q)
            ys.append(P["y"][q])
            ss.append(P["step"][q])
            rels.append(np.full(len(q), P["rel"]) * (P["roww"][q] if "roww" in P else 1.0))
            tis.append(P["ti"][q])
        ytr, str_, rel, tir = np.concatenate(ys), np.concatenate(ss), np.concatenate(rels), np.concatenate(tis)
        if args.ext_balance and any(P["name"] == "2025" for P in parts):
            src = np.concatenate([np.zeros(n_o, bool)] + [np.full(len(q), P["name"] == "2025") for P, q in zip(parts, sel)])
            T = int(str_.max()) + 1
            cnt = lambda m: np.bincount(str_[m], weights=rel[m], minlength=T)
            Ps, Ns = cnt(src & (ytr == 1)), cnt(src & (ytr == 0))
            Po, No = cnt(~src & (ytr == 1)), cnt(~src & (ytr == 0))
            mult = np.clip((Ps / np.maximum(Po, 1e-12)) / np.maximum(Ns / np.maximum(No, 1e-12), 1e-12), 0.0, 3.0)
            neg = src & (ytr == 0)
            rel = rel.copy()
            rel[neg] *= mult[str_[neg]]
            print(f"fold {f}: --ext-balance: 2025 negative rows re-weighted, median factor {np.median(mult[str_[neg]]):.2f}", flush=True)
        is_orig = np.arange(len(ytr)) < n_o
        bm_va = None
        Xva_ = X[va]
        if args.xsec:
            Xva_ = xsec_fold(f, Xtr, str_, Xva_, step[va], names)
        if parts:
            print(f"fold {f}: training rows {n_o:,} original + "
                  + " + ".join(f"{len(q):,} {P['name']}" for P, q in zip(parts, sel)), flush=True)
        if args.objective == "rank":
            m = fit_rank(Xtr, ytr, str_, is_orig)
        else:
            if args.no_weights:
                w = None
            elif parts:
                w = A.step_weights_mixed(ytr, str_, is_orig, rel)
            else:
                w = step_weights(ytr, str_)          # identical to the original code path
            if args.near_k > 0:
                if w is None:
                    sys.exit("--near-k needs weights (drop --no-weights)")
                nm_ = near_mult(ytr, str_, tir)
                print(f"fold {f}: --near-k {args.near_k} ({args.near_mode}, floor {args.near_floor:g}): "
                      f"{100 * (nm_ < 1).mean():.1f}% of training rows down-weighted, mean multiplier "
                      f"{nm_[nm_ < 1].mean() if (nm_ < 1).any() else 1:.2f}", flush=True)
                w = w * nm_
                w = w / max(w.mean(), 1e-12)
            if DETM is not None:
                if w is None:
                    sys.exit("--detw needs weights (drop --no-weights)")
                dm = np.concatenate([DETM[sid[ti_o]]] + [P["detm"][q] for P, q in zip(parts, sel)])
                w = det_apply(w, ytr, str_, dm, f)
            if args.backend == "svm":
                best_sc, best_raw, kb, curve = svm_fold(Xtr, ytr, w, str_, Xva_, y[va], step[va], svm_cols, names, f)
                best_k = grid[kb]
                m = None
            else:
                bm_tr = None
                if args.base_margin:
                    bm_tr, bm_va = base_margin_fold(f, Xtr, ytr, w if w is not None else np.ones(len(ytr)), str_,
                                                    Xva_, step[va], names)
                    print(f"fold {f}: --base-margin alone on validation: TS-AUC {ts_auc(bm_va, y[va], step[va]):.4f}",
                          flush=True)
                m = fit(Xtr, ytr, w, bm_tr)
                del bm_tr
                if args.prune_keep:
                    c1 = curve_of(m, valid_obj(Xva_), y[va], step[va], grid)
                    keep, g1 = prune_columns(m, len(names))
                    p1_curves.append(c1)
                    kept.append(keep)
                    dropped = np.setdiff1d(np.argsort(-g1, kind="stable")[:len(keep) + 5], keep)
                    print(f"fold {f}: prune pass 1 (all {len(names)} columns): best TS-AUC {c1.max():.4f} at round "
                          f"{grid[int(np.argmax(c1))]} | at round {grid[i_pr]} {c1[i_pr]:.4f}; refit on the top {len(keep)} "
                          f"({100 * g1[keep].sum() / max(g1.sum(), 1e-12):.1f}% of the ranking gain; "
                          f"{int((g1 == 0).sum())} columns unused); first dropped: "
                          f"{', '.join(names[j] for j in dropped)}  ({time.time() - t0:.0f}s)", flush=True)
                    del m
                    m = fit(Xtr[:, keep], ytr, w)   # fancy indexing: a C-contiguous copy of the kept columns
            del w
        del Xtr, ytr, str_, rel, is_orig, tir
        if m is not None:
            vobj = valid_obj(Xva_ if keep is None else Xva_[:, keep])
            # --base-margin: the starting logit is added once here (the validation DMatrix carries none, so the
            # per-chunk predictions stay tree sums only)
            raw = bm_va.astype(np.float64) if bm_va is not None else np.zeros(int(va.sum()))
            prev = 0
            curve = []
            best_sc, best_raw = -1.0, None
            for k in grid:
                raw = raw + predict_chunk(m, vobj, prev, k)
                prev = k
                sc = ts_auc(raw, y[va], step[va])
                curve.append(sc)
                if sc > best_sc:
                    best_sc, best_raw, best_k = sc, raw.copy(), k
            if keep is None:
                imp_total += gain(m, len(names))
            else:   # refit columns f0..f{K-1} are keep[0..K-1]; pruned columns stay at 0
                imp_total[keep] += gain(m, len(keep))
        oof[va] = best_raw
        if TTA_SHIFTS and m is not None:
            tta_fold(f, m, best_k, Xva_, va, va_s, parts, y, step, sid, oof_tta, tta_res)
        curves.append(curve)
        print(f"fold {f}: best TS-AUC {best_sc:.4f} at round {best_k} | at last round {curve[-1]:.4f} | {time.time() - t0:.0f}s", flush=True)
        if base_oof is not None:
            b = ts_auc(base_oof[va], y[va], step[va])
            diffs.append(best_sc - b)
            print(f"fold {f}: baseline {b:.4f} -> this run {best_sc:.4f}   paired difference {best_sc - b:+.4f}", flush=True)
            if args.prune_keep:
                p1_vs_base.append(p1_curves[-1].max() - b)
        if args.prune_keep:
            print(f"fold {f}: prune at round {grid[i_pr]}: pass 1 {p1_curves[-1][i_pr]:.4f} -> refit {curve[i_pr]:.4f}   "
                  f"paired difference {curve[i_pr] - p1_curves[-1][i_pr]:+.4f}", flush=True)

    curves = np.array(curves)
    mean_curve = curves.mean(axis=0)
    kbest = int(np.argmax(mean_curve))
    print("\nTS-AUC vs boosting rounds, averaged over folds:")
    print("  " + "  ".join(f"{k}:{v:.4f}" for k, v in zip(grid, mean_curve)))
    sd = curves[:, kbest].std() if len(curves) > 1 else float("nan")
    print(f"\nHEADLINE  mean TS-AUC = {mean_curve[kbest]:.4f}  (std over folds {sd:.4f}) at {grid[kbest]} rounds")
    if diffs:
        d = np.array(diffs)
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
        print(f"PAIRED vs {args.compare} (per-fold best round in both): {d.mean():+.4f} +- {se:.4f} (standard error), "
              f"{int((d > 0).sum())}/{len(d)} folds better")
    if args.prune_keep:
        c1s = np.array(p1_curves)
        d = curves[:, i_pr] - c1s[:, i_pr]
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
        print(f"PRUNE K={args.prune_keep}: paired at round {grid[i_pr]} vs the in-run all-column pass 1 (reported, not the "
              f"gate): {d.mean():+.4f} +- {se:.4f}, {int((d > 0).sum())}/{len(d)} folds better")
        if p1_vs_base:
            print(f"PRUNE determinism check: pass 1 per-fold best minus the --compare file's: max |diff| "
                  f"{np.abs(p1_vs_base).max():.2e} (0 = pass 1 reproduces the baseline run)")
        sets = [set(k.tolist()) for k in kept]
        print(f"PRUNE column sets over folds: kept in every fold {len(set.intersection(*sets))}, in any fold "
              f"{len(set.union(*sets))} (of {len(names)})")
        with open(os.path.join(OUT, f"prune_keep{TAG}.txt"), "w", encoding="utf-8") as fh:
            for f, k in enumerate(kept):
                fh.write(f"fold {f}: " + ",".join(names[j] for j in k) + "\n")

    if tta_res:
        tta_report(tta_res, oof_tta, y, step, sid)
    done = ~np.isnan(oof)
    print("\nOOF TS-AUC by online-step bucket (pooled over folds run):")
    for lo, hi in ((0, 100), (100, 300), (300, 600), (600, 10000)):
        mk = done & (step >= lo) & (step < hi)
        if mk.sum() > 0:
            print(f"  steps [{lo:4d}, {hi:5d}):  {ts_auc(oof[mk], y[mk], step[mk]):.4f}   ({int(mk.sum()):,} rows)")

    if args.backend == "svm":   # no tree importances; keep importance files for the trees only
        np.savez(os.path.join(OUT, f"oof{TAG}.npz"), oof=oof, y=y, step=step, sid=sid)
        print(f"\nsaved {OUT}/oof{TAG}.npz")
        return
    imp = imp_total / max(imp_total.sum(), 1e-12)
    order = np.argsort(-imp)
    print("\nTop 25 features by gain share (%):")
    for j in order[:25]:
        print(f"  {names[j]:24s} {100 * imp[j]:5.2f}")
    fam, inc = {}, {}
    for j, nm in enumerate(names):
        parts = nm.split("_", 1)
        if parts[0] in ("full", "glrmax", "glrmin", "scanlogL", "scaneff") or parts[0].startswith(("win", "kwin")):
            fam[parts[0]] = fam.get(parts[0], 0) + imp[j]
            inc[parts[1]] = inc.get(parts[1], 0) + imp[j]
        else:
            fam["other"] = fam.get("other", 0) + imp[j]
    print("\nImportance by feature family (%):", {k: round(100 * v, 1) for k, v in sorted(fam.items(), key=lambda kv: -kv[1])})
    print("Importance by increment series, top 12 (%):", [(k, round(100 * v, 1)) for k, v in sorted(inc.items(), key=lambda kv: -kv[1])[:12]])
    with open(os.path.join(OUT, f"importance{TAG}.csv"), "w", encoding="utf-8") as fh:
        fh.write("feature,gain_share\n")
        for j in order:
            fh.write(f"{names[j]},{imp[j]:.6f}\n")
    np.savez(os.path.join(OUT, f"oof{TAG}.npz"), oof=oof, y=y, step=step, sid=sid)
    print(f"\nsaved {OUT}/importance{TAG}.csv and {OUT}/oof{TAG}.npz")


if __name__ == "__main__":
    main()
