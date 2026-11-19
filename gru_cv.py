"""GRU sequence model for per-step break probability, cross-validated with TS-AUC.

Run:    python gru_cv.py --folds 1 --epochs 15                  # quick first look (GPU: a few minutes)
        python gru_cv.py --folds 5 --epochs 15 --tag base        # full cross-validation
        python gru_cv.py --folds 1 --ramp 30 --tag ramp30        # a sensitivity variant (see below)
        python gru_cv.py --folds 5 --blend cv_out/oof_loc.npz    # blend with the LightGBM run from cv.py
Output: cv_out/gru_summary<tag>.txt, cv_out/oof_gru<tag>.npz (same row layout as cv.py's oof files);
        with --arch tcn: cv_out/tcn_summary<tag>.txt, cv_out/oof_tcn<tag>.npz

What the network sees at every step (computed from the HISTORY-fitted AR(10) predictor, so everything is
causal and cheap to reproduce at inference):
    z, AR(10) residual e, e^2-1, |e|-0.8, e_t*e_t-1, 1{|e|>3}         (6 channels)
    history descriptors: log n, skew, kurtosis, lag-1 autocorr, AR phi, log error scale (6 constant channels)
    is_online flag and log(1+t)                                        (2 channels)
  optional extra channels (flags you can also sweep):
    --tails         7 tail indicators: 1{|e|>2.5, 3.5, 4, 5} and 1{|z|>3, 4, 5}
    --feat-top K    the K most important streaming features of the LightGBM model (from cv_out/importance_loc.csv),
                    computed by the same kernel the submission uses, so inference stays cheap
    --vol           3 volatility-state channels (sbrt/volstate.py): the residual measured against a frozen variance
                    forecast from the end of the history, its square, and the log forecast
The last --burn history steps are fed first (no loss) so the hidden state starts from "normal".

SENSITIVITY. TS-AUC only compares the RANK of scores across series at the same step, so a decision threshold
or any monotone rescaling of the output changes nothing. How eagerly the model reacts is decided in training:
    --ramp R        positives right after a break are nearly indistinguishable from normal data; their loss
                    weight ramps up over R steps (R=0: off). Larger R = learn from clear evidence only.
    --pos-boost B   multiply the weight of positive rows (B>1: more eager / more false alarms, B<1: more cautious)
    --mem-bias M    initial bias of the GRU update gate: larger M = longer memory = accumulates evidence over
                    more steps (like a smaller EWMA decay), smaller M = reacts to recent points
    --focal G       focal loss exponent: concentrate on hard rows (G=0: off)
    --hidden/--layers/--dropout/--wd   capacity and regularisation
    --burn          how much history the state is warmed up on
Judge every variant by TS-AUC overall AND by step range (printed): early-step and late-step behaviour move differently.
"""
import argparse
import copy
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hashlib

import numpy as np
from joblib import Parallel, delayed
from scipy.signal import lfilter
from sklearn.model_selection import StratifiedKFold

from sbrt import augment as A
from sbrt import synth as SY
from sbrt import external as XT
from sbrt import features as F
from sbrt import volstate as VS
from sbrt import evidence as EV
from sbrt import disttest as DT
from sbrt import bocpd as BO
from sbrt import nbm as NBM
from sbrt.data import load_train
from sbrt.metric import ts_auc

ap = argparse.ArgumentParser()
ap.add_argument("--folds", type=int, default=1)
ap.add_argument("--epochs", type=int, default=15)
ap.add_argument("--hidden", type=int, default=64)
ap.add_argument("--layers", type=int, default=1)
ap.add_argument("--dropout", type=float, default=0.1)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--wd", type=float, default=1e-4)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--burn", type=int, default=100)
ap.add_argument("--ramp", type=int, default=0)
ap.add_argument("--pos-boost", type=float, default=1.0)
ap.add_argument("--mem-bias", type=float, default=1.0)
ap.add_argument("--focal", type=float, default=0.0)
ap.add_argument("--no-ctx", action="store_true", help="zero the 6 history-descriptor channels")
ap.add_argument("--tails", action="store_true",
                help="add 7 tail-indicator channels: 1{|e|>2.5, 3.5, 4, 5} and 1{|z|>3, 4, 5}")
ap.add_argument("--feat-top", type=int, default=0,
                help="add the top-K streaming features of the LightGBM model (ranked by its importance file) "
                     "as extra input channels at every online step")
ap.add_argument("--feat-file", type=str, default=os.path.join("cv_out", "importance_loc.csv"),
                help="importance csv written by cv.py (feature,gain_share), used by --feat-top")
ap.add_argument("--feat-src", choices=["kernel", "all"], default="kernel",
                help="where --feat-top may pick its features: kernel = the 270 kernel features (as before); all = also "
                     "the evidence (6 compact), volatility-state (13) and distribution-test (16, no dist_z_) features, "
                     "e.g. with --feat-file cv_out/importance_aug2_evc_vol_df.csv (the current best trees)")
ap.add_argument("--patience", type=int, default=4)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
                help="auto = GPU if available (with a loud warning if not); cuda = require the GPU")
ap.add_argument("--n-series", type=int, default=0, help="debug: use only this many series")
ap.add_argument("--blend", type=str, default="")
ap.add_argument("--tag", type=str, default="")
ap.add_argument("--threads", type=int, default=8, help="CPU threads for building inputs (one-time, then cached)")
ap.add_argument("--aug", type=int, default=0,
                help="boundary re-cut variants per training series (sbrt/augment.py; 0 = off). Variants train only "
                     "with their parent series; validation and the OOF file use original series only")
ap.add_argument("--aug-seed", type=int, default=0)
ap.add_argument("--aug-min-shift", type=int, default=20)
ap.add_argument("--aug-trim", type=float, default=0.0,
                help="probability that a re-cut variant also drops the START of its history (new length drawn from "
                     "the original history-length distribution; labels unchanged)")
ap.add_argument("--aug-trim-only", type=int, default=0,
                help="extra variants per training series with the original boundary/online segment and a history "
                     "trimmed by at least 20%%")
ap.add_argument("--aug-rel", type=float, default=1.0, help="weight of a variant row relative to an original row")
ap.add_argument("--compare", type=str, default="",
                help="OOF file of a baseline GRU run (e.g. cv_out/oof_gru_sw_2348ae47_cv5.npz): paired per-fold difference")
ap.add_argument("--synth", type=int, default=0, help="synthetic-break pseudo-series per training series (sbrt/synth.py; 0 = off)")
ap.add_argument("--synth-scale", type=float, default=0.5, help="size scale of synthetic breaks (synth_calibrate.py)")
ap.add_argument("--synth-pzero", type=float, default=0.25)
ap.add_argument("--synth-seed", type=int, default=0)
ap.add_argument("--synth-rel", type=float, default=1.0, help="weight of a synthetic row relative to an original row")
ap.add_argument("--synth-types", type=str, default="all",
                help="comma list of break types to inject (variance,mean,dependence,tails,skew,volclust,splice) or all")
ap.add_argument("--ext", type=int, default=0,
                help="cuts per 2025 offline-edition series added as extra REAL training series (sbrt/external.py; "
                     "0 = off; run ext2025.py once first)")
ap.add_argument("--ext-dir", type=str, default=os.path.join("data", "ext2025"))
ap.add_argument("--ext-seed", type=int, default=0)
ap.add_argument("--ext-hmin", type=int, default=1000, help="shortest history allowed when cutting a 2025 series")
ap.add_argument("--ext-rel", type=float, default=1.0, help="weight of a 2025 row relative to an original row")
ap.add_argument("--rank-loss", type=float, default=0.0,
                help="share of a pairwise ranking loss (0..1): within each batch, every positive vs every negative "
                     "series at the SAME step (what TS-AUC compares); the rest stays weighted cross-entropy")
ap.add_argument("--aux-tsb", type=float, default=0.0,
                help="weight of an auxiliary head predicting log(1 + time since break) on positive steps")
ap.add_argument("--aux-type", type=float, default=0.0,
                help="weight of an auxiliary head classifying the break type on positive steps of synthetic series "
                     "(needs --synth; real breaks have no type label)")
ap.add_argument("--gate", choices=["none", "prior", "static", "dynamic"], default="none",
                help="per-channel input weights whose RATIOS are what matter (they average to 1): "
                     "prior = fixed sqrt(importance ratio) of the tree model for the top-K feature channels; "
                     "static = learned per series from its 6 history descriptors (which statistics to trust depends "
                     "on the kind of series); dynamic = learned per series and step from all inputs. "
                     "Learned gates start at equal weights, so training starts from the ungated model")
ap.add_argument("--gate-hidden", type=int, default=32)
ap.add_argument("--bocpd", action="store_true",
                help="add the 10 BOCPD posterior features (sbrt/bocpd.py) as input channels: signed log, standardised "
                     "on every 10th original series, clipped to +-6, zeros in the burn-in; placed after the volatility "
                     "channels and before the top-K feature channels")
ap.add_argument("--vol", action="store_true",
                help="add 3 volatility-state channels (sbrt/volstate.py): u = AR residual / frozen end-of-history "
                     "variance forecast, min(u^2, 25) - 1, and log of that forecast (in-sample GARCH in the burn-in)")
ap.add_argument("--nbm", action="store_true",
                help="add the 6 learned normal-behaviour-model features (sbrt/nbm.py) as input channels, computed PER FOLD "
                     "exactly as cv.py --nbm (same pretrained fold weights data/nbm_v1_cv_f<f>_<cfg>.pt, prior from the "
                     "fold's training breaks), every online step; signed log, standardised on every 10th original "
                     "training series of the fold, clipped +-6, zeros in the burn-in; placed after the BOCPD channels")
ap.add_argument("--nbm-epochs", type=int, default=12)   # the --nbm-* defaults and cache names mirror cv.py
ap.add_argument("--nbm-hidden", type=int, default=64)
ap.add_argument("--nbm-lr", type=float, default=2e-3)
ap.add_argument("--nbm-batch", type=int, default=48)
ap.add_argument("--nbm-wd", type=float, default=1e-4)
ap.add_argument("--nbm-loss-from", type=int, default=50)
ap.add_argument("--nbm-multi", type=int, default=1,
                help="NBM pretrained with heads for 1..M steps ahead, only the 1-step head kept (HANDOFF 41)")
ap.add_argument("--nbm-hpca", type=int, default=0,
                help="add K channels: the fold NBM's 64-d GRU hidden state after each online point (causal), projected on its first K principal components (PCA per fold on the online parts of every 10th original training series, no labels), standardised, clipped +-6, zeros in the burn-in; placed right after the 6 NBM channels. Needs --nbm. HANDOFF section 24")
ap.add_argument("--arch", choices=["gru", "tcn", "qattn"], default="gru",
                help="sequence core: gru (as before) or tcn = causal dilated 1-D convolutions (residual blocks of two "
                     "convs, dilations 1, 2, 4, ... 2^(levels-1)) plus a causal running mean of the block output, "
                     "projected back to --hidden; same input layer and heads. OOF file oof_tcn<tag>.npz. "
                     "qattn = query attention (HANDOFF section 39): per step, learned feature queries attend over "
                     "tokens of the channel groups; a causal transformer over the steps (ALiBi recency heads); "
                     "learned break queries attend over all steps <= t. OOF file oof_qattn<tag>.npz")
ap.add_argument("--qa-d", type=int, default=64, help="qattn: token width")
ap.add_argument("--qa-heads", type=int, default=4, help="qattn: attention heads (ALiBi windows ~8, 32, 128, all)")
ap.add_argument("--qa-layers", type=int, default=2, help="qattn: causal transformer blocks over the steps")
ap.add_argument("--qa-queries", type=int, default=4, help="qattn: learned break queries")
ap.add_argument("--qa-fq", type=int, default=2, help="qattn: learned feature queries per step")
ap.add_argument("--qa-group", type=int, default=5, help="qattn: kernel features per token (1 = one token each)")
ap.add_argument("--warmup", type=int, default=0,
                help="linear learning-rate warmup over the first N optimiser steps (~375 per epoch with --aug 2), "
                     "multiplied onto the usual epoch cosine; 0 = off (as before). HANDOFF section 39")
ap.add_argument("--tcn-kernel", type=int, default=3)
ap.add_argument("--tcn-levels", type=int, default=9, help="receptive field 1 + 2*(kernel-1)*(2^levels - 1) steps")
ap.add_argument("--swa", type=str, default="",
                help="A-B: also score each fold with (1) the WEIGHTS averaged over the models after epochs A..B "
                     "(equal weights) and (2) the LOGITS of those epochs averaged, and save the OOF of the fixed "
                     "epoch --fixed-ep; none of the three looks at the validation fold. No early stop: training "
                     "stops after max(B, --fixed-ep). Extra OOF files oof_<arch><tag>_wavg / _lavg / _ep<E>.npz; "
                     "the main OOF stays the best epoch (selected on the validation fold). HANDOFF section 29")
ap.add_argument("--fixed-ep", type=int, default=7, help="with --swa: the fixed epoch saved as the production-like "
                                                         "reference (= submission_template GRU_EPOCHS)")
ap.add_argument("--nbm-joint", type=float, default=0.0,
                help="joint fine-tuning (HANDOFF section 42): the fold NBM becomes a trainable part of the detector. "
                     "Its hidden-state PCA channels (--nbm-hpca; the PCA basis stays fixed) are recomputed every batch "
                     "from the current NBM weights, so the detector loss trains the NBM too. Value = the NBM's "
                     "learning rate as a multiple of --lr (0 = off, as before). Truncated backprop: the history except "
                     "its last --burn rows runs without gradient. The 6 nb_ channels stay on the frozen NBM")
ap.add_argument("--nbm-joint-nll", type=float, default=0.2,
                help="with --nbm-joint: weight of the NBM's own Student-t NLL (next AR residual, as in its "
                     "pretraining) on the pre-break rows of the gradient window; keeps it a normal-behaviour model")
ap.add_argument("--nbm-hist", choices=["none", "raw", "white"], default="none",
                help="needs --nbm-hpca. raw: also fill the K hidden-state channels in the 100-step history burn-in with the "
                     "NBM's real history states (same PCA / scaling), so the GRU sees the state before the online part. "
                     "white: as raw, and every hidden-state channel (burn-in and online) is re-expressed relative to "
                     "the series' own history states (minus their mean, over their sd floored at 0.2). HANDOFF section 46")
ap.add_argument("--nbm-gates", action="store_true",
                help="add 3 channels from the NBM's GRU gates after each point: mean share of memory overwritten "
                     "(1 - update gate), mean reset gate, log size of the state change; standardised per fold on "
                     "training online parts, clipped +-6; filled in the burn-in too when --nbm-hist is not none. Needs "
                     "--nbm. HANDOFF section 46")
ap.add_argument("--distill", type=float, default=0.0,
                help="distillation weight a: on ORIGINAL training series the target becomes (1 - a) * label + a * p, "
                     "p = the teacher blend (--distill-file) Platt-calibrated per fold on that fold's training rows "
                     "with the metric's class-balanced step weights, interpolated between the every-3rd-step rows. "
                     "Re-cut / extra series keep hard labels. HANDOFF section 44")
ap.add_argument("--distill-file", type=str, default=os.path.join("cv_out", "teacher_blend.npz"),
                help="teacher scores from distill_teacher.py (z per sid/step row, same layout as the OOF files)")
args = ap.parse_args()
SWA = bool(args.swa)
if SWA:
    SWA_A, SWA_B = (int(v) for v in args.swa.split("-"))
    if not (1 <= SWA_A <= SWA_B):
        sys.exit("--swa needs A-B with 1 <= A <= B")
    SWA_END = max(SWA_B, args.fixed_ep)
    if SWA_END > args.epochs:
        sys.exit("--swa / --fixed-ep go past --epochs")
    SWA_KEYS = ("wavg", "lavg", f"ep{args.fixed_ep}")

import torch
import torch.nn as nn

torch.set_float32_matmul_precision("high")  # TF32 matmuls on Ampere+ GPUs
torch.set_num_threads(max(1, min(args.threads, 4)))  # the GPU does the training; keep CPU-side torch light
G = {}  # device-resident data (inputs, labels, weights), filled in main()
TAG = ("_" + args.tag) if args.tag else ""
OUT = "cv_out"
os.makedirs(OUT, exist_ok=True)
SUB = 3        # same step subsampling as cv.py, so the numbers are comparable
AR = 10
CIN = (6 + 6 + 2 + (7 if args.tails else 0) + (VS.N_GRU if args.vol else 0) + (BO.N_BO if args.bocpd else 0)
       + (NBM.N_NB if args.nbm else 0) + args.nbm_hpca + (NBM.N_GATE if args.nbm_gates else 0) + args.feat_top)
CTX0 = 6 + (7 if args.tails else 0)          # first of the 6 history-descriptor channels (see build_inputs)
NB0 = CTX0 + 6 + 2 + (VS.N_GRU if args.vol else 0) + (BO.N_BO if args.bocpd else 0)   # first NBM channel (--nbm)
HP0 = NB0 + NBM.N_NB                          # first NBM hidden-state PCA channel (--nbm-hpca)
NG = NBM.N_GATE if args.nbm_gates else 0
GT0 = HP0 + args.nbm_hpca                     # first NBM gate-activity channel (--nbm-gates)
if args.nbm_hist != "none" and not args.nbm_hpca:
    sys.exit("--nbm-hist needs --nbm-hpca")
if args.nbm_gates and not args.nbm:
    sys.exit("--nbm-gates needs --nbm")
if args.nbm_hpca and not args.nbm:
    sys.exit("--nbm-hpca needs --nbm")
JOINT = args.nbm_joint > 0
if JOINT and (args.nbm_hist != "none" or args.nbm_gates):
    sys.exit("--nbm-joint cannot be combined with --nbm-hist / --nbm-gates")
if JOINT and not args.nbm_hpca:
    sys.exit("--nbm-joint needs --nbm-hpca")


class Tee:
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")

    def write(self, s):
        sys.__stdout__.write(s)
        self.f.write(s)

    def flush(self):
        sys.__stdout__.flush()
        self.f.flush()


ARCH = args.arch   # output-name prefix: oof_gru<tag> / oof_tcn<tag> / oof_qattn<tag>
sys.stdout = Tee(os.path.join(OUT, f"{ARCH}_summary{TAG}.txt"))


# ---------------------------------------------------------------------------------------------
# inputs
def build_inputs(h, o, burn):
    h = np.asarray(h, dtype=np.float64)
    o = np.asarray(o, dtype=np.float64)
    mu, sd = h.mean(), h.std() + 1e-12
    zh, zo = (h - mu) / sd, (o - mu) / sd
    s = np.r_[zh, zo]
    nh = len(h)
    Xl = np.column_stack([zh[AR - 1 - i: nh - 1 - i] for i in range(AR)])
    coef = np.linalg.lstsq(Xl, zh[AR:], rcond=None)[0]
    e = lfilter(np.r_[1.0, -coef], [1.0], s)
    se = max(float(e[2 * AR:nh].std()), 1e-8)
    r = e / se
    st = nh - burn
    rr = r[st:]
    rp = np.r_[r[st - 1], rr[:-1]]
    ch = np.stack([
        np.clip(s[st:], -8, 8),
        np.clip(rr, -8, 8),
        np.minimum(rr ** 2, 25.0) - 1.0,
        np.minimum(np.abs(rr), 8.0) - 0.8,
        np.clip(rr * rp, -25, 25),
        (np.abs(rr) > 3.0).astype(np.float64),
    ], axis=1)
    if args.tails:  # sharp threshold features a GRU finds hard to build from raw residuals
        zc = s[st:]
        tl_ind = np.stack([np.abs(rr) > 2.5, np.abs(rr) > 3.5, np.abs(rr) > 4.0, np.abs(rr) > 5.0,
                           np.abs(zc) > 3.0, np.abs(zc) > 4.0, np.abs(zc) > 5.0], axis=1).astype(np.float64)
        ch = np.hstack([ch, tl_ind])
    if args.no_ctx:
        ctx = np.zeros(6)
    else:
        rho1 = float(np.dot(zh[:-1], zh[1:]) / max(float(np.dot(zh, zh)), 1e-12))
        ctx = np.array([np.log(nh) - 8.0, np.clip(np.mean(zh ** 3), -5, 5) / 2.0,
                        np.log1p(max(float(np.mean(zh ** 4)) - 3.0, 0.0)) - 1.0, rho1, coef[0], np.log(se)])
    T = len(o)
    online = np.r_[np.zeros(burn), np.ones(T)]
    tl = np.r_[np.zeros(burn), np.log1p(np.arange(T)) / 7.0]
    X = np.hstack([ch, np.tile(ctx, (burn + T, 1)), online[:, None], tl[:, None]])
    return np.nan_to_num(X).astype(np.float32)


def add_vol_channels(Xs, items):
    """--vol: append the 3 volatility-state channels to every input matrix. They go after log_t and BEFORE the top-K
    feature channels, so CTX0 (static gate) and the 'last feat_top channels' rule of the prior gate stay valid."""
    if not args.vol:
        return Xs
    t0 = time.time()
    vc = Parallel(n_jobs=args.threads, backend="threading")(delayed(VS.gru_channels)(h, o, args.burn) for h, o in items)
    print(f"volatility-state channels for {len(items):,} series in {time.time() - t0:.0f}s")
    return [np.hstack([x, v]) for x, v in zip(Xs, vc)]


def vol_stub():
    return (f"_vol{VS.VERSION}" if args.vol else "") + (f"_bo{BO.VERSION}" if args.bocpd else "")


def _bo_raw(h, o):
    M = BO.run_series_bocpd(h, o)
    return (np.sign(M) * np.log1p(np.abs(M))).astype(np.float32)


BO_STATS = {}


def add_bo_channels(Xs, items, ref_items):
    """--bocpd: append the 10 BOCPD channels (standardisation constants from every 10th ORIGINAL series)."""
    if not args.bocpd:
        return Xs
    t0 = time.time()
    if "mu" not in BO_STATS:
        ref = Parallel(n_jobs=args.threads, backend="threading")(delayed(_bo_raw)(h, o) for h, o in ref_items[::10])
        sample = np.concatenate(ref)
        BO_STATS["mu"], BO_STATS["sd"] = sample.mean(axis=0), sample.std(axis=0) + 1e-6
    raws = Parallel(n_jobs=args.threads, backend="threading")(delayed(_bo_raw)(h, o) for h, o in items)
    out = []
    for x, r in zip(Xs, raws):
        fe = np.clip((r - BO_STATS["mu"]) / BO_STATS["sd"], -6, 6).astype(np.float32)
        out.append(np.hstack([x, np.vstack([np.zeros((args.burn, BO.N_BO), np.float32), fe])]))
    print(f"BOCPD channels for {len(items):,} series in {time.time() - t0:.0f}s")
    return out


def nbm_cfg():
    """Same configuration string as cv.py, so both scripts share the per-fold NBM weight files."""
    cfg = f"e{args.nbm_epochs}"
    if (args.nbm_hidden, args.nbm_lr, args.nbm_batch, args.nbm_wd, args.nbm_loss_from) != (64, 2e-3, 48, 1e-4, 50):
        cfg += f"_h{args.nbm_hidden}_lr{args.nbm_lr:g}_b{args.nbm_batch}_wd{args.nbm_wd:g}_lf{args.nbm_loss_from}"
    if args.nbm_multi != 1:   # new name only when used, so the existing caches keep theirs
        cfg += f"_m{args.nbm_multi}"
    return cfg


def _nbm_raw(sums, prior):
    M = NBM.features_from_sums(sums, prior, 1)
    return (np.sign(M) * np.log1p(np.abs(M))).astype(np.float32)


def nbm_fold_raw(f, tr_orig, S, tau_orig, sources):
    """--nbm: fold-f NBM features (signed log, every online step) for every original series, then every extra
    source in the order of the device rows. Weights: the fold's pretrained NBM, shared with cv.py --nbm (trained here
    the same way if missing). Prior: fitted on the fold's original training breaks, as in cv.py. Cached per fold."""
    use_cache = args.n_series == 0
    cfg = nbm_cfg()
    stub = "".join(st for _, _, st in sources)
    cache = os.path.join("data", f"gru_nbm{NBM.VERSION}_f{f}_{cfg}{stub}.npz")
    count = len(S) + sum(len(it) for _, it, _ in sources)
    if use_cache and os.path.exists(cache):
        z = np.load(cache)
        if int(z["n"]) == count:
            flat, off = z["flat"], z["off"]
            print(f"fold {f}: NBM channels loaded from {cache}", flush=True)
            return [flat[off[i]:off[i + 1]] for i in range(len(off) - 1)]
    t0 = time.time()
    wpath = os.path.join("data", f"nbm_v{NBM.VERSION}_cv_f{f}_{cfg}.pt")
    if use_cache and os.path.exists(wpath):
        state = torch.load(wpath, map_location="cpu")
    else:
        rs = np.random.default_rng(1000 + f)
        trp = rs.permutation(tr_orig)
        nva = max(1, len(trp) // 10)
        state = NBM.train_model([S[i][1] for i in trp[nva:]], [S[i][1] for i in trp[:nva]],
                                epochs=args.nbm_epochs, seed=f, verbose=False, hidden=args.nbm_hidden,
                                lr=args.nbm_lr, batch=args.nbm_batch, wd=args.nbm_wd, loss_from=args.nbm_loss_from,
                                multi=args.nbm_multi)
        if use_cache:
            torch.save(state, wpath)
        print(f"fold {f}: NBM pretrained in {time.time() - t0:.0f}s", flush=True)
    G["nbm_state"] = state            # reused by --nbm-joint when the weights are not on disk (smoke runs)
    M = NBM.model_arrays(state)
    G["nbm_M"] = M                    # reused by --nbm-hpca when the weights are not on disk (smoke runs)
    so = NBM.sums_parallel([(q[1], q[2]) for q in S], M, args.threads)
    prior, nfit = NBM.fit_prior_w([NBM.w_of(so[i]) for i in tr_orig], tau_orig[tr_orig])
    out = Parallel(n_jobs=args.threads, backend="threading")(delayed(_nbm_raw)(q, prior) for q in so)
    del so
    for name, items, _ in sources:
        sp = NBM.sums_parallel(items, M, args.threads)
        out += Parallel(n_jobs=args.threads, backend="threading")(delayed(_nbm_raw)(q, prior) for q in sp)
        del sp
    print(f"fold {f}: NBM channels for {count:,} series in {time.time() - t0:.0f}s; prior from {nfit} training breaks "
          f"(kappa0 {prior[0]:.0f}, alpha0 {prior[1]:.2f}, beta0 {prior[2]:.2f})", flush=True)
    if use_cache:
        np.savez(cache, flat=np.concatenate(out), off=np.r_[0, np.cumsum([len(x) for x in out])], n=count)
    return out


def set_nbm_channels(raws, tr_orig, lens):
    """Write one fold's NBM channels into the device inputs (rows' online part only; burn-in and padding stay 0).
    Standardisation constants from every 10th ORIGINAL TRAINING series of the fold (unsupervised scale only)."""
    sample = np.concatenate([raws[i] for i in tr_orig[::10]])
    mu, sd = sample.mean(axis=0), sample.std(axis=0) + 1e-6
    del sample
    Xd = G["X"]
    for i, r in enumerate(raws):
        fe = np.clip((r - mu) / sd, -6, 6).astype(np.float32)
        Xd[i, args.burn: args.burn + lens[i], NB0: NB0 + NBM.N_NB] = torch.from_numpy(fe).to(G["dev"], non_blocking=False)


def hpca_basis(M, S, tr_orig):
    """PCA of the NBM hidden states on the online parts of every 10th original training series: mean, the first K
    components (K x H), the projections' sd, and the explained-variance ratios."""
    samp = Parallel(n_jobs=args.threads, backend="threading")(
        delayed(NBM.run_hidden)(S[i][1], S[i][2], M) for i in tr_orig[::10])
    Hs = np.concatenate(samp)
    del samp
    mean = Hs.mean(0)
    U, sv, Vt = np.linalg.svd(Hs - mean, full_matrices=False)
    comp = Vt[:args.nbm_hpca]
    ev = (sv ** 2) / (sv ** 2).sum()
    proj = (Hs - mean) @ comp.T
    sd = proj.std(0) + 1e-6
    return mean, comp, sd, ev


def nbm_hidden_fold(f, tr_orig, S, sources, lens):
    """--nbm-hpca: fold-f NBM hidden states projected on K principal components (fitted on the online parts of
    every 10th original training series), standardised; every original series then every extra source. Cached."""
    K = args.nbm_hpca
    use_cache = args.n_series == 0
    cfg = nbm_cfg()
    stub = "".join(st for _, _, st in sources)
    cache = os.path.join("data", f"gru_nbmh{K}v1_f{f}_{cfg}{stub}.npz")
    count = len(S) + sum(len(it) for _, it, _ in sources)
    if use_cache and os.path.exists(cache):
        z = np.load(cache)
        if int(z["n"]) == count:
            flat, off = z["flat"], z["off"]
            print(f"fold {f}: NBM hidden-state PCA channels loaded from {cache}", flush=True)
            return [flat[off[i]:off[i + 1]] for i in range(len(off) - 1)]
    t0 = time.time()
    wpath = os.path.join("data", f"nbm_v{NBM.VERSION}_cv_f{f}_{cfg}.pt")
    if os.path.exists(wpath) and use_cache:
        M = NBM.model_arrays(torch.load(wpath, map_location="cpu"))
    elif "nbm_M" in G:
        M = G["nbm_M"]
    else:
        sys.exit(f"--nbm-hpca: no NBM weights for fold {f} ({wpath})")
    mean, comp, sd, ev = hpca_basis(M, S, tr_orig)
    G["hpca_basis"] = (f, mean, comp, sd)
    print(f"fold {f}: NBM hidden-state PCA from {len(tr_orig[::10])} training series: first {K} components explain "
          f"{100 * ev[:K].sum():.1f}% of the variance ({', '.join(f'{100 * v:.1f}' for v in ev[:K])})", flush=True)
    items = [(q[1], q[2]) for q in S] + [it for _, its, _ in sources for it in its]
    out = Parallel(n_jobs=args.threads, backend="threading")(
        delayed(NBM.run_hidden)(h, o, M, mean, comp) for h, o in items)
    out = [np.clip(o / sd, -6, 6).astype(np.float32) for o in out]
    print(f"fold {f}: NBM hidden-state PCA channels for {count:,} series in {time.time() - t0:.0f}s", flush=True)
    if use_cache:
        np.savez(cache, flat=np.concatenate(out), off=np.r_[0, np.cumsum([len(x) for x in out])], n=count)
    return out


def set_hpca_channels(hp, lens):
    Xd = G["X"]
    for i, r in enumerate(hp):
        Xd[i, args.burn: args.burn + lens[i], HP0: HP0 + args.nbm_hpca] = torch.from_numpy(r).to(G["dev"])


def _extra_item(h, o, M, mean, comp, sd, burn, K, mode, gates):
    """One series for --nbm-hist / --nbm-gates: rows (burn + n): K hidden-state columns (burn-in rows when mode !=
    none; online rows too, re-expressed per series, when mode == white) then 3 raw gate columns."""
    Hh, Gh, Ho, Go = NBM.run_hidden_full(h, o, M)
    n_e, n = len(Hh), len(Ho)
    out = np.zeros((burn + n, K + NBM.N_GATE))
    take = min(burn, n_e)                        # burn-in row j = history point len(h) - burn + j
    if K:
        ph = ((Hh - mean) @ comp.T) / sd
        hb = ph[n_e - take:]
        if mode == "white":
            ref = ph[NBM.LOSS_FROM:] if n_e > NBM.LOSS_FROM + 10 else ph
            mu, sg = ref.mean(0), np.maximum(ref.std(0), 0.2)
            hb = (hb - mu) / sg
            out[burn:, :K] = (((Ho - mean) @ comp.T) / sd - mu) / sg
        out[burn - take: burn, :K] = hb
    if gates:
        out[burn - take: burn, K:] = Gh[n_e - take:]
        out[burn:, K:] = Go
    return out


def nbm_extra_fold(f, tr_orig, S, sources, lens):
    """--nbm-hist / --nbm-gates: per series (rows burn + n) the history burn-in hidden-state channels (or the
    per-series whitened ones) and the gate channels, gates standardised on the online rows of every 10th original
    training series. Cached per fold."""
    K = args.nbm_hpca if args.nbm_hist != "none" else 0
    use_cache = args.n_series == 0
    cfg = nbm_cfg()
    stub = "".join(st for _, _, st in sources)
    cache = os.path.join("data", f"gru_nbmx{K}{args.nbm_hist}_g{int(args.nbm_gates)}_b{args.burn}v1_f{f}_{cfg}{stub}.npz")
    count = len(S) + sum(len(it) for _, it, _ in sources)
    if use_cache and os.path.exists(cache):
        z = np.load(cache)
        if int(z["n"]) == count:
            flat, off = z["flat"], z["off"]
            print(f"fold {f}: --nbm-hist / --nbm-gates channels loaded from {cache}", flush=True)
            return [flat[off[i]:off[i + 1]] for i in range(len(off) - 1)]
    t0 = time.time()
    wpath = os.path.join("data", f"nbm_v{NBM.VERSION}_cv_f{f}_{cfg}.pt")
    if os.path.exists(wpath) and use_cache:
        M = NBM.model_arrays(torch.load(wpath, map_location="cpu"))
    elif "nbm_M" in G:
        M = G["nbm_M"]
    else:
        sys.exit(f"--nbm-hist / --nbm-gates: no NBM weights for fold {f} ({wpath})")
    if K:
        mean, comp, sd, _ = hpca_basis(M, S, tr_orig)    # deterministic: the same basis as the cached hpca channels
    else:
        mean, comp, sd = None, None, None
    items = [(q[1], q[2]) for q in S] + [it for _, its, _ in sources for it in its]
    out = Parallel(n_jobs=args.threads, backend="threading")(
        delayed(_extra_item)(h, o, M, mean, comp, sd, args.burn, K, args.nbm_hist, args.nbm_gates) for h, o in items)
    if args.nbm_gates:
        ref = np.concatenate([out[i][args.burn:, K:] for i in tr_orig[::10]])
        gm, gs = ref.mean(0), ref.std(0) + 1e-6
        print(f"fold {f}: --nbm-gates online means {np.round(gm, 3).tolist()} sd {np.round(gs, 3).tolist()}", flush=True)
        lo = 0 if args.nbm_hist != "none" else args.burn
        for r in out:
            r[lo:, K:] = (r[lo:, K:] - gm) / gs
    out = [np.clip(r, -6, 6).astype(np.float32) for r in out]
    print(f"fold {f}: --nbm-hist {args.nbm_hist} / --nbm-gates {args.nbm_gates} channels for {count:,} series in "
          f"{time.time() - t0:.0f}s", flush=True)
    if use_cache:
        np.savez(cache, flat=np.concatenate(out), off=np.r_[0, np.cumsum([len(x) for x in out])], n=count)
    return out


def set_extra_channels(ex, lens):
    """Write nbm_extra_fold's rows: hidden-state burn-in rows (and the whitened online rows), gate channels."""
    Xd = G["X"]
    K = args.nbm_hpca if args.nbm_hist != "none" else 0
    for i, r in enumerate(ex):
        if K:
            hi = args.burn + lens[i] if args.nbm_hist == "white" else args.burn
            Xd[i, :hi, HP0: HP0 + K] = torch.from_numpy(r[:hi, :K]).to(G["dev"])
        if NG:
            Xd[i, : args.burn + lens[i], GT0: GT0 + NG] = torch.from_numpy(r[:, K:]).to(G["dev"])


def _seq_item(h, o):
    """The NBM's raw rows for one series, as sbrt/nbm.build_seq (z and AR residual e from row AR on; history
    z-scored and AR(10)-fitted exactly as features.make_params), the 6 history descriptors, and the history rows."""
    h = np.asarray(h, dtype=np.float64)
    mu, sd = float(h.mean()), float(h.std())
    if not sd > 0:
        sd = 1.0
    zh = (h - mu) / sd
    nh = len(zh)
    Xl = np.column_stack([zh[NBM.AR - 1 - i: nh - 1 - i] for i in range(NBM.AR)])
    coef, *_ = np.linalg.lstsq(Xl, zh[NBM.AR:], rcond=None)
    se = max(float((zh[NBM.AR:] - Xl @ coef).std()), 1e-8)
    s = np.r_[zh, (np.asarray(o, dtype=np.float64) - mu) / sd]
    n = len(s)
    Xl = np.column_stack([s[NBM.AR - 1 - i: n - 1 - i] for i in range(NBM.AR)])
    e = np.clip((s[NBM.AR:] - Xl @ coef) / se, -10.0, 10.0)
    return s[NBM.AR:].astype(np.float32), e.astype(np.float32), NBM._ctx(zh, coef, se).astype(np.float32), nh - NBM.AR


def nbm_seq_data(S, sources):
    """--nbm-joint: the NBM's input rows for every device row (originals, then each extra source). Fold independent;
    cached. Stored as flat float32 z / e plus offsets, the 6 descriptors and the number of history rows."""
    stub = "".join(st for _, _, st in sources)
    cache = os.path.join("data", f"gru_nbmseq_v1{stub}.npz")
    count = len(S) + sum(len(it) for _, it, _ in sources)
    if args.n_series == 0 and os.path.exists(cache):
        z = np.load(cache)
        if int(z["n"]) == count:
            G.update(js=z["s"], je=z["e"], joff=z["off"], jctx=z["ctx"], jnh=z["nh"])
            print(f"--nbm-joint: NBM input rows loaded from {cache}", flush=True)
            return
    t0 = time.time()
    items = [(q[1], q[2]) for q in S] + [it for _, its, _ in sources for it in its]
    out = Parallel(n_jobs=args.threads, backend="threading")(delayed(_seq_item)(h, o) for h, o in items)
    js = np.concatenate([r[0] for r in out])
    je = np.concatenate([r[1] for r in out])
    off = np.r_[0, np.cumsum([len(r[0]) for r in out])]
    ctx = np.stack([r[2] for r in out])
    nh = np.array([r[3] for r in out], dtype=np.int64)
    del out
    G.update(js=js, je=je, joff=off, jctx=ctx, jnh=nh)
    print(f"--nbm-joint: NBM input rows for {count:,} series ({len(js) / 1e6:.0f}M rows) in {time.time() - t0:.0f}s",
          flush=True)
    if args.n_series == 0:
        np.savez(cache, s=js, e=je, off=off, ctx=ctx, nh=nh, n=count)


def fold_nbm_state(f):
    wpath = os.path.join("data", f"nbm_v{NBM.VERSION}_cv_f{f}_{nbm_cfg()}.pt")
    if args.n_series == 0 and os.path.exists(wpath):
        return torch.load(wpath, map_location="cpu")
    if "nbm_state" in G:
        return G["nbm_state"]
    sys.exit(f"--nbm-joint: no NBM weights for fold {f} ({wpath})")


class NbmJoint(nn.Module):
    """The fold's pretrained NBM as a trainable submodule of the detector (--nbm-joint, HANDOFF section 42).
    forward(ids, Lb, y) returns the K hidden-state PCA channels at the detector's positions (zeros in the burn-in
    and padding, as set_hpca_channels writes them) and, with labels y, the NBM's Student-t NLL on the pre-break rows
    of the gradient window. Detector position k <-> NBM row nh - burn + k (row nh + j = online point j)."""

    def __init__(self, state, mean, comp, sd):
        super().__init__()
        _, NbmNet = NBM._torch()
        self.net = NbmNet(state["gru.weight_hh_l0"].shape[1])
        self.net.load_state_dict(state)
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("comp", torch.as_tensor(np.ascontiguousarray(comp.T), dtype=torch.float32))   # (H, K)
        self.register_buffer("sd", torch.as_tensor(sd, dtype=torch.float32))

    def forward(self, ids, Lb, y=None):
        dev = self.mean.device
        B = len(ids)
        nh, ln = G["jnh"][ids], G["lens"][ids]
        split = nh - args.burn                       # rows before the gradient window (no gradient)
        Lf = -(-int((split + Lb + 1).max()) // 512) * 512   # repeated shapes: no VRAM spill on Windows (section 4)
        s = np.zeros((B, Lf), np.float32)
        e = np.zeros((B, Lf), np.float32)
        for k, i in enumerate(ids):
            a, b = G["joff"][i], G["joff"][i + 1]
            s[k, : b - a] = G["js"][a:b]
            e[k, : b - a] = G["je"][a:b]
        s = torch.from_numpy(s).to(dev)
        e = torch.from_numpy(e).to(dev)
        ep = torch.cat([torch.zeros(B, 1, device=dev), e[:, :-1]], 1)
        ctx = torch.from_numpy(G["jctx"][ids]).to(dev)[:, None, :].expand(-1, Lf, -1)
        X = torch.cat([s.clamp(-8, 8)[..., None], e.clamp(-8, 8)[..., None], e.square().clamp(max=25)[..., None] - 1,
                       e.abs().clamp(max=8)[..., None] - 0.8, (e * ep).clamp(-25, 25)[..., None], ctx], -1)
        with torch.no_grad():
            pk = nn.utils.rnn.pack_padded_sequence(self.net.inp(X), torch.as_tensor(split), batch_first=True,
                                                   enforce_sorted=False)
            _, h0 = self.net.gru(pk)
        pos = torch.as_tensor(split[:, None] + np.arange(Lb)[None, :], device=dev)
        Xg = X.gather(1, pos[..., None].expand(-1, -1, X.shape[-1]))
        out, _ = self.net.gru(self.net.inp(Xg), h0)
        hp = (((out - self.mean) @ self.comp) / self.sd).clamp(-6, 6)
        kk = torch.arange(Lb, device=dev)[None, :]
        lt = torch.as_tensor(ln, device=dev)[:, None]
        hp = hp * ((kk >= args.burn) & (kk < args.burn + lt))[..., None]
        if y is None:
            return hp, None
        r = self.net.out(out)
        nll = NBM.t_nll(torch, e.gather(1, pos + 1), r[..., 0], r[..., 1].clamp(-4, 4), self.net.nu())
        y1 = torch.cat([y[:, 1:], torch.ones(B, 1, device=dev)], 1)       # label of the predicted (next) row
        msk = ((kk + 1 < args.burn + lt) & (y1 < 0.5)).float()
        return hp, (nll * msk).sum() / msk.sum().clamp(min=1)


def joint_inputs(model, x, ids, y=None):
    """Replace the cached (frozen-NBM) hidden-state channels of a batch by the current NBM's."""
    hp, nll = model.nbmj(np.asarray(ids), x.shape[1], y)
    return torch.cat([x[..., :HP0], hp, x[..., HP0 + args.nbm_hpca:]], dim=-1), nll


DIST_COLS = [j for j, n in enumerate(DT.FEATURE_NAMES) if not n.startswith("dist_z_")]
ALL_NAMES = (list(F.FEATURE_NAMES) + list(EV.FEATURE_NAMES[:EV.N_COMPACT]) + list(VS.FEATURE_NAMES)
             + [DT.FEATURE_NAMES[j] for j in DIST_COLS])   # the tree feature order of cv.py --ev compact --vol --dist full --drop dist_z_


def src_names():
    return F.FEATURE_NAMES if args.feat_src == "kernel" else ALL_NAMES


def top_feature_cols():
    """Names + column indices (into src_names()) of the K most important streaming features (ctx_/time features
    excluded: the network already receives those as channels)."""
    if not os.path.exists(args.feat_file):
        sys.exit(f"--feat-top needs {args.feat_file} (written by cv.py); run cv.py first")
    names = []
    with open(args.feat_file, encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            nm = line.split(",")[0]
            if nm in src_names() and not nm.startswith(("ctx_", "t_log", "z_now")):
                names.append(nm)
    names = names[: args.feat_top]
    if len(names) < args.feat_top:
        sys.exit(f"only {len(names)} usable features in {args.feat_file} for --feat-src {args.feat_src}")
    return names, [src_names().index(nm) for nm in names]


def _feat_raw(h, o, cols):
    if args.feat_src == "kernel":
        M = F.run_series(h, o)[:, cols]
    else:   # only the modules the chosen columns need, every online step
        nk, ne, nv = F.N_FEAT, EV.N_COMPACT, VS.N_VOL
        cols = np.asarray(cols)
        parts = [F.run_series(h, o)]
        parts.append(EV.run_series_ev(h, o)[:, :ne] if ((cols >= nk) & (cols < nk + ne)).any() else np.zeros((len(o), ne)))
        parts.append(VS.run_series_vol(h, o) if ((cols >= nk + ne) & (cols < nk + ne + nv)).any() else np.zeros((len(o), nv)))
        parts.append(DT.run_series_dist(h, o)[:, DIST_COLS] if (cols >= nk + ne + nv).any() else np.zeros((len(o), len(DIST_COLS))))
        M = np.hstack(parts)[:, cols]
    return (np.sign(M) * np.log1p(np.abs(M))).astype(np.float32)


def load_inputs(S):
    spec = ""
    if args.feat_top:
        names, cols = top_feature_cols()
        spec = f"_f{args.feat_top}_{hashlib.md5(','.join(names).encode()).hexdigest()[:6]}"
        print(f"--feat-top {args.feat_top}: first channels from the model's importance list: {names[:8]} ...")
    cache = os.path.join("data", f"gru_in_b{args.burn}{'_noctx' if args.no_ctx else ''}"
                                 f"{'_tails' if args.tails else ''}{vol_stub()}{spec}.npz")
    if args.n_series == 0 and os.path.exists(cache):
        z = np.load(cache)
        flat, off = z["flat"], z["off"]
        return [flat[off[i]:off[i + 1]] for i in range(len(off) - 1)]
    t0 = time.time()
    Xs = [build_inputs(s[1], s[2], args.burn) for s in S]
    Xs = add_vol_channels(Xs, [(s[1], s[2]) for s in S])
    Xs = add_bo_channels(Xs, [(s[1], s[2]) for s in S], [(s[1], s[2]) for s in S])
    if args.feat_top:
        raws = Parallel(n_jobs=args.threads, backend="threading")(delayed(_feat_raw)(s[1], s[2], cols) for s in S)
        sample = np.concatenate(raws[::10])
        mu, sd = sample.mean(axis=0), sample.std(axis=0) + 1e-6  # unsupervised scale only
        for i in range(len(Xs)):
            fe = np.clip((raws[i] - mu) / sd, -6, 6).astype(np.float32)
            Xs[i] = np.hstack([Xs[i], np.vstack([np.zeros((args.burn, args.feat_top), np.float32), fe])])
        del raws
    print(f"built GRU inputs for {len(Xs)} series in {time.time() - t0:.0f}s ({Xs[0].shape[1]} channels)")
    if args.n_series == 0:
        off = np.r_[0, np.cumsum([len(x) for x in Xs])]
        np.savez(cache, flat=np.concatenate(Xs), off=off)
    return Xs


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


def extra_sources(S, tau):
    """Sources of extra TRAINING series: [(name, [(parent, hist, online, tau, kind)], cache stub, rel weight)].
    kind = index into sbrt.synth.TYPES for a visible synthetic break, -1 otherwise."""
    out = []
    if args.aug > 0:
        V = A.make_variants(S, tau, args.aug, seed=args.aug_seed, min_shift=args.aug_min_shift,
                            trim_p=args.aug_trim, n_trim_only=args.aug_trim_only)
        hv = np.array([len(v[2]) for v in V])
        print(f"re-cut variants: {len(V):,}; history length quantiles 10/50/90% originals "
              f"{np.percentile([len(q[1]) for q in S], [10, 50, 90]).astype(int)} variants "
              f"{np.percentile(hv, [10, 50, 90]).astype(int)}")
        out.append(("re-cut", [(v[0], v[2], v[3], v[4], -1) for v in V],
                    f"_aug{args.aug}_s{args.aug_seed}_m{args.aug_min_shift}" + (f"_tr{args.aug_trim:g}_x{args.aug_trim_only}" if (args.aug_trim > 0 or args.aug_trim_only > 0) else ""), args.aug_rel))
    if args.synth > 0:
        V = SY.make_synthetic(S, tau, args.synth, seed=args.synth_seed, scale=args.synth_scale,
                              p_zero=args.synth_pzero, folds=SY.fold_ids(tau), types=synth_types())
        out.append(("synthetic", [(v[0], v[3], v[4], v[5], SY.TYPES.index(v[1]) if (v[1] and v[2] > 0) else -1)
                                  for v in V],
                    f"_synth{args.synth}_v1_s{args.synth_seed}_sc{args.synth_scale:g}_pz{args.synth_pzero:g}{synth_stub()}",
                    args.synth_rel))
    if args.ext > 0:  # parent = the Real-Time twin found by ext2025.py, or -1 (trains in every fold)
        V = XT.make_external(args.ext_dir, args.ext, seed=args.ext_seed, hmin=args.ext_hmin, limit=args.n_series)
        out.append(("2025", [(p, h, o, t, -1) for p, h, o, t in V],
                    f"_ext{args.ext}_s{args.ext_seed}_h{args.ext_hmin}", args.ext_rel))
    return out


def extra_inputs(S, name, V, stub):
    """GRU inputs of one source of extra series (cached). The top-K feature channels use exactly the
    standardisation constants of load_inputs (recomputed from the same every-10th original series)."""
    vlen = np.array([len(v[2]) for v in V], dtype=np.int64)
    spec = ""
    if args.feat_top:
        names, cols = top_feature_cols()
        spec = f"_f{args.feat_top}_{hashlib.md5(','.join(names).encode()).hexdigest()[:6]}"
    cache = os.path.join("data", f"gru_in_b{args.burn}{'_noctx' if args.no_ctx else ''}"
                                 f"{'_tails' if args.tails else ''}{vol_stub()}{spec}{stub}.npz")
    if args.n_series == 0 and os.path.exists(cache):
        z = np.load(cache)
        if np.array_equal(z["vlen"], vlen):
            flat, off = z["flat"], z["off"]
            print(f"{name}: loaded {len(V):,} series from {cache}")
            return [flat[off[i]:off[i + 1]] for i in range(len(off) - 1)]
        print(f"{name}: cache does not match the current settings -> rebuilding")
    t0 = time.time()
    Xv = [build_inputs(v[1], v[2], args.burn) for v in V]
    Xv = add_vol_channels(Xv, [(v[1], v[2]) for v in V])
    Xv = add_bo_channels(Xv, [(v[1], v[2]) for v in V], [(q[1], q[2]) for q in S])
    if args.feat_top:
        ref = Parallel(n_jobs=args.threads, backend="threading")(delayed(_feat_raw)(q[1], q[2], cols) for q in S[::10])
        sample = np.concatenate(ref)
        mu, sd = sample.mean(axis=0), sample.std(axis=0) + 1e-6
        del ref, sample
        raws = Parallel(n_jobs=args.threads, backend="threading")(delayed(_feat_raw)(v[1], v[2], cols) for v in V)
        for i in range(len(Xv)):
            fe = np.clip((raws[i] - mu) / sd, -6, 6).astype(np.float32)
            Xv[i] = np.hstack([Xv[i], np.vstack([np.zeros((args.burn, args.feat_top), np.float32), fe])])
        del raws
    print(f"{name}: built GRU inputs for {len(V):,} series in {time.time() - t0:.0f}s")
    if args.n_series == 0:
        off = np.r_[0, np.cumsum([len(x) for x in Xv])]
        np.savez(cache, flat=np.concatenate(Xv), off=off, vlen=vlen)
    return Xv


def channel_names():
    nm = ["z", "e", "e2m1", "abs_e", "e_eprev", "e_gt3"]
    if args.tails:
        nm += ["e_gt2.5", "e_gt3.5", "e_gt4", "e_gt5", "z_gt3", "z_gt4", "z_gt5"]
    nm += ["ctx_logn", "ctx_skew", "ctx_kurt", "ctx_rho1", "ctx_phi", "ctx_logse", "online", "log_t"]
    if args.vol:
        nm += list(VS.GRU_NAMES)
    if args.bocpd:
        nm += list(BO.FEATURE_NAMES)
    if args.nbm:
        nm += list(NBM.FEATURE_NAMES)
    nm += [f"nbh_pc{k}" for k in range(args.nbm_hpca)]
    if args.nbm_gates:
        nm += list(NBM.GATE_NAMES)
    if args.feat_top:
        nm += top_feature_cols()[0]
    assert len(nm) == CIN
    return nm


def prior_gate_weights():
    """1 for the raw channels; sqrt(gain / mean gain) of the tree model (clipped to 0.5-2) for the top-K features."""
    w = np.ones(CIN)
    if args.feat_top:
        gain = {}
        with open(args.feat_file, encoding="utf-8") as fh:
            next(fh)
            for line in fh:
                k, v = line.strip().split(",")
                gain[k] = float(v)
        g = np.array([gain[n] for n in top_feature_cols()[0]])
        w[CIN - args.feat_top:] = np.clip(np.sqrt(g / g.mean()), 0.5, 2.0)
    return w


# ---------------------------------------------------------------------------------------------
# model
class TCN(nn.Module):
    """Causal dilated convolutions on (B, L, H): the output at step t sees only steps <= t (left padding only, no
    batch norm, so right padding of shorter series cannot leak in). The block output is joined with its causal
    running mean (accumulated evidence, like a recurrent state) and projected back to H."""

    def __init__(self, H, k, levels, p):
        super().__init__()
        self.k = k
        self.dil = [2 ** i for i in range(levels)]
        self.c1 = nn.ModuleList(nn.Conv1d(H, H, k, dilation=d) for d in self.dil)
        self.c2 = nn.ModuleList(nn.Conv1d(H, H, k, dilation=d) for d in self.dil)
        self.drop = nn.Dropout(p)
        self.mix = nn.Sequential(nn.Linear(2 * H, H), nn.GELU())

    def forward(self, x):
        h = x.transpose(1, 2)                                   # (B, H, L)
        for d, c1, c2 in zip(self.dil, self.c1, self.c2):
            pad = (self.k - 1) * d
            r = self.drop(nn.functional.gelu(c1(nn.functional.pad(h, (pad, 0)))))
            r = self.drop(nn.functional.gelu(c2(nn.functional.pad(r, (pad, 0)))))
            h = h + r
        h = h.transpose(1, 2)                                   # (B, L, H)
        cnt = torch.arange(1, h.shape[1] + 1, device=h.device, dtype=h.dtype)[None, :, None]
        return self.mix(torch.cat([h, h.cumsum(1) / cnt], -1)), None


def qa_groups():
    """Channel groups (start, stop) that become one token each, in the build_inputs / CIN order."""
    g = [(0, 6)]
    if args.tails:
        g.append((6, 13))
    g += [(CTX0, CTX0 + 6), (CTX0 + 6, CTX0 + 8)]
    c = CTX0 + 8
    for k in ((VS.N_GRU if args.vol else 0), (BO.N_BO if args.bocpd else 0), (NBM.N_NB if args.nbm else 0),
              args.nbm_hpca, NG):
        if k:
            g.append((c, c + k))
            c += k
    for a in range(c, CIN, args.qa_group):
        g.append((a, min(a + args.qa_group, CIN)))
    assert g[-1][1] == CIN
    return g


class CausalAttn(nn.Module):
    """Multi-head attention of queries at step t over keys at steps <= t; the additive mask carries the ALiBi
    recency penalty and the causal -inf. Queries may come in M stacked copies of the batch (break queries)."""

    def __init__(self, d, heads, p):
        super().__init__()
        self.h, self.p = heads, p
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d, 2 * d)
        self.o = nn.Linear(d, d)

    def forward(self, xq, xkv, mask):
        B, L, d = xkv.shape
        M = xq.shape[0] // B
        q = self.q(xq).view(M * B, L, self.h, d // self.h).transpose(1, 2)
        k, v = self.kv(xkv).view(B, L, 2, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        if M > 1:
            k, v = k.repeat(M, 1, 1, 1), v.repeat(M, 1, 1, 1)
        o = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                                       dropout_p=self.p if self.training else 0.0)
        return self.o(o.transpose(1, 2).reshape(M * B, L, d))


class QAttn(nn.Module):
    """Query attention over feature tokens and time (HANDOFF section 39). Everything at step t sees only steps
    <= t: per-step token mixing, causal masks, LayerNorm per token, no normalisation across time or batch, so the
    right padding of shorter series cannot leak in.
      1. tokens: each channel group (raw z/residuals, history descriptors, time, vol, BOCPD, NBM, NBM hidden PCA,
         the kernel features in --qa-group chunks) -> its own Linear -> LayerNorm, + a group embedding
      2. feature queries: --qa-fq learned queries attend over the step's tokens ("which statistics matter now")
      3. causal transformer over the steps; heads have ALiBi slopes 1/8, 1/32, 1/128, 0 (recency windows ~8, 32,
         128 steps and the whole stream)
      4. break queries: --qa-queries learned queries ("did a break happen?"), conditioned on the current state,
         attend over all steps <= t; [state, query answers] -> H for the usual head."""

    def __init__(self, H, p):
        super().__init__()
        d, nh = args.qa_d, args.qa_heads
        self.groups = qa_groups()
        self.tok = nn.ModuleList(nn.Linear(b - a, d) for a, b in self.groups)
        self.tok_ln = nn.LayerNorm(d)
        self.gemb = nn.Parameter(torch.randn(len(self.groups), d) * 0.02)
        self.fq = nn.Parameter(torch.randn(args.qa_fq, d) * 0.02)
        self.fkv = nn.Linear(d, 2 * d)    # feature attention by einsum: B*L "sequences" of K tokens exceed the
        self.fo = nn.Linear(d, d)         # fused kernels' 65,535-batch limit
        self.fmix = nn.Sequential(nn.Linear((args.qa_fq + 1) * d, d), nn.GELU())
        self.ln1 = nn.ModuleList(nn.LayerNorm(d) for _ in range(args.qa_layers))
        self.att = nn.ModuleList(CausalAttn(d, nh, p) for _ in range(args.qa_layers))
        self.ln2 = nn.ModuleList(nn.LayerNorm(d) for _ in range(args.qa_layers))
        self.mlp = nn.ModuleList(nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(p), nn.Linear(2 * d, d))
                                 for _ in range(args.qa_layers))
        self.drop = nn.Dropout(p)
        self.bq = nn.Parameter(torch.randn(args.qa_queries, d) * 0.02)
        self.qln = nn.LayerNorm(d)
        self.qatt = CausalAttn(d, nh, p)
        self.out = nn.Sequential(nn.Linear((args.qa_queries + 1) * d, H), nn.GELU())
        sl = [2.0 ** -(3 + 2 * i) for i in range(nh - 1)] + [0.0]
        self.register_buffer("slopes", torch.tensor(sl, dtype=torch.float32))

    def mask(self, L, dev):
        pos = torch.arange(L, device=dev)
        dist = (pos[:, None] - pos[None, :]).float()                     # i - j: query step minus key step
        m = -self.slopes[:, None, None] * dist.clamp(min=0)[None]
        return m.masked_fill(dist[None] < 0, float("-inf"))              # (heads, L, L)

    def forward(self, x):
        B, L, _ = x.shape
        d = self.gemb.shape[1]
        t = torch.stack([lin(x[..., a:b]) for lin, (a, b) in zip(self.tok, self.groups)], 2)    # (B, L, K, d)
        t = self.tok_ln(t) + self.gemb
        nh = self.slopes.shape[0]
        k, v = self.fkv(t).view(B, L, t.shape[2], 2, nh, d // nh).unbind(3)                   # (B, L, K, h, e)
        q = self.fq.view(-1, nh, d // nh)                                                      # (F, h, e)
        w = torch.softmax(torch.einsum("blkhe,fhe->blfhk", k, q) / (d // nh) ** 0.5, -1)
        fo = self.fo(torch.einsum("blfhk,blkhe->blfhe", self.drop(w), v).reshape(B, L, -1, d))
        h = self.fmix(torch.cat([fo.reshape(B, L, -1), t.mean(2)], -1))                      # (B, L, d)
        m = self.mask(L, x.device)
        for ln1, att, ln2, mlp in zip(self.ln1, self.att, self.ln2, self.mlp):
            z = ln1(h)
            h = h + self.drop(att(z, z, m))
            h = h + self.drop(mlp(ln2(h)))
        z = self.qln(h)
        nq = self.bq.shape[0]
        xq = (z[None] + self.bq[:, None, None, :]).reshape(nq * B, L, d)
        a = self.qatt(xq, z, m).view(nq, B, L, d).permute(1, 2, 0, 3).reshape(B, L, nq * d)
        return self.out(torch.cat([h, a], -1)), None


def qa_causal_check(model, ids):
    """Change every input after step t and require the outputs at steps <= t to stay the same (eval mode)."""
    was = model.training
    model.eval()
    with torch.no_grad():
        x, _, _ = get_batch(ids, False)
        y0 = model(x)
        worst = 0.0
        for t in (args.burn - 1, args.burn + 5, args.burn + 50, x.shape[1] // 2):
            if t + 1 >= x.shape[1]:
                continue
            x2 = x.clone()
            x2[:, t + 1:] = torch.randn_like(x2[:, t + 1:]) * 3.0
            worst = max(worst, float((model(x2)[:, :t + 1] - y0[:, :t + 1]).abs().max()))
    model.train(was)
    print(f"qattn causality check: max change of past outputs when the future is replaced = {worst:.2e}"
          + ("  OK" if worst < 1e-4 else "  FAILED"), flush=True)
    if worst >= 1e-4:
        sys.exit("qattn leaks the future")


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        H = args.hidden
        # qattn embeds the channel groups itself (raw channels in, no shared input layer)
        self.inp = nn.Identity() if args.arch == "qattn" else nn.Sequential(nn.Linear(CIN, H), nn.GELU())
        if args.arch == "qattn":
            self.gru = QAttn(H, args.dropout)
        elif args.arch == "tcn":
            self.gru = TCN(H, args.tcn_kernel, args.tcn_levels, args.dropout)
        else:
            self.gru = nn.GRU(H, H, num_layers=args.layers, batch_first=True,
                              dropout=args.dropout if args.layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(H, H), nn.GELU(), nn.Dropout(args.dropout), nn.Linear(H, 1))
        # auxiliary heads (training only; the submission uses the main head alone)
        self.tsb = nn.Linear(H, 1) if args.aux_tsb > 0 else None
        self.btype = nn.Linear(H, len(SY.TYPES)) if args.aux_type > 0 else None
        # input gate (created last so the ungated model keeps exactly the same initial weights)
        self.gate_mode = args.gate
        if args.gate == "prior":
            self.register_buffer("gw", torch.as_tensor(prior_gate_weights(), dtype=torch.float32))
        elif args.gate in ("static", "dynamic"):
            nin = 6 if args.gate == "static" else CIN
            self.gnet = nn.Sequential(nn.Linear(nin, args.gate_hidden), nn.GELU(), nn.Linear(args.gate_hidden, CIN))
            nn.init.zeros_(self.gnet[2].weight)
            nn.init.zeros_(self.gnet[2].bias)
        # memory knob: GRU h_t = (1-z)*n + z*h_{t-1}; a larger update-gate bias keeps the old state longer
        for name, p in (self.gru.named_parameters() if args.arch == "gru" else ()):
            if name.startswith("bias_ih"):
                with torch.no_grad():
                    p[H:2 * H] += args.mem_bias

    def gate(self, x):
        """Channel weights (B, L, CIN) or (CIN,), averaging 1 over channels."""
        if self.gate_mode == "prior":
            return self.gw
        src = x[..., CTX0:CTX0 + 6] if self.gate_mode == "static" else x
        return CIN * torch.softmax(self.gnet(src), dim=-1)

    def forward(self, x, aux=False):
        if self.gate_mode != "none":
            x = x * self.gate(x)
        out, _ = self.gru(self.inp(x))
        main = self.head(out).squeeze(-1)
        if not aux:
            return main
        return (main, self.tsb(out).squeeze(-1) if self.tsb is not None else None,
                self.btype(out) if self.btype is not None else None)


def get_batch(ids, with_labels):
    """All data already lives on the device as padded tensors, so a batch is just an index + slice
    (no per-batch CPU work, no host->device copies)."""
    Lb = int(G["tot"][ids].max())
    idx = torch.as_tensor(ids, dtype=torch.long, device=G["dev"])
    x = G["X"][idx, :Lb]
    if not with_labels:
        return x, None, None
    return x, G["Y"][idx, :Lb], G["W"][idx, :Lb], idx, Lb


def make_batches(idx, lens, batch, rng, shuffle):
    idx = np.asarray(idx)
    key = lens[idx] + (rng.random(len(idx)) * 0.15 * lens[idx] if shuffle else 0)
    order = idx[np.argsort(key)]
    batches = [order[i:i + batch] for i in range(0, len(order), batch)]
    if shuffle:
        rng.shuffle(batches)
    return batches


def per_step_z(s, step, mask):
    T = int(step.max()) + 1
    cnt = np.maximum(np.bincount(step[mask], minlength=T), 1)
    m = np.bincount(step[mask], weights=s[mask], minlength=T) / cnt
    v = np.bincount(step[mask], weights=s[mask] ** 2, minlength=T) / cnt - m ** 2
    return (s - m[step]) / np.sqrt(np.maximum(v[step], 1e-12))


def main():
    S, tau_all = load_train()
    pick = None
    if args.n_series:
        rs = np.random.default_rng(0)
        pick = np.sort(rs.choice(len(S), args.n_series, replace=False))
        S = [S[i] for i in pick]
        tau_all = tau_all[pick]
    n = len(S)
    if args.device == "cuda" and not torch.cuda.is_available():
        sys.exit("--device cuda requested, but this PyTorch build has no CUDA (torch %s). Install the GPU build:\n"
                 "  pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130" % torch.__version__)
    dev = torch.device("cuda" if (args.device in ("auto", "cuda") and torch.cuda.is_available()) else "cpu")
    if dev.type == "cpu" and args.device == "auto":
        print("=" * 78 + f"\nWARNING: running on the CPU (torch {torch.__version__} has no CUDA) -- this will be slow.\n"
              "         GPU build:  pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130\n"
              + "=" * 78)
    print(f"series {n} | device {dev}" + (f" ({torch.cuda.get_device_name(0)})" if dev.type == "cuda" else ""))
    print(f"settings: {vars(args)}")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    Xs = load_inputs(S)
    lens = np.array([len(s[2]) for s in S])
    n_orig = n
    parent = np.arange(n_orig)
    rel_of = np.ones(n_orig)
    kind_of = np.full(n_orig, -1, dtype=np.int64)
    tau_orig = tau_all.copy()
    nbm_sources = []   # (name, [(hist, online)], cache stub) in device-row order, for --nbm
    for name, V, stub, rel in extra_sources(S, tau_orig):
        nbm_sources.append((name, [(v[1], v[2]) for v in V], stub))
        # extra series are appended as rows n_orig.. of the padded device tensors; folds, validation and the
        # OOF file stay on the original series (rows 0..n_orig-1)
        Xv = extra_inputs(S, name, V, stub)
        Xs = list(Xs) + Xv
        del Xv
        len_v = np.array([len(v[2]) for v in V])
        tau_v = np.array([v[3] for v in V])
        lens = np.r_[lens, len_v]
        tau_all = np.r_[tau_all, tau_v]
        parent = np.r_[parent, np.array([v[0] for v in V])]
        rel_of = np.r_[rel_of, np.full(len(V), rel)]
        kind_of = np.r_[kind_of, np.array([v[4] for v in V], dtype=np.int64)]
        print(f"distribution check, original series vs {name} series:")
        A.describe(lens[:n_orig], tau_orig, len_v, tau_v)
    n = len(Xs)
    G["lens"] = lens
    if JOINT:
        nbm_seq_data(S, nbm_sources)
        if not np.array_equal(G["jnh"][:n_orig], np.array([len(s[1]) for s in S]) - NBM.AR):
            sys.exit("--nbm-joint: NBM input rows do not match the series")
        if (G["jnh"] <= args.burn).any():
            sys.exit(f"--nbm-joint: {(G['jnh'] <= args.burn).sum()} series have <= --burn history rows")
    if args.aux_type > 0 and not (kind_of >= 0).any():
        sys.exit("--aux-type needs synthetic series with visible breaks: add --synth N")
    Y = [((np.arange(L) >= t) & (t >= 0)).astype(np.float32) for L, t in zip(lens, tau_all)]
    # put every input and label on the device ONCE (padded); batches are then device-side slices
    tot = np.array([len(x) for x in Xs])
    Lmax = int(tot.max())
    Xp = np.zeros((n, Lmax, CIN), dtype=np.float32)
    Yp = np.zeros((n, Lmax), dtype=np.float32)
    for i, x in enumerate(Xs):
        if args.nbm:   # cached inputs have no NBM channels: leave them zero here, filled per fold
            Xp[i, : len(x), :NB0] = x[:, :NB0]
            Xp[i, : len(x), NB0 + NBM.N_NB + args.nbm_hpca + NG:] = x[:, NB0:]
        else:
            Xp[i, : len(x)] = x
        Yp[i, args.burn: len(x)] = Y[i]
    G.update(dev=dev, tot=tot, X=torch.from_numpy(Xp).to(dev), Y=torch.from_numpy(Yp).to(dev),
             R=torch.as_tensor(rel_of, dtype=torch.float32, device=dev),
             K=torch.as_tensor(kind_of, dtype=torch.long, device=dev))
    del Xp, Yp
    if args.aux_tsb > 0:  # log(1 + time since break) / 7 on positive steps, 0 elsewhere
        Dp = np.zeros((n, Lmax), dtype=np.float32)
        for i in range(n):
            if tau_all[i] >= 0:
                d = np.arange(lens[i]) - tau_all[i]
                Dp[i, args.burn: args.burn + lens[i]] = np.where(d >= 0, np.log1p(np.maximum(d, 0)) / 7.0, 0.0)
        G["D"] = torch.from_numpy(Dp).to(dev)
        del Dp
    print(f"inputs resident on {dev}: {G['X'].numel() * 4 / 1e6:.0f} MB", flush=True)
    has_break = (tau_all[:n_orig] >= 0).astype(int)
    steps_sub = [np.arange(0, L, SUB) for L in lens[:n_orig]]
    sid_sub = np.concatenate([np.full(len(s), i) for i, s in enumerate(steps_sub)])
    step_sub = np.concatenate(steps_sub)
    y_sub = np.concatenate([Y[i][steps_sub[i]] for i in range(n_orig)]).astype(np.int8)
    row_start = np.r_[0, np.cumsum([len(s) for s in steps_sub])]
    oof = np.full(len(y_sub), np.nan)
    base_oof, diffs = None, []
    if args.compare:
        c = np.load(args.compare)
        if not (np.array_equal(c["sid"], sid_sub) and np.array_equal(c["step"], step_sub)):
            sys.exit(f"--compare file {args.compare} has a different row layout")
        base_oof = c["oof"]
    def load_teacher(path):
        tz = dict(np.load(path))
        if args.n_series:   # debug subset: keep the picked series' rows, renumber sid like the subset
            keep = np.isin(tz["sid"], pick)
            tz = {k: v[keep] for k, v in tz.items()}
            tz["sid"] = np.searchsorted(pick, tz["sid"])
        if not (np.array_equal(tz["sid"], sid_sub) and np.array_equal(tz["step"], step_sub)):
            sys.exit(f"--distill-file {path} has a different row layout")
        return tz["z"]

    teach_z = None
    PER_FOLD_TEACHER = "{f}" in args.distill_file   # clean teacher: one cv.py --lpo file per outer fold
    if args.distill > 0 and not PER_FOLD_TEACHER:
        teach_z = load_teacher(args.distill_file)
        print(f"--distill {args.distill:g}: teacher {args.distill_file}, TS-AUC on all rows "
              f"{ts_auc(teach_z, y_sub, step_sub):.4f} (LEAKY probe: in-sample blend weights, members saw the "
              f"validation folds)", flush=True)

    def distill_targets(f, tr_orig):
        """Device tensor of BCE targets: hard labels everywhere, (1 - a) y + a p on the original training series."""
        from sklearn.linear_model import LogisticRegression
        nonlocal teach_z
        mk = np.isin(sid_sub, tr_orig)
        if PER_FOLD_TEACHER:
            path = args.distill_file.format(f=f)
            teach_z = load_teacher(path)
            if np.isnan(teach_z[mk]).any():
                sys.exit(f"{path}: NaN on fold {f}'s training rows (not a cv.py --lpo {f} teacher)")
            if not np.isnan(teach_z[~mk]).all():
                sys.exit(f"{path}: scores fold {f}'s validation rows, so it is not leave-fold-{f}-out")
            print(f"fold {f}: --distill CLEAN teacher {path}, TS-AUC on the training rows "
                  f"{ts_auc(teach_z[mk], y_sub[mk], step_sub[mk]):.4f}", flush=True)
        z, yy, st = teach_z[mk], y_sub[mk].astype(float), step_sub[mk]
        T = int(st.max()) + 1
        npos = np.bincount(st, weights=yy, minlength=T)
        nneg = np.bincount(st, minlength=T) - npos
        sw = np.where(yy > 0, nneg[st], npos[st])        # class-balanced per step, step weight ~ npos*nneg
        sw = sw / sw.mean()
        ls = np.log1p(st) / 7.0
        feats = lambda zz, ll: np.c_[zz, zz * ll, ll]
        lr = LogisticRegression(C=1e4, max_iter=1000).fit(feats(z, ls), yy, sample_weight=sw)
        print(f"fold {f}: --distill Platt on {mk.sum():,} training rows: coef {np.round(lr.coef_[0], 3).tolist()} "
              f"intercept {lr.intercept_[0]:+.3f}", flush=True)
        Yd = G["Y"].clone()
        gap, cnt = 0.0, 0
        for i in tr_orig:
            r = slice(row_start[i], row_start[i + 1])
            q = lr.decision_function(feats(teach_z[r], np.log1p(steps_sub[i]) / 7.0))
            p = 1.0 / (1.0 + np.exp(-np.interp(np.arange(lens[i]), steps_sub[i], q)))
            soft = Y[i] + args.distill * (p - Y[i])
            gap += float(np.abs(p - Y[i]).sum())
            cnt += lens[i]
            Yd[i, args.burn: args.burn + lens[i]] = torch.as_tensor(soft, dtype=torch.float32, device=dev)
        print(f"fold {f}: --distill mean |p - label| on original training steps {gap / max(cnt, 1):.3f}", flush=True)
        return Yd

    skf = StratifiedKFold(5, shuffle=True, random_state=0)  # identical to cv.py
    fold_scores = []
    if SWA:
        swa_oof = {k: np.full(len(y_sub), np.nan) for k in SWA_KEYS}
        swa_scores = {k: [] for k in SWA_KEYS}
        swa_base = {k: [] for k in SWA_KEYS}
        swa_vs_fix = {k: [] for k in SWA_KEYS[:2]}

    def va_score(va, logits):
        return ts_auc(np.concatenate([logits[i][steps_sub[i]] for i in va]),
                      np.concatenate([Y[i][steps_sub[i]] for i in va]).astype(np.int8),
                      np.concatenate([steps_sub[i] for i in va]))

    for f, (tr, va) in enumerate(skf.split(np.arange(n_orig), has_break)):
        if f >= args.folds:
            break
        t0 = time.time()
        tr_orig = tr
        if n > n_orig:
            tr = np.r_[tr_orig, n_orig + np.flatnonzero(np.isin(parent[n_orig:], tr_orig) | (parent[n_orig:] < 0))]
            print(f"fold {f}: training series {len(tr_orig):,} original + {len(tr) - len(tr_orig):,} extra", flush=True)
        # metric-matched weights: the per-step class targets come from the ORIGINAL training series only
        # (npos*nneg/2 per class); that target is shared among all rows of the step and class, original rows with
        # share 1 and variant rows with share --aug-rel. Without variants this is exactly pos -> nneg/2, neg -> npos/2.
        Tm = int(lens[tr].max())
        npos = np.zeros(Tm)
        nall = np.zeros(Tm)
        for i in tr_orig:
            nall[: lens[i]] += 1
            npos[: lens[i]] += Y[i]
        nneg = nall - npos
        rpos = np.zeros(Tm)
        rneg = np.zeros(Tm)
        for i in tr:
            r = rel_of[i]
            rpos[: lens[i]] += r * Y[i]
            rneg[: lens[i]] += r * (1.0 - Y[i])
        tgt = npos * nneg / 2.0
        wpos = tgt / np.maximum(rpos, 1e-12)
        wneg = tgt / np.maximum(rneg, 1e-12)
        W = {}
        tot, cnt = 0.0, 0
        for i in tr:
            T = lens[i]
            pos = Y[i] > 0
            r = rel_of[i]
            w = r * np.where(pos, wpos[:T], wneg[:T])
            if pos.any():
                d = np.arange(T) - tau_all[i]
                ramp = np.minimum(1.0, (d + 1) / args.ramp) if args.ramp > 0 else 1.0
                w = np.where(pos, w * ramp * args.pos_boost, w)
            W[i] = w.astype(np.float32)
            tot += float(w.sum())
            cnt += T
        scale = cnt / max(tot, 1e-12)
        Wp = np.zeros((n, Lmax), dtype=np.float32)
        for i in tr:
            Wp[i, args.burn: args.burn + lens[i]] = W[i] * scale
        G["W"] = torch.from_numpy(Wp).to(dev)
        del Wp
        if args.distill > 0:
            G["YD"] = distill_targets(f, tr_orig)
        if args.nbm:
            raws = nbm_fold_raw(f, tr_orig, S, tau_orig, nbm_sources)
            set_nbm_channels(raws, tr_orig, lens)
            if args.nbm_hpca:
                hp = nbm_hidden_fold(f, tr_orig, S, nbm_sources, lens)
                set_hpca_channels(hp, lens)
                del hp
            if args.nbm_hist != "none" or args.nbm_gates:
                ex = nbm_extra_fold(f, tr_orig, S, nbm_sources, lens)
                set_extra_channels(ex, lens)
                del ex
            del raws
        model = Net().to(dev)
        if JOINT:
            hb = G.get("hpca_basis")
            if hb is not None and hb[0] == f:
                mean, comp, sd = hb[1:]
            else:
                mean, comp, sd, _ = hpca_basis(NBM.model_arrays(fold_nbm_state(f)), S, tr_orig)
            with torch.random.fork_rng(devices=[]):   # the NBM's (overwritten) init must not shift later folds' RNG
                model.nbmj = NbmJoint(fold_nbm_state(f), mean, comp, sd).to(dev)
            jw0 = [p.detach().clone() for p in model.nbmj.net.parameters()]
            with torch.no_grad():
                chk = tr_orig[:: max(1, len(tr_orig) // 48)][:48]
                xc, _, _ = get_batch(chk, False)
                ref = xc[..., HP0: HP0 + args.nbm_hpca]
                hc, _ = model.nbmj(chk, xc.shape[1])
                # a basis recomputed here (cached frozen channels) may differ from the cached one by component signs
                flip = (hc * ref).sum((0, 1)) < 0
                model.nbmj.comp[:, flip] *= -1
                hc, _ = model.nbmj(chk, xc.shape[1])
                dmax = float((hc - ref).abs().max())
            print(f"fold {f}: --nbm-joint {args.nbm_joint:g} (NBM lr {args.lr * args.nbm_joint:.1e}, NLL weight "
                  f"{args.nbm_joint_nll:g}); start check torch vs cached numba channels on {len(chk)} series: "
                  f"{int(flip.sum())} component sign(s) flipped, max |diff| {dmax:.1e}", flush=True)
        if args.arch == "qattn" and f == 0:
            qa_causal_check(model, va[:16])
            print(f"qattn: {len(model.gru.groups)} tokens per step, "
                  f"{sum(p.numel() for p in model.parameters()):,} parameters", flush=True)
        if JOINT:
            jp = {id(p) for p in model.nbmj.parameters()}
            opt = torch.optim.AdamW(
                [{"params": [p for p in model.parameters() if id(p) not in jp], "mult": 1.0},
                 {"params": list(model.nbmj.parameters()), "lr": args.lr * args.nbm_joint,
                  "weight_decay": args.nbm_wd, "mult": args.nbm_joint}], lr=args.lr, weight_decay=args.wd)
        else:
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        gstep = 0   # optimiser steps so far (--warmup)
        best, bad, best_logits, best_ep = -1.0, 0, None, 0
        if SWA:
            wsum, wcnt, lsum, fix_logits = None, 0, {}, None
        for ep in range(args.epochs):
            model.train()
            losses = []
            jnlls = []
            for b in make_batches(tr, lens + args.burn, args.batch, rng, True):
                x, y, w, idx, Lb = get_batch(b, True)
                jnll = None
                if JOINT:
                    x, jnll = joint_inputs(model, x, b, y)
                logit, tsb, btype = model(x, aux=True)
                yt = G["YD"][idx, :Lb] if args.distill > 0 else y   # soft targets for the BCE only
                bce = nn.functional.binary_cross_entropy_with_logits(logit, yt, reduction="none")
                if args.focal > 0:
                    p = torch.sigmoid(logit)
                    pt = torch.where(y > 0.5, p, 1 - p)
                    bce = bce * (1 - pt).pow(args.focal)
                loss = (bce * w).sum() / w.sum().clamp(min=1e-8)
                if args.rank_loss > 0:
                    # every positive vs every negative series of the batch at the SAME step, both in their online
                    # part (w > 0); pair weight = product of the two series' relative weights. Pair counts already
                    # grow with npos*nneg per step, which is the metric's own step weight.
                    valid = w > 0
                    pos = (y > 0.5) & valid
                    neg = (y < 0.5) & valid
                    r = G["R"][idx]
                    pw = (pos[:, None, :] & neg[None, :, :]).float() * (r[:, None, None] * r[None, :, None])
                    d = logit[:, None, :] - logit[None, :, :]
                    rank = (nn.functional.softplus(-d) * pw).sum() / pw.sum().clamp(min=1e-8)
                    loss = (1 - args.rank_loss) * loss + args.rank_loss * rank
                if tsb is not None:
                    m = (y > 0.5) & (w > 0)
                    loss = loss + args.aux_tsb * (((tsb - G["D"][idx, :Lb]) ** 2) * m).sum() / m.sum().clamp(min=1)
                if btype is not None:
                    # masked mean without boolean indexing (no GPU->CPU sync per batch)
                    k = G["K"][idx]
                    m = ((y > 0.5) & (w > 0) & (k[:, None] >= 0)).reshape(-1).float()
                    tgt = k.clamp(min=0)[:, None].expand(-1, Lb).reshape(-1)
                    ce = nn.functional.cross_entropy(btype.reshape(-1, btype.shape[-1]), tgt, reduction="none")
                    loss = loss + args.aux_type * (ce * m).sum() / m.sum().clamp(min=1)
                opt.zero_grad()
                if jnll is None:
                    loss.backward()
                else:   # the printed train loss stays the detector loss alone
                    (loss + args.nbm_joint_nll * jnll).backward()
                    jnlls.append(jnll.detach())
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if gstep < args.warmup:   # linear warmup on top of the epoch cosine (closed form, see --warmup)
                    lr_now = args.lr * 0.5 * (1 + np.cos(np.pi * ep / args.epochs)) * (gstep + 1) / args.warmup
                    for g in opt.param_groups:
                        g["lr"] = lr_now * g.get("mult", 1.0)
                elif gstep == args.warmup and args.warmup > 0:
                    for g in opt.param_groups:
                        g["lr"] = args.lr * 0.5 * (1 + np.cos(np.pi * ep / args.epochs)) * g.get("mult", 1.0)
                gstep += 1
                opt.step()
                losses.append(loss.detach())  # no .item() here: it would force a GPU sync every batch
            train_loss = float(torch.stack(losses).mean())
            sched.step()
            model.eval()
            val_logits = {}
            with torch.no_grad():
                for b in make_batches(va, lens + args.burn, 128, rng, False):
                    x, _, _ = get_batch(b, False)
                    if JOINT:
                        x, _ = joint_inputs(model, x, b)
                    lg = model(x).cpu().numpy()
                    for k, i in enumerate(b):
                        val_logits[i] = lg[k, args.burn: args.burn + lens[i]]
            s_all = np.concatenate([val_logits[i][steps_sub[i]] for i in va])
            y_all = np.concatenate([Y[i][steps_sub[i]] for i in va]).astype(np.int8)
            st_all = np.concatenate([steps_sub[i] for i in va])
            sc = ts_auc(s_all, y_all, st_all)
            flag = ""
            if sc > best:
                best, bad, best_logits, flag, best_ep = sc, 0, val_logits, "  *best", ep + 1
            else:
                bad += 1
            if SWA:
                e1 = ep + 1
                if e1 == args.fixed_ep:
                    fix_logits = val_logits
                if SWA_A <= e1 <= SWA_B:
                    sd = model.state_dict()
                    if wsum is None:
                        wsum = {k: v.detach().double().clone() for k, v in sd.items() if v.is_floating_point()}
                    else:
                        for k in wsum:
                            wsum[k] += sd[k].detach().double()
                    wcnt += 1
                    for i, lg in val_logits.items():
                        lsum[i] = lsum.get(i, 0.0) + lg.astype(np.float64)
            jtxt = f"  NBM NLL {float(torch.stack(jnlls).mean()):.4f}" if jnlls else ""
            print(f"fold {f} epoch {ep + 1:2d}  train loss {train_loss:.4f}{jtxt}  val TS-AUC {sc:.4f}{flag}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
            if SWA:
                if ep + 1 >= SWA_END:
                    break
            elif bad >= args.patience:
                print(f"  early stop (no improvement for {args.patience} epochs)")
                break
        fold_scores.append(best)
        if JOINT:
            dw = sum(float((p.detach() - q).square().sum()) for p, q in zip(model.nbmj.net.parameters(), jw0))
            w0 = sum(float(q.square().sum()) for q in jw0)
            print(f"fold {f}: --nbm-joint: NBM weights moved {100 * (dw / w0) ** 0.5:.2f}% (relative L2, last epoch)",
                  flush=True)
        if args.gate in ("static", "dynamic"):
            with torch.no_grad():
                b = make_batches(va, lens + args.burn, 256, rng, False)[0]
                x, _, _ = get_batch(b, False)
                gm = model.gate(x)[:, args.burn:].reshape(-1, CIN).mean(0).cpu().numpy()
            cn = channel_names()
            o = np.argsort(-gm)
            print(f"fold {f}: mean learned channel weight (1 = neutral), top: "
                  + ", ".join(f"{cn[j]} {gm[j]:.2f}" for j in o[:8])
                  + " | bottom: " + ", ".join(f"{cn[j]} {gm[j]:.2f}" for j in o[-5:]), flush=True)
        for i in va:
            oof[row_start[i]: row_start[i + 1]] = best_logits[i][steps_sub[i]]
        print(f"fold {f}: best TS-AUC {best:.4f} at epoch {best_ep}  ({time.time() - t0:.0f}s)", flush=True)
        if SWA:
            avg = copy.deepcopy(model)
            sd = model.state_dict()
            avg.load_state_dict({k: ((wsum[k] / wcnt).to(v.dtype) if k in wsum else v) for k, v in sd.items()})
            avg.eval()
            wav_logits = {}
            with torch.no_grad():
                for b in make_batches(va, lens + args.burn, 128, rng, False):
                    x, _, _ = get_batch(b, False)
                    if JOINT:
                        x, _ = joint_inputs(avg, x, b)
                    lg = avg(x).cpu().numpy()
                    for k, i in enumerate(b):
                        wav_logits[i] = lg[k, args.burn: args.burn + lens[i]]
            del avg
            got = {"wavg": wav_logits, "lavg": {i: v / wcnt for i, v in lsum.items()}, SWA_KEYS[2]: fix_logits}
            sc_k = {k: va_score(va, got[k]) for k in SWA_KEYS}
            for k in SWA_KEYS:
                swa_scores[k].append(sc_k[k])
                for i in va:
                    swa_oof[k][row_start[i]: row_start[i + 1]] = got[k][i][steps_sub[i]]
            for k in SWA_KEYS[:2]:
                swa_vs_fix[k].append(sc_k[k] - sc_k[SWA_KEYS[2]])
            line = "  ".join(f"{k} {sc_k[k]:.4f}" for k in SWA_KEYS)
            print(f"fold {f}: epochs {SWA_A}-{SWA_B} averaged ({wcnt} models): {line}   "
                  f"wavg - {SWA_KEYS[2]} {sc_k['wavg'] - sc_k[SWA_KEYS[2]]:+.4f}  "
                  f"lavg - {SWA_KEYS[2]} {sc_k['lavg'] - sc_k[SWA_KEYS[2]]:+.4f}  "
                  f"wavg - best {sc_k['wavg'] - best:+.4f}", flush=True)
        if base_oof is not None:
            mk = np.isin(sid_sub, va)
            b = ts_auc(base_oof[mk], y_sub[mk], step_sub[mk])
            diffs.append(best - b)
            print(f"fold {f}: baseline {b:.4f} -> this run {best:.4f}   paired difference {best - b:+.4f}", flush=True)
            if SWA:
                for k in SWA_KEYS:
                    swa_base[k].append(swa_scores[k][-1] - b)

    if dev.type == "cuda":
        print(f"\npeak GPU memory used: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    print(f"\nHEADLINE  {ARCH.upper()} mean TS-AUC = {np.mean(fold_scores):.4f}  (std over folds {np.std(fold_scores):.4f}, "
          f"{len(fold_scores)} fold(s))   LightGBM reference (270 features): about 0.6155")
    if diffs:
        d = np.array(diffs)
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
        print(f"PAIRED vs {args.compare}: {d.mean():+.4f} +- {se:.4f} (standard error), "
              f"{int((d > 0).sum())}/{len(d)} folds better")
    done = ~np.isnan(oof)
    print("\nOOF TS-AUC by online-step bucket (pooled over folds run):")
    for lo, hi in ((0, 100), (100, 300), (300, 600), (600, 10000)):
        mk = done & (step_sub >= lo) & (step_sub < hi)
        if mk.sum():
            print(f"  steps [{lo:4d}, {hi:5d}):  {ts_auc(oof[mk], y_sub[mk], step_sub[mk]):.4f}")
    np.savez(os.path.join(OUT, f"oof_{ARCH}{TAG}.npz"), oof=oof, y=y_sub, step=step_sub, sid=sid_sub)
    print(f"\nsaved {OUT}/oof_{ARCH}{TAG}.npz")
    if SWA:
        def _pm(v):
            d = np.array(v)
            se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
            return f"{d.mean():+.4f} +- {se:.4f} (SE), {int((d > 0).sum())}/{len(d)} folds > 0"
        print(f"\nEPOCH AVERAGING (epochs {SWA_A}-{SWA_B}; no validation-fold selection in any of these):")
        for k in SWA_KEYS:
            print(f"  {k:5s} mean TS-AUC {np.mean(swa_scores[k]):.4f}"
                  + (f"   vs --compare {_pm(swa_base[k])}" if swa_base[k] else ""))
        for k in SWA_KEYS[:2]:
            print(f"  PAIRED {k} - {SWA_KEYS[2]} (same run): {_pm(swa_vs_fix[k])}")
        for k in SWA_KEYS:
            fn = os.path.join(OUT, f"oof_{ARCH}{TAG}_{k}.npz")
            np.savez(fn, oof=swa_oof[k], y=y_sub, step=step_sub, sid=sid_sub)
            print(f"saved {fn}")

    if args.blend and not args.n_series:
        g = np.load(args.blend)
        og = g["oof"]
        both = done & ~np.isnan(og)
        if both.sum() == 0:
            print("\nblend: no overlapping out-of-fold rows")
        else:
            zg, zr = per_step_z(og, step_sub, both), per_step_z(oof, step_sub, both)
            print(f"\nBLEND with {args.blend} on {int(both.sum()):,} rows (scores standardised within each step):")
            for wg in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0):
                sc = ts_auc((1 - wg) * zg[both] + wg * zr[both], y_sub[both], step_sub[both])
                tag = "  <- LightGBM only" if wg == 0.0 else ("  <- GRU only" if wg == 1.0 else "")
                print(f"  weight on GRU {wg:.1f}: TS-AUC {sc:.4f}{tag}")
            print("A blend only helps if some mixture beats BOTH ends by more than the fold noise (~0.003).")


if __name__ == "__main__":
    main()
