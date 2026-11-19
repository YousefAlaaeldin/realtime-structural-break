

# =============================================================================================
# Submission entry points (appended to the embedded feature modules by make_submission.py)
#
# Model: blend of three members (HANDOFF sections 16-19, 26, 29, 44, 24 of the condensed file), CV TS-AUC 0.6531
#   NBM    a small GRU pretrained (no labels) on the training histories to predict the next AR residual; its online
#          prediction errors go through the BOCPD posterior with an empirical-Bayes prior fitted on training breaks
#   trees  XGBoost on 321 streaming features (kernel 270 + evidence 6 + volatility state 13 + distribution 16
#          + BOCPD 10 + NBM 6); two seeds, combined like the GRU pair
#   GRU    two seeds, 61 input channels per step (residual channels, history context, volatility state, BOCPD,
#          NBM, the NBM's hidden state projected on 8 principal components, top-20 kernel features),
#          numpy/numba forward pass at inference
#   SVM    least-squares SVM, RBF kernel via 8,192 random Fourier features on the top-100 tree features + the 8 NBM
#          hidden-state PCA channels (HANDOFF 27)
#   blend  0.4 trees + 0.5 GRU pair + 0.1 SVM on globally scaled margins/logits, then a sigmoid
# Training uses every original series plus 2 boundary re-cut variants per series, with metric-matched weights.
# =============================================================================================
import json
import math
import os
import time
from typing import Iterable, List, Optional, Tuple

import numpy as np
from joblib import Parallel, delayed
from numba import njit
from scipy.signal import lfilter

THREADS = min(8, os.cpu_count() or 1)
# read by the Crunch runner: score the test series in this many worker processes (each series is independent,
# so outputs are unchanged). The cloud CPU instances have 16 vCPUs and ~5M online points (HANDOFF 48); each worker
# needs about 0.5 GB and one CPU, since infer() limits numpy's BLAS to one thread per worker (_limit_blas)
INFER_PARALLELISM = 12
SUB = 3
AUG_N, AUG_SEED, AUG_MIN_SHIFT = 2, 0, 20
AR_G = 10

# cv.py --bocpd --nbm: the fold-averaged curve peaks at 150-250 rounds (0.6417 / 0.6414 / 0.6411) and declines slowly
TREE_ROUNDS = 200
TREE_PARAMS = dict(objective="binary:logistic", tree_method="hist", learning_rate=0.03, grow_policy="lossguide",
                   max_leaves=31, max_depth=0, min_child_weight=300.0, subsample=0.8, colsample_bytree=0.5,
                   reg_lambda=10.0, max_bin=255, seed=0, verbosity=1)
# HANDOFF 44.1: the average of two XGBoost seeds (same data and settings) gives +0.0015 in the trees, 0.6489 blend
TREE_SEEDS = (0, 1)

GRU_HIDDEN, GRU_DROPOUT, GRU_LR, GRU_WD, GRU_BATCH, GRU_BURN = 64, 0.1, 2e-3, 1e-4, 64, 100
# the gru_cv.py --vol --bocpd --nbm --nbm-hpca 8 runs peaked at epochs 4-8 (median 7) with the 25-epoch cosine
# schedule (the --nbm runs without the hidden state: 3-9, median 6)
GRU_EPOCHS, GRU_TMAX, GRU_MEM_BIAS, GRU_SEEDS = 8, 25, 1.0, (0, 1)
# HANDOFF 29: each seed's GRU = the equal-weight average of the weights after epochs GRU_SWA_A..GRU_SWA_B
# (gru_cv.py --swa 5-8; float64 sum; no batch norm, so nothing to recompute)
GRU_SWA_A, GRU_SWA_B = 5, 8
# HANDOFF 24: the NBM's 64-d GRU hidden state after each online point, projected on its first NBM_HPCA principal
# components (fitted on the online parts of every 10th original training series), / sd, clipped +-6
NBM_HPCA = 8

SVM_D, SVM_GMUL, SVM_LAM, SVM_SEED, SVM_CHUNK = 8192, 0.5, 0.3, 0, 32768
BLEND_W = (0.4, 0.4, 0.2)    # HANDOFF 29 blend (0.4 / 0.5 / 0.1 before)

# NBM pretraining (cv.py --nbm --nbm-batch 24 --nbm-epochs 25; the other settings are the defaults); 10% of the
# histories held out for early stopping, split as in cv.py's fold 0
NBM_EPOCHS, NBM_BATCH, NBM_LR, NBM_HIDDEN, NBM_WD, NBM_LOSS_FROM, NBM_SEED = 25, 24, 2e-3, 64, 1e-4, 50, 0

EV_N = EV.N_COMPACT
DIST_COLS = [j for j, n in enumerate(DT.FEATURE_NAMES) if not n.startswith("dist_z_")]
TREE_NAMES = (list(F.FEATURE_NAMES) + list(EV.FEATURE_NAMES[:EV_N]) + list(VS.FEATURE_NAMES)
              + [DT.FEATURE_NAMES[j] for j in DIST_COLS] + list(BO.FEATURE_NAMES) + list(NBM.FEATURE_NAMES))
N_TREE = len(TREE_NAMES)
O_DIST = F.N_FEAT + EV_N + VS.N_VOL                     # first column of each block in a tree row
O_BO = O_DIST + len(DIST_COLS)
O_NB = O_BO + BO.N_BO
SVM_COLS = [TREE_NAMES.index(n) for n in SVM_NAMES]
GRU_TOP_COLS = [F.FEATURE_NAMES.index(n) for n in GRU_TOP_NAMES]
G_VOL = 14                                              # first channel of each block in a GRU input row
G_BO = G_VOL + VS.N_GRU
G_NB = G_BO + BO.N_BO
G_HP = G_NB + NBM.N_NB
G_TOP = G_HP + NBM_HPCA
GRU_CIN = G_TOP + len(GRU_TOP_NAMES)

CONFIG = dict(version="blend-v5-swa", tree_rounds=TREE_ROUNDS, tree_seeds=list(TREE_SEEDS), nbm_hpca=NBM_HPCA,
              svm_hp=NBM_HPCA,
              gru_epochs=GRU_EPOCHS, gru_swa=[GRU_SWA_A, GRU_SWA_B], gru_seeds=list(GRU_SEEDS),
              svm=[SVM_D, SVM_GMUL, SVM_LAM, SVM_SEED], aug=[AUG_N, AUG_SEED, AUG_MIN_SHIFT], blend=list(BLEND_W),
              nbm=[NBM_EPOCHS, NBM_BATCH, NBM_LR, NBM_HIDDEN, NBM_WD, NBM_LOSS_FROM, NBM_SEED],
              n_tree=N_TREE, gru_cin=GRU_CIN, svm_cols=SVM_NAMES)
FILES = ("meta.json",) + tuple(f"trees_s{s}.json" for s in TREE_SEEDS) + ("svm.npz", "gru.npz", "nbm.npz")


def _clean(a):
    a = np.asarray(a, dtype=np.float64)
    if not np.isfinite(a).all():
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    return a


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------------------------
# features: batch (training) versions, identical to the streaming ones on the kept rows
def _slog(a):
    return (np.sign(a) * np.log1p(np.abs(a))).astype(np.float32)


def _series_feats(h, o, nbm, prior, sums=None):
    """Tree rows (every SUB-th step, N_TREE float32 columns) and the GRU's raw (signed-log) channels at every step:
    (top-K kernel features, BOCPD, NBM). `sums` = NBM.run_stream(h, o, nbm) if already computed."""
    K = F.run_series(h, o)
    E = EV.run_series_ev(h, o)[::SUB, :EV_N]
    V = VS.run_series_vol(h, o)[::SUB]
    D = DT.run_series_dist(h, o, SUB)[:, DIST_COLS]
    B = BO.run_series_bocpd(h, o)                                  # every step (the GRU needs all of them)
    N = NBM.features_from_sums(sums if sums is not None else NBM.run_stream(h, o, nbm), prior, 1)
    rows = np.hstack([K[::SUB].astype(np.float32), E.astype(np.float32), V.astype(np.float32), D.astype(np.float32),
                      B[::SUB].astype(np.float32), N[::SUB].astype(np.float32)])
    return rows, (_slog(K[:, GRU_TOP_COLS]), _slog(B), _slog(N))


def _gru_hist_consts(h):
    h = np.asarray(h, dtype=np.float64)
    mu, sd = h.mean(), h.std() + 1e-12
    zh = (h - mu) / sd
    nh = len(h)
    Xl = np.column_stack([zh[AR_G - 1 - i: nh - 1 - i] for i in range(AR_G)])
    coef = np.linalg.lstsq(Xl, zh[AR_G:], rcond=None)[0]
    e = lfilter(np.r_[1.0, -coef], [1.0], zh)
    se = max(float(e[2 * AR_G:nh].std()), 1e-8)
    rho1 = float(np.dot(zh[:-1], zh[1:]) / max(float(np.dot(zh, zh)), 1e-12))
    ctx = np.array([np.log(nh) - 8.0, np.clip(np.mean(zh ** 3), -5, 5) / 2.0,
                    np.log1p(max(float(np.mean(zh ** 4)) - 3.0, 0.0)) - 1.0, rho1, coef[0], np.log(se)])
    return mu, sd, zh, coef, se, ctx


def _gru_base(h, o, burn):
    """The 14 base channels (port of gru_cv.build_inputs without --tails): burn-in history rows + online rows."""
    mu, sd, zh, coef, se, ctx = _gru_hist_consts(h)
    zo = (np.asarray(o, dtype=np.float64) - mu) / sd
    s = np.r_[zh, zo]
    nh = len(zh)
    r = lfilter(np.r_[1.0, -coef], [1.0], s) / se
    st = nh - burn
    rr = r[st:]
    rp = np.r_[r[st - 1], rr[:-1]]
    ch = np.stack([np.clip(s[st:], -8, 8), np.clip(rr, -8, 8), np.minimum(rr ** 2, 25.0) - 1.0,
                   np.minimum(np.abs(rr), 8.0) - 0.8, np.clip(rr * rp, -25, 25),
                   (np.abs(rr) > 3.0).astype(np.float64)], axis=1)
    T = len(zo)
    online = np.r_[np.zeros(burn), np.ones(T)]
    tl = np.r_[np.zeros(burn), np.log1p(np.arange(T)) / 7.0]
    X = np.hstack([ch, np.tile(ctx, (burn + T, 1)), online[:, None], tl[:, None]])
    return np.nan_to_num(X).astype(np.float32)


GRU_STD = ("b", "n", "t")    # standardisation constants (<k>mu, <k>sd) for the BOCPD, NBM and top-K channels
HP_KEYS = ("h_mean", "h_comp", "h_sd")   # NBM hidden-state PCA: mean (64), components (8 x 64), scales (8)


@njit(nogil=True)
def _hp_proj(Hs, mean, comp, sd, out):
    """out[i] = clip(comp (Hs[i] - mean) / sd, +-6); the same loop for the batch rows and the streaming row."""
    for i in range(Hs.shape[0]):
        for k in range(comp.shape[0]):
            s = 0.0
            for j in range(comp.shape[1]):
                s += comp[k, j] * (Hs[i, j] - mean[j])
            s = s / sd[k]
            out[i, k] = min(max(s, -6.0), 6.0)


def _nbm_hp(h, o, nbm, P):
    """The NBM_HPCA hidden-state channels of one series' online points (float32, T x NBM_HPCA), batch path."""
    Hs = NBM.run_hidden(h, o, nbm)
    out = np.zeros((len(Hs), NBM_HPCA))
    _hp_proj(Hs, P["h_mean"], P["h_comp"], P["h_sd"], out)
    return out.astype(np.float32)


def _gru_inputs(h, o, raws, C, hp):
    """Full GRU input matrix (burn + T, GRU_CIN): base 14, volatility 3, BOCPD 10, NBM 6, NBM hidden-state PCA 8
    (`hp`, from _nbm_hp), top-K features; BOCPD / NBM / top-K are standardised signed logs, clipped at +-6; zeros in
    the burn-in for those four blocks (gru_cv.py --vol --bocpd --nbm --nbm-hpca 8)."""
    top, bo, nb = raws
    z = lambda c: np.zeros((GRU_BURN, c), np.float32)
    std = lambda raw, k: np.clip((raw - C[k + "mu"]) / C[k + "sd"], -6, 6).astype(np.float32)
    parts = [_gru_base(h, o, GRU_BURN), VS.gru_channels(h, o, GRU_BURN),
             np.vstack([z(bo.shape[1]), std(bo, "b")]), np.vstack([z(nb.shape[1]), std(nb, "n")]),
             np.vstack([z(NBM_HPCA), np.asarray(hp, np.float32)]), np.vstack([z(top.shape[1]), std(top, "t")])]
    return np.hstack(parts)


# ---------------------------------------------------------------------------------------------
# GRU forward pass in numba (inference), same equations as torch.nn.GRU (gates r, z, n) with exact GELU
@njit(nogil=True)
def _gelu(v):
    return 0.5 * v * (1.0 + math.erf(v / math.sqrt(2.0)))


@njit(nogil=True)
def _gru_step(x, h, Wi, bi, Wih, bih, Whh, bhh, a, gi, gh):
    H = h.shape[0]
    for j in range(H):
        s = bi[j]
        for k in range(x.shape[0]):
            s += Wi[j, k] * x[k]
        a[j] = _gelu(s)
    for j in range(3 * H):
        s1 = bih[j]
        s2 = bhh[j]
        for k in range(H):
            s1 += Wih[j, k] * a[k]
            s2 += Whh[j, k] * h[k]
        gi[j] = s1
        gh[j] = s2
    for j in range(H):
        r = 1.0 / (1.0 + math.exp(-(gi[j] + gh[j])))
        z = 1.0 / (1.0 + math.exp(-(gi[H + j] + gh[H + j])))
        n = math.tanh(gi[2 * H + j] + r * gh[2 * H + j])
        h[j] = (1.0 - z) * n + z * h[j]


@njit(nogil=True)
def _gru_head(h, W1, b1, W2, b2, t):
    H = h.shape[0]
    for j in range(H):
        s = b1[j]
        for k in range(H):
            s += W1[j, k] * h[k]
        t[j] = _gelu(s)
    out = b2[0]
    for k in range(H):
        out += W2[0, k] * t[k]
    return out


class _GruNet:
    """One trained GRU (float64 copies of the torch weights) with its own hidden state."""
    KEYS = ("inp.0.weight", "inp.0.bias", "gru.weight_ih_l0", "gru.bias_ih_l0", "gru.weight_hh_l0", "gru.bias_hh_l0",
            "head.0.weight", "head.0.bias", "head.3.weight", "head.3.bias")

    def __init__(self, W):
        self.W = [np.ascontiguousarray(W[k], dtype=np.float64) for k in self.KEYS]
        self.H = self.W[1].shape[0]

    def new_state(self):
        """Hidden state plus private scratch buffers (several series/threads can share one net)."""
        H = self.H
        return [np.zeros(H), np.zeros(H), np.zeros(3 * H), np.zeros(3 * H), np.zeros(H)]

    def step(self, x, st):
        Wi, bi, Wih, bih, Whh, bhh = self.W[:6]
        _gru_step(x, st[0], Wi, bi, Wih, bih, Whh, bhh, st[1], st[2], st[3])

    def head(self, st):
        return _gru_head(st[0], self.W[6], self.W[7], self.W[8], self.W[9], st[4])

    def run(self, X):
        """Logits for every row of X (burn-in rows included), hidden state from zero."""
        st = self.new_state()
        X = np.asarray(X, dtype=np.float64)
        out = np.zeros(len(X))
        for i in range(len(X)):
            self.step(X[i], st)
            out[i] = self.head(st)
        return out


class _GruIn:
    """Streaming GRU inputs for one series; the burn-in rows come from the batch builder."""

    def __init__(self, hist, C):
        self.mu, self.sd, zh, self.coef, self.se, self.ctx = _gru_hist_consts(hist)
        self.burn = np.hstack([_gru_base(hist, [], GRU_BURN), VS.gru_channels(hist, [], GRU_BURN),
                               np.zeros((GRU_BURN, GRU_CIN - G_BO), np.float32)])
        self.lag = list(zh[-AR_G:][::-1])                       # lag[i] = s_{t-1-i}
        e = lfilter(np.r_[1.0, -self.coef], [1.0], zh)
        self.rprev = float(e[-1] / self.se)
        self.C = C
        self.t = 0
        self.x = np.zeros(GRU_CIN)

    def _std(self, raw, k):
        return np.clip((_slog(raw) - self.C[k + "mu"]) / self.C[k + "sd"], -6, 6).astype(np.float32)

    def push(self, x, kfeat, vol_out, bo_out, nb_out, hp):
        s = (x - self.mu) / self.sd
        e = s
        for i in range(AR_G):
            e -= self.coef[i] * self.lag[i]
        r = e / self.se
        v = np.zeros(GRU_CIN)
        v[0] = min(max(s, -8.0), 8.0)
        v[1] = min(max(r, -8.0), 8.0)
        v[2] = min(r * r, 25.0) - 1.0
        v[3] = min(abs(r), 8.0) - 0.8
        v[4] = min(max(r * self.rprev, -25.0), 25.0)
        v[5] = 1.0 if abs(r) > 3.0 else 0.0
        v[6:12] = self.ctx
        v[12] = 1.0
        v[13] = math.log1p(self.t) / 7.0
        v = np.nan_to_num(v[:14]).astype(np.float32)
        u = vol_out[VS.N_VOL]
        vc = np.nan_to_num(np.array([u, min(u * u, VS.QCAP) - 1.0, vol_out[VS.N_VOL + 1]])).astype(np.float32)
        self.x[:G_VOL], self.x[G_VOL:G_BO] = v, vc
        self.x[G_BO:G_NB] = self._std(bo_out, "b")
        self.x[G_NB:G_HP] = self._std(nb_out, "n")
        self.x[G_HP:G_TOP] = hp
        self.x[G_TOP:] = self._std(kfeat[GRU_TOP_COLS], "t")
        self.lag = [s] + self.lag[:-1]
        self.rprev = r
        self.t += 1
        return self.x


# ---------------------------------------------------------------------------------------------
# trees: the XGBoost model flattened into arrays and evaluated by numba (XGBoost's per-row predict call costs
# ~250 us; this ~10 us). Same rule as XGBoost: go left if x < threshold (float32), missing -> default direction.
def _flatten_trees(booster):
    model = json.loads(booster.save_raw("json"))["learner"]
    trees = model["gradient_booster"]["model"]["trees"]
    bs = model["learner_model_param"]["base_score"].strip("[]")
    base = float(bs)
    obj = model["objective"]["name"]
    if obj.startswith("binary:logistic"):
        base = math.log(base / (1.0 - base))
    off = [0]
    for t in trees:
        off.append(off[-1] + len(t["left_children"]))
    cat = lambda key, dt: np.concatenate([np.asarray(t[key], dtype=dt) for t in trees])
    return dict(off=np.array(off, dtype=np.int64), left=cat("left_children", np.int64),
                right=cat("right_children", np.int64), feat=cat("split_indices", np.int64),
                thr=cat("split_conditions", np.float32), dleft=cat("default_left", np.int64), base=base)


@njit(nogil=True)
def _trees_margin(x, off, left, right, feat, thr, dleft, base):
    s = base
    for t in range(len(off) - 1):
        o = off[t]
        n = 0
        while left[o + n] != -1:
            v = x[feat[o + n]]
            if v != v:
                n = left[o + n] if dleft[o + n] == 1 else right[o + n]
            elif v < thr[o + n]:
                n = left[o + n]
            else:
                n = right[o + n]
        s += thr[o + n]
    return s


def _trees_predict(T, X):
    X = np.asarray(X, dtype=np.float32)
    return np.array([_trees_margin(X[i], T["off"], T["left"], T["right"], T["feat"], T["thr"], T["dleft"], T["base"])
                     for i in range(len(X))])


# ---------------------------------------------------------------------------------------------
# SVM
def _svm_prep(Xc, mu, sd):
    Z = np.sign(Xc) * np.log1p(np.abs(Xc))
    return np.clip((Z - mu) / sd, -6.0, 6.0).astype(np.float32)


def _svm_in(rows, hp):
    """SVM input rows (float32): the SVM_COLS tree columns, then the NBM_HPCA hidden-state channels at the same steps
    (cv.py --nbm-hpca 8 appends nbh_pc0..7 after the top-100 columns; HANDOFF 27)."""
    return np.hstack([np.asarray(rows, np.float32)[:, SVM_COLS], np.asarray(hp, np.float32)])


def _svm_score(S, Xc):
    """Decision values of the SVM for input rows Xc (from _svm_in)."""
    out = np.zeros(len(Xc))
    sc = math.sqrt(2.0 / SVM_D)
    for i in range(0, len(Xc), 8192):
        z = _svm_prep(Xc[i:i + 8192], S["mu"], S["sd"])
        out[i:i + 8192] = (np.cos(z @ S["Om"] + S["be"]) * sc) @ S["c"][:SVM_D] + S["c"][SVM_D]
    return out


def _svm_row(S, xc):
    """Decision value for one input row (SVM_COLS tree columns + hidden-state channels, as _svm_in)."""
    z = _svm_prep(xc, S["mu"], S["sd"])
    return float((np.cos(S["OmT"] @ z + S["be"]) * math.sqrt(2.0 / SVM_D)) @ S["c"][:SVM_D] + S["c"][SVM_D])


def _train_svm(X, HPr, w, y, torch, dev):
    """X: tree rows; HPr: the hidden-state channels of the same rows. gamma = SVM_GMUL / inputs (114), as cv.py."""
    K = len(SVM_COLS) + NBM_HPCA
    s1, s2 = np.zeros(K), np.zeros(K)
    for i in range(0, len(X), SVM_CHUNK):
        Z = _svm_in(X[i:i + SVM_CHUNK], HPr[i:i + SVM_CHUNK]).astype(np.float64)
        Z = np.sign(Z) * np.log1p(np.abs(Z))
        s1 += Z.sum(0)
        s2 += (Z * Z).sum(0)
    mu = s1 / len(X)
    sd = np.sqrt(np.maximum(s2 / len(X) - mu * mu, 0)) + 1e-6
    rng = np.random.default_rng(SVM_SEED)
    gamma = SVM_GMUL / K
    Om = (rng.standard_normal((K, SVM_D)) * np.sqrt(2.0 * gamma)).astype(np.float32)
    be = rng.uniform(0, 2 * np.pi, SVM_D).astype(np.float32)
    sc = float(np.sqrt(2.0 / SVM_D))
    Omt, bet = torch.as_tensor(Om, device=dev), torch.as_tensor(be, device=dev)
    A = torch.zeros(SVM_D + 1, SVM_D + 1, dtype=torch.float64, device=dev)
    bv = torch.zeros(SVM_D + 1, dtype=torch.float64, device=dev)
    wt = w.astype(np.float32)
    ypm = (2.0 * y - 1.0).astype(np.float32)
    for i in range(0, len(X), SVM_CHUNK):
        z = torch.as_tensor(_svm_prep(_svm_in(X[i:i + SVM_CHUNK], HPr[i:i + SVM_CHUNK]), mu, sd), device=dev)
        Fm = torch.cat([torch.cos(z @ Omt + bet) * sc, torch.ones(len(z), 1, device=dev)], dim=1)
        wc = torch.as_tensor(wt[i:i + SVM_CHUNK], device=dev)
        A += (Fm.T @ (Fm * wc[:, None])).double()
        bv += (Fm.T @ (wc * torch.as_tensor(ypm[i:i + SVM_CHUNK], device=dev))).double()
    A, bv = A.cpu().numpy(), bv.cpu().numpy()
    lam = SVM_LAM * float(np.trace(A)) / (SVM_D + 1)
    A[np.diag_indices_from(A)] += lam
    L = np.linalg.cholesky(A)
    c = np.linalg.solve(L.T, np.linalg.solve(L, bv))
    return dict(mu=mu, sd=sd, Om=Om, be=be, c=c)


# ---------------------------------------------------------------------------------------------
# GRU training (port of gru_cv.py: same network, weights, batching and schedule; fixed epochs instead of
# early stopping: the CV runs peaked at epochs 6-12, median 8, with the 25-epoch cosine schedule)
def _make_net(torch):
    nn = torch.nn
    H = GRU_HIDDEN

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Sequential(nn.Linear(GRU_CIN, H), nn.GELU())
            self.gru = nn.GRU(H, H, num_layers=1, batch_first=True)
            self.head = nn.Sequential(nn.Linear(H, H), nn.GELU(), nn.Dropout(GRU_DROPOUT), nn.Linear(H, 1))
            for name, p in self.gru.named_parameters():
                if name.startswith("bias_ih"):
                    with torch.no_grad():
                        p[H:2 * H] += GRU_MEM_BIAS

        def forward(self, x):
            out, _ = self.gru(self.inp(x))
            return self.head(out).squeeze(-1)

    return Net()


def _make_batches(idx, lens, batch, rng, shuffle):
    idx = np.asarray(idx)
    key = lens[idx] + (rng.random(len(idx)) * 0.15 * lens[idx] if shuffle else 0)
    order = idx[np.argsort(key)]
    batches = [order[i:i + batch] for i in range(0, len(order), batch)]
    if shuffle:
        rng.shuffle(batches)
    return batches


def _train_gru(Xs, lens, taus, n_orig, seed, torch, dev):
    nn = torch.nn
    n = len(Xs)
    burn = GRU_BURN
    Y = [((np.arange(L) >= t) & (t >= 0)).astype(np.float32) for L, t in zip(lens, taus)]
    tot = np.array([len(x) for x in Xs])
    Lmax = int(tot.max())
    Xp = np.zeros((n, Lmax, GRU_CIN), dtype=np.float32)
    Yp = np.zeros((n, Lmax), dtype=np.float32)
    for i, x in enumerate(Xs):
        Xp[i, :len(x)] = x
        Yp[i, burn:len(x)] = Y[i]
    Xg, Yg = torch.from_numpy(Xp).to(dev), torch.from_numpy(Yp).to(dev)
    del Xp, Yp
    # metric-matched weights, per-step class targets from the original series (gru_cv.py, --aug-rel 1)
    Tm = int(lens.max())
    npos, nall = np.zeros(Tm), np.zeros(Tm)
    for i in range(n_orig):
        nall[:lens[i]] += 1
        npos[:lens[i]] += Y[i]
    nneg = nall - npos
    rpos, rneg = np.zeros(Tm), np.zeros(Tm)
    for i in range(n):
        rpos[:lens[i]] += Y[i]
        rneg[:lens[i]] += 1.0 - Y[i]
    tgt = npos * nneg / 2.0
    wpos, wneg = tgt / np.maximum(rpos, 1e-12), tgt / np.maximum(rneg, 1e-12)
    Wl, s_tot, s_cnt = [], 0.0, 0
    for i in range(n):
        w = np.where(Y[i] > 0, wpos[:lens[i]], wneg[:lens[i]])
        Wl.append(w.astype(np.float32))
        s_tot += float(w.sum())
        s_cnt += lens[i]
    scale = s_cnt / max(s_tot, 1e-12)
    Wp = np.zeros((n, Lmax), dtype=np.float32)
    for i in range(n):
        Wp[i, burn:burn + lens[i]] = Wl[i] * scale
    Wg = torch.from_numpy(Wp).to(dev)
    del Wp
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = _make_net(torch).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=GRU_LR, weight_decay=GRU_WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=GRU_TMAX)
    wsum, wcnt = None, 0
    for ep in range(GRU_EPOCHS):
        model.train()
        losses = []
        for b in _make_batches(np.arange(n), lens + burn, GRU_BATCH, rng, True):
            Lb = int(tot[b].max())
            idx = torch.as_tensor(b, dtype=torch.long, device=dev)
            x, y, w = Xg[idx, :Lb], Yg[idx, :Lb], Wg[idx, :Lb]
            logit = model(x)
            bce = nn.functional.binary_cross_entropy_with_logits(logit, y, reduction="none")
            loss = (bce * w).sum() / w.sum().clamp(min=1e-8)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.detach())
        sched.step()
        _log(f"GRU seed {seed} epoch {ep + 1}/{GRU_EPOCHS}: train loss {float(torch.stack(losses).mean()):.4f}")
        if GRU_SWA_A <= ep + 1 <= GRU_SWA_B:
            sd = model.state_dict()
            if wsum is None:
                wsum = {k: v.detach().double().clone() for k, v in sd.items() if v.is_floating_point()}
            else:
                for k in wsum:
                    wsum[k] += sd[k].detach().double()
            wcnt += 1
    sd = model.state_dict()
    model.load_state_dict({k: ((wsum[k] / wcnt).to(v.dtype) if k in wsum else v) for k, v in sd.items()})
    _log(f"GRU seed {seed}: weights averaged over epochs {GRU_SWA_A}-{GRU_SWA_B} ({wcnt} models)")
    model.eval()
    W = {k: v.detach().cpu().numpy().astype(np.float64) for k, v in model.state_dict().items()}
    # in-sample logits of the original series at the kept steps (blend scale)
    outs = []
    with torch.no_grad():
        for b in _make_batches(np.arange(n_orig), lens + burn, 128, rng, False):
            Lb = int(tot[b].max())
            lg = model(Xg[torch.as_tensor(b, dtype=torch.long, device=dev), :Lb]).cpu().numpy()
            for k, i in enumerate(b):
                outs.append((i, lg[k, burn:burn + lens[i]][::SUB]))
    del Xg, Yg, Wg
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    outs.sort(key=lambda t: t[0])
    return W, np.concatenate([o for _, o in outs])


# ---------------------------------------------------------------------------------------------
def train(
    datasets: List[Tuple[int, List[float], List[float], Optional[int]]],
    model_directory_path: str,
):
    """Train the three blend members on all training series (+2 boundary re-cut variants each) and save them.
    Skips training if a complete model set with the same configuration is already in model_directory_path."""
    os.makedirs(model_directory_path, exist_ok=True)
    meta_path = os.path.join(model_directory_path, "meta.json")
    if all(os.path.exists(os.path.join(model_directory_path, f)) for f in FILES):
        try:
            if json.load(open(meta_path))["config"] == CONFIG:
                _log("complete model set with this configuration found -> skipping training")
                return
        except Exception:
            pass
    import torch
    import xgboost as xgb
    t_start = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    _log(f"training on {len(datasets)} series; device {dev}")

    S = [(i, _clean(h), _clean(o)) for i, (_, h, o, _t) in enumerate(datasets)]
    tau = np.array([-1 if t is None else int(t) for _, _, _, t in datasets])
    V = AUG.make_variants(S, tau, AUG_N, seed=AUG_SEED, min_shift=AUG_MIN_SHIFT)
    items = [(h, o, int(tau[i])) for i, h, o in S] + [(v[2], v[3], int(v[4])) for v in V]
    n_orig = len(S)
    _log(f"{n_orig} original + {len(V)} re-cut series")

    # 0. NBM: pretrain on the ORIGINAL histories (10% held out for early stopping), run it over every series, fit the
    #    empirical-Bayes prior on the original training breaks (cv.py nbm_fold)
    t0 = time.time()
    trp = np.random.default_rng(1000 + NBM_SEED).permutation(n_orig)
    nva = max(1, n_orig // 10)
    _log(f"NBM pretraining on {dev}: {n_orig - nva} + {nva} histories, {NBM_EPOCHS} epochs")
    state = NBM.train_model([S[i][1] for i in trp[nva:]], [S[i][1] for i in trp[:nva]], epochs=NBM_EPOCHS,
                            batch=NBM_BATCH, lr=NBM_LR, seed=NBM_SEED, verbose=True, hidden=NBM_HIDDEN, wd=NBM_WD,
                            loss_from=NBM_LOSS_FROM)
    nbm_w = {k: v.numpy() for k, v in state.items()}
    nbm = NBM.model_arrays(nbm_w)
    t1 = time.time()
    sums = NBM.sums_parallel([(h, o) for h, o, _ in items], nbm, THREADS)
    prior, nfit = NBM.fit_prior_w([NBM.w_of(sums[i]) for i in range(n_orig)], tau[:n_orig])
    np.savez(os.path.join(model_directory_path, "nbm.npz"), prior=np.asarray(prior, dtype=np.float64), **nbm_w)
    _log(f"NBM: pretrained in {t1 - t0:.0f}s, streams {time.time() - t1:.0f}s; prior from {nfit} breaks "
         f"(kappa0 {prior[0]:.0f}, alpha0 {prior[1]:.2f}, beta0 {prior[2]:.2f})")
    # 0b. NBM hidden-state PCA (gru_cv.py nbm_hidden_fold): every 10th original series' online hidden states
    t1 = time.time()
    samp = Parallel(n_jobs=THREADS, backend="threading")(
        delayed(NBM.run_hidden)(items[i][0], items[i][1], nbm) for i in range(0, n_orig, 10))
    Hs = np.concatenate(samp)
    del samp
    h_mean = Hs.mean(0)
    sv, Vt = np.linalg.svd(Hs - h_mean, full_matrices=False)[1:]
    h_comp = np.ascontiguousarray(Vt[:NBM_HPCA])
    h_sd = ((Hs - h_mean) @ h_comp.T).std(0) + 1e-6
    ev = (sv ** 2) / (sv ** 2).sum()
    del Hs
    HP = dict(h_mean=h_mean, h_comp=h_comp, h_sd=h_sd)
    hps = Parallel(n_jobs=THREADS, backend="threading")(delayed(_nbm_hp)(h, o, nbm, HP) for h, o, _ in items)
    _log(f"NBM hidden-state PCA: {NBM_HPCA} components explain {100 * ev[:NBM_HPCA].sum():.1f}% of the variance; "
         f"channels for {len(items)} series ({time.time() - t1:.0f}s)")

    # 1. features (tree rows every SUB-th step; GRU raw channels every step)
    t0 = time.time()
    counts = np.array([(len(o) + SUB - 1) // SUB for _, o, _ in items])
    off = np.r_[0, np.cumsum(counts)]
    X = np.empty((int(off[-1]), N_TREE), dtype=np.float32)
    raws = [None] * len(items)

    def _fill(i):
        rows, raw = _series_feats(items[i][0], items[i][1], nbm, prior, sums[i])
        X[off[i]:off[i + 1]] = rows
        raws[i] = raw
        sums[i] = None
    Parallel(n_jobs=THREADS, backend="threading")(delayed(_fill)(i) for i in range(len(items)))
    del sums
    step = np.concatenate([np.arange(0, len(o), SUB) for _, o, _ in items])
    ti = np.repeat(np.array([t for _, _, t in items]), counts)
    y = ((ti >= 0) & (step >= ti)).astype(np.int8)
    is_orig = np.arange(len(y)) < off[n_orig]
    w = AUG.step_weights_mixed(y, step, is_orig, 1.0)
    _log(f"features: {X.shape[0]:,} rows x {N_TREE} ({time.time() - t_start:.0f}s)")

    # 2. trees: one booster per seed on the same data and settings (HANDOFF 44.1)
    t0 = time.time()
    p = dict(TREE_PARAMS, device="cuda" if dev.type == "cuda" else "cpu", nthread=THREADS)
    dtrain = xgb.QuantileDMatrix(X, label=y, weight=w, max_bin=255)
    boosters = [xgb.train(dict(p, seed=sd_), dtrain, TREE_ROUNDS) for sd_ in TREE_SEEDS]
    del dtrain                                    # free the training matrix before copying the original rows
    dorig = xgb.DMatrix(X[is_orig], nthread=THREADS)
    m_trees = []
    for sd_, booster in zip(TREE_SEEDS, boosters):
        booster.set_param({"device": "cpu", "nthread": THREADS})
        m_trees.append(booster.predict(dorig, output_margin=True))
        booster.save_model(os.path.join(model_directory_path, f"trees_s{sd_}.json"))
    del dorig, boosters
    _log(f"trees: {len(TREE_SEEDS)} seeds x {TREE_ROUNDS} rounds ({time.time() - t0:.0f}s)")

    # 3. SVM
    t0 = time.time()
    HPr = np.concatenate([hp[::SUB] for hp in hps])        # hidden-state channels at the tree rows' steps
    assert len(HPr) == len(X)
    Sv = _train_svm(X, HPr, w, y, torch, dev)
    m_svm = _svm_score(Sv, _svm_in(X[is_orig], HPr[is_orig]))
    np.savez(os.path.join(model_directory_path, "svm.npz"), **Sv)
    _log(f"SVM: D={SVM_D}, {len(SVM_COLS)} tree columns + {NBM_HPCA} hidden-state channels ({time.time() - t0:.0f}s)")
    del X, HPr, y, w, step, ti, is_orig

    # 4. GRU pair
    t0 = time.time()
    C = {}
    for j, k in ((1, "b"), (2, "n"), (0, "t")):   # constants from every 10th original series (gru_cv.py)
        sample = np.concatenate([r[j] for r in raws[:n_orig][::10]])
        C[k + "mu"], C[k + "sd"] = sample.mean(axis=0), sample.std(axis=0) + 1e-6
    del sample
    Xs = Parallel(n_jobs=THREADS, backend="threading")(
        delayed(_gru_inputs)(h, o, raws[i], C, hps[i]) for i, (h, o, _) in enumerate(items))
    del raws, hps
    lens = np.array([len(o) for _, o, _ in items])
    taus = np.array([t for _, _, t in items])
    _log(f"GRU inputs built ({time.time() - t0:.0f}s)")
    gw, g_ins = {}, []
    for sd_ in GRU_SEEDS:
        W, g = _train_gru(Xs, lens, taus, n_orig, sd_, torch, dev)
        for k, v in W.items():
            gw[f"s{sd_}|{k}"] = v
        g_ins.append(g)
    np.savez(os.path.join(model_directory_path, "gru.npz"), **C, **HP, **gw)
    _log(f"GRU: {len(GRU_SEEDS)} seeds x {GRU_EPOCHS} epochs ({time.time() - t0:.0f}s)")

    # 5. blend scales (in-sample, original rows, every SUB-th step), as blend.py: each model / its sd; a member with
    #    several seeds (trees, GRU) is the sum of seed / its sd, then divided by that sum's sd
    sg = [float(g.std()) for g in g_ins]
    gsum = sum(g / s for g, s in zip(g_ins, sg))
    st = [float(m.std()) for m in m_trees]
    tsum = sum(m / s for m, s in zip(m_trees, st))
    meta = dict(config=CONFIG, sd_tree=st, sd_tsum=float(tsum.std()), sd_gru=sg, sd_gsum=float(gsum.std()),
                sd_svm=float(m_svm.std()), trained_series=n_orig, train_seconds=round(time.time() - t_start))
    json.dump(meta, open(meta_path, "w"), indent=1)
    _log(f"done in {time.time() - t_start:.0f}s; blend scales {meta}")


# ---------------------------------------------------------------------------------------------
def load_models(model_directory_path):
    import xgboost as xgb
    meta = json.load(open(os.path.join(model_directory_path, "meta.json")))
    boosters = [xgb.Booster(model_file=os.path.join(model_directory_path, f"trees_s{s}.json")) for s in TREE_SEEDS]
    for b in boosters:
        b.set_param({"device": "cpu", "nthread": 1})
    Sv = dict(np.load(os.path.join(model_directory_path, "svm.npz")))
    G = np.load(os.path.join(model_directory_path, "gru.npz"))
    nets = [_GruNet({k: G[f"s{s}|{k}"] for k in _GruNet.KEYS}) for s in GRU_SEEDS]
    T = [_flatten_trees(b) for b in boosters]
    Sv["OmT"] = np.ascontiguousarray(Sv["Om"].T)
    C = {k + s: G[k + s] for k in GRU_STD for s in ("mu", "sd")}
    HP = {k: np.ascontiguousarray(G[k], dtype=np.float64) for k in HP_KEYS}
    Nz = dict(np.load(os.path.join(model_directory_path, "nbm.npz")))
    return dict(meta=meta, boosters=boosters, trees=T, svm=Sv, nets=nets, gstd=C, hpca=HP,
                nbm=NBM.model_arrays(Nz), nbm_prior=np.asarray(Nz["prior"], dtype=np.float64))


def blend_scores(M, m_trees, g_list, m_svm):
    """Blended score (before the sigmoid) from the members' raw margins / logits (one per tree seed and GRU seed)."""
    me = M["meta"]
    t = sum(mi / si for mi, si in zip(m_trees, me["sd_tree"])) / me["sd_tsum"]
    g = sum(gi / si for gi, si in zip(g_list, me["sd_gru"])) / me["sd_gsum"]
    return BLEND_W[0] * t + BLEND_W[1] * g + BLEND_W[2] * m_svm / me["sd_svm"]


def fill_row(r, k, e, v, d, b, nb):
    """One tree row (float32) from the streams' outputs, in the training column order."""
    r[:F.N_FEAT] = k
    r[F.N_FEAT:F.N_FEAT + EV_N] = e[:EV_N]
    r[F.N_FEAT + EV_N:O_DIST] = v
    r[O_DIST:O_BO] = d[DIST_COLS]
    r[O_BO:O_NB] = b
    r[O_NB:] = nb


class SeriesScorer:
    """All streaming state for one series; push(x) returns the blended break probability for that point."""

    def __init__(self, hist, M):
        self.M = M
        self.fs, self.es, self.vs = F.Stream(hist), EV.EvStream(hist), VS.VolStream(hist)
        self.ds = DT.DistStream(hist)
        self.bs = BO.BoStream(hist)
        self.ns = NBM.NbmStream(hist, M["nbm"], M["nbm_prior"])
        self.gi = _GruIn(hist, M["gstd"])
        self.hs = []
        for net in M["nets"]:
            h = net.new_state()
            for r in self.gi.burn:
                net.step(np.asarray(r, dtype=np.float64), h)
            self.hs.append(h)
        self.row = np.zeros((1, N_TREE), dtype=np.float32)
        self.hp = np.zeros((1, NBM_HPCA))
        self.svx = np.zeros(len(SVM_COLS) + NBM_HPCA, dtype=np.float32)   # SVM input row (_svm_in layout)
        self.last = None

    def push(self, x):
        k = self.fs.push(x)
        e = self.es.push(x)
        v = self.vs.push(x)
        d = self.ds.push(x)
        b = self.bs.push(x)
        nb = self.ns.push(x)
        r = self.row[0]
        fill_row(r, k, e, v, d, b, nb)
        m_trees = [float(_trees_margin(r, T["off"], T["left"], T["right"], T["feat"], T["thr"], T["dleft"], T["base"]))
                   for T in self.M["trees"]]
        P = self.M["hpca"]
        _hp_proj(self.ns.h[None, :], P["h_mean"], P["h_comp"], P["h_sd"], self.hp)   # state after this point
        hp32 = self.hp[0].astype(np.float32)
        self.svx[:len(SVM_COLS)] = r[SVM_COLS]
        self.svx[len(SVM_COLS):] = hp32
        m_svm = _svm_row(self.M["svm"], self.svx)
        xg = np.asarray(self.gi.push(x, k, self.vs.out, b, nb, hp32), dtype=np.float64)
        gl = []
        for net, h in zip(self.M["nets"], self.hs):
            net.step(xg, h)
            gl.append(net.head(h))
        self.last = (m_trees, gl, m_svm)
        s = blend_scores(self.M, m_trees, gl, m_svm)
        return 1.0 / (1.0 + math.exp(-min(max(s, -50.0), 50.0)))


def infer(
    datasets: Iterable[Tuple[List[float], Iterable[float]]],
    model_directory_path: str,
):
    """Yield one score in [0, 1] per online point: the blended probability that a break has already happened,
    computed only from the history and the online points seen so far."""
    limiter = _limit_blas()
    try:
        t0 = time.time()
        M = load_models(model_directory_path)
        _log(f"infer pid {os.getpid()}: models loaded in {time.time() - t0:.1f}s")

        yield  # signal readiness

        # wall = everything between yields (incl. the runner's per-point round trip); own = our compute only
        # (series setup + push). own << wall means the runner, not this code, is the bottleneck (HANDOFF 49)
        t0, n_series, n_points, own = time.time(), 0, 0, 0.0
        clk = time.perf_counter
        for x_historical, x_online in datasets:
            c = clk()
            sc = SeriesScorer(_clean(x_historical), M)
            own += clk() - c
            for point in x_online:
                c = clk()
                x = float(point)
                if not np.isfinite(x):
                    x = 0.0
                n_points += 1
                p = sc.push(x)
                own += clk() - c
                yield p
            n_series += 1
            if n_series % INFER_LOG_EVERY == 0:
                _log_speed("", n_series, n_points, time.time() - t0, own)
        _log_speed("done, ", n_series, n_points, time.time() - t0, own)
    finally:
        if limiter is not None:
            limiter.restore_original_limits()


def _log_speed(tag, n_series, n_points, wall, own):
    n = max(n_points, 1)
    _log(f"infer pid {os.getpid()}: {tag}{n_series} series, {n_points:,} points, {wall:.0f}s, "
         f"{wall / n * 1e6:.0f} us/point wall, {own / n * 1e6:.0f} us/point own")


INFER_LOG_EVERY = 20    # progress line every this many series per worker (cloud log keeps ~1000/workers head lines; at 12 workers ~42 lines each, first one after ~10k points)


def _limit_blas():
    """One BLAS/OpenMP thread in this inference worker (HANDOFF 48). Otherwise numpy's BLAS starts one thread per CPU
    in EVERY worker process; with INFER_PARALLELISM workers those threads busy-wait between the per-point SVM products
    and starve the workers (first cloud run 2026-10-17: 4 workers x 16 threads on 16 vCPUs, over 2 h for ~5M points).
    Returns the limiter (restored when infer() ends) or None if threadpoolctl is unavailable."""
    try:
        from threadpoolctl import threadpool_info, threadpool_limits
        before = [(d["internal_api"], d["num_threads"]) for d in threadpool_info()]
        limiter = threadpool_limits(limits=1)
        after = [(d["internal_api"], d["num_threads"]) for d in threadpool_info()]
        _log(f"infer pid {os.getpid()}: BLAS/OpenMP threads {before} -> {after}")
        return limiter
    except Exception as e:      # never fail inference over this
        _log(f"infer pid {os.getpid()}: could not limit BLAS threads ({e!r})")
        return None
