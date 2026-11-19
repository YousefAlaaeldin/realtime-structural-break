"""Stage 1 for the binned-entropy DROP: is it a usable streaming break detector, alone or on top of the blend?

Background (HANDOFF section 30): entropy does not RISE at breaks, but the Shannon entropy of the residual over the
history's 10 decile bins FALLS after breaks and stays down. In the history every bin holds 10%, so entropy is at its
maximum; any change of distribution (mostly variance, up or down) makes the bins uneven. The entropy deficit
log(10) - H(p) equals KL(p || uniform), and 2 n KL ~ the chi-square of the counts, so this is close to the kernel's
`chi2_full` / `chi2_decay` (decile chi-square of the RAW series z: all online points, and a 0.97-decayed version;
importance ranks 61 and 152 of 321). New here: the residual e instead of z, fixed windows, and a running max.

Streaming features (causal: at online step t only points <= t are used), for v in {e, z}:
  all_v      (2 n KL(p_online || uniform) - 9) / sqrt(18), all online points so far (same scale as chi2_full)
  w{W}_v     entropy deficit over the last W points (window may reach into the history for small t), z-scored
             with the same rolling deficit over the series' own history; W = 30 / 50 / 100
  max{W}_v   running max of w{W}_v since the start of the online part (a drop that happened stays remembered)
and chi_z = the kernel's chi2_full formula, to show that all_z is the same information.

Reports:
  1. single-feature TS-AUC (all 10,000 series, rows = every 3rd online step like the OOF files), by step range,
     and on the series whose variance did not change (|log var ratio of e after tau| < 0.1, controls by pseudo-tau);
  2. Spearman correlation of each feature with chi_z;
  3. INCREMENTAL over the current best blend (0.5 trees_nbm + 0.3 GRU_bo pair + 0.2 SVM_nbm, out-of-fold):
     score = blend + lam * x, both standardised per online step, lam cross-fitted on the other 4 folds (same 5
     folds as blend.py --honest) from a grid that includes 0; x = each feature, and COMBO = the best linear
     combination of all entropy features (logistic regression on the training folds). Paired gain per fold vs the
     blend alone. (A LightGBM stacker was tried first and dropped: even its control lost 0.012 vs the plain blend.)

DECISION RULE (written before the run): adopt into cv.py only if the candidate beats the control by >= +0.001
TS-AUC with >= 4/5 folds better (the blend-level rule used for free features). Otherwise stop.
Expectation (stated before running): low. The event study's best post-break AUC for this statistic was about 0.56,
and the trees already see chi2 on z, the variance scans and the NBM.

Run:  python diagnostics/entropy_gate.py [n_series] [threads]     (default all 10,000 series, 8 processes)
Out:  cv_out/entropy_gate.txt
"""
import os
import sys
import time

import numpy as np
from scipy.signal import lfilter
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
os.chdir(HERE)

AR = 10
WS = (30, 50, 100)
VAR_SLICE = 0.1
BLEND = (("cv_out/oof_aug2_df_bo_nbm_bs24_ep25.npz",), ("cv_out/oof_gru_aug2_vol_bo.npz",
         "cv_out/oof_gru_aug2_vol_bo_s1.npz"), ("cv_out/oof_svm_nbm.npz",))
WEIGHTS = (0.5, 0.3, 0.2)
NAMES = ["chi_z"] + [f"{p}_{v}" for v in ("e", "z") for p in ["all"] + [f"w{W}" for W in WS] + [f"max{W}" for W in WS]]


def _deficit_rolling(b, W):
    """log(10) - H of the bin counts in the trailing window [t-W+1, t]; NaN until W points exist."""
    oh = np.zeros((len(b) + 1, 10), np.int32)
    oh[np.arange(1, len(b) + 1), b] = 1
    cs = np.cumsum(oh, axis=0)
    out = np.full(len(b), np.nan)
    t = np.arange(W - 1, len(b))
    p = (cs[t + 1] - cs[t + 1 - W]) / W
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -np.sum(np.where(p > 0, p * np.log(p), 0.0), axis=1)
    out[t] = np.log(10) - h
    return out


def series_feats(h, o, t, seed, pseudo):
    h = np.asarray(h, np.float64)
    o = np.asarray(o, np.float64)
    nh, no = len(h), len(o)
    mu, sd = h.mean(), h.std() + 1e-12
    z = np.r_[(h - mu) / sd, (o - mu) / sd]
    zh = z[:nh]
    Xl = np.column_stack([zh[AR - 1 - k: nh - 1 - k] for k in range(AR)])
    coef = np.linalg.lstsq(Xl, zh[AR:], rcond=None)[0]
    e = lfilter(np.r_[1.0, -coef], [1.0], z)
    se = max(float(e[2 * AR:nh].std()), 1e-8)
    e = np.clip(e / se, -10, 10)

    # variance-unchanged flag (oracle, used only to slice the report)
    tt = t if t >= 0 else int(np.random.default_rng(seed).choice(pseudo[pseudo < no])) if (pseudo < no).any() else 0
    post = e[nh + tt: nh + min(no, tt + 300)]
    vm = len(post) >= 30 and abs(np.log(post.var() / e[2 * AR:nh].var())) < VAR_SLICE

    rows = np.arange(0, no, 3)
    F = np.zeros((len(rows), len(NAMES)), np.float32)
    # v = z also reproduces the kernel's chi2_full (deciles of the history z)
    for v, x, h0 in (("z", z, 0), ("e", e, 2 * AR)):
        edges = np.quantile(x[h0:nh], np.linspace(0.1, 0.9, 9))
        b = (x[:, None] > edges[None, :]).sum(1)   # kernel convention: bin = number of edges strictly below
        bo = b[nh:]
        cnt = np.cumsum(np.eye(10, dtype=np.int32)[bo], axis=0)[rows].astype(np.float64)
        n = (rows + 1).astype(np.float64)
        if v == "z":
            F[:, 0] = (np.sum((cnt - n[:, None] / 10) ** 2 / (n[:, None] / 10), axis=1) - 9.0) / 4.243
        p = cnt / n[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            kl = np.sum(np.where(p > 0, p * np.log(p * 10), 0.0), axis=1)
        allc = (2 * n * kl - 9.0) / np.sqrt(18.0)
        ws, ms = [], []
        for W in WS:
            D = _deficit_rolling(b, W)
            dh = D[h0 + W - 1:nh]
            dh = dh[np.isfinite(dh)]
            m, s = (dh.mean(), max(dh.std(), 1e-6)) if len(dh) > 20 else (0.0, 1.0)
            Do = (D[nh:] - m) / s
            ws.append(Do[rows])
            ms.append(np.maximum.accumulate(Do)[rows])
        blk = [allc] + ws + ms
        base = 1 if v == "e" else 1 + 7
        for j, a in enumerate(blk):
            F[:, base + j] = a
    lab = (rows >= t).astype(np.int8) if t >= 0 else np.zeros(len(rows), np.int8)
    return F, lab, rows, vm


def main():
    import warnings
    from joblib import Parallel, delayed
    from sklearn.model_selection import StratifiedKFold

    from sbrt.data import load_train
    from sbrt.metric import ts_auc

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    n_series = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    threads = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    log = open(os.path.join("cv_out", "entropy_gate.txt"), "w", encoding="utf-8")

    def say(s=""):
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    S, tau = load_train()
    N = n_series or len(S)
    pseudo = np.asarray(tau[tau >= 0], np.int64)
    t0 = time.time()
    res = Parallel(n_jobs=threads, backend="loky", batch_size=32)(
        delayed(series_feats)(S[i][1], S[i][2], int(tau[i]), 9000 + i, pseudo) for i in range(N))
    F = np.concatenate([r[0] for r in res])
    y = np.concatenate([r[1] for r in res])
    step = np.concatenate([r[2] for r in res])
    sid = np.concatenate([np.full(len(r[2]), i) for i, r in enumerate(res)])
    vm_s = np.array([r[3] for r in res])
    say(f"entropy features for {N} series, {len(y)} rows, in {time.time() - t0:.0f}s")

    # ---- the current best blend, aligned to these rows
    groups = []
    for files in BLEND:
        zs = [np.load(f) for f in files]
        groups.append(sum(z["oof"] / z["oof"].std() for z in zs))
    ref = np.load(BLEND[0][0])
    keep = ref["sid"] < N
    if not (np.array_equal(ref["sid"][keep], sid) and np.array_equal(ref["step"][keep], step)):
        sys.exit("row layout differs from the OOF files")
    if not np.array_equal(ref["y"][keep], y):
        sys.exit("labels differ from the OOF files (sid order?)")
    blend = sum(w * g / g.std() for w, g in zip(WEIGHTS, groups))[keep]
    say(f"row layout and labels match the OOF files; blend TS-AUC on these rows {ts_auc(blend, y, step):.4f}")

    # ---- 1/2. single features
    vm = vm_s[sid]
    rng_b = ((0, 100), (100, 300), (300, 700), (700, 10 ** 6))
    say("")
    say(f"{'feature':10s} {'TS-AUC':>7s} {'0-100':>7s} {'100-300':>8s} {'300-700':>8s} {'700+':>7s} "
        f"{'var-same':>9s} {'rho(chi_z)':>10s}")
    sub = np.random.default_rng(0).choice(len(y), min(300000, len(y)), replace=False)
    for j, nm in enumerate(NAMES + ["BLEND"]):
        x = blend if nm == "BLEND" else F[:, j].astype(np.float64)
        x = np.nan_to_num(x)
        parts = [ts_auc(x[(step >= a) & (step < b)], y[(step >= a) & (step < b)], step[(step >= a) & (step < b)])
                 for a, b in rng_b]
        rho = spearmanr(x[sub], F[sub, 0]).correlation
        say(f"{nm:10s} {ts_auc(x, y, step):7.4f} " + " ".join(f"{p:7.4f}" for p in parts)
            + f" {ts_auc(x[vm], y[vm], step[vm]):9.4f} {rho:10.3f}")

    # ---- 3. incremental over the blend
    # score = blend + lam * x, both standardised per online step with constants from the TRAINING folds (shippable);
    # lam picked on the training folds from a grid that includes 0, so "no gain" reproduces the blend exactly.
    # x = each single feature, and COMBO = a logistic regression of y on all entropy features (fit on training folds,
    # per-step standardised inputs), i.e. the best linear entropy score.
    from sklearn.linear_model import LogisticRegression
    fold_of = np.zeros(len(tau), int)
    for f, (_, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(
            np.arange(len(tau)), (tau >= 0).astype(int))):
        fold_of[va] = f
    fr = fold_of[sid]
    T = int(step.max()) + 1
    LAMS = np.array([-0.3, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3])

    def pstd(x, tr):
        n = np.maximum(np.bincount(step[tr], minlength=T), 1)
        m = np.bincount(step[tr], weights=x[tr], minlength=T) / n
        v = np.bincount(step[tr], weights=x[tr] ** 2, minlength=T) / n - m * m
        return (x - m[step]) / np.sqrt(np.maximum(v[step], 1e-12))

    Fx = np.nan_to_num(F[:, 1:].astype(np.float64))
    cand = {nm: Fx[:, j] for j, nm in enumerate(NAMES[1:])}
    cand["chi_z"] = np.nan_to_num(F[:, 0].astype(np.float64))
    say("")
    say("incremental over the blend: score = blend + lam * x (per-step standardised; lam cross-fitted, grid incl. 0)")
    results = {}
    for nm in list(cand) + ["COMBO"]:
        d, lams = [], []
        for f in range(5):
            tr, va = fr != f, fr == f
            if not va.any() or not tr.any():
                continue
            bz = pstd(blend, tr)
            if nm == "COMBO":
                Z = np.column_stack([pstd(Fx[:, j], tr) for j in range(Fx.shape[1])])
                Z = np.clip(Z, -8, 8)
                sub = np.flatnonzero(tr)[::5]
                lr = LogisticRegression(C=0.1, max_iter=500).fit(np.column_stack([bz[sub], Z[sub]]), y[sub])
                x = pstd(Z @ lr.coef_[0, 1:], tr)
            else:
                x = np.clip(pstd(cand[nm], tr), -8, 8)
            sc_tr = [ts_auc(bz[tr] + l * x[tr], y[tr], step[tr]) for l in LAMS]
            lam = LAMS[int(np.argmax(sc_tr))]
            lams.append(lam)
            d.append(ts_auc(bz[va] + lam * x[va], y[va], step[va]) - ts_auc(bz[va], y[va], step[va]))
        d = np.array(d)
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
        results[nm] = d
        say(f"  {nm:10s} gain {d.mean():+.5f} +- {se:.5f}  {int((d > 0).sum())}/{len(d)} folds better  "
            f"(lam per fold {', '.join(f'{l:+.2f}' for l in lams)})")
    best = max(results, key=lambda k: results[k].mean())
    d = results[best]
    ok = d.mean() >= 0.001 and (d > 0).sum() >= 4
    say(f"  best: {best} {d.mean():+.5f}, {int((d > 0).sum())}/{len(d)} folds")
    say(f"  DECISION (>= +0.001 and >= 4/5 folds; note the best of {len(results)} candidates is slightly optimistic): "
        f"{'ADOPT -> wire into cv.py' if ok else 'REJECT -> stop'}")


if __name__ == "__main__":
    main()
