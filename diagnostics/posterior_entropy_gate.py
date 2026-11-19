"""Two untested entropy ideas, gated against the current best blend (0.6473).

A. Entropy of the BREAK-LOCATION posterior (bocpd.py / nbm.py). At every step the posterior already weighs every
   candidate break time j (given that a break happened by t): q_j ∝ pi_j exp(LR_j). The models get its odds
   (logodds) and its mean (tsb) but not how CONCENTRATED it is. A real break piles the evidence on one location
   (low entropy); noise spreads it (high entropy). Per stream, at every 3rd online step:
     pe_ent   entropy of q (nats)
     pe_kl    KL(q || prior over j given a break by t): how much the data sharpened the belief about WHERE
              (= E_q[LR] - log E_prior[exp(LR)])
     pe_sdl   posterior sd of log(1 + time since the break) (a spread measure next to tsb, which is the mean)
   Streams: bo_w, bo_wu (bocpd.py, fixed EB prior) and, with --nbm, nb (the NBM stream with the fold's own model
   and the prior fitted on the fold's training breaks, exactly as cv.py --nbm; out-of-fold for every series).
   Check: the recomputed bo_w logodds must equal data/bo1_orig.npz column 0 (same rows).

B. History entropy as CONTEXT (one number per series): spectral entropy (Welch, normalised) of the history z and of
   its AR(10) residual e (whiteness), permutation entropy order 3/4/5 of z and order 4 of e. History statistics carry
   no label signal by themselves (HANDOFF section 3), so a single-feature AUC is ~0.5 by construction. They can help
   only by correcting how the blend scores different kinds of series:
     additive        score = blend + lam * c       (blend scores systematically too high/low for some series types)
     multiplicative  score = blend * exp(g * c)    (trust the evidence more/less for some series types)

Gate for every candidate: single-feature TS-AUC (all, by step range, variance-unchanged slice), Spearman rho with the
existing bo_w logodds / tsb, and the INCREMENTAL gain over the blend with lam (or g) cross-fitted on the other 4
folds from a grid that includes 0 (cv.py's folds), per-step standardisation from the training folds; plus COMBO
(logistic mix of all posterior-entropy columns).
DECISION RULE (written before the full run): a candidate passes only with gain >= +0.001, >= 4/5 folds better AND
> 2.5 SE (about 22 candidates: the best of that many pure-noise candidates lands about +2 SE up). Passing means
CONFIRM, not adopt: posterior-shape features -> a paired cv.py tree run; a context transform -> rerun with
--fold-seed 1 and the same candidate must win again.
Controls (not candidates, never picked as "best"):
  POSITIVE  nb_logodds (out-of-fold, from cv.py's per-fold NBM caches) gated on the PRE-NBM blend (0.6438). The
            trees with the NBM added +0.0032 to the blend, so this shows whether the linear lam gate can see a
            feature of that kind. If it shows about 0, a REJECT of family A means "the linear screen cannot see it"
            and the honest next step for A is a cv.py tree run anyway (the idea is an interaction: same odds,
            sharper location). If it shows >= +0.001, a REJECT of A is a real reject.
  NEGATIVE  log(history length) in add and mult modes. spec_e / pe4_e have sd 0.002-0.003, the size of their
            estimation bias, which depends on history length (1,000-5,000 points). If log(nh) does as well as a
            context candidate, that candidate is a length effect, not entropy (and the trees already have ctx_logn).
Changes made after the 200-800-series smoke runs, before any full run: the lam grid was widened from +-0.3 to
+-0.6 (spec_e mult sat at the edge), and the controls and the 2.5 SE condition were added.

Run:  python diagnostics/posterior_entropy_gate.py [--n N] [--threads T] [--nbm]
      without --nbm: about 1-2 min (CPU). With --nbm: re-runs the 5 fold NBMs over all 10,000 series
      (about 2 min per fold at 16 threads; the fold weights data/nbm_v1_cv_f*_e25_h64_lr0.002_b24_wd0.0001_lf50.pt
      must exist, which they do after the adopted cv.py --nbm run).
      --fold-seed S re-draws the folds used to cross-fit lam (confirmation run for a context transform).
Out:  cv_out/posterior_entropy_gate[_nbm].txt
"""
import argparse
import math
import os
import sys
import time

import numpy as np
from numba import njit
from scipy.signal import lfilter, welch
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
os.chdir(HERE)

from sbrt import bocpd as BO  # noqa: E402

AR = 10
SUB = 3
VAR_SLICE = 0.1
BLEND = (("cv_out/oof_aug2_df_bo_nbm_bs24_ep25.npz",), ("cv_out/oof_gru_aug2_vol_bo.npz",
         "cv_out/oof_gru_aug2_vol_bo_s1.npz"), ("cv_out/oof_svm_nbm.npz",))
WEIGHTS = (0.5, 0.3, 0.2)
NBM_CFG = "e25_h64_lr0.002_b24_wd0.0001_lf50"
PSTATS = ("logodds", "tsb", "ent", "kl", "sdl")
CTX = ("spec_z", "spec_e", "pe3_z", "pe4_z", "pe5_z", "pe4_e")


@njit(cache=True, nogil=True)
def post_ent(C1, C2, k, n, prior, LOGH, LP, buf, out, o):
    """bocpd._posterior plus the shape of q_j: logodds, tsb, entropy, KL to the prior over j, sd of log(1+m)."""
    k0, a0, b0 = prior[0], prior[1], prior[2]
    base = a0 * math.log(b0) - math.lgamma(a0)
    mx = -1e300
    mx0 = -1e300
    for j in range(n):
        m = n - j
        S1 = C1[k, n] - C1[k, j]
        S2 = C2[k, n] - C2[k, j]
        kn = k0 + m
        an = a0 + 0.5 * m
        bn = b0 + 0.5 * (S2 - S1 * S1 / kn)
        if bn < 1e-12:
            bn = 1e-12
        lr = 0.5 * math.log(k0 / kn) + base - an * math.log(bn) + math.lgamma(an) + 0.5 * S2
        p0 = LP[j] + LOGH[j]
        buf[j] = p0 + lr
        if buf[j] > mx:
            mx = buf[j]
        if p0 > mx0:
            mx0 = p0
    s = 0.0
    s0 = 0.0
    ea = 0.0
    el = 0.0
    el2 = 0.0
    for j in range(n):
        wgt = math.exp(buf[j] - mx)
        s += wgt
        s0 += math.exp(LP[j] + LOGH[j] - mx0)
        ea += wgt * (buf[j] - mx)
        x = math.log(1.0 + (n - j))
        el += wgt * x
        el2 += wgt * x * x
    lse = mx + math.log(s)
    lse0 = mx0 + math.log(s0)
    out[o] = lse - LP[n]
    mean_l = el / s
    out[o + 1] = mean_l
    ent = math.log(s) - ea / s
    out[o + 2] = ent
    # KL(q || q0) = sum q (log q - log q0); log q_j = buf_j - lse, log q0_j = p0_j - lse0, buf_j - p0_j = lr_j
    # -> sum q lr - lse + lse0 ; and sum q lr = sum q buf - sum q p0. Compute via entropy identities:
    # sum q (buf - lse) = -ent  ->  KL = -ent - sum q (p0 - lse0)
    sp0 = 0.0
    for j in range(n):
        sp0 += math.exp(buf[j] - mx) * (LP[j] + LOGH[j])
    out[o + 3] = -ent - (sp0 / s - lse0)
    v = el2 / s - mean_l * mean_l
    out[o + 4] = math.sqrt(v if v > 0 else 0.0)


@njit(cache=True, nogil=True)
def rows_from_sums(C1, C2, nk, n_pts, prior, LOGH, LP, buf, sub, out):
    r = 0
    for t in range(0, n_pts, sub):
        for k in range(nk):
            post_ent(C1, C2, k, t + 1, prior, LOGH, LP, buf, out[r], 5 * k)
        r += 1


def bo_feats(h, o):
    Q, refs, ST, C1, C2 = BO.make_bo_params(h)
    o = np.asarray(o, np.float64)
    BO._bo_sums(o, Q, refs, C1, C2, ST, np.array(BO.WIDE), BO.LOGH, BO.LP, np.zeros(BO.CAP + 1), np.zeros(BO.N_BO))
    n = len(o)
    out = np.zeros(((n + SUB - 1) // SUB, 10))
    rows_from_sums(C1, C2, 2, n, np.array((BO.KAPPA0, BO.ALPHA0, BO.BETA0)), BO.LOGH, BO.LP, np.zeros(n + 1), SUB, out)
    return out


def _pe(x, m):
    W = np.lib.stride_tricks.sliding_window_view(x, m)
    r = W.argsort(axis=1, kind="stable")
    code = (r * (m ** np.arange(m))).sum(1)
    _, c = np.unique(code, return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log(p)).sum() / math.log(math.factorial(m)))


def _spec(x):
    f, P = welch(x - x.mean(), nperseg=min(256, len(x)))
    P = P[1:]
    p = P / P.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / math.log(len(P)))


def series_part(h, o, t, seed, pseudo):
    h = np.asarray(h, np.float64)
    o = np.asarray(o, np.float64)
    nh, no = len(h), len(o)
    zh = (h - h.mean()) / (h.std() + 1e-12)
    Xl = np.column_stack([zh[AR - 1 - k: nh - 1 - k] for k in range(AR)])
    coef = np.linalg.lstsq(Xl, zh[AR:], rcond=None)[0]
    z = np.r_[zh, (o - h.mean()) / (h.std() + 1e-12)]
    e = lfilter(np.r_[1.0, -coef], [1.0], z)
    eh = e[2 * AR:nh]
    ctx = np.array([_spec(zh), _spec(eh), _pe(zh, 3), _pe(zh, 4), _pe(zh, 5), _pe(eh, 4)])
    tt = t if t >= 0 else (int(np.random.default_rng(seed).choice(pseudo[pseudo < no])) if (pseudo < no).any() else 0)
    post = e[nh + tt: nh + min(no, tt + 300)]
    vm = len(post) >= 30 and abs(np.log(post.var() / eh.var())) < VAR_SLICE
    return bo_feats(h, o).astype(np.float32), ctx, vm


def main():
    import warnings
    from joblib import Parallel, delayed
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    from sbrt.data import load_train
    from sbrt.metric import ts_auc

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--nbm", action="store_true")
    ap.add_argument("--fold-seed", type=int, default=0)
    args = ap.parse_args()
    log = open(os.path.join("cv_out", f"posterior_entropy_gate{'_nbm' if args.nbm else ''}"
                            f"{f'_fs{args.fold_seed}' if args.fold_seed else ''}.txt"), "w",
               encoding="utf-8")

    def say(s=""):
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    S, tau = load_train()
    N = args.n or len(S)
    pseudo = np.asarray(tau[tau >= 0], np.int64)
    t0 = time.time()
    res = Parallel(n_jobs=args.threads, backend="loky", batch_size=16)(
        delayed(series_part)(S[i][1], S[i][2], int(tau[i]), 9000 + i, pseudo) for i in range(N))
    B = np.concatenate([r[0] for r in res])
    CT = np.stack([r[1] for r in res])
    vm_s = np.array([r[2] for r in res])
    nrow = np.array([len(r[0]) for r in res])
    sid = np.repeat(np.arange(N), nrow)
    say(f"BOCPD posterior shape + history entropies for {N} series ({len(sid)} rows) in {time.time() - t0:.0f}s")

    ref = np.load(BLEND[0][0])
    keep = ref["sid"] < N
    y, step = ref["y"][keep].astype(np.int8), ref["step"][keep]
    if not np.array_equal(ref["sid"][keep], sid):
        sys.exit("row layout differs from the OOF files")
    bo_cache = np.load(os.path.join("data", "bo1_orig.npz"))["B"][keep]
    dmax = float(np.abs(B[:, 0] - bo_cache[:, 0]).max())
    say(f"check: recomputed bo_w_logodds vs data/bo1_orig.npz max |diff| {dmax:.2e} "
        f"({'OK' if dmax < 1e-3 else 'MISMATCH'}); bo_w_tsb max |diff| {float(np.abs(B[:, 1] - bo_cache[:, 2]).max()):.2e}")
    groups = []
    for files in BLEND:
        zs = [np.load(f) for f in files]
        groups.append(sum(z["oof"] / z["oof"].std() for z in zs))
    blend = sum(w * g / g.std() for w, g in zip(WEIGHTS, groups))[keep]
    say(f"blend TS-AUC on these rows {ts_auc(blend, y, step):.4f}")

    cols = {}
    for k, v in enumerate(("w", "wu")):
        for j, st in enumerate(PSTATS):
            cols[f"bo_{v}_{st}"] = B[:, 5 * k + j].astype(np.float64)

    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    folds = list(skf.split(np.arange(len(tau)), (tau >= 0).astype(int)))
    fold_of = np.zeros(len(tau), int)
    for f, (_, va) in enumerate(folds):
        fold_of[va] = f
    fr_cv = fold_of[sid]                       # cv.py's folds (the NBM models and OOF caches follow these)
    if args.fold_seed:
        fo2 = np.zeros(len(tau), int)
        for f, (_, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=args.fold_seed).split(
                np.arange(len(tau)), (tau >= 0).astype(int))):
            fo2[va] = f
        fr = fo2[sid]
    else:
        fr = fr_cv

    if args.nbm:
        import torch
        from sbrt import nbm as NBM
        nb = np.zeros((len(sid), 5))
        starts = np.r_[0, np.cumsum(nrow)]
        for f, (tr_s, va_s) in enumerate(folds):
            wpath = os.path.join("data", f"nbm_v{NBM.VERSION}_cv_f{f}_{NBM_CFG}.pt")
            if not os.path.exists(wpath):
                sys.exit(f"missing {wpath} (run cv.py --nbm --nbm-batch 24 --nbm-epochs 25 first)")
            t1 = time.time()
            M = NBM.model_arrays(torch.load(wpath, map_location="cpu"))
            tr_s = tr_s[tr_s < N]
            va_s = va_s[va_s < N]
            need = np.r_[tr_s, va_s]
            sums = NBM.sums_parallel([(S[i][1], S[i][2]) for i in need], M, args.threads)
            sd = dict(zip(need.tolist(), sums))
            prior, nfit = NBM.fit_prior_w([NBM.w_of(sd[i]) for i in tr_s], tau[tr_s])
            pr = np.asarray(prior, np.float64)
            for i in va_s:
                C1, C2, _ = sd[i]
                n = C1.shape[1] - 1
                out = np.zeros(((n + SUB - 1) // SUB, 5))
                rows_from_sums(C1, C2, 1, n, pr, BO.LOGH, BO.LP, np.zeros(n + 1), SUB, out)
                nb[starts[i]:starts[i + 1]] = out
            say(f"fold {f}: NBM stream for {len(need)} series, prior from {nfit} breaks (kappa0 {pr[0]:.0f}), "
                f"{time.time() - t1:.0f}s")
        nbc = np.load(os.path.join("data", f"nbm{NBM.VERSION}_f0_{NBM_CFG}_orig.npz"))["B"][keep]
        m0 = fr_cv == 0
        say(f"check: recomputed nb_logodds vs cv.py fold-0 cache on fold-0 rows max |diff| "
            f"{float(np.abs(nb[m0, 0] - nbc[m0, 0]).max()):.2e} (should be ~0 on the full run; with --n the prior "
            f"is fitted on fewer breaks and differs)")
        for j, st in enumerate(PSTATS):
            cols[f"nb_{st}"] = nb[:, j]

    vm = vm_s[sid]
    rb = ((0, 100), (100, 300), (300, 700), (700, 10 ** 6))
    sub = np.random.default_rng(0).choice(len(y), min(300000, len(y)), replace=False)
    say("")
    say("A. posterior shape (logodds / tsb are the EXISTING features, shown for reference)")
    say(f"{'feature':14s} {'TS-AUC':>7s} {'0-100':>7s} {'100-300':>8s} {'300-700':>8s} {'700+':>7s} {'var-same':>9s}"
        f" {'rho(lo)':>8s} {'rho(tsb)':>8s}")
    for nm, x in list(cols.items()) + [("BLEND", blend)]:
        parts = []
        for a, b in rb:
            m = (step >= a) & (step < b)
            parts.append(ts_auc(x[m], y[m], step[m]))
        r1 = spearmanr(x[sub], cols["bo_w_logodds"][sub]).correlation
        r2 = spearmanr(x[sub], cols["bo_w_tsb"][sub]).correlation
        say(f"{nm:14s} {ts_auc(x, y, step):7.4f} " + " ".join(f"{p:7.4f}" for p in parts)
            + f" {ts_auc(x[vm], y[vm], step[vm]):9.4f} {r1:8.3f} {r2:8.3f}")

    say("")
    say("B. history entropy as context (one value per series; single AUC should be ~0.5 by construction)")
    lognh = np.log(np.array([len(S[i][1]) for i in range(N)], np.float64))
    for j, nm in enumerate(CTX):
        c = CT[:, j][sid]
        say(f"   {nm:8s} TS-AUC {ts_auc(c, y, step):.4f}   mean {CT[:, j].mean():.3f} sd {CT[:, j].std():.3f}"
            f"   rho with log(history length) {spearmanr(CT[:, j], lognh).correlation:+.3f}")

    # ---- incremental
    T = int(step.max()) + 1
    LAMS = np.array([-0.6, -0.4, -0.3, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6])

    def pstd(x, tr):
        n = np.maximum(np.bincount(step[tr], minlength=T), 1)
        m = np.bincount(step[tr], weights=x[tr], minlength=T) / n
        v = np.bincount(step[tr], weights=x[tr] ** 2, minlength=T) / n - m * m
        return (x - m[step]) / np.sqrt(np.maximum(v[step], 1e-12))

    def gate(make, mode, base=None):
        base = blend if base is None else base
        d, ls = [], []
        for f in range(5):
            tr, va = fr != f, fr == f
            if not va.any():
                continue
            bz = pstd(base, tr)
            x = make(tr, bz)
            comb = (lambda l, m: bz[m] + l * x[m]) if mode == "add" else (lambda l, m: bz[m] * np.exp(l * x[m]))
            lam = LAMS[int(np.argmax([ts_auc(comb(l, tr), y[tr], step[tr]) for l in LAMS]))]
            ls.append(lam)
            d.append(ts_auc(comb(lam, va), y[va], step[va]) - ts_auc(bz[va], y[va], step[va]))
        return np.array(d), ls

    say("")
    say("incremental over the blend (lam cross-fitted on the other 4 folds, grid incl. 0; per-step standardised)")
    results = {}
    newcols = [k for k in cols if not k.endswith(("logodds", "tsb"))]
    for nm in newcols:
        results[nm] = gate(lambda tr, bz, nm=nm: np.clip(pstd(cols[nm], tr), -8, 8), "add")

    def combo(tr, bz):
        Z = np.clip(np.column_stack([pstd(cols[k], tr) for k in newcols]), -8, 8)
        s_ = np.flatnonzero(tr)[::5]
        lr = LogisticRegression(C=0.1, max_iter=1000).fit(np.column_stack([bz[s_], Z[s_]]), y[s_])
        return np.clip(pstd(Z @ lr.coef_[0, 1:], tr), -8, 8)
    results["COMBO_post"] = gate(combo, "add")
    for j, nm in enumerate(CTX):
        c_all = CT[:, j][sid]

        def mk(tr, bz, c_all=c_all):
            mu, sd = c_all[tr].mean(), c_all[tr].std() + 1e-12
            return np.clip((c_all - mu) / sd, -4, 4)
        results[f"{nm} add"] = gate(mk, "add")
        results[f"{nm} mult"] = gate(mk, "mult")
    controls = {}

    def mk_len(tr, bz):
        x = lognh[sid]
        return np.clip((x - x[tr].mean()) / (x[tr].std() + 1e-12), -4, 4)
    controls["NEG log(nh) add"] = gate(mk_len, "add")
    controls["NEG log(nh) mult"] = gate(mk_len, "mult")
    if N == len(S):
        nbo = np.zeros(len(sid))
        for f in range(5):
            q = os.path.join("data", f"nbm1_f{f}_{NBM_CFG}_orig.npz")
            if not os.path.exists(q):
                nbo = None
                break
            m = fr_cv == f
            nbo[m] = np.load(q)["B"][:, 0][m]
        if nbo is not None:
            zs = [np.load(f_) for f_ in ("cv_out/oof_aug2_df_bo.npz",)]
            g_old = [sum(z["oof"] / z["oof"].std() for z in zs),
                     sum(np.load(f_)["oof"] / np.load(f_)["oof"].std() for f_ in
                         ("cv_out/oof_gru_aug2_vol_bo.npz", "cv_out/oof_gru_aug2_vol_bo_s1.npz")),
                     np.load("cv_out/oof_svm.npz")["oof"]]
            old = sum(w * g / g.std() for w, g in zip(WEIGHTS, g_old))
            say(f"pre-NBM blend TS-AUC {ts_auc(old, y, step):.4f} (expected about 0.6438)")
            controls["POS nb_logodds on pre-NBM blend"] = gate(
                lambda tr, bz: np.clip(pstd(nbo, tr), -8, 8), "add", base=old)
    else:
        say("(positive control needs the full 10,000 series: skipped)")

    def show(nm, d, ls):
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
        say(f"  {nm:32s} gain {d.mean():+.5f} +- {se:.5f}  {int((d > 0).sum())}/{len(d)} folds  "
            f"(lam {', '.join(f'{l:+.2f}' for l in ls)})")
        return se
    ses = {nm: show(nm, d, ls) for nm, (d, ls) in results.items()}
    say("  controls:")
    for nm, (d, ls) in controls.items():
        show(nm, d, ls)
    best = max(results, key=lambda k: results[k][0].mean())
    d = results[best][0]
    ok = d.mean() >= 0.001 and (d > 0).sum() >= 4 and d.mean() > 2.5 * ses[best]
    say(f"  best: {best} {d.mean():+.5f} +- {ses[best]:.5f}, {int((d > 0).sum())}/{len(d)} folds")
    say(f"  DECISION (>= +0.001, >= 4/5 folds, > 2.5 SE; best of {len(results)} candidates): "
        f"{'PASS -> CONFIRM (see the docstring)' if ok else 'REJECT'}")


if __name__ == "__main__":
    main()
