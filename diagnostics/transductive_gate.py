"""Transductive NBM gate: learn from the TEST SET's own unlabelled data during inference (owner's idea 2026-10-20).

WHY. In the cloud, infer(datasets, ...) is one generator per worker that streams test series one after another, so a
model may carry state from series to series (never look-ahead WITHIN a series). Everything a finished or current
series reveals before its online points is unlabelled but legal: its history (break-free, test distribution) and,
once its last online point has been scored, its whole online segment (unlabelled; about half contain a break).
Question: does adapting the shared normal-behaviour model (sbrt/nbm.py) on that pool improve the NBM detector?

WHAT IS ALREADY CLOSED (and why this is different).
  - per-series NBM fine-tuning on each series' OWN history (diagnostics/nbm_tta_gate.py, old-file section 40): held-out
    history NLL got worse; 130 head parameters overfit a few thousand points.
  - more NBM pretraining data (diagnostics/nbm_datacurve.py, HANDOFF section 15): 25% / 50% / 100% of a fold's
    histories give the same nb_logodds and valid NLL; the NBM is at its information limit, not data-limited.
  New here: POOLED adaptation of the shared NBM on the scored set's histories (A) and finished online segments (B).

WHAT CV CAN AND CANNOT SHOW (honest limitation, read before interpreting).
  In the default (unshifted) split the eval histories 5000-6999 come from the SAME distribution as the pretraining
  histories 0-4499. Then pooled adaptation is just "more same-distribution unlabelled data", which section 15 already
  measured at ~0. The default run therefore mostly re-tests section 15 (expected NULL). A transductive gain can only
  come from a train-to-test DISTRIBUTION SHIFT, which a random CV split does not contain. Two probes:
    1. SHIFT PROBE (always printed): history statistics of data/X_test.reduced.parquet (100 series, sbrt/data.py
       load_test_reduced) vs all 10,000 training series (two-sample KS), and the base NBM's per-series history NLL on
       test-reduced vs on the eval series. Only 100 test series: the probe can detect only a LARGE shift; a null
       probe is weak evidence of no shift.
    2. SHIFTED SPLIT (--shift kurt, pre-registered as the informative run): sort the 10,000 training series by a
       history statistic; pretrain + prior fit on the 5,000 LOWEST, evaluate on the 2,000 HIGHEST (the middle 3,000
       are unused). This manufactures a shift so we can see whether adaptation helps when there IS one.
       Also available: --shift rho1 (lag-1 autocorrelation), --shift arch (lag-1 autocorrelation of z^2).

ARMS (each on the SAME pretrained NBM per seed: production config bs24_ep25 = width 64, batch 24, lr 2e-3, wd 1e-4,
25 epochs, loss from position 50; pretrained on histories 0-4499, validation 4500-4999, as nbm_sweep / nbm_datacurve):
  base              no adaptation.
  pool_hist_all     CEILING, NOT causal: one fine-tune on ALL eval histories (no labels, no online data). Not
                    shippable as is (the stream never shows later series' histories early); it bounds A from above.
  pool_hist         A, causal: the eval series are streamed in a fixed order (by id), cut into blocks of --block
                    series (default 250). Block 0 is scored with base; block j >= 1 with the base NBM fine-tuned on
                    the histories of blocks 0..j-1 (re-fine-tuned from base each time on the cumulative pool:
                    deterministic and independent of the block before). Series k never uses series >= its own block.
  pool_hist_online  B, causal: as pool_hist, plus the finished ONLINE segments of blocks 0..j-1, their loss rows
                    down-weighted by --online-w (0.3; ~50% of segments contain a break). Rows of an online target
                    (row p predicts e[p+1]; p >= nh-1) get the down-weight. The block's own series are never in its
                    pool. NOTE: training on post-break stretches can teach the NBM to FOLLOW post-break variance
                    (adapt faster after a break), which would weaken detection; a loss is a possible and meaningful
                    outcome, not a bug.
  Pseudo-label / semi-supervised refits of the trees on finished series (other half of B) are not in this gate: they
  need the whole tree stack in the stream and the trees are data-limited by LABELLED data (re-cuts helped, synthetic
  breaks did not), so a self-training loop on its own noisy labels is a weak bet; build only if this gate shows the
  unlabelled test data carry anything the training data do not.
  --shards K (default 1) splits the stream into K interleaved independent streams (position k -> shard k % K), each
  with its own blocks and pools: production has ~10,000 test series over INFER_PARALLELISM = 12 workers, i.e. about
  830 series per worker, so a single 2,000-series stream is the FAVOURABLE case (best shot: a fail closes the idea; a
  pass needs a re-check with --shards 2 or more).

FINE-TUNE CONFIG (pre-registered, ONE config, not tuned on the eval set): start from the pretrained weights (all
parameters including the Student-t nu), AdamW lr 5e-4 (1/4 of the pretraining peak) with OneCycle (10% warm-up),
wd 1e-4, batch 24, 4 epochs, grad-clip 1.0, loss from position 50, batching as NB.train_model (sorted by length, padded
to 512, cache emptied per batch). 10% of each pool's series (seeded per seed/shard/block) are held out; the epoch
(0..4, where 0 = the base weights) with the lowest held-out HISTORY NLL is kept, so adaptation is only used when it
predicts unseen pool histories better. The chosen epoch of every block is logged ("never moved" vs "moved and
helped / hurt" are different outcomes).

PRIOR. The decision uses the base net's empirical-Bayes prior (fitted on the fit series' breaks) for every arm: at
test time the prior cannot be refitted (no labels). Diagnostic for the ceiling arm only: the prior refitted on the fit
series with the adapted net (separates "prior mismatch" from "the NBM got worse"). nb_lsd is prior-free.

METRIC (the NBM gate of nbm_sweep / nbm_datacurve / nbm_tta_gate): single-feature TS-AUC of nb_logodds on the eval
series (every 3rd step; all, by step range, variance-unchanged slice), nb_lvr, nb_lsd; PAIRED difference vs base with
a series-bootstrap SE (all eval series, and the series of blocks >= 1 only, since block 0 equals base); per-series
history NLL arm minus base (for the causal arms these histories were NOT in the pool: genuinely held out; for the
ceiling arm in-sample); no-break whitened online mean / variance (calibration, should be ~1); per-block mean
nb_logodds on no-break series (every step ranks series scored by DIFFERENT nets, so a level shift between blocks
changes TS-AUC by itself).

DECISION RULE (written 2026-10-20, before any run):
  1. A causal arm (pool_hist, pool_hist_online) PASSES only if on BOTH pretraining seeds (0, 1) its paired nb_logodds
     vs base over all eval series is >= +0.003 and > 2 bootstrap SE, and its no-break whitened variance is within
     1.00 +- 0.05 (or, when base itself is outside that band, as it can be under a manufactured shift, no further
     from 1 than base's).
  2. If the ceiling arm pool_hist_all does not reach >= +0.003 and > 2 SE on both seeds, A and B are CLOSED for that
     split regardless of the causal arms (they can only be weaker, up to noise).
  3. Default split: a PASS -> re-check with --shards 2, then build the causal block adaptation of the fold NBM on the
     validation stream in a NEW cv.py variant and run the paired cv.py --nbm trees under the standing pass rule
     (HANDOFF section 5), subject to the determinism blocker below.
  4. Shifted split (--shift kurt): same bar. A pass here but not in the default split means "helps only under shift":
     a cv.py build is justified only if the SHIFT PROBE fires, pre-registered as: a KS p < 0.01 on any of the 6
     named history statistics (log_nh, skew, exkurt, rho1, arch1, tail3) of test-reduced vs all training series, OR
     the base-NBM history NLL gap (test-reduced minus the eval series of the DEFAULT split, i.e. held-out training
     histories of the full training distribution) beyond 3 SE in either direction on both seeds. The NLL part is
     read from the default-split run only: in a shifted split the net is pretrained on one end of the training
     distribution, so its NLL gap measures "test is not that end", not train-to-test shift (printed, not used).
     No pass even under the manufactured shift -> CLOSED outright.
  Expectation: default split ~0 (section 15); the shifted split is the mechanism check (does pooled adaptation help
  at all when there is a shift?).
  Amended after the smoke runs, before any full run (2026-10-20): the variance clause is relative to base when base
  is outside the band (base 0.917 in the shifted smoke); the NLL trigger is two-sided and taken from the default
  split only.

C (cross-series score recalibration) is DROPPED, reasons: (i) TS-AUC ranks series within each step, so any per-step
monotone transform shared by all series changes nothing; (ii) the series-specific version, subtracting each series'
expected no-break score given its history, was tested WITH labels in section 32 (diagnostics/drift_gate.py: no-break
detector values are unpredictable from the history, R^2 ~0, richer history inputs <= +0.05); learning the same
expectation from earlier test series' UNLABELLED scores (a 50/50 mixture with breaks) can only be weaker; (iii)
per-step cross-sectional normalisation (--xsec, section 22) and self-replay calibration (section 14) are closed.
Volatility state (sbrt/volstate.py) is not adapted: it has no shared learned parameters (per-series grid GARCH; one
constant, the slow half-life 200), so "pooled adaptation" would be a new estimator (empirical-Bayes GARCH shrinkage),
not an adaptation.

DETERMINISM / SHIPPING BLOCKER (independent of this gate). HANDOFF section 2: the platform re-runs on 10% of the data
and requires the same outputs to 1e-8. Any state carried ACROSS series makes a series' output depend on which series
preceded it in its worker's stream; if the re-run streams a 10% subset (or the runner assigns series to the 12
workers in a different way), the outputs differ and the determinism check fails. main.py's own comment relies on
"each series is independent, so outputs are unchanged" across workers. The files do not say how the runner shards
series or what the re-run feeds; a pass here needs the owner to confirm that protocol before anything ships.
Inference cost if shipped: torch fine-tunes on the CPU inside each worker (no GPU in the cloud).

    python diagnostics/transductive_gate.py                    # default split, seeds 0,1 (GPU); cv_out/transductive_gate.txt
    python diagnostics/transductive_gate.py --shift kurt       # manufactured shift; cv_out/transductive_gate_shift_kurt.txt
    python diagnostics/transductive_gate.py --shards 2         # re-check of a pass (smaller pools per stream)
    python diagnostics/transductive_gate.py --smoke            # tiny end-to-end check; cv_out/transductive_gate_smoke.txt
"""
import argparse
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
os.chdir(os.path.dirname(HERE))

import numpy as np  # noqa: E402

from sbrt import bocpd as BO  # noqa: E402
from sbrt import nbm as NB  # noqa: E402
from nbm_sweep import gpu_sums  # noqa: E402
from nbm_tta_gate import Log  # noqa: E402

CFG = dict(hidden=64, lr=2e-3, epochs=25, batch=24, wd=1e-4, loss_from=NB.LOSS_FROM)   # adopted bs24_ep25
FT = dict(lr=5e-4, epochs=4, sel_frac=0.1)                                             # pre-registered fine-tune
ARMS = ["base", "pool_hist_all", "pool_hist", "pool_hist_online"]
CAUSAL = ("pool_hist", "pool_hist_online")
STAT_NAMES = ["log_nh", "skew", "exkurt", "rho1", "arch1", "tail3"]


# ---------------------------------------------------------------------------------------------
def hist_stats(h):
    """6 history descriptors of a (z-scored) history: the shift probe and the --shift sort keys."""
    z = np.asarray(h, dtype=np.float64)
    z = (z - z.mean()) / max(z.std(), 1e-12)
    q = z * z - 1.0
    return np.array([np.log(len(z)), np.mean(z ** 3), np.mean(z ** 4) - 3.0,
                     float(np.dot(z[:-1], z[1:]) / np.dot(z, z)),
                     float(np.dot(q[:-1], q[1:]) / max(np.dot(q, q), 1e-12)), np.mean(np.abs(z) > 3)])


def index_sets(args, stats):
    """(pretrain, valid, fit, eval) series indices; eval in stream order (by id)."""
    if args.shift == "none":
        if args.smoke:
            return np.arange(0, 300), np.arange(300, 360), np.arange(0, 400), np.arange(5000, 5200)
        return np.arange(0, 4500), np.arange(4500, 5000), np.arange(0, 5000), np.arange(5000, 5000 + args.n_eval)
    key = stats[:, {"kurt": 2, "rho1": 3, "arch": 4}[args.shift]]
    order = np.argsort(key, kind="stable")
    n_low, n_tr, n_ev = (400, 340, 200) if args.smoke else (5000, 4500, args.n_eval)
    low, high = np.sort(order[:n_low]), np.sort(order[-n_ev:])
    return low[:n_tr], low[n_tr:], low, high


def items_hist(b):
    """Pretraining-style item (inputs, targets, row weights) from the history part of a build_seq triple."""
    X, e, nh = b
    return X[:nh - 1], e[1:nh].astype(np.float32), np.ones(nh - 1, np.float32)


def items_full(b, w_online):
    """History + finished online segment; rows with an online target (p >= nh-1) weigh w_online."""
    X, e, nh = b
    w = np.ones(len(e) - 1, np.float32)
    w[nh - 1:] = w_online
    return X[:-1], e[1:].astype(np.float32), w


def finetune(state0, items, sel, seed, torch, NBM_, dev, epochs):
    """Continue training the pretrained NBM on `items`; keep the epoch (0 = base) with the lowest NLL on `sel`
    (history rows only). Returns (state, chosen epoch, selection-NLL curve)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = NBM_(CFG["hidden"]).to(dev)
    net.load_state_dict(state0)
    bs, lf = CFG["batch"], CFG["loss_from"]

    def batches(D, shuffle):
        order = np.argsort([len(d[1]) for d in D], kind="stable")
        bl = [order[i:i + bs] for i in range(0, len(order), bs)]
        if shuffle:
            rng.shuffle(bl)
        for b in bl:
            x = NB._pad([D[i][0] for i in b], dev, torch)
            y = NB._pad([D[i][1][:, None] for i in b], dev, torch)[..., 0]
            msk = NB._pad([D[i][2][:, None] for i in b], dev, torch)[..., 0]
            msk[:, :lf] = 0.0
            yield x, y, msk
            del x, y, msk
            if dev.type == "cuda":
                torch.cuda.empty_cache()

    def evaluate(D):
        net.eval()
        tot, cnt = 0.0, 0.0
        with torch.no_grad():
            for x, y, msk in batches(D, False):
                m, ls = net(x)
                tot += float((NB.t_nll(torch, y, m, ls, net.nu()) * msk).sum())
                cnt += float(msk.sum())
        net.train()
        return tot / max(cnt, 1.0)

    curve = [evaluate(sel)]
    best, best_ep, best_state = curve[0], 0, state0
    if epochs <= 0 or not items:
        return best_state, best_ep, curve
    opt = torch.optim.AdamW(net.parameters(), lr=FT["lr"], weight_decay=CFG["wd"])
    steps = epochs * ((len(items) + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=FT["lr"], total_steps=steps, pct_start=0.1)
    net.train()
    for ep in range(1, epochs + 1):
        for x, y, msk in batches(items, True):
            m, ls = net(x)
            loss = (NB.t_nll(torch, y, m, ls, net.nu()) * msk).sum() / msk.sum().clamp(min=1.0)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
        v = evaluate(sel)
        curve.append(v)
        if v < best:
            best, best_ep, best_state = v, ep, {k: t.detach().cpu().clone() for k, t in net.state_dict().items()}
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return best_state, best_ep, curve


def series_hist_nll(net, built, torch, dev, batch=64):
    """Per-series mean Student-t NLL of the history targets (positions >= loss_from), one value per series."""
    out = np.zeros(len(built))
    order = np.argsort([len(b[1]) for b in built], kind="stable")
    with torch.no_grad():
        for k in range(0, len(order), batch):
            b = order[k:k + batch]
            D = [items_hist(built[i]) for i in b]
            x = NB._pad([d[0] for d in D], dev, torch)
            y = NB._pad([d[1][:, None] for d in D], dev, torch)[..., 0]
            msk = NB._pad([d[2][:, None] for d in D], dev, torch)[..., 0]
            msk[:, :CFG["loss_from"]] = 0.0
            m, ls = net(x)
            v = (NB.t_nll(torch, y, m, ls, net.nu()) * msk).sum(1) / msk.sum(1).clamp(min=1.0)
            out[b] = v.cpu().numpy()
            del x, y, msk
            if dev.type == "cuda":
                torch.cuda.empty_cache()
    return out


def paired_boot(a, b, y, step, starts, series, n_boot, rng):
    """Paired TS-AUC difference a - b over the rows of `series` and its series-bootstrap SE."""
    from sbrt.metric import ts_auc

    def rows(sel):
        return np.concatenate([np.arange(starts[i], starts[i + 1]) for i in sel])
    idx = rows(series)
    d0 = ts_auc(a[idx], y[idx], step[idx]) - ts_auc(b[idx], y[idx], step[idx])
    ds = []
    for _ in range(n_boot):
        idx = rows(series[rng.integers(0, len(series), len(series))])
        ds.append(ts_auc(a[idx], y[idx], step[idx]) - ts_auc(b[idx], y[idx], step[idx]))
    return d0, float(np.std(ds, ddof=1)) if n_boot > 1 else float("nan")


# ---------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="0,1", help="pretraining seeds of the shared NBM")
    ap.add_argument("--arms", type=str, default=",".join(ARMS))
    ap.add_argument("--shift", type=str, default="none", choices=["none", "kurt", "rho1", "arch"])
    ap.add_argument("--block", type=int, default=250, help="series per block in each stream")
    ap.add_argument("--shards", type=int, default=1, help="independent interleaved streams (production: ~830 series each)")
    ap.add_argument("--online-w", type=float, default=0.3, help="pool_hist_online: weight of finished online rows")
    ap.add_argument("--n-eval", type=int, default=2000)
    ap.add_argument("--boot", type=int, default=50, help="series-bootstrap draws for the paired SE")
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end check (~300 series, 1 epoch)")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()
    arms = [a for a in args.arms.split(",") if a]
    if "base" not in arms or any(a not in ARMS for a in arms):
        sys.exit(f"arms must include base; known {ARMS}")
    if not args.out:
        args.out = os.path.join("cv_out", "transductive_gate" + ("" if args.shift == "none" else f"_shift_{args.shift}")
                                + (f"_sh{args.shards}" if args.shards > 1 else "") + ("_smoke" if args.smoke else "")
                                + ".txt")
    pre_ep = 1 if args.smoke else CFG["epochs"]
    ft_ep = 1 if args.smoke else FT["epochs"]
    block = 50 if args.smoke else args.block
    n_boot = 10 if args.smoke else args.boot
    seeds = [0] if args.smoke else [int(s) for s in args.seeds.split(",") if s]

    import torch
    from scipy.stats import ks_2samp
    from sbrt.data import load_test_reduced, load_train
    from sbrt.metric import ts_auc

    log = Log(args.out)
    t_all = time.time()
    S_all, tau_all = load_train()
    S_te, _ = load_test_reduced()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st_tr = np.array([hist_stats(s[1]) for s in S_all])
    st_te = np.array([hist_stats(s[1]) for s in S_te])
    tr_ids, va_ids, fit_ids, ev_ids = index_sets(args, st_tr)
    log(f"transductive NBM gate | shift {args.shift} | pretrain {len(tr_ids)} / valid {len(va_ids)} histories | prior fit "
        f"{len(fit_ids)} | eval {len(ev_ids)} series in stream order, {args.shards} shard(s), blocks of {block} | NBM {CFG} "
        f"pretrain epochs {pre_ep} | fine-tune {FT} epochs {ft_ep} | online weight {args.online_w} | arms {arms} | "
        f"seeds {seeds} | device {dev}")

    # ---- shift probe, part 1: history statistics of test-reduced vs training (KS) ----
    log(f"\nSHIFT PROBE 1: history statistics, test-reduced ({len(S_te)} series) vs training ({len(S_all)}); "
        "pre-registered trigger: any KS p < 0.01")
    probe_ks = []
    for k, nm in enumerate(STAT_NAMES):
        p = float(ks_2samp(st_te[:, k], st_tr[:, k]).pvalue)
        probe_ks.append(p)
        log(f"  {nm:7s} train mean {st_tr[:, k].mean():+.4f} median {np.median(st_tr[:, k]):+.4f} | test-reduced mean "
            f"{st_te[:, k].mean():+.4f} median {np.median(st_te[:, k]):+.4f} | KS p {p:.3g}")
    p_on = float(ks_2samp([len(s[2]) for s in S_te], [len(s[2]) for s in S_all]).pvalue)
    log(f"  (online length KS p {p_on:.3g}, not part of the trigger)")
    if args.shift != "none":
        k = {"kurt": 2, "rho1": 3, "arch": 4}[args.shift]
        log(f"  manufactured shift on {STAT_NAMES[k]}: pretrain mean {st_tr[tr_ids, k].mean():+.4f}, eval mean "
            f"{st_tr[ev_ids, k].mean():+.4f} (KS p {ks_2samp(st_tr[tr_ids, k], st_tr[ev_ids, k]).pvalue:.3g}); other stats "
            + ", ".join(f"{STAT_NAMES[j]} {st_tr[tr_ids, j].mean():+.3f}->{st_tr[ev_ids, j].mean():+.3f}"
                        for j in range(len(STAT_NAMES)) if j != k))

    # ---- model-independent inputs ----
    tr = NB.prep_histories([S_all[i][1] for i in tr_ids])
    va = NB.prep_histories([S_all[i][1] for i in va_ids])
    va_b = [NB.build_seq(S_all[i][1], np.zeros(0)) for i in va_ids]
    fit_b = [NB.build_seq(S_all[i][1], S_all[i][2]) for i in fit_ids]
    S, tau = [S_all[i] for i in ev_ids], tau_all[ev_ids]
    ev_b = [NB.build_seq(s[1], s[2]) for s in S]
    te_b = [NB.build_seq(s[1], np.zeros(0)) for s in S_te]
    step = np.concatenate([np.arange(0, len(s[2]), 3) for s in S])
    nrow = np.array([(len(s[2]) + 2) // 3 for s in S])
    sid = np.repeat(np.arange(len(S)), nrow)
    starts = np.r_[0, np.cumsum(nrow)]
    y = ((tau[sid] >= 0) & (step >= tau[sid])).astype(np.int8)
    vr = np.full(len(S), np.nan)
    for i, (s, t) in enumerate(zip(S, tau)):
        if t >= 0 and len(s[2]) - t >= 30:
            vr[i] = BO.whitened_online(s[1], s[2])[0][t:t + 300].var()
    in_slice = (tau[sid] < 0) | ((vr > 0.85) & (vr < 1.15))[sid]
    nobreak = np.flatnonzero(tau < 0)
    # stream layout: position k -> shard k % K, block = (index within its shard) // block
    shard = np.arange(len(S)) % args.shards
    blk = np.zeros(len(S), np.int64)
    for s_ in range(args.shards):
        pos = np.flatnonzero(shard == s_)
        blk[pos] = np.arange(len(pos)) // block
    late = np.flatnonzero(blk >= 1)
    log(f"inputs prepared ({time.time() - t_all:.0f}s): {len(nobreak)} no-break eval series; {len(late)} in blocks >= 1; "
        f"max block {blk.max()}")

    def auc(x, lo=0, hi=10 ** 6, mask=None):
        mk = (step >= lo) & (step < hi)
        if mask is not None:
            mk &= mask
        return ts_auc(x[mk], y[mk], step[mk])

    def feats(sums, prior):
        return np.concatenate([NB.features_from_sums(q, prior, 3) for q in sums])

    _, NBM_ = NB._torch()

    def net_of(state):
        n = NBM_(CFG["hidden"]).to(dev)
        n.load_state_dict(state)
        n.eval()
        return n

    summary = {a: [] for a in arms}
    probe_nll = []
    for seed in seeds:
        t0 = time.time()
        state0, vnll = NB.train_model(tr, va, epochs=pre_ep, batch=CFG["batch"], lr=CFG["lr"], seed=seed, verbose=False,
                                      hidden=CFG["hidden"], wd=CFG["wd"], loss_from=CFG["loss_from"], return_nll=True,
                                      prepared=True)
        net0 = net_of(state0)
        log(f"\nseed {seed}: base NBM pretrained in {time.time() - t0:.0f}s (valid NLL {vnll:.5f}, nu "
            f"{float(net0.nu().detach()):.2f})")
        sf0 = gpu_sums(net0, fit_b, torch, dev)
        prior0, nfit = NB.fit_prior_w([np.diff(q[0][0]) for q in sf0], tau_all[fit_ids])
        del sf0
        log(f"  base prior from {nfit} fit breaks: kappa0 {prior0[0]:.0f}, alpha0 {prior0[1]:.2f}, beta0 {prior0[2]:.2f}")

        # ---- shift probe, part 2: base-NBM per-series history NLL ----
        nll_va = series_hist_nll(net0, va_b, torch, dev)
        nll_ev0 = series_hist_nll(net0, ev_b, torch, dev)
        nll_te = series_hist_nll(net0, te_b, torch, dev)
        ref, ref_nm = (nll_ev0, "eval") if args.shift == "none" else (nll_va, "valid")
        gap = nll_te.mean() - ref.mean()
        se_gap = np.sqrt(nll_te.var(ddof=1) / len(nll_te) + ref.var(ddof=1) / len(ref))
        probe_nll.append(gap / se_gap)
        log(f"  SHIFT PROBE 2 (base NBM per-series history NLL): valid (pretrain distribution) {nll_va.mean():.4f}, eval "
            f"{nll_ev0.mean():.4f}, test-reduced {nll_te.mean():.4f}; test-reduced minus {ref_nm} {gap:+.4f} +- "
            f"{se_gap:.4f} ({gap / se_gap:+.1f} SE; trigger |z| > 3); eval minus valid {nll_ev0.mean() - nll_va.mean():+.4f} "
            "(size of the split's own shift, for scale)")

        # ---- adapted nets per arm ----
        nets = {}   # arm -> {(shard, block): state}, missing key = base
        if "pool_hist_all" in arms:
            t1 = time.time()
            rs = np.random.default_rng(10_000 + seed)
            perm = rs.permutation(len(S))
            nsel = max(1, int(round(FT["sel_frac"] * len(S))))
            stt, ep_, curve = finetune(state0, [items_hist(ev_b[i]) for i in perm[nsel:]],
                                       [items_hist(ev_b[i]) for i in perm[:nsel]], 10_000 + seed, torch, NBM_, dev, ft_ep)
            nets["pool_hist_all"] = {(s_, j): stt for s_ in range(args.shards) for j in range(blk.max() + 1)}
            log(f"  pool_hist_all: fine-tuned on {len(S) - nsel} eval histories ({nsel} for selection) in "
                f"{time.time() - t1:.0f}s; chosen epoch {ep_}; selection NLL " + " ".join(f"{v:.4f}" for v in curve))
        for arm in [a for a in arms if a in CAUSAL]:
            t1 = time.time()
            nets[arm] = {}
            msgs = []
            for s_ in range(args.shards):
                for j in range(1, int(blk[shard == s_].max()) + 1):
                    pool = np.flatnonzero((shard == s_) & (blk < j))
                    sd = 100_000 * (1 + CAUSAL.index(arm)) + 1000 * seed + 100 * s_ + j
                    rs = np.random.default_rng(sd)
                    perm = pool[rs.permutation(len(pool))]
                    nsel = max(1, int(round(FT["sel_frac"] * len(pool))))
                    sel = [items_hist(ev_b[i]) for i in perm[:nsel]]
                    if arm == "pool_hist":
                        it = [items_hist(ev_b[i]) for i in perm[nsel:]]
                    else:
                        it = [items_full(ev_b[i], args.online_w) for i in perm[nsel:]]
                    stt, ep_, curve = finetune(state0, it, sel, sd, torch, NBM_, dev, ft_ep)
                    nets[arm][(s_, j)] = stt
                    msgs.append(f"s{s_}b{j}:{len(pool)}->ep{ep_} ({curve[0]:.4f}->{min(curve):.4f})")
            log(f"  {arm}: {len(msgs)} fine-tunes in {time.time() - t1:.0f}s; per block pool size -> chosen epoch "
                f"(selection NLL base -> best): " + ", ".join(msgs))

        # ---- score every arm ----
        feats_by, ws_by = {}, {}
        for arm in arms:
            t1 = time.time()
            sums = [None] * len(S)
            nll_arm = np.zeros(len(S))
            groups = {}
            for i in range(len(S)):
                key = (int(shard[i]), int(blk[i]))
                groups.setdefault(key if arm != "base" and key in nets.get(arm, {}) else None, []).append(i)
            for key, idx in groups.items():
                net = net0 if key is None else net_of(nets[arm][key])
                sub = [ev_b[i] for i in idx]
                for i, q in zip(idx, gpu_sums(net, sub, torch, dev)):
                    sums[i] = q
                nll_arm[idx] = series_hist_nll(net, sub, torch, dev)
                if key is not None:
                    del net
            Fm = feats(sums, prior0)
            feats_by[arm] = Fm
            wnb = np.concatenate([np.diff(sums[i][0][0]) for i in nobreak])
            r = dict(nb_logodds=auc(Fm[:, 0]), lo_0_100=auc(Fm[:, 0], 0, 100), lo_100_300=auc(Fm[:, 0], 100, 300),
                     lo_300_700=auc(Fm[:, 0], 300, 700), lo_700p=auc(Fm[:, 0], 700), varsame=auc(Fm[:, 0], mask=in_slice),
                     nb_lvr=auc(Fm[:, 2]), nb_lsd=auc(Fm[:, 4]), nb_var=float(wnb.var()), nb_mean=float(wnb.mean()))
            on = late if arm in CAUSAL else np.arange(len(S))
            dn = nll_arm[on] - nll_ev0[on]
            r["dnll"] = (float(dn.mean()), float(dn.std(ddof=1) / np.sqrt(len(dn))) if len(dn) > 1 else float("nan"))
            per_blk = []
            for j in range(int(blk.max()) + 1):
                rr = np.isin(sid, nobreak[blk[nobreak] == j])
                per_blk.append(float(Fm[rr, 0].mean()) if rr.any() else float("nan"))
            summary[arm].append(r)
            log(f"  {arm:16s} nb_logodds {r['nb_logodds']:.4f} (0-100 {r['lo_0_100']:.4f}, 100-300 {r['lo_100_300']:.4f}, "
                f"300-700 {r['lo_300_700']:.4f}, 700+ {r['lo_700p']:.4f}, var-same {r['varsame']:.4f})  nb_lvr "
                f"{r['nb_lvr']:.4f}  nb_lsd {r['nb_lsd']:.4f} | no-break whitened mean {r['nb_mean']:+.3f} var "
                f"{r['nb_var']:.3f} | history NLL arm - base {r['dnll'][0]:+.5f} +- {r['dnll'][1]:.5f} "
                f"({'held out, blocks >= 1' if arm in CAUSAL else ('in-sample' if arm != 'base' else '-')}) | "
                f"{time.time() - t1:.0f}s")
            log(f"  {'':16s} no-break mean nb_logodds per block: " + " ".join(f"{v:+.3f}" for v in per_blk))

        # ---- paired differences vs base ----
        rng = np.random.default_rng(seed)
        for arm in arms:
            if arm == "base":
                continue
            d, s_ = paired_boot(feats_by[arm][:, 0], feats_by["base"][:, 0], y, step, starts, np.arange(len(S)), n_boot, rng)
            summary[arm][-1]["d_all"] = (d, s_)
            line = f"  paired {arm} - base: all series {d:+.4f} +- {s_:.4f}"
            if arm in CAUSAL and len(late):
                d2, s2 = paired_boot(feats_by[arm][:, 0], feats_by["base"][:, 0], y, step, starts, late, n_boot, rng)
                line += f" | blocks >= 1 only {d2:+.4f} +- {s2:.4f}"
            log(line + f" (series bootstrap, {n_boot} draws)")
        if "pool_hist_all" in arms:   # diagnostic: prior refitted with the adapted net (ceiling arm only)
            net = net_of(nets["pool_hist_all"][(0, 0)])
            sf = gpu_sums(net, fit_b, torch, dev)
            prior1, _ = NB.fit_prior_w([np.diff(q[0][0]) for q in sf], tau_all[fit_ids])
            Fm1 = feats(gpu_sums(net, ev_b, torch, dev), prior1)
            del net, sf
            log(f"  diagnostic pool_hist_all with its own refitted prior (kappa0 {prior1[0]:.0f}, alpha0 {prior1[1]:.2f}, "
                f"beta0 {prior1[2]:.2f}): nb_logodds {auc(Fm1[:, 0]):.4f} vs base-prior {summary['pool_hist_all'][-1]['nb_logodds']:.4f}")
        del net0
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    # ---- verdict ----
    log("\nVERDICT (rule written before the run: on BOTH seeds paired nb_logodds vs base >= +0.003 and > 2 SE, no-break "
        "whitened variance in 1.00 +- 0.05 or no worse than base's; ceiling failing closes the causal arms for this split)")
    passed = {}
    for arm in arms:
        if arm == "base":
            continue
        rs = summary[arm]
        ok = len(rs) >= 2 and all(r["d_all"][0] >= 0.003 and r["d_all"][0] > 2 * r["d_all"][1]
                                  and (abs(r["nb_var"] - 1.0) <= 0.05 or abs(r["nb_var"] - 1.0) <= abs(b["nb_var"] - 1.0))
                                  for r, b in zip(rs, summary["base"]))
        passed[arm] = ok
        log(f"  {arm:16s} " + " | ".join(f"seed {sd}: {r['d_all'][0]:+.4f} +- {r['d_all'][1]:.4f}, var {r['nb_var']:.3f}"
                                         for sd, r in zip(seeds, rs)) + f"  -> {'PASS' if ok else 'FAIL'}")
    if "pool_hist_all" in passed and not passed["pool_hist_all"]:
        log("  ceiling pool_hist_all FAILS -> A and B CLOSED for this split, whatever the causal arms show")
    nll_fire = args.shift == "none" and len(probe_nll) >= 2 and all(abs(z) > 3 for z in probe_nll)
    fired = any(p < 0.01 for p in probe_ks) or nll_fire
    log(f"  shift probe: KS min p {min(probe_ks):.3g} ({STAT_NAMES[int(np.argmin(probe_ks))]}), NLL gap z per seed "
        + ", ".join(f"{z:+.1f}" for z in probe_nll)
        + ("" if args.shift == "none" else " (NOT used here: the NLL part of the trigger is read from the default-split run)")
        + f" -> {'FIRES' if fired else 'does not fire'} (only 100 test series: weak evidence either way)")
    if any(passed.get(a, False) for a in CAUSAL) and not (passed.get("pool_hist_all", True) is False):
        nxt = ("re-check with --shards 2, then a cv.py build" if args.shift == "none"
               else ("cv.py build justified (KS probe fires)" if fired else
                     "cv.py build only if the default-split run's NLL probe fires (the KS probe does not)"))
        log(f"  next: {nxt}; shipping also needs the owner to confirm the determinism re-run protocol (docstring)")
    log(f"\ntotal {time.time() - t_all:.0f}s; log {args.out}" + ("  (SMOKE: numbers meaningless)" if args.smoke else ""))


if __name__ == "__main__":
    main()
