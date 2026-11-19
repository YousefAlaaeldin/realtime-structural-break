"""Self-replay calibration of the FINAL blend score (owner's idea 2026-10-17; HANDOFF section 14). A one-fold test on
the production fold-0 model set in verify_models/fold0 (trained by verify_submission.py on fold 0's 8,000 training
series, so fold 0's 2,000 validation series are out of sample). Nothing is retrained.

Idea: the history is break-free, so replaying its end as a fake online segment shows how the SHIPPED blend scores this
series when there is no break. At real online step t, the blend logit is corrected by the series' own replay behaviour
("is this alarm unusually high for this series at this online age?").

Replays (the sbrt/replay.py scheme of the old HANDOFF section 18): for L in LENS with n_h - L >= 500,
    pseudo-history = h[:n_h - L],   pseudo-online = h[n_h - L : n_h - L + min(L, n_online)]
both re-z-scored with the pseudo-history's mean and sd (ddof 0), as the real histories are. Truncating a replay to the
real online length changes nothing (every score is causal) and saves compute. Every replay is scored by the
production batch path (identical to infer(); verify_submission.py checks the parity).

Calibration at real kept row k (online step 3k) of series i:
    d_ij[k] = r_ij[k] - (mean over all series' replays with the SAME L at row k)   per-(L, row) pooling: each L has its
              own pseudo-history shortfall, which would otherwise leak into the deviation as "history length"
    aligned = replays that reach row k; carry = also shorter replays, with the d of their last row
    n = replays used, dev = mean_j d_ij[k], v = var_j d_ij[k] (ddof 1)
    shift:  s' = s - lam * n / (n + kap) * dev
    z:      s' = (1 - a) * (s - c_k) / sd_k + a * (s - c_k - n / (n + kap) * dev) / sqrt(((n - 1) v + 10 vp_k) / (n - 1 + 10))
            c_k, sd_k = median and sd of the real scores at row k over all series (a per-row affine map: rank-neutral);
            vp_k = mean within-series replay variance at row k (heavy shrinkage: a per-series sd from ~2 draws is noise)
    hmin 500 or 1000: replays with a shorter pseudo-history are dropped (training histories are all >= ~1000 points)
Grid (27): raw + shift (lam 0.5 / 1, kap 1 / 3 / 10, aligned / carry, hmin 500 / 1000) + z (a 0.5 / 1, kap 3, carry,
hmin 500). Everything after the scoring is label-free except the TS-AUC itself.

Selection (cross-fitted): 5 random halves of the series (stratified by break / no break); the best config on one half
(raw is a candidate) is scored against raw on the other half, then the halves swap: 10 held-out gains.
Mechanism (no-break series): per step, the Spearman correlation across series between the real blend logit and the
series' replay deviation, averaged within step buckets; the replay-vs-real null offset per bucket.

DECISION RULE (written 2026-10-17 before any run; HANDOFF section 14): GO only if the mean of the 10 cross-fitted
held-out gains is >= +0.001 AND at least 8 of the 10 are > 0; otherwise CLOSED. GO is not adoption: it means
production model sets for folds 1-4 (verify_submission.py --fold k, about 25-35 min each with the GPU) and this test
on each fold.

Run:  python diagnostics/replay_blend.py                   fold-0 validation, 2,000 series (scores cached in cv_out/)
      python diagnostics/replay_blend.py --n-val 20         smoke test
      python diagnostics/replay_blend.py --analyse-only     the analysis again from the cached scores
"""
import argparse
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy.stats import rankdata  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

from sbrt.data import load_train  # noqa: E402
from sbrt.metric import ts_auc  # noqa: E402

LENS = (50, 100, 200, 400, 700, 999)
HMIN_SCORE = 500
SUB = 3
BUCKETS = ((0, 100), (100, 300), (300, 600), (600, 10000))
N_SPLITS = 5
N_BOOT = 100

ap = argparse.ArgumentParser()
ap.add_argument("--fold", type=int, default=0)
ap.add_argument("--n-val", type=int, default=0, help="score only the first N validation series (smoke test)")
ap.add_argument("--threads", type=int, default=12)
ap.add_argument("--analyse-only", action="store_true", help="skip the scoring; read the cached scores")
args = ap.parse_args()
SUFFIX = f"_f{args.fold}" + (f"_n{args.n_val}" if args.n_val else "")
CACHE = os.path.join("cv_out", f"replay_blend_scores{SUFFIX}.npz")
REPORT = os.path.join("cv_out", f"replay_blend{SUFFIX}.txt")
_fh = None


def say(msg=""):
    print(msg, flush=True)
    _fh.write(msg + "\n")
    _fh.flush()


# ---------------------------------------------------------------------------------------------
# scoring (production fold-0 models, batch path)
def blend_logit(main, M, h, o):
    """The production blend score before the sigmoid at kept rows (steps 0, 3, 6, ...); verify_submission's batch path."""
    rows, raw = main._series_feats(h, o, M["nbm"], M["nbm_prior"])
    mt = [b.inplace_predict(rows, predict_type="margin") for b in M["boosters"]]
    ms = main._svm_score(M["svm"], rows[:, main.SVM_COLS])
    Xg = main._gru_inputs(h, o, raw, M["gstd"])
    g = [net.run(Xg)[main.GRU_BURN:][::main.SUB] for net in M["nets"]]
    return np.asarray(main.blend_scores(M, mt, g, ms), dtype=np.float64)


def score_series(main, M, h, o):
    h, o = main._clean(h), main._clean(o)
    real = blend_logit(main, M, h, o)
    reps = []
    for L in LENS:
        c = len(h) - L
        if c < HMIN_SCORE:
            continue
        hh, oo = h[:c], h[c:c + min(L, len(o))]
        mu, sd = hh.mean(), hh.std()
        reps.append((L, c, blend_logit(main, M, (hh - mu) / sd, (oo - mu) / sd)))
    return real, reps


def run_scoring(S, va):
    import main
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)   # parallel over series; one BLAS thread per call (as infer() does)
    mdir = os.path.join("verify_models", f"fold{args.fold}")
    if not os.path.exists(os.path.join(mdir, "meta.json")):
        sys.exit(f"{mdir} has no model set: run verify_submission.py --fold {args.fold} first")
    M = main.load_models(mdir)
    if M["meta"].get("config", {}).get("version") != main.CONFIG["version"]:
        sys.exit(f"{mdir} was trained for config {M['meta'].get('config', {}).get('version')}, main.py is "
                 f"{main.CONFIG['version']}")
    n_pts = sum(len(S[i][2]) for i in va)
    say(f"scoring {len(va)} validation series of fold {args.fold} ({n_pts:,} real online points) and their replays "
        f"(L {LENS}, pseudo-history >= {HMIN_SCORE}) with the model set in {mdir}; {args.threads} threads")
    out, t0, chunk = [], time.time(), 100
    for a in range(0, len(va), chunk):
        part = va[a:a + chunk]
        out += Parallel(n_jobs=args.threads, backend="threading")(
            delayed(score_series)(main, M, S[i][1], S[i][2]) for i in part)
        el = time.time() - t0
        say(f"  {a + len(part):5d} / {len(va)} series  {el:6.0f}s  (about {el / (a + len(part)) * (len(va) - a - len(part)):.0f}s left)")
    real = [r for r, _ in out]
    rep = [(j, L, c, x) for j, (_, reps) in enumerate(out) for L, c, x in reps]
    np.savez(CACHE, va=va, real=np.concatenate(real), real_off=np.r_[0, np.cumsum([len(r) for r in real])],
             rep_ser=np.array([q[0] for q in rep]), rep_L=np.array([q[1] for q in rep]),
             rep_c=np.array([q[2] for q in rep]), rep=np.concatenate([q[3] for q in rep]),
             rep_off=np.r_[0, np.cumsum([len(q[3]) for q in rep])], seconds=time.time() - t0)
    say(f"scores saved to {CACHE} ({time.time() - t0:.0f}s)")


# ---------------------------------------------------------------------------------------------
# analysis
def calib_inputs(Z, hmin, carry):
    """Per real row: n replays used, mean deviation dev, deviation variance v (ddof 1); plus per-row pooled vp."""
    roff, rep_off = Z["real_off"], Z["rep_off"]
    ns = len(roff) - 1
    keep = np.flatnonzero(Z["rep_c"] >= hmin)
    K = int(np.diff(roff).max())
    # per-(L, row) pooled replay mean over all series
    mu = {}
    for L in LENS:
        s1, cnt = np.zeros(K), np.zeros(K)
        for j in keep[Z["rep_L"][keep] == L]:
            x = Z["rep"][rep_off[j]:rep_off[j + 1]]
            s1[:len(x)] += x
            cnt[:len(x)] += 1
        mu[L] = np.where(cnt > 0, s1 / np.maximum(cnt, 1), 0.0)
    by_ser = [[] for _ in range(ns)]
    for j in keep:
        by_ser[Z["rep_ser"][j]].append(j)
    n = np.zeros(roff[-1])
    dev = np.zeros(roff[-1])
    v = np.zeros(roff[-1])
    for i in range(ns):
        Ki = roff[i + 1] - roff[i]
        if not by_ser[i]:
            continue
        D = np.full((len(by_ser[i]), Ki), np.nan)
        for r, j in enumerate(by_ser[i]):
            x = Z["rep"][rep_off[j]:rep_off[j + 1]]
            d = x - mu[Z["rep_L"][j]][:len(x)]
            D[r, :len(d)] = d
            if carry and len(d) < Ki:
                D[r, len(d):] = d[-1]
        ok = ~np.isnan(D)
        c = ok.sum(0)
        s1 = np.where(ok, D, 0.0).sum(0)
        m = s1 / np.maximum(c, 1)
        s2 = np.where(ok, (D - m) ** 2, 0.0).sum(0)
        sl = slice(roff[i], roff[i + 1])
        n[sl], dev[sl] = c, np.where(c > 0, m, 0.0)
        v[sl] = np.where(c > 1, s2 / np.maximum(c - 1, 1), 0.0)
    return n, dev, v


def bucket_mean(arr, K):
    """Mean of a per-row array over the rows of each step bucket (nan when a bucket has no finite value)."""
    st = np.arange(K) * SUB
    out = []
    for lo, hi in BUCKETS:
        x = arr[(st >= lo) & (st < hi)]
        x = x[np.isfinite(x)]
        out.append(x.mean() if len(x) else np.nan)
    return out


def per_row(values, k, K, mask=None, fn=np.mean):
    out = np.full(K, np.nan)
    for kk in range(K):
        sel = (k == kk) if mask is None else ((k == kk) & mask)
        if sel.any():
            out[kk] = fn(values[sel])
    return out


def analyse(S, tau):
    Z = dict(np.load(CACHE))
    va, roff = Z["va"], Z["real_off"]
    ns = len(va)
    lens = np.diff(roff)
    sidx = np.repeat(np.arange(ns), lens)
    k = np.concatenate([np.arange(m) for m in lens])
    step = k * SUB
    t_ser = tau[va]
    y = ((t_ser[sidx] >= 0) & (step >= t_ser[sidx])).astype(np.int8)
    s = Z["real"]
    K = int(lens.max())
    say(f"\nanalysis: {ns} series, {len(s):,} real rows, {len(Z['rep_L']):,} replays; scoring took {float(Z['seconds']):.0f}s")
    rows_in = [(step >= lo) & (step < hi) for lo, hi in BUCKETS]
    nobrk = t_ser[sidx] < 0

    # per-row centring for the z variant (label-free)
    order = np.argsort(k, kind="stable")
    bnd = np.r_[0, np.flatnonzero(np.diff(k[order])) + 1, len(k)]
    c_k, sd_k = np.zeros(K), np.ones(K)
    for a, b in zip(bnd[:-1], bnd[1:]):
        x = s[order[a:b]]
        c_k[k[order[a]]] = np.median(x)
        sd_k[k[order[a]]] = max(x.std(), 1e-9)

    inputs = {}
    for hmin in (500, 1000):
        for carry in (0, 1):
            inputs[(hmin, carry)] = calib_inputs(Z, hmin, carry)

    # ---- mechanism
    say("\n[mechanism] replays per real row (mean n), by step bucket")
    say("  " + "".join(f"{f'{lo}-{hi if hi < 10000 else 999}':>12s}" for lo, hi in BUCKETS) + "   (hmin, carry)")
    for key, (n, _, _) in inputs.items():
        say("  " + "".join(f"{n[m].mean():12.2f}" for m in rows_in) + f"   {key}")
    say("\n[mechanism] no-break series: Spearman correlation across series, per step, between the real blend logit and "
        "the series' replay deviation (rows with n >= 1), averaged within buckets; 0 = the replays say nothing about "
        "this series' real no-break score")
    for key, (n, dev, _) in inputs.items():
        cors = np.full(K, np.nan)
        for a, b in zip(bnd[:-1], bnd[1:]):
            r = order[a:b]
            r = r[nobrk[r] & (n[r] >= 1)]
            if len(r) >= 30:
                cors[k[r[0]]] = np.corrcoef(rankdata(s[r]), rankdata(dev[r]))[0, 1]
        say("  " + "".join(f"{x:12.3f}" for x in bucket_mean(cors, K)) + f"   {key}")
    say("\n[mechanism] null offset: pooled replay logit minus the real no-break logit at the same row, in units of the "
        "real no-break sd at that row (bias of the replay null), by pseudo-history length")
    rep_off = Z["rep_off"]
    for name, sel in (("pseudo-history 500-999", (Z["rep_c"] >= 500) & (Z["rep_c"] < 1000)),
                      ("pseudo-history >= 1000", Z["rep_c"] >= 1000)):
        s1, cnt = np.zeros(K), np.zeros(K)
        for j in np.flatnonzero(sel):
            x = Z["rep"][rep_off[j]:rep_off[j + 1]]
            s1[:len(x)] += x
            cnt[:len(x)] += 1
        rm = np.where(cnt > 0, s1 / np.maximum(cnt, 1), np.nan)
        nm = per_row(s, k, K, nobrk)
        nsd = per_row(s, k, K, nobrk, np.std)
        off = (rm - nm) / np.where(nsd > 1e-9, nsd, np.nan)
        say("  " + "".join(f"{x:12.3f}" for x in bucket_mean(off, K)) + f"   {name} ({int(sel.sum())} replays)")
    n5, dev5, _ = inputs[(500, 1)]
    say("\n[mechanism] scale: cross-sectional sd of the real no-break logit vs sd of the replay deviation (carry, hmin 500)")
    say("  " + "".join(f"{s[m & nobrk].std():6.3f}/{dev5[m & nobrk].std():5.3f}" for m in rows_in))

    # ---- configurations
    cfgs = [("raw",)]
    for hmin in (500, 1000):
        for carry in (0, 1):
            for kap in (1, 3, 10):
                for lam in (0.5, 1.0):
                    cfgs.append(("shift", hmin, carry, kap, lam))
    for a in (0.5, 1.0):
        cfgs.append(("z", 500, 1, 3, a))

    vp_cache = {}

    def score_of(cfg):
        if cfg[0] == "raw":
            return s
        _, hmin, carry, kap, par = cfg
        n, dev, v = inputs[(hmin, carry)]
        w = n / (n + kap)
        if cfg[0] == "shift":
            return s - par * w * dev
        if (hmin, carry) not in vp_cache:
            vp = per_row(v, k, K, n >= 2)
            vp = np.where(np.isnan(vp), np.nanmedian(vp), vp)
            vp_cache[(hmin, carry)] = vp
        vp = vp_cache[(hmin, carry)][k]
        dof = np.maximum(n - 1, 0)
        sig = np.sqrt(np.maximum((dof * v + 10.0 * vp) / (dof + 10.0), 1e-12))
        return (1 - par) * (s - c_k[k]) / sd_k[k] + par * (s - c_k[k] - w * dev) / sig

    def name(cfg):
        if cfg[0] == "raw":
            return "raw"
        t, hmin, carry, kap, par = cfg
        return f"{t} hmin{hmin} {'carry' if carry else 'align'} kap{kap} {'lam' if t == 'shift' else 'a'}{par:g}"

    scores = [score_of(c) for c in cfgs]

    # ---- cross-fitted selection
    has_brk = (t_ser >= 0).astype(int)
    held, chosen = [], []
    t0 = time.time()
    for r in range(N_SPLITS):
        A, B = next(StratifiedKFold(2, shuffle=True, random_state=100 + r).split(np.arange(ns), has_brk))
        for tune, test in ((A, B), (B, A)):
            mt, me = np.isin(sidx, tune), np.isin(sidx, test)
            auc_t = [ts_auc(x[mt], y[mt], step[mt]) for x in scores]
            best = int(np.argmax(auc_t))
            g = ts_auc(scores[best][me], y[me], step[me]) - ts_auc(s[me], y[me], step[me])
            held.append(g)
            chosen.append(best)
            say(f"  split {r} half {len(held) % 2 or 2}: chosen on the tuning half: {name(cfgs[best]):34s} "
                f"(tuning gain {auc_t[best] - auc_t[0]:+.4f})   held-out gain {g:+.4f}")
    held = np.array(held)
    say(f"cross-fitting done in {time.time() - t0:.0f}s")

    # ---- full-data table (in-sample: optimistic, for orientation only)
    full = np.array([ts_auc(x, y, step) for x in scores])
    say(f"\nall {ns} series, in-sample (selection on these rows; optimistic): raw TS-AUC {full[0]:.4f}")
    for j in np.argsort(-full)[:8]:
        say(f"  {name(cfgs[j]):34s} {full[j]:.4f}  gain {full[j] - full[0]:+.4f}  "
            + "  ".join(f"[{lo},{hi if hi < 10000 else 999}) {ts_auc(scores[j][m], y[m], step[m]) - ts_auc(s[m], y[m], step[m]):+.4f}"
                        for (lo, hi), m in zip(BUCKETS, rows_in)))
    vals, cnts = np.unique(chosen, return_counts=True)
    top = int(vals[np.argmax(cnts)])
    if top != 0:
        rng = np.random.default_rng(0)
        starts, ends = roff[:-1], roff[1:]
        bs = []
        for _ in range(N_BOOT):
            pick = rng.integers(0, ns, ns)
            idx = np.concatenate([np.arange(starts[i], ends[i]) for i in pick])
            bs.append(ts_auc(scores[top][idx], y[idx], step[idx]) - ts_auc(s[idx], y[idx], step[idx]))
        say(f"\nmost often chosen: {name(cfgs[top])} ({cnts.max()}/{len(chosen)}): in-sample gain "
            f"{full[top] - full[0]:+.4f}, series-bootstrap sd {np.std(bs):.4f} ({N_BOOT} resamples)")
    else:
        say(f"\nmost often chosen: raw ({cnts.max()}/{len(chosen)})")

    # ---- decision
    npos = int((held > 0).sum())
    say(f"\nCROSS-FITTED held-out gains ({len(held)}): " + " ".join(f"{g:+.4f}" for g in held))
    say(f"  mean {held.mean():+.4f}, {npos}/{len(held)} > 0")
    ok = held.mean() >= 0.001 and npos >= 8
    say(f"DECISION (rule written before the run: mean >= +0.001 and >= 8/10 > 0): "
        + ("GO -> confirm on folds 1-4 (production model sets via verify_submission.py --fold k)" if ok else "CLOSED"))
    if args.n_val:
        say("(smoke run: the numbers above are meaningless at this size)")


def main_():
    global _fh
    _fh = open(REPORT, "a" if args.analyse_only else "w", encoding="utf-8")
    S, tau = load_train()
    va = list(StratifiedKFold(5, shuffle=True, random_state=0).split(np.arange(len(tau)), (tau >= 0).astype(int)))[args.fold][1]
    if args.n_val:
        va = va[:args.n_val]
    if not args.analyse_only:
        run_scoring(S, va)
    elif not os.path.exists(CACHE):
        sys.exit(f"no cached scores {CACHE}: run without --analyse-only first")
    analyse(S, tau)
    say(f"\nreport: {REPORT}")


if __name__ == "__main__":
    main_()
