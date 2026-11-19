"""Chronos-2 ENCODER drift gate (HANDOFF section 35; written 2026-10-15 before any run).

Question: does the frozen Chronos-2 encoder (pretrained on a large outside corpus) see a break that the production
trees do not? Same machinery as the embedding gate (section 33: windows, leave-out history null, anchor / lag cosines,
CUSUMs, BOCPD posterior on whitened blocks, causality check), only the encoder changes.

Why a new instrument: the section 33 gate (base = bo + vol + nb + step + MMD) gave +0.004 to +0.0047 for emb / femb,
and both came out at +0.0001 in the paired cv.py runs (section 33.7). That base is too weak for this kind of feature.
Here the base is the PRODUCTION model: the out-of-fold score of the current best trees
(cv_out/oof_aug2_df_bo_nbm_bs24_ep25.npz) + step. Rows are aligned by (series, step) and asserted equal.

Stages (in this order):
  1. CALIBRATION (CPU, no Chronos): base vs base + femb and base + emb (saved gate columns). Both are known to add
     about 0 in production, so the instrument must show that:
       TRUSTED if both gains are < +0.0015. Otherwise the script stops (override: --force) because a Chronos result
       on an instrument that passes known-null features means nothing.
     AMENDED after the smoke run, before any full run (the smoke printed TRUSTED with femb -0.034 / emb -0.068, i.e.
     a one-sided rule cannot tell a sound instrument from one that wrecks any added column):
       - null controls two-sided: |gain| <= 0.0015 for both femb and emb;
       - hard stop if the LightGBM refit of [oof, step] scores more than 0.003 below the OOF score alone;
       - POSITIVE control (sensitivity): base' = OOF of cv_out/oof_aug2_df_bo.npz + step, + the 6 nb_ columns of the
         gate NBM (data/backcast_refs_*.npz, NBF; gate model nbm_v1.pt, not per-fold nets). Production measured
         +0.0049 for the NBM over that file; required here >= +0.002 and > 2 SE, else the instrument is blind.
       All three must hold for TRUSTED.
  2. CPU COST (4 pinned threads, fp32): ms per single-window encode and per 256-window batch at every L, and the
     implied seconds per series (one encode per L every SUB = 3 online steps + the history null once). A pass that
     cannot run on the platform is worthless, so the number is printed next to the verdict.
  3. Chronos features on the GPU, then single-feature table and the incremental test.

Encoder arm (fixed before any run):
  - input: channel z of each window (the history-standardised series), window lengths L = 48, 144, 240 (multiples of
    the 16-point patch, of HSTRIDE 6 and of SUB 3, so no padding).
  - Chronos-2 standardises each input (instance norm + arcsinh), so the embedding of a window has that window's level
    and scale REMOVED: it sees shape and dependence only, not variance. Deliberate: variance is what the trees already
    have. The variance-matched slice is therefore the column to watch.
  - pooling: mean of the context-patch tokens of the last encoder layer (the [REG] and the future-patch token are
    dropped). One pooling only, to keep the decision set to one arm.
  - projection: centred and PCA-whitened to 64 dimensions with constants from fit-set HISTORY windows only, then
    L2-normalised (raw transformer states are anisotropic, so raw cosines sit near 1).
  - 9 columns per L (d, dmean, dcus, lag, lagcus, pb_logodds/tsb/lvr/mu) = 27 columns, `chr<L>_<col>`.

KILL RULE (written before any run): incremental LightGBM, 8 folds by series x 3 seeds, base = [oof, step]:
  KC1 (decision) base + chr vs base: >= +0.003, > 2.5 SE, >= 6/8 folds -> paired cv.py run (only if the CPU cost
      fits the platform budget). Otherwise Chronos encoder drift is CLOSED.
  Reported, not deciding: base + chr + emb vs base + emb (does Chronos add beyond the small CNN), and (amended
  after the smoke) base + chr pb-only (the 12 pb_* columns: one encode every L steps + the history null, the only
  part cheap enough to ship). A KC1 pass that the pb-only set does not carry cannot ship as measured.
Expectation: KC1 killed (every embedding variant so far is about 0 in production); a pass would be a surprise.

Run (PowerShell, `$env:KMP_DUPLICATE_LIB_OK="TRUE"` first; nothing else on the GPU):
    python diagnostics/chronos_embed_gate.py --calibrate-only     # stage 1 only, CPU, ~10-20 min
    python diagnostics/chronos_embed_gate.py                      # all stages
    python diagnostics/chronos_embed_gate.py --eval 2000:2600 --fit 0:300 --folds 2 --lgb-seeds 1 --out chronos_embed_gate_smoke
Out: cv_out/<out>.txt, cv_out/<out>.npz (chr columns), per-L feature caches data/chr_emb_L<L>_<fit>_<eval>.npz
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "diagnostics"))
os.chdir(HERE)

import _backcast_common as C   # noqa: E402
import embedding_gate as EG    # noqa: E402

MODEL = os.path.join(HERE, "data", "hf", "chronos-2")
LS = (48, 144, 240)
PCA_DIM = 64
PCA_SERIES = 300          # fit series whose history windows give the centring / PCA constants
CAL_LIMIT = 0.0015
POS_MIN = 0.002
EV_GATE = (2000, 10000)   # rows of the saved emb / femb gate files


# ---------------------------------------------------------------------------------------------
# Chronos encoder

def load_model(device):
    import torch
    from chronos import Chronos2Pipeline
    pipe = Chronos2Pipeline.from_pretrained(MODEL, device_map=device, dtype=torch.float32)
    m = pipe.model
    m.eval()
    return m


def encode_mean(model, X, batch):
    """X (N, L) float32 -> (N, 768) mean of the context-patch tokens of the last encoder layer."""
    import torch
    dev = model.device
    out = np.empty((len(X), model.config.d_model), np.float32)
    with torch.inference_mode():
        for k in range(0, len(X), batch):
            x = torch.as_tensor(X[k:k + batch], device=dev, dtype=torch.float32)
            enc, _, _, ncp = model.encode(context=x)
            out[k:k + batch] = enc[0][:, :ncp].mean(1).float().cpu().numpy()
    if dev.type == "cuda":
        torch.cuda.empty_cache()      # varying last-batch shapes grow the cache (HANDOFF 29.5: sysmem fallback)
    return out


def chronos_embedder(model, pf, L, batch, say):
    """embed(list of (N, 4, L) window arrays) -> list of (N, PCA_DIM) unit vectors; constants from fit histories."""
    t1 = time.time()
    Ws = []
    for p in pf[:PCA_SERIES]:
        S, ih, _, _ = EG.grid(p, L)
        Ws.append(EG.windows(p, S[ih], L)[:, 0])
    R = encode_mean(model, np.concatenate(Ws), batch).astype(np.float64)
    mu = R.mean(0)
    U, s, Vt = np.linalg.svd(R - mu, full_matrices=False)
    ev = s ** 2 / (len(R) - 1)
    P = (Vt[:PCA_DIM].T / np.sqrt(ev[:PCA_DIM] + 1e-9)).astype(np.float32)
    say(f"      L{L}: PCA from {len(R):,} history windows of {min(len(pf), PCA_SERIES)} fit series: top {PCA_DIM} "
        f"explain {ev[:PCA_DIM].sum() / ev.sum():.1%}; raw cosine to the mean {np.mean((R / np.linalg.norm(R, axis=1, keepdims=True)) @ (mu / np.linalg.norm(mu))):.3f}; {time.time() - t1:.0f}s")
    mu32 = mu.astype(np.float32)

    def embed(Ws):
        sizes = [len(W) for W in Ws]
        X = np.concatenate([W[:, 0] for W in Ws]).astype(np.float32)
        E = (encode_mean(model, X, batch) - mu32) @ P
        E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
        return np.split(E, np.cumsum(sizes)[:-1])
    return embed


def cpu_cost(say, n_online, threads=4):
    import torch
    old = torch.get_num_threads()
    torch.set_num_threads(threads)
    m = load_model("cpu")
    rng = np.random.default_rng(0)
    say(f"\nCPU cost ({threads} threads, fp32; mean online length {n_online:.0f}):")
    per_series = 0.0
    for L in LS:
        x1 = rng.standard_normal((1, L)).astype(np.float32)
        xb = rng.standard_normal((256, L)).astype(np.float32)
        encode_mean(m, x1, 1)                                   # warm-up
        t = time.perf_counter()
        for _ in range(20):
            encode_mean(m, x1, 1)
        t1 = (time.perf_counter() - t) / 20
        t = time.perf_counter()
        encode_mean(m, xb, 256)
        tb = (time.perf_counter() - t) / 256
        n_hist = EG.REF // EG.HSTRIDE
        sec = (n_online / C.SUB) * t1 + n_hist * tb
        per_series += sec
        say(f"  L{L}: single window {t1 * 1e3:.1f} ms, batched {tb * 1e3:.2f} ms/window -> {sec:.2f} s per series "
            f"({n_online / C.SUB:.0f} streaming encodes + {n_hist} history windows)")
    say(f"  all L: {per_series:.2f} s per series = {per_series * 1e4 / 3600:.1f} h per 10,000 series "
        f"(budget: 15 h per two weeks for everything). Encoding every 3rd step; coarser steps cut the streaming part.")
    del m
    torch.set_num_threads(old)
    return per_series


# ---------------------------------------------------------------------------------------------

def saved_columns(ev, step, sid_local):
    """femb / emb columns of the section 33 gate files and the gate NBM columns, restricted to the eval range."""
    if not (EV_GATE[0] <= ev[0] and ev[1] <= EV_GATE[1]):
        return None, None, None
    ze = np.load(os.path.join(HERE, "cv_out", "embedding_gate.npz"))
    zf = np.load(os.path.join(HERE, "cv_out", "embedding_feat_gate.npz"))
    g = ze["sid"] + EV_GATE[0]
    m = (g >= ev[0]) & (g < ev[1])
    assert np.array_equal(ze["step"][m], step) and np.array_equal(g[m] - ev[0], sid_local), "emb rows differ"
    assert np.array_equal(zf["step"][m], step), "femb rows differ"
    nbf = np.load(os.path.join(HERE, "data", f"backcast_refs_0_2000_{EV_GATE[0]}_{EV_GATE[1]}.npz"))["NBF"]
    assert len(nbf) == len(g), "NBM reference rows differ"
    return ze["emb"][m], zf["femb"][m], nbf[m]


def oof_base(path, ev, step, sid_local, y):
    z = np.load(path)
    m = (z["sid"] >= ev[0]) & (z["sid"] < ev[1])
    assert np.array_equal(z["step"][m], step) and np.array_equal(z["sid"][m] - ev[0], sid_local) \
        and np.array_equal(z["y"][m], y), "OOF rows do not match the gate layout"
    return z["oof"][m].astype(np.float32)


def diff(sc, a, b):
    d = np.array(sc[a]) - np.array(sc[b])
    return d.mean(), d.std(ddof=1) / np.sqrt(len(d)), int((d > 0).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", default="0:2000")
    ap.add_argument("--eval", default="2000:10000")
    ap.add_argument("--folds", type=int, default=8)
    ap.add_argument("--lgb-seeds", type=int, default=3)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--batch", type=int, default=2048, help="GPU encode batch (windows)")
    ap.add_argument("--oof", default="cv_out/oof_aug2_df_bo_nbm_bs24_ep25.npz")
    ap.add_argument("--calibrate-only", action="store_true")
    ap.add_argument("--force", action="store_true", help="continue even if the calibration fails")
    ap.add_argument("--out", default="chronos_embed_gate")
    args = ap.parse_args()
    from joblib import Parallel, delayed
    from scipy.stats import spearmanr

    from sbrt.data import load_train

    EG.LS = LS                        # prep() sizes the kept history by max(LS)
    say = C.Log(os.path.join(HERE, "cv_out", f"{args.out}.txt"))
    fit = tuple(map(int, args.fit.split(":")))
    ev = tuple(map(int, args.eval.split(":")))
    S_all, tau_all = load_train()
    t0 = time.time()
    step, sid, y, in_slice = C.layout(S_all, tau_all, ev)
    base = np.column_stack([oof_base(os.path.join(HERE, args.oof), ev, step, sid, y), step]).astype(np.float32)
    emb, femb, nbf = saved_columns(ev, step, sid)
    base_bo = np.column_stack([oof_base(os.path.join(HERE, "cv_out", "oof_aug2_df_bo.npz"), ev, step, sid, y),
                               step]).astype(np.float32)
    say(f"Chronos-2 encoder drift gate | fit {fit[0]}-{fit[1] - 1} | eval {ev[0]}-{ev[1] - 1} | rows {len(y):,} | "
        f"L {LS} | base = OOF of {os.path.basename(args.oof)} + step | {args.folds} folds x {args.lgb_seeds} seeds")
    say(f"  OOF score alone on these rows: {C.fmt(C.auc_row(base[:, 0], y, step, in_slice))}   ({C.HDR})")

    # ---- stage 1: calibration
    sc = {}
    if emb is not None:
        say("\nSTAGE 1: calibration on known-null features (production paired runs: femb +0.0001, emb ~0)")
        sets = {"base": base, "+femb": np.column_stack([base, femb]).astype(np.float32),
                "+emb": np.column_stack([base, emb]).astype(np.float32)}
        C.incremental(sets, y, step, sid, args.folds, args.lgb_seeds, args.threads, say, scores_out=sc)
        say("  positive control: base' = OOF of oof_aug2_df_bo + step, + gate NBM columns (production: +0.0049)")
        scp = {}
        C.incremental({"base": base_bo, "+nb": np.column_stack([base_bo, nbf]).astype(np.float32)},
                      y, step, sid, args.folds, args.lgb_seeds, args.threads, say, scores_out=scp)
        oof_alone = C.auc_row(base[:, 0], y, step, in_slice)[0]
        base_ok = np.mean(sc["base"]) >= oof_alone - 0.003
        if not base_ok:
            say(f"  BASE DAMAGED: LightGBM base {np.mean(sc['base']):.4f} is below the OOF score alone {oof_alone:.4f} "
                "by more than 0.003 (the refit loses information)")
        g_f, g_e = diff(sc, "+femb", "base")[0], diff(sc, "+emb", "base")[0]
        pm, pse, _ = diff(scp, "+nb", "base")
        null_ok = abs(g_f) <= CAL_LIMIT and abs(g_e) <= CAL_LIMIT
        pos_ok = pm >= POS_MIN and pm > 2 * pse
        trusted = null_ok and pos_ok and base_ok
        say(f"  calibration: nulls femb {g_f:+.4f}, emb {g_e:+.4f} (need |gain| <= {CAL_LIMIT}: "
            f"{'ok' if null_ok else 'FAIL'}); positive nb {pm:+.4f} +- {pse:.4f} (need >= +{POS_MIN}, > 2 SE: "
            f"{'ok' if pos_ok else 'FAIL'}); base {'ok' if base_ok else 'FAIL'} -> "
            f"{'TRUSTED' if trusted else 'NOT TRUSTED'}")
        if not trusted and not args.force and not args.calibrate_only:
            say("  stopping (use --force to continue anyway)")
            return
    else:
        trusted = None
        say("\nSTAGE 1 skipped: eval range outside the saved gate rows")
    if args.calibrate_only:
        say(f"\ntotal {time.time() - t0:.0f}s; log {say.path}")
        return

    # ---- stage 2: CPU cost
    n_online = float(np.mean([len(S_all[i][2]) for i in range(ev[0], ev[1])]))
    sec = cpu_cost(say, n_online)

    # ---- stage 3: Chronos features
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    model = load_model(dev)
    say(f"\nSTAGE 3: Chronos features on {dev}")
    idx = list(range(fit[0], fit[1])) + list(range(ev[0], ev[1]))
    nfit = fit[1] - fit[0]
    preps = Parallel(n_jobs=args.threads)(delayed(EG.prep)(S_all[i][1], S_all[i][2], k < nfit)
                                          for k, i in enumerate(idx))
    pf, pe = preps[:nfit], preps[nfit:]
    tau_f, tau_e = tau_all[fit[0]:fit[1]], tau_all[ev[0]:ev[1]]
    feats = []
    for L in LS:
        cache = os.path.join(HERE, "data", f"chr_emb_L{L}_{fit[0]}_{fit[1]}_{ev[0]}_{ev[1]}.npz")
        if os.path.exists(cache):
            feats.append(np.load(cache)["F"])
            say(f"      L{L}: cache {os.path.basename(cache)}")
            continue
        t1 = time.time()
        embed = chronos_embedder(model, pf, L, args.batch, say)
        Fm, prior = EG.arm_features(embed, pf, tau_f, pe, tau_e, L, say)
        if L == LS[0]:
            EG.causality_check(embed, S_all, ev, prior, L, say)
        np.savez(cache, F=Fm.astype(np.float32))
        feats.append(Fm.astype(np.float32))
        say(f"      L{L}: {time.time() - t1:.0f}s ({len(pf) + len(pe)} series)")
    del model
    chr_ = np.column_stack(feats).astype(np.float32)
    names = [f"chr{L}_{c}" for L in LS for c in EG.COLS]
    assert len(chr_) == len(y)
    np.savez(os.path.join(HERE, "cv_out", f"{args.out}.npz"), chr=chr_, step=step, y=y, sid=sid)

    say("\nsingle-feature TS-AUC (* = flipped)               " + C.HDR + "   |rho| with OOF")
    sub = np.random.default_rng(0).permutation(len(y))[:200000]
    for c, nm in enumerate(names):
        r = C.auc_row(chr_[:, c], y, step, in_slice)
        flip = r[0] < 0.5
        if flip:
            r = 1.0 - r
        say(f"  {nm:22s}{'*' if flip else ' '}      {C.fmt(r)}   {abs(spearmanr(chr_[sub, c], base[sub, 0])[0]):.2f}")
    if emb is not None:
        for arm, M in (("emb", emb), ("femb", femb)):
            best = max((max(a, 1 - a), j) for j, a in
                       ((j, C.auc_row(M[:, j], y, step, in_slice)[0]) for j in range(M.shape[1])))
            say(f"  best {arm} column (reference): {best[0]:.4f}")

    say(f"\nSTAGE 3 incremental: {args.folds} folds x {args.lgb_seeds} seeds")
    pb = [j for j, nm in enumerate(names) if "_pb_" in nm]
    sets = {"base": base, "+chr": np.column_stack([base, chr_]).astype(np.float32),
            "+chr_pb": np.column_stack([base, chr_[:, pb]]).astype(np.float32)}
    if emb is not None:
        sets["+chr+emb"] = np.column_stack([base, chr_, emb]).astype(np.float32)
    sc2 = {}
    C.incremental(sets, y, step, sid, args.folds, args.lgb_seeds, args.threads, say, scores_out=sc2)
    m, se, npos = diff(sc2, "+chr", "base")
    ok = m >= 0.003 and m > 2.5 * se and npos >= max(1, round(args.folds * 0.75))

    say("\nVERDICTS (rules in the docstring, written before any run)")
    if ev[1] - ev[0] < 4000:
        say("  SMOKE RUN (< 4,000 eval series): pipeline check only; do not act on it.")
    if trusted is not None:
        say(f"  instrument: {'TRUSTED' if trusted else 'NOT TRUSTED (run forced)'} (femb / emb gains below)")
        for k in ("+femb", "+emb"):
            a = diff(sc, k, "base")
            say(f"    base {k} vs base: {a[0]:+.4f} +- {a[1]:.4f} ({a[2]}/{args.folds})")
        a = diff(scp, "+nb", "base")
        say(f"    positive control +nb vs base': {a[0]:+.4f} +- {a[1]:.4f} ({a[2]}/{args.folds})")
    a = diff(sc2, "+chr_pb", "base")
    say(f"  (reported) base + chr pb-only (cheap, shippable part) vs base: {a[0]:+.4f} +- {a[1]:.4f} ({a[2]}/{args.folds})")
    if "+chr+emb" in sc2 and "+emb" in sc:
        d = np.array(sc2["+chr+emb"]) - np.array(sc["+emb"])
        say(f"  (reported) base + chr + emb vs base + emb: {d.mean():+.4f} +- {d.std(ddof=1) / np.sqrt(len(d)):.4f} "
            f"({(d > 0).sum()}/{len(d)})")
    say(f"  KC1 base + chr vs base: {m:+.4f} +- {se:.4f} ({m / max(se, 1e-9):.1f} SE), {npos}/{args.folds} folds -> "
        f"{'PASSES: paired cv.py run next, if the CPU cost fits' if ok else 'killed: Chronos encoder drift CLOSED'}")
    say(f"  CPU cost: {sec:.2f} s per series ({sec * 1e4 / 3600:.1f} h per 10,000 series)")
    say(f"\ntotal {time.time() - t0:.0f}s; log {say.path}")


if __name__ == "__main__":
    main()
