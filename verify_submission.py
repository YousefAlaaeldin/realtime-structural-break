"""Verify the generated main.py before any crunch test / submission.

  (b) quality      main.train() on one CV fold's training series, then batch-score that fold's validation series and
                   compare TS-AUC (per member and blended) with the research out-of-fold scores on the SAME rows
  (a) parity       main.infer()'s streaming path vs the batch path on a few validation series (every member), the
                   streaming GRU inputs vs the batch GRU inputs, and the numba GRU vs a torch GRU with the same weights
  (c) determinism  two runs of main.infer() give identical outputs
  (d) timing       microseconds per online point, per component, and the projection to ~8M cloud steps

Run:   python verify_submission.py                       full check on fold 0 (trains about 15-25 min with the GPU)
       python verify_submission.py --n-train 600 --n-val 200 --parity 5      quick smoke test
Models go to verify_models/fold<k>[_n<N>]; main.train() reuses a complete set with the same configuration.
"""
import argparse
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
os.chdir(here)

import numpy as np  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

import main  # noqa: E402
from sbrt.data import load_train  # noqa: E402
from sbrt.metric import ts_auc  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--fold", type=int, default=0)
ap.add_argument("--n-train", type=int, default=0, help="use only the first N training series (smoke test)")
ap.add_argument("--n-val", type=int, default=0, help="score only the first N validation series")
ap.add_argument("--parity", type=int, default=20, help="validation series streamed through infer() for parity")
ap.add_argument("--threads", type=int, default=8)
args = ap.parse_args()

RESEARCH = dict(tree=["cv_out/oof_aug2_df_bo_nbm_bs24_ep25.npz", "cv_out/oof_aug2_df_bo_nbm_bs24_ep25_s1.npz"],
                gru=["cv_out/oof_gru_aug2_vol_bo_nbm_hpca8_swa_wavg.npz",
                     "cv_out/oof_gru_aug2_vol_bo_nbm_hpca8_swa_s1_wavg.npz"],
                svm=["cv_out/oof_svm_nbm_hpca8.npz"])


def batch_scores(M, h, o):
    """Production models on one series, batch path: tree margins and GRU logits (all seeds), SVM, at kept steps;
    also the GRU input rows and the SVM input rows (for the parity check)."""
    rows, raw = main._series_feats(h, o, M["nbm"], M["nbm_prior"])
    m_tree = [b.inplace_predict(rows, predict_type="margin") for b in M["boosters"]]
    hp = main._nbm_hp(h, o, M["nbm"], M["hpca"])
    Xs = main._svm_in(rows, hp[::main.SUB])
    m_svm = main._svm_score(M["svm"], Xs)
    Xg = main._gru_inputs(h, o, raw, M["gstd"], hp)
    g = [net.run(Xg)[main.GRU_BURN:][::main.SUB] for net in M["nets"]]
    return m_tree, g, m_svm, Xg, Xs


def main_():
    S, tau = load_train()
    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(np.arange(len(tau)), (tau >= 0).astype(int)))
    tr, va = folds[args.fold]
    if args.n_train:
        tr = tr[:args.n_train]
    if args.n_val:
        va = va[:args.n_val]
    mdir = os.path.join("verify_models", f"fold{args.fold}" + (f"_n{args.n_train}" if args.n_train else ""))
    print(f"fold {args.fold}: {len(tr)} training series, {len(va)} validation series; models in {mdir}", flush=True)

    # ---- train (skipped by main.train if a complete model set with this configuration exists)
    t0 = time.time()
    main.train([(int(S[i][0]), S[i][1], S[i][2], None if tau[i] < 0 else int(tau[i])) for i in tr], mdir)
    print(f"train(): {time.time() - t0:.0f}s", flush=True)
    M = main.load_models(mdir)
    for b in M["boosters"]:
        b.set_param({"nthread": args.threads})

    # ---- (b) quality: batch-score the validation series
    t0 = time.time()
    res = Parallel(n_jobs=args.threads, backend="threading")(delayed(batch_scores)(M, S[i][1], S[i][2]) for i in va)
    print(f"batch scoring of {len(va)} series: {time.time() - t0:.0f}s", flush=True)
    m_tree = [np.concatenate([r[0][k] for r in res]) for k in range(len(main.TREE_SEEDS))]
    g = [np.concatenate([r[1][k] for r in res]) for k in range(len(main.GRU_SEEDS))]
    m_svm = np.concatenate([r[2] for r in res])
    blend = main.blend_scores(M, m_tree, g, m_svm)
    step = np.concatenate([np.arange(0, len(S[i][2]), main.SUB) for i in va])
    sid = np.repeat(va, [(len(S[i][2]) + main.SUB - 1) // main.SUB for i in va])
    y = ((tau[sid] >= 0) & (step >= tau[sid])).astype(np.int8)
    gpair = sum(gi / si for gi, si in zip(g, M["meta"]["sd_gru"]))
    tpair = sum(mi / si for mi, si in zip(m_tree, M["meta"]["sd_tree"]))

    oof = {}
    for k, files in RESEARCH.items():
        zs = [np.load(f) for f in files]
        oof[k] = sum(z["oof"] / z["oof"].std() for z in zs)
        ref = zs[0]
    oof["blend"] = sum(w * oof[k] / oof[k].std() for w, k in zip(main.BLEND_W, ("tree", "gru", "svm")))
    mask = np.isin(ref["sid"], va)
    assert np.array_equal(ref["sid"][mask], sid) and np.array_equal(ref["step"][mask], step), "row layout mismatch"
    assert np.array_equal(ref["y"][mask], y)
    print(f"\n(b) QUALITY on fold {args.fold} validation rows ({len(y):,} rows): production (fixed rounds/epochs, "
          f"trained on {len(tr)} series) vs research out-of-fold")
    for nm, p, k in (("tree pair", tpair, "tree"), ("GRU pair", gpair, "gru"), ("SVM", m_svm, "svm"), ("BLEND", blend, "blend")):
        a, b = ts_auc(p, y, step), ts_auc(oof[k][mask], y, step)
        print(f"  {nm:9s} production {a:.4f}   research {b:.4f}   difference {a - b:+.4f}")

    # ---- (a) parity: streaming infer path vs batch path
    import torch
    print(f"\n(a) PARITY on {min(args.parity, len(va))} validation series (streaming vs batch, max abs difference)")
    dmax = dict(tree=0.0, svm=0.0, svm_in=0.0, gru=0.0, gru_in=0.0, prob=0.0)
    for j in range(min(args.parity, len(va))):
        h, o = S[va[j]][1], S[va[j]][2]
        mt, gb, ms, Xg, Xs = res[j]
        sc = main.SeriesScorer(main._clean(h), M)
        st, sg, ss, sp, xin, sin = [], [], [], [], [], []
        for x in o:
            sp.append(sc.push(float(x)))
            st.append(sc.last[0])
            sg.append(sc.last[1])
            ss.append(sc.last[2])
            xin.append(sc.gi.x.copy())
            sin.append(sc.svx.copy())
        st, ss, sg = np.array(st)[::main.SUB], np.array(ss)[::main.SUB], np.array(sg)[::main.SUB]
        dmax["tree"] = max(dmax["tree"], max(np.abs(st[:, k] - mt[k]).max() for k in range(len(mt))))
        dmax["svm"] = max(dmax["svm"], np.abs(ss - ms).max())
        dmax["svm_in"] = max(dmax["svm_in"], np.abs(np.array(sin)[::main.SUB] - Xs).max())
        dmax["gru"] = max(dmax["gru"], max(np.abs(sg[:, k] - gb[k]).max() for k in range(len(gb))))
        dmax["gru_in"] = max(dmax["gru_in"], np.abs(np.array(xin) - Xg[main.GRU_BURN:]).max())
        pb = 1 / (1 + np.exp(-main.blend_scores(M, mt, gb, ms)))
        dmax["prob"] = max(dmax["prob"], np.abs(np.array(sp)[::main.SUB] - pb).max())
    for k, v in dmax.items():
        print(f"  {k:7s} {v:.2e}")
    # numba GRU vs torch GRU with the same weights
    net = main._make_net(torch).double().eval()
    G = np.load(os.path.join(mdir, "gru.npz"))
    net.load_state_dict({k: torch.as_tensor(G[f"s{main.GRU_SEEDS[0]}|{k}"]) for k in net.state_dict()})
    Xg = res[0][3]
    with torch.no_grad():
        tl = net(torch.as_tensor(Xg[None].astype(np.float64)))[0].numpy()
    print(f"  numba GRU vs torch GRU (same weights, one series): {np.abs(tl - M['nets'][0].run(Xg)).max():.2e}")
    ok = all(v < 1e-4 for v in dmax.values())
    print("  PARITY OK" if ok else "  PARITY FAILED")

    # ---- (c) determinism and (d) timing through the real infer() generator
    sub = [(S[i][1], S[i][2]) for i in va[:5]]

    def run_infer():
        gen = main.infer(iter([(list(h), list(o)) for h, o in sub]), mdir)
        next(gen)
        return np.array(list(gen))
    t0 = time.time()
    p1 = run_infer()
    dt = time.time() - t0
    p2 = run_infer()
    n_pts = sum(len(o) for _, o in sub)
    print(f"\n(c) DETERMINISM: two infer() runs on {len(sub)} series ({n_pts} points): "
          f"{'identical' if np.array_equal(p1, p2) else 'DIFFERENT, max diff %.2e' % np.abs(p1 - p2).max()}; "
          f"outputs in [{p1.min():.3f}, {p1.max():.3f}]")
    # (d) per-component timing on the same series
    tm = dict(kernel=0.0, evidence=0.0, volstate=0.0, dist=0.0, bocpd=0.0, nbm=0.0, trees=0.0, svm=0.0, gru=0.0,
              setup=0.0)
    for h, o in sub:
        t = time.perf_counter()
        sc = main.SeriesScorer(main._clean(h), M)
        tm["setup"] += time.perf_counter() - t
        for x in o:
            x = float(x)
            t = time.perf_counter(); k = sc.fs.push(x); tm["kernel"] += time.perf_counter() - t
            t = time.perf_counter(); e = sc.es.push(x); tm["evidence"] += time.perf_counter() - t
            t = time.perf_counter(); v = sc.vs.push(x); tm["volstate"] += time.perf_counter() - t
            t = time.perf_counter(); d = sc.ds.push(x); tm["dist"] += time.perf_counter() - t
            t = time.perf_counter(); b = sc.bs.push(x); tm["bocpd"] += time.perf_counter() - t
            t = time.perf_counter(); nb = sc.ns.push(x); tm["nbm"] += time.perf_counter() - t
            r = sc.row[0]
            main.fill_row(r, k, e, v, d, b, nb)
            t = time.perf_counter()
            for T in M["trees"]:
                main._trees_margin(r, T["off"], T["left"], T["right"], T["feat"], T["thr"], T["dleft"], T["base"])
            tm["trees"] += time.perf_counter() - t
            t = time.perf_counter()
            P = M["hpca"]
            main._hp_proj(sc.ns.h[None, :], P["h_mean"], P["h_comp"], P["h_sd"], sc.hp)
            hp32 = sc.hp[0].astype(np.float32)
            tg = time.perf_counter() - t                  # projection counted with the GRU, as before
            t = time.perf_counter()
            sc.svx[:len(main.SVM_COLS)] = r[main.SVM_COLS]
            sc.svx[len(main.SVM_COLS):] = hp32
            main._svm_row(M["svm"], sc.svx)
            tm["svm"] += time.perf_counter() - t
            t = time.perf_counter() - tg
            xg = np.asarray(sc.gi.push(x, k, sc.vs.out, b, nb, hp32), dtype=np.float64)
            for net_, hh in zip(M["nets"], sc.hs):
                net_.step(xg, hh)
                net_.head(hh)
            tm["gru"] += time.perf_counter() - t
    print(f"\n(d) TIMING per online point (5 series, {n_pts} points; online lengths matter: the scans grow with t)")
    tot = 0.0
    for k, v in tm.items():
        us = 1e6 * v / n_pts
        tot += us
        print(f"  {k:9s} {us:8.0f} us")
    print(f"  total     {tot:8.0f} us  (infer() end to end {1e6 * dt / n_pts:.0f} us) -> about "
          f"{tot * 8e6 / 6e7:.0f} min for ~8M cloud points in one process")


if __name__ == "__main__":
    main_()
