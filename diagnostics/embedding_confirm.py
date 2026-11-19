"""Confirmation of the embedding gate's weak positive (HANDOFF section 33): the learned embedding gave +0.0019 +- 0.0014
over the plain base (+0.0026 +- 0.0018 over base + ctl), 3/4 folds, i.e. 1.4 SE. Same saved features
(cv_out/embedding_gate.npz), same rows, same LightGBM settings; only the instrument is sharper: 8 folds x 3 seeds.
No retraining, no new features.

RULE (written 2026-10-15 after seeing the 4-fold result, before this run):
  CONFIRMED  base + emb vs base (base = bo + vol + nb + step + MMD): >= +0.002, > 2.5 SE, >= 6/8 folds.
  If CONFIRMED -> the learned embedding earns a paired cv.py run (production feature set, 5 folds, usual bar).
  Otherwise -> closed. A result like +0.0015 +- 0.0007 would mean "probably real, too small to build".
  Also reported (not deciding): base + ctl vs base, base + ctl + emb vs base + ctl.
ADDED after embedding_feat_gate.py (femb = contrastive MLP on the 15 window statistics: +0.0029 +- 0.0013 over + ctl,
+0.0022 +- 0.0016 over the base, 4 folds), before this run, same rule:
  CONFIRMED-femb  base + femb vs base: >= +0.002, > 2.5 SE, >= 6/8 folds -> femb earns the paired cv.py run (cheap at
  inference: 15 statistics + a 15-64-64-16 MLP per step). Also reported: base + femb + emb vs base + femb.

Run (CPU; my estimate 15-30 min, 6 sets x 24 fits):
    python diagnostics/embedding_confirm.py
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", default="0:2000")
    ap.add_argument("--eval", default="2000:10000")
    ap.add_argument("--folds", type=int, default=8)
    ap.add_argument("--lgb-seeds", type=int, default=3)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--emb-npz", default="cv_out/embedding_gate.npz")
    ap.add_argument("--femb-npz", default="cv_out/embedding_feat_gate.npz")
    ap.add_argument("--out", default="embedding_confirm")
    args = ap.parse_args()
    from sbrt.data import load_train

    say = C.Log(os.path.join(HERE, "cv_out", f"{args.out}.txt"))
    fit = tuple(map(int, args.fit.split(":")))
    ev = tuple(map(int, args.eval.split(":")))
    S_all, tau_all = load_train()
    t0 = time.time()
    Bo, V, NBF = C.references(S_all, tau_all, fit, ev, args.threads, say)
    MM = EG.mmd_reference(S_all, ev, args.threads, say)
    z = np.load(os.path.join(HERE, args.emb_npz))
    step, y, sid, ctl, emb = z["step"], z["y"], z["sid"], z["ctl"], z["emb"]
    assert len(step) == len(Bo) == len(MM), "row mismatch between the saved features and the references"
    zf = np.load(os.path.join(HERE, args.femb_npz))
    assert np.array_equal(zf["step"], step), "femb npz rows do not match"
    femb = zf["femb"]
    say(f"embedding confirmation | {len(y):,} rows | {args.folds} folds x {args.lgb_seeds} seeds | features {args.emb_npz}")

    base = np.column_stack([Bo, V, NBF, step, MM]).astype(np.float32)
    sets = {"base": base,
            "+emb": np.column_stack([base, emb]).astype(np.float32),
            "+ctl": np.column_stack([base, ctl]).astype(np.float32),
            "+ctl+emb": np.column_stack([base, ctl, emb]).astype(np.float32),
            "+femb": np.column_stack([base, femb]).astype(np.float32),
            "+femb+emb": np.column_stack([base, femb, emb]).astype(np.float32)}
    sc = {}
    C.incremental(sets, y, step, sid, args.folds, args.lgb_seeds, args.threads, say, scores_out=sc)
    d = np.array(sc["+ctl+emb"]) - np.array(sc["+ctl"])
    say(f"  +emb over +ctl:  {d.mean():+.4f} +- {d.std(ddof=1) / np.sqrt(len(d)):.4f}  ({(d > 0).sum()}/{len(d)} folds)")
    d = np.array(sc["+femb+emb"]) - np.array(sc["+femb"])
    say(f"  +emb over +femb: {d.mean():+.4f} +- {d.std(ddof=1) / np.sqrt(len(d)):.4f}  ({(d > 0).sum()}/{len(d)} folds)")

    say("\nVERDICTS (rules in the docstring, written before this run)")
    for key, label in (("+emb", "raw-window emb"), ("+femb", "learned-statistics femb")):
        d = np.array(sc[key]) - np.array(sc["base"])
        m, se, npos = d.mean(), d.std(ddof=1) / np.sqrt(len(d)), int((d > 0).sum())
        ok = m >= 0.002 and m > 2.5 * se and npos >= 6
        say(f"  base + {key[1:]} vs base ({label}): {m:+.4f} +- {se:.4f} ({m / max(se, 1e-9):.1f} SE), {npos}/{len(d)} "
            f"folds -> {'CONFIRMED: paired cv.py run next' if ok else 'not confirmed'}")
    say(f"\ntotal {time.time() - t0:.0f}s; log {say.path}")


if __name__ == "__main__":
    main()
