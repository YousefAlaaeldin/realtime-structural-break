"""Teacher scores for GRU distillation (gru_cv.py --distill, HANDOFF section 44).

Combines out-of-fold member scores exactly like blend.py (each file / its sd, files in a group summed, group / its sd,
weighted sum) and saves the blended score z per (sid, step) row. gru_cv.py calibrates z per fold on that fold's
training rows and mixes it into the training targets.

LEAKY PROBE: the OOF score of a fold-k training row came from member models that were trained on fold k's
validation series, so a gain from this teacher is an upper bound; the clean teacher needs an inner CV per outer fold.
CLEAN teacher: feed the cv.py --lpo K files (NaN on fold K's rows) with --allow-nan, one output per outer fold K, and
give gru_cv.py --distill-file a name with {f} in it.
A group left empty (e.g. --b with no files) is skipped; --w still lists 3 weights.

Run:    python distill_teacher.py   (defaults = the 0.6527 reference blend, weights 0.4 / 0.4 / 0.2)
        python distill_teacher.py --constant --out cv_out/teacher_const.npz   (label-smoothing control)
        python distill_teacher.py --a <trees lpo0 s0> <trees lpo0 s1> --b --c <svm lpo0> --allow-nan --out cv_out/teacher_lpo_f0.npz
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sbrt.metric import ts_auc  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--a", nargs="+", default=["cv_out/oof_aug2_df_bo_nbm_bs24_ep25.npz",
                                           "cv_out/oof_aug2_df_bo_nbm_bs24_ep25_s1.npz"])
ap.add_argument("--b", nargs="*", default=["cv_out/oof_gru_aug2_vol_bo_nbm_hpca8_swa_wavg.npz",
                                           "cv_out/oof_gru_aug2_vol_bo_nbm_hpca8_swa_s1_wavg.npz"])
ap.add_argument("--c", nargs="*", default=["cv_out/oof_svm_nbm_hpca8.npz"])
ap.add_argument("--w", nargs=3, type=float, default=[0.4, 0.4, 0.2])
ap.add_argument("--out", default="cv_out/teacher_blend.npz")
ap.add_argument("--constant", action="store_true",
                help="control: z = 0 everywhere, so the per-fold Platt fit returns the per-step prior and --distill "
                     "becomes pure label smoothing through the same code path")
ap.add_argument("--allow-nan", action="store_true", help="accept partial OOF files (cv.py --lpo); sd over scored rows")
args = ap.parse_args()

ref = None
parts = []
for files, wt in zip((args.a, args.b, args.c), args.w):
    if not files:
        continue
    g = 0.0
    for fn in files:
        z = np.load(fn)
        if ref is None:
            ref = z
        elif not (np.array_equal(z["sid"], ref["sid"]) and np.array_equal(z["step"], ref["step"])):
            sys.exit(f"{fn}: different row layout")
        if np.isnan(z["oof"]).any() and not args.allow_nan:
            sys.exit(f"{fn}: has NaN rows (not a full 5-fold OOF file)")
        g = g + z["oof"] / np.nanstd(z["oof"])
    parts.append(wt * g / np.nanstd(g))
z = sum(parts)
ok = ~np.isnan(z)
print(f"scored rows {ok.sum():,} of {len(z):,}")
if args.constant:
    z = np.zeros_like(z)
    print("--constant: z = 0 (label-smoothing control)")
if not args.constant:
    print(f"teacher TS-AUC (scored rows): {ts_auc(z[ok], ref['y'][ok], ref['step'][ok]):.4f}")
np.savez(args.out, z=z.astype(np.float64), y=ref["y"], step=ref["step"], sid=ref["sid"])
print(f"saved {args.out}")
