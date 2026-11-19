"""Blend out-of-fold scores of 2 to 4 models and report TS-AUC.

Run:    python blend.py --a cv_out/oof_aug2_gpu.npz --b cv_out/oof_gru_aug2.npz cv_out/oof_gru_aug2_s1.npz
        python blend.py --a cv_out/oof_aug2_gpu.npz --b cv_out/oof_gru_aug2.npz --c cv_out/oof_rank.npz

Each group (--a, --b, --c) is one model; several files in a group (e.g. seeds) are averaged first. Two ways of
putting the models on one scale, both usable at inference time:
  global    divide each model's score by its overall standard deviation (two constants to ship)
  per-step  standardise each model's score with the mean and std of ITS OWN out-of-fold scores at that online step
            (a small table to ship). Needed for ranking-trained models, whose score scale is free at every step.

--honest  the weight search above picks the weights on the same rows it scores, so its best number is slightly
          optimistic. --honest re-does it with CROSS-FITTED weights (chosen on 4 folds, scored on the 5th), for
          (1) one set of global weights and (2) separate weights per online-step range (0-100, 100-300, 300-700,
          700+). TS-AUC only compares series at the same step, so per-range weights are valid and easy to ship.
          Prints the honest scores, a paired per-fold comparison of (2) vs (1), and the weights to ship (chosen on
          all rows).
"""
import argparse
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sbrt.metric import ts_auc  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--a", nargs="+", required=True)
ap.add_argument("--b", nargs="+", required=True)
ap.add_argument("--c", nargs="+", default=[])
ap.add_argument("--d", nargs="+", default=[])
ap.add_argument("--ref-a", nargs="+", default=[],
                help="baseline blend for a paired per-fold comparison (same group syntax; e.g. the current best)")
ap.add_argument("--ref-b", nargs="+", default=[])
ap.add_argument("--ref-c", nargs="+", default=[])
ap.add_argument("--ref-d", nargs="+", default=[])
ap.add_argument("--honest", action="store_true",
                help="cross-fitted weight search, global and per step range (see the docstring)")
ap.add_argument("--honest-paired", action="store_true",
                help="with --ref-*: paired per-fold comparison with CROSS-FITTED weights for both blends (weights "
                     "chosen on 4 folds, scored on the 5th). Use this to judge a new MEMBER: in-sample weights can "
                     "only gain from an extra member")
args = ap.parse_args()


def load_group(files):
    zs = [np.load(f) for f in files]
    for z in zs[1:]:
        if not (np.array_equal(z["sid"], zs[0]["sid"]) and np.array_equal(z["step"], zs[0]["step"])):
            sys.exit("files have different row layouts")
    return sum(z["oof"] / z["oof"].std() for z in zs), zs[0]


def per_step(x, step):
    T = int(step.max()) + 1
    n = np.maximum(np.bincount(step, minlength=T), 1)
    m = np.bincount(step, weights=x, minlength=T) / n
    v = np.bincount(step, weights=x * x, minlength=T) / n - m * m
    return (x - m[step]) / np.sqrt(np.maximum(v[step], 1e-12))


groups = [g for g in (args.a, args.b, args.c, args.d) if g]
G, ref = zip(*[load_group(g) for g in groups])
y, step = ref[0]["y"], ref[0]["step"]
for r in ref[1:]:
    if not np.array_equal(r["sid"], ref[0]["sid"]):
        sys.exit("groups have different row layouts")
for i, (g, files) in enumerate(zip(G, groups)):
    print(f"model {'abcd'[i]}: {ts_auc(g, y, step):.4f}   ({', '.join(files)})")

grid = np.round(np.arange(0, 1.0001, 0.1), 2)
BUCKETS = ((0, 100), (100, 300), (300, 700), (700, 100000))


def honest():
    from sklearn.model_selection import StratifiedKFold
    from sbrt.data import load_train
    tau = load_train()[1]
    sid = ref[0]["sid"]
    fold_of = np.zeros(len(tau), int)
    for f, (_, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(np.arange(len(tau)),
                                                                                       (tau >= 0).astype(int))):
        fold_of[va] = f
    fr = fold_of[sid]
    S_ = [g / g.std() for g in G]
    combos = [w for w in itertools.product(grid, repeat=len(S_)) if abs(sum(w) - 1) < 1e-9]
    blend = lambda w, m: sum(wi * si[m] for wi, si in zip(w, S_))

    def best_on(m):
        return max(combos, key=lambda w: ts_auc(blend(w, m), y[m], step[m]))

    print(f"\nHONEST weight search ({len(combos)} weight combinations, cross-fitted over 5 folds)")
    glob, buck, insamp = [], [], []
    for f in range(5):
        tr, va = fr != f, fr == f
        if not va.any():
            continue
        w = best_on(tr)
        glob.append(ts_auc(blend(w, va), y[va], step[va]))
        insamp.append(max(ts_auc(blend(c, va), y[va], step[va]) for c in combos))
        sc = np.zeros(int(va.sum()))
        idx = np.flatnonzero(va)
        wb = []
        for lo, hi in BUCKETS:
            bt = tr & (step >= lo) & (step < hi)
            bv = (step[idx] >= lo) & (step[idx] < hi)
            wk = best_on(bt)
            wb.append(wk)
            sc[bv] = blend(wk, idx[bv])
        buck.append(ts_auc(sc, y[va], step[va]))
        print(f"  fold {f}: global weights {tuple(map(float, w))} -> {glob[-1]:.4f} (in-sample best on this fold "
              f"{insamp[-1]:.4f});  per-range weights -> {buck[-1]:.4f}", flush=True)
    glob, buck, insamp = map(np.array, (glob, buck, insamp))
    d = buck - glob
    print(f"  honest global weights: {glob.mean():.4f}   (in-sample per fold {insamp.mean():.4f}: optimism "
          f"{insamp.mean() - glob.mean():+.4f})")
    print(f"  honest per-range weights: {buck.mean():.4f};  paired vs global {d.mean():+.4f} +- "
          f"{d.std(ddof=1) / np.sqrt(len(d)):.4f}, {int((d > 0).sum())}/{len(d)} folds better")
    allm = np.ones(len(y), bool)
    print(f"  weights to ship (chosen on all rows): global {dict(zip('abcd', map(float, best_on(allm))))}")
    for lo, hi in BUCKETS:
        m = (step >= lo) & (step < hi)
        print(f"    steps {lo}-{hi if hi < 100000 else 'end'}: {dict(zip('abcd', map(float, best_on(m))))}")


if args.honest:
    honest()


def best_global(Gs):
    """Best global-scaling blend of a list of model scores: (TS-AUC, weights, blended scores)."""
    S_ = [g / g.std() for g in Gs]
    best = (-1.0, None, None)
    for w in itertools.product(grid, repeat=len(S_)):
        if abs(sum(w) - 1) > 1e-9:
            continue
        s = sum(wi * si for wi, si in zip(w, S_))
        sc = ts_auc(s, y, step)
        if sc > best[0]:
            best = (sc, w, s)
    return best


if args.ref_a:
    from sklearn.model_selection import StratifiedKFold
    from sbrt.data import load_train
    refs = [g for g in (args.ref_a, args.ref_b, args.ref_c, args.ref_d) if g]
    RG, rref = zip(*[load_group(g) for g in refs])
    if not np.array_equal(rref[0]["sid"], ref[0]["sid"]):
        sys.exit("reference blend has a different row layout")
    new_sc, new_w, new_s = best_global(list(G))
    old_sc, old_w, old_s = best_global(list(RG))
    tau = load_train()[1]
    sid = ref[0]["sid"]
    d = []
    for _, va in StratifiedKFold(5, shuffle=True, random_state=0).split(np.arange(len(tau)), (tau >= 0).astype(int)):
        m = np.isin(sid, va)
        if m.any():
            d.append(ts_auc(new_s[m], y[m], step[m]) - ts_auc(old_s[m], y[m], step[m]))
    d = np.array(d)
    print(f"\nPAIRED (global scaling, each blend at its own best weights): this blend {new_sc:.4f} "
          f"{dict(zip('abcd', map(float, new_w)))} vs reference {old_sc:.4f} {dict(zip('abcd', map(float, old_w)))}")
    print(f"  per fold {np.round(d, 4)}  ->  {d.mean():+.4f} +- {d.std(ddof=1) / np.sqrt(len(d)):.4f}, "
          f"{int((d > 0).sum())}/{len(d)} folds better")
    if args.honest_paired:
        # in-sample weights favour the blend with MORE members (its grid contains the smaller one's), so a new
        # member always looks >= 0 above. Cross-fitted: weights chosen on 4 folds, scored on the 5th, both blends.
        fr = np.zeros(len(tau), int)
        for f, (_, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(np.arange(len(tau)),
                                                                                           (tau >= 0).astype(int))):
            fr[va] = f
        fr = fr[sid]

        def honest_folds(Gs):
            S_ = [g / g.std() for g in Gs]
            combos = [w for w in itertools.product(grid, repeat=len(S_)) if abs(sum(w) - 1) < 1e-9]
            out, ws = [], []
            for f in range(5):
                tr, va = fr != f, fr == f
                if not va.any():
                    continue
                w = max(combos, key=lambda c: ts_auc(sum(ci * si[tr] for ci, si in zip(c, S_)), y[tr], step[tr]))
                out.append(ts_auc(sum(wi * si[va] for wi, si in zip(w, S_)), y[va], step[va]))
                ws.append(tuple(map(float, w)))
            return np.array(out), ws

        hn, wn = honest_folds(list(G))
        ho, wo = honest_folds(list(RG))
        dh = hn - ho
        print(f"\nHONEST PAIRED (cross-fitted weights, the adoption test): this blend {hn.mean():.4f} vs reference "
              f"{ho.mean():.4f}")
        for f in range(len(dh)):
            print(f"  fold {f}: {hn[f]:.4f} {wn[f]} vs {ho[f]:.4f} {wo[f]}  {dh[f]:+.4f}")
        print(f"  ->  {dh.mean():+.4f} +- {dh.std(ddof=1) / np.sqrt(len(dh)):.4f}, "
              f"{int((dh > 0).sum())}/{len(dh)} folds better")

for label, S in (("global", [g / g.std() for g in G]), ("per-step", [per_step(g, step) for g in G])):
    best = (0, None)
    rows = []
    for w in itertools.product(grid, repeat=len(S)):
        if abs(sum(w) - 1) > 1e-9:
            continue
        sc = ts_auc(sum(wi * si for wi, si in zip(w, S)), y, step)
        rows.append((sc, w))
        best = max(best, (sc, w))
    print(f"\n{label} scaling: best TS-AUC {best[0]:.4f} at weights {dict(zip('abcd', map(float, best[1])))}")
    if len(S) == 2:
        print("  " + "  ".join(f"b={w[1]:.1f}:{sc:.4f}" for sc, w in rows))
    else:
        for sc, w in sorted(rows, reverse=True)[:6]:
            print(f"  {dict(zip('abcd', map(float, w)))}: {sc:.4f}")
