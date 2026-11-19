"""Stacking: learn how to combine the members' out-of-fold scores, instead of fixed blend weights.

Inputs per row (online step of a series): each member's out-of-fold score (groups --a/--b/--c/--d as in blend.py,
several files in a group are averaged after dividing by their sd; each group is then divided by its sd), plus
context the fixed weights cannot use:
    --ctx none     members only
    --ctx step     + t_log (how far into the online segment)
    --ctx hist     + t_log + the 5 history descriptors (ctx_logn, ctx_skew, ctx_kurt, ctx_rho1, ctx_phi)
    --ctx vol      + the above + the 4 volatility-state constants (vol_lvh200, vol_lvh1000, vol_logs2T, vol_persist)
Model: a small LightGBM (binary, metric-matched step weights as in cv.py), monotone increasing in every member score
by default (more evidence from a member never lowers the stacked score; a strong regulariser).

Honest by construction: CROSS-FITTED over the same 5 folds as cv.py: the stacker for fold f is trained on the
out-of-fold rows of the other 4 folds and scores fold f (the members' OOF scores for fold f never saw fold f either).
The headline round is the best shared round of the AVERAGE curve, and the paired comparison against the fixed-weight
blend (--ref-w, default 0.4/0.4/0.2 on the same scaled scores) uses that single round in every fold.

Run:  python stack.py --a cv_out/oof_aug2_evc_vol_df.npz --b cv_out/oof_gru_aug2_vol.npz cv_out/oof_gru_aug2_vol_s1.npz
                      --c cv_out/oof_svm.npz --ctx hist
Out:  cv_out/stack_<tag>.txt and cv_out/oof_stack_<tag>.npz (same row layout; can be blended / compared further)
"""
import argparse
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import lightgbm as lgb  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

from sbrt import features as F  # noqa: E402
from sbrt import volstate as VS  # noqa: E402
from sbrt.data import load_train  # noqa: E402
from sbrt.metric import ts_auc  # noqa: E402

ap = argparse.ArgumentParser()
for g in "abcd":
    ap.add_argument(f"--{g}", nargs="+", default=[], required=g in "ab")
ap.add_argument("--ctx", choices=["none", "step", "hist", "vol"], default="hist")
ap.add_argument("--ref-w", type=str, default="0.4,0.4,0.2", help="fixed blend weights to compare against")
ap.add_argument("--leaves", type=int, default=15)
ap.add_argument("--min-leaf", type=int, default=5000)
ap.add_argument("--lr", type=float, default=0.05)
ap.add_argument("--rounds", type=int, default=400)
ap.add_argument("--grid", type=int, default=25)
ap.add_argument("--no-mono", action="store_true", help="drop the monotone constraint on the member scores")
ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--tag", type=str, default="")
args = ap.parse_args()
TAG = args.tag or args.ctx
os.makedirs("cv_out", exist_ok=True)
log = open(os.path.join("cv_out", f"stack_{TAG}.txt"), "w", encoding="utf-8")


def say(s=""):
    print(s, flush=True)
    log.write(s + "\n")
    log.flush()


def load_group(files):
    zs = [np.load(f) for f in files]
    for z in zs[1:]:
        if not (np.array_equal(z["sid"], zs[0]["sid"]) and np.array_equal(z["step"], zs[0]["step"])):
            sys.exit("files have different row layouts")
    s = sum(z["oof"] / z["oof"].std() for z in zs)
    return s / s.std(), zs[0]


def step_weights(y, step):
    T = int(step.max()) + 1
    npos = np.bincount(step, weights=y, minlength=T)
    n = np.bincount(step, minlength=T).astype(float)
    w = np.where(y == 1, (n - npos)[step] / 2.0, npos[step] / 2.0)
    return w / max(w.mean(), 1e-12)


def main():
    groups = [getattr(args, g) for g in "abcd" if getattr(args, g)]
    G, refs = zip(*[load_group(g) for g in groups])
    ref = refs[0]
    for r in refs[1:]:
        if not np.array_equal(r["sid"], ref["sid"]):
            sys.exit("groups have different row layouts")
    y, step, sid = ref["y"], ref["step"], ref["sid"]
    cols, names = list(G), [f"m_{'abcd'[i]}" for i in range(len(G))]
    if args.ctx != "none":
        z = np.load(os.path.join("data", f"feat_{F.N_FEAT}.npz"))
        if not (np.array_equal(z["sid"], sid) and np.array_equal(z["step"], step)):
            sys.exit("feature cache row layout differs from the OOF files")
        want = ["t_log"] + ([n for n in F.FEATURE_NAMES if n.startswith("ctx_")] if args.ctx in ("hist", "vol") else [])
        X = z["X"]
        for n in want:
            cols.append(X[:, F.FEATURE_NAMES.index(n)].astype(np.float64))
            names.append(n)
        del X, z
    if args.ctx == "vol":
        V = np.load(os.path.join("data", f"vol{VS.VERSION}_orig.npz"))["V"]
        if len(V) != len(y):
            sys.exit("volatility-state cache row count differs")
        for n in ("vol_lvh200", "vol_lvh1000", "vol_logs2T", "vol_persist"):
            cols.append(V[:, VS.FEATURE_NAMES.index(n)].astype(np.float64))
            names.append(n)
    Xs = np.column_stack(cols).astype(np.float32)
    nm = len(G)
    rw = [float(v) for v in args.ref_w.split(",")]
    if len(rw) != nm:
        sys.exit(f"--ref-w needs {nm} weights")
    ref_score = sum(w * g for w, g in zip(rw, G))
    say(f"stacker inputs ({Xs.shape[1]}): {names}")
    say(f"members alone: " + ", ".join(f"{n} {ts_auc(g, y, step):.4f}" for n, g in zip(names, G))
        + f";  fixed blend {rw}: {ts_auc(ref_score, y, step):.4f}")

    params = dict(objective="binary", learning_rate=args.lr, num_leaves=args.leaves, min_data_in_leaf=args.min_leaf,
                  feature_fraction=1.0, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, seed=0,
                  deterministic=True, force_row_wise=True, num_threads=args.threads, verbose=-1)
    if not args.no_mono:
        params["monotone_constraints"] = [1] * nm + [0] * (Xs.shape[1] - nm)
    _, tau = load_train()
    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(np.arange(len(tau)), (tau >= 0).astype(int)))
    grid = list(range(args.grid, args.rounds + 1, args.grid))
    curves, raws, refsc, masks = [], [], [], []
    for f, (_, va_s) in enumerate(folds):
        va = np.isin(sid, va_s)
        tr = ~va
        m = lgb.train(params, lgb.Dataset(Xs[tr], y[tr], weight=step_weights(y[tr], step[tr])), args.rounds)
        raw = np.zeros(int(va.sum()))
        curve, prev, per_round = [], 0, []
        for k in grid:
            raw = raw + m.predict(Xs[va], start_iteration=prev, num_iteration=k - prev, raw_score=True)
            prev = k
            curve.append(ts_auc(raw, y[va], step[va]))
            per_round.append(raw.copy())
        curves.append(curve)
        raws.append(per_round)
        refsc.append(ts_auc(ref_score[va], y[va], step[va]))
        masks.append(va)
        j = int(np.argmax(curve))
        say(f"fold {f}: stack best {curve[j]:.4f} at round {grid[j]} | fixed blend {refsc[-1]:.4f}")
    mean_curve = np.mean(curves, axis=0)
    kb = int(np.argmax(mean_curve))
    say("\naverage curve: " + "  ".join(f"{k}:{v:.4f}" for k, v in zip(grid, mean_curve)))
    d = np.array([c[kb] for c in curves]) - np.array(refsc)
    say(f"\nHEADLINE stack {mean_curve[kb]:.4f} at {grid[kb]} rounds (one round for all folds) vs fixed blend "
        f"{np.mean(refsc):.4f}")
    say(f"PAIRED stack - fixed blend: {d.mean():+.4f} +- {d.std(ddof=1) / np.sqrt(len(d)):.4f}, "
        f"{int((d > 0).sum())}/5 folds better   (per fold {np.round(d, 4)})")
    oof = np.full(len(y), np.nan)
    for va, pr in zip(masks, raws):
        oof[va] = pr[kb]
    say("\nby online-step range (pooled):   stack   fixed blend")
    for lo, hi in ((0, 100), (100, 300), (300, 700), (700, 100000)):
        mk = (step >= lo) & (step < hi)
        say(f"  steps {lo:4d}-{hi if hi < 100000 else 'end':>4}:   {ts_auc(oof[mk], y[mk], step[mk]):.4f}   "
            f"{ts_auc(ref_score[mk], y[mk], step[mk]):.4f}")
    np.savez(os.path.join("cv_out", f"oof_stack_{TAG}.npz"), oof=oof, y=y, step=step, sid=sid)
    say(f"\nsaved cv_out/oof_stack_{TAG}.npz")


if __name__ == "__main__":
    main()
