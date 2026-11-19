"""Search ALL existing out-of-fold files for the best blend, with an honest (nested) estimate of what searching buys.

Owner's idea 2026-10-18: members rejected on their own might still help in combination, and the current blend should
not be assumed optimal. So the default search starts FROM SCRATCH and grows blends one file at a time:

  size 1: every single file on its own; the best is the first peak (a single file can stay the winner).
  size 2: EVERY pair of files (weight on the second 0.1-0.9); the --beam best pairs are kept.
  size 3, 4, ...: for every kept blend s, every file c and every w in WGRID: s' = (1 - w) * s + w * c / sd(c); keep
              the --beam best distinct blends. A blend that beats the peak becomes the new peak.
  stop when a step raises the peak by < --min-gain, or after --max-add additions. A file may be picked again (that
  raises its weight). --beam is how many candidate blends are carried to the next size, NOT a minimum size.
The search on all rows also prints the top 40 pairs and the best partner of every file.
--start current instead starts from the shipped blend (trees pair, GRU pair, SVM at 0.5 / 0.3 / 0.2) and only adds.

Picking from ~74 files on the same 5 folds finds luck, so every number that decides anything is NESTED:
  outer loop over the 5 CV folds: run the search on the other 4 folds, score the blend it found on the held-out fold
  next to the CURRENT blend (and the best single file) on that fold -> 5 honest paired gains.
Then the same search on ALL rows gives the blend to ship if the honest test passes (its in-sample score is optimistic).

Candidates: every cv_out/oof_*.npz with the full 10,000-series row layout and no NaN (full 5-fold runs), or --only.
The current blend's weights were chosen on all rows: a small leak in favour of the CURRENT arm.
DECISION RULE (written before any run): the searched blend is a candidate for production only if its NESTED gain vs
the current blend is >= +0.0005, in >= 4/5 folds, and > 2 SE (standing bar, section 5 of HANDOFF). Also look at the
per-fold picks: if each fold finds different files, the gain is not a stable combination.

Run:  python diagnostics/blend_search.py                     -> diagnostics/blend_search.txt (from scratch)
      python diagnostics/blend_search.py --start current --out diagnostics/blend_search_current.txt
      python diagnostics/blend_search.py --smoke             (8 candidates, 1 outer fold, checks the fast metric)
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
from numba import njit, prange

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from sbrt.metric import ts_auc  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--start", choices=["scratch", "current"], default="scratch")
ap.add_argument("--beam", type=int, default=10, help="candidate blends carried to the next size (not a minimum size)")
ap.add_argument("--min-gain", type=float, default=0.0002)
ap.add_argument("--max-add", type=int, default=10)
ap.add_argument("--only", nargs="+", default=[], help="restrict candidates to these files")
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                help="cuda = batched GPU metric (~25x faster); auto = cuda if available")
ap.add_argument("--batch", type=int, default=48, help="blends per GPU call (~2.5 GB of GPU memory per 32)")
ap.add_argument("--out", default=os.path.join(ROOT, "diagnostics", "blend_search.txt"))
args = ap.parse_args()
WGRID = np.array([0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5])   # weight of a file added to a blend of 2+ files
PGRID = np.arange(1, 10) / 10                              # weight of the second file of a pair
CV = os.path.join(ROOT, "cv_out")
CURRENT = [(["oof_aug2_df_bo_nbm_bs24_ep25.npz", "oof_aug2_df_bo_nbm_bs24_ep25_s1.npz"], 0.5),
           (["oof_gru_aug2_vol_bo_nbm.npz", "oof_gru_aug2_vol_bo_nbm_s1.npz"], 0.3),
           (["oof_svm_nbm.npz"], 0.2)]
SMOKE = ("aug2_df_bo_nbm_bs24_ep25", "aug2_df_bo_nbm_bs24_ep25_s1", "gru_aug2_vol_bo_nbm", "svm_nbm",
         "tcn_aug2_vol_bo_nbm", "aug2_df_bo_nbm_bs24_ep25_LR015", "aug2_df_bo_nbm_bs24_ep25_pc", "gru_aug2_vol")


class Tee:
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")

    def write(self, s):
        sys.__stdout__.write(s)
        self.f.write(s)
        self.f.flush()

    def flush(self):
        sys.__stdout__.flush()


@njit(parallel=True, cache=True)
def _auc(s, y, starts, ends):
    G = len(starts)
    num = np.zeros(G)
    den = np.zeros(G)
    for g in prange(G):
        a, b = starts[g], ends[g]
        n = b - a
        npos = 0
        for i in range(a, b):
            npos += y[i]
        nneg = n - npos
        if npos == 0 or nneg == 0:
            continue
        o = np.argsort(s[a:b])
        rs = 0.0
        i = 0
        while i < n:   # average ranks over ties, like scipy rankdata
            j = i
            while j + 1 < n and s[a + o[j + 1]] == s[a + o[i]]:
                j += 1
            r = 0.5 * (i + j) + 1.0
            for k in range(i, j + 1):
                if y[a + o[k]] == 1:
                    rs += r
            i = j + 1
        num[g] = (rs - npos * (npos + 1) / 2.0)
        den[g] = npos * nneg
    return num.sum() / den.sum()


class Rows:
    """A row subset pre-sorted by step, so each TS-AUC call is one parallel pass."""

    def __init__(self, idx, step, y):
        o = idx[np.argsort(step[idx], kind="stable")]
        b = np.flatnonzero(np.diff(step[o])) + 1
        self.o = o
        self.y = y[o].astype(np.int64)
        self.starts = np.r_[0, b].astype(np.int64)
        self.ends = np.r_[b, len(o)].astype(np.int64)

    def score(self, s_sorted):
        return _auc(s_sorted, self.y, self.starts, self.ends)


class CpuMixer:
    """Scores blends (1 - w) * base + w * file_c on one row subset with the numba metric."""

    def __init__(self, rows, C, current):
        self.rows = rows
        self.Cr = C[:, rows.o].astype(np.float64)
        self.cur = current[rows.o].astype(np.float64)

    def file(self, c):
        return self.Cr[c]

    def score(self, s):
        return self.rows.score(s)

    def mix(self, base, cs, ws):
        return np.array([[self.rows.score((1 - w) * base + w * self.Cr[c]) for w in ws] for c in cs])

    def vec(self, base, c, w):
        return (1 - w) * base + w * self.Cr[c]


class GpuMixer(CpuMixer):
    """Same on the GPU, many blends per call: sort each blend's scores, then stable-sort by online step; the rank of
    a row inside its step is its position minus the step's first position. Ties are not averaged (continuous scores;
    checked equal to the CPU metric to ~1e-8 at start-up)."""

    def __init__(self, rows, C, current, batch=48):
        import torch
        self.t = torch
        self.rows = rows
        self.B = batch
        d = self.dev = torch.device("cuda")
        n = rows.ends - rows.starts
        gid = np.repeat(np.arange(len(n)), n)
        npos = np.add.reduceat(rows.y, rows.starts)
        self.C0 = float((npos * (npos + 1) / 2).sum())
        self.den = float((npos * (n - npos)).sum())
        self.G = torch.as_tensor(gid, device=d)
        self.R = torch.as_tensor(np.arange(len(gid)) - rows.starts[gid] + 1, device=d)
        self.Y = torch.as_tensor(rows.y.astype(bool), device=d)
        self.Cr = torch.as_tensor(C[:, rows.o], dtype=torch.float32, device=d)
        self.cur = torch.as_tensor(current[rows.o], dtype=torch.float32, device=d)

    def _batch(self, S):
        t = self.t
        _, p1 = t.sort(S, dim=1, stable=True)
        _, p2 = t.sort(self.G[p1], dim=1, stable=True)
        tot = t.where(self.Y[t.gather(p1, 1, p2)], self.R[None, :], 0).sum(1)
        return ((tot.double() - self.C0) / self.den).cpu().numpy()

    def score(self, s):
        return float(self._batch(s[None])[0])

    def mix(self, base, cs, ws):
        t = self.t
        cw = [(c, w) for c in cs for w in ws]
        out = []
        for k in range(0, len(cw), self.B):
            ch = cw[k:k + self.B]
            ci = t.as_tensor([c for c, _ in ch], device=self.dev)
            wi = t.as_tensor([w for _, w in ch], dtype=t.float32, device=self.dev)[:, None]
            out.append(self._batch((1 - wi) * base[None] + wi * self.Cr[ci]))
        return np.concatenate(out).reshape(len(cs), len(ws)) if cw else np.zeros((0, len(ws)))


def main():
    sys.stdout = Tee(args.out)
    t0 = time.time()
    print(f"settings: {vars(args)}")
    ref = np.load(os.path.join(CV, CURRENT[2][0][0]))
    sid, step, y = ref["sid"], ref["step"], ref["y"]

    def zs(files):
        zz = [np.load(os.path.join(CV, f)) for f in files]
        g = sum(z["oof"] / z["oof"].std() for z in zz)
        return g / g.std()

    current = sum(w * zs(f) for f, w in CURRENT)
    files = args.only or sorted(os.path.basename(p) for p in glob.glob(os.path.join(CV, "oof_*.npz")))
    names, C = [], []
    for f in files:
        z = np.load(os.path.join(CV, f))
        if len(z["sid"]) != len(sid) or not (np.array_equal(z["sid"], sid) and np.array_equal(z["step"], step)):
            continue
        if np.isnan(z["oof"]).any():
            continue
        if args.smoke and f[4:-4] not in SMOKE:
            continue
        names.append(f[4:-4])
        C.append((z["oof"] / z["oof"].std()).astype(np.float32))
    K = len(names)
    C = np.stack(C)
    print(f"{K} candidate files (full 5-fold layout, no NaN); {len(y):,} rows; loaded in {time.time() - t0:.0f}s")

    from sklearn.model_selection import StratifiedKFold   # folds exactly as cv.py / blend.py
    from sbrt.data import load_train
    tau = load_train()[1]
    fold_of = np.zeros(len(tau), int)
    for f, (_, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(np.arange(len(tau)),
                                                                                       (tau >= 0).astype(int))):
        fold_of[va] = f
    fr = fold_of[sid]

    allr = Rows(np.arange(len(y)), step, y)
    fast = allr.score(current[allr.o].astype(np.float64))
    slow = ts_auc(current, y, step)
    print(f"current blend TS-AUC {slow:.6f} (sbrt.metric) vs {fast:.6f} (fast metric)")
    assert abs(fast - slow) < 1e-9
    global Mixer
    use_gpu = args.device == "cuda"
    if args.device == "auto":
        try:
            import torch
            use_gpu = torch.cuda.is_available()
        except ImportError:
            use_gpu = False
    if use_gpu:
        Mixer = lambda r, C_, cur_: GpuMixer(r, C_, cur_, args.batch)   # noqa: E731
        g = Mixer(allr, C, current)
        ws = np.array([0.0, 0.3, 0.7, 1.0])
        sg = g.mix(g.cur, [0, K - 1], ws)
        cm = CpuMixer(allr, C, current)
        sc = cm.mix(cm.cur, [0, K - 1], ws)
        print(f"GPU metric vs CPU metric on 8 blends: max |diff| {np.abs(sg - sc).max():.1e}")
        assert np.abs(sg - sc).max() < 1e-6
        del g, cm
    else:
        Mixer = CpuMixer
    print(f"metric backend: {'GPU (batched)' if use_gpu else 'CPU (numba)'}")
    single = np.array([allr.score(C[i][allr.o].astype(np.float64)) for i in range(K)])
    print("\nsingle-file TS-AUC (all rows):")
    for i in np.argsort(-single):
        print(f"  {single[i]:.4f}  {names[i]}")

    def search(rows, label, show_pairs=False):
        """Singles -> every pair -> beam over larger blends; returns the peak (score, weights over files, current wt)."""
        mx = Mixer(rows, C, current)

        def onehot(*iw):
            wv = np.zeros(K)
            for i, w in iw:
                wv[i] += w
            return wv

        print(f"\n[{label}]", flush=True)
        if args.start == "scratch":
            sc1 = mx.mix(mx.file(0), range(K), [1.0])[:, 0]
            peak = (sc1.max(), None, onehot((int(np.argmax(sc1)), 1.0)), 0.0)
            print(f"  size 1 (single files): best {peak[0]:.4f}  {describe(peak[2], 0.0)}", flush=True)
            # size 2: EVERY pair of files, weight on the second one 0.1-0.9
            pairs = []
            for i in range(K - 1):
                js = list(range(i + 1, K))
                sc = mx.mix(mx.file(i), js, PGRID)
                for k, j in enumerate(js):
                    pairs.append((sc[k].max(), i, j, PGRID[int(np.argmax(sc[k]))]))
            pairs.sort(key=lambda t: -t[0])
            beam = [(sc, mx.vec(mx.file(i), j, w), onehot((i, 1 - w), (j, w)), 0.0)
                    for sc, i, j, w in pairs[:args.beam]]
            print(f"  size 2 (all {len(pairs):,} pairs): best {beam[0][0]:.4f} ({beam[0][0] - peak[0]:+.4f} vs best "
                  f"single)  {describe(beam[0][2], 0.0)}  [{time.time() - t0:.0f}s]", flush=True)
            if show_pairs:
                print_pairs(pairs, sc1)
            if beam[0][0] - peak[0] < args.min_gain:
                print(f"  stop: no pair beats the best single by {args.min_gain}")
                return peak[0], peak[2], peak[3]
            peak = beam[0]
        else:
            beam = [(mx.score(mx.cur), mx.cur, np.zeros(K), 1.0)]
            peak = beam[0]
            print(f"  start: current blend {peak[0]:.4f}", flush=True)
        for rnd in range(args.max_add):
            cand = []
            for sc_b, s, wv, wc in beam:
                sc = mx.mix(s, range(K), WGRID)
                for c in range(K):
                    for k, w in enumerate(WGRID):
                        cand.append((sc[c, k], s, wv, wc, c, w))
            cand.sort(key=lambda t: -t[0])
            new, seen = [], set()
            for sc, s, wv, wc, c, w in cand:
                wv2 = (1 - w) * wv
                wv2[c] += w
                key = (round(sc, 7), tuple(np.flatnonzero(wv2 > 1e-9)))
                if key in seen:
                    continue
                seen.add(key)
                new.append((sc, mx.vec(s, c, w), wv2, (1 - w) * wc))
                if len(new) == args.beam:
                    break
            gain = new[0][0] - peak[0]
            nf = int((new[0][2] > 1e-9).sum()) + int(new[0][3] > 1e-9)
            print(f"  addition {rnd + 1} (blend of {nf} files): best {new[0][0]:.4f} ({gain:+.4f} vs peak)  "
                  f"{describe(new[0][2], new[0][3])}  [{time.time() - t0:.0f}s]", flush=True)
            if gain < args.min_gain:
                print(f"  stop: gain < {args.min_gain}; peak {peak[0]:.4f}")
                break
            beam = new
            peak = new[0]
        return peak[0], peak[2], peak[3]

    def print_pairs(pairs, sc1):
        print("  top 40 pairs (in-sample):")
        for bs, i, j, bw in pairs[:40]:
            print(f"    {bs:.4f}  {names[i]} {1 - bw:.1f} + {names[j]} {bw:.1f}   (singles {sc1[i]:.4f} / {sc1[j]:.4f})")
        best_partner = {}
        for bs, i, j, bw in pairs:
            for a, b in ((i, j), (j, i)):
                if bs > best_partner.get(a, (-1,))[0]:
                    best_partner[a] = (bs, b)
        print("  best partner of every file (pair score, gain over the better single):")
        for i in np.argsort(-sc1):
            bs, b = best_partner[i]
            print(f"    {names[i]:45s} {sc1[i]:.4f} -> with {names[b]:45s} {bs:.4f} ({bs - max(sc1[i], sc1[b]):+.4f})")

    def describe(wv, wc):
        parts = ([f"CURRENT {wc:.2f}"] if wc > 1e-9 else []) + \
                [f"{names[i]} {wv[i]:.2f}" for i in np.argsort(-wv) if wv[i] > 1e-9]
        return " + ".join(parts)

    def blend_of(wv, wc, idx):
        return C[:, idx].astype(np.float64).T @ wv + wc * current[idx]

    gains, gains_single, rows_out = [], [], []
    for f in range(1 if args.smoke else 5):
        tr = Rows(np.flatnonzero(fr != f), step, y)
        va = Rows(np.flatnonzero(fr == f), step, y)
        sc_tr, wv, wc = search(tr, f"outer fold {f}: search on the other 4 folds", show_pairs=args.smoke)
        best1 = int(np.argmax([tr.score(C[i][tr.o].astype(np.float64)) for i in range(K)]))
        cur = va.score(current[va.o].astype(np.float64))
        found = va.score(blend_of(wv, wc, va.o))
        one = va.score(C[best1][va.o].astype(np.float64))
        gains.append(found - cur)
        gains_single.append(found - one)
        rows_out.append((cur, found, one, describe(wv, wc)))
        print(f"  HELD-OUT fold {f}: current blend {cur:.4f} | searched blend {found:.4f} ({found - cur:+.4f}) | "
              f"best single file {one:.4f}", flush=True)

    g = np.array(gains)
    print("\n" + "=" * 110)
    print(f"NESTED (honest) gain of the {args.start} search vs the CURRENT blend, per held-out fold: "
          + " ".join(f"{x:+.4f}" for x in g))
    if len(g) > 1:
        se = g.std(ddof=1) / np.sqrt(len(g))
        print(f"  mean {g.mean():+.4f} +- {se:.4f} (SE), {int((g > 0).sum())}/{len(g)} folds better;  current "
              f"{np.mean([r[0] for r in rows_out]):.4f} -> searched {np.mean([r[1] for r in rows_out]):.4f} "
              f"(mean of fold scores); searched vs best single file {np.mean(gains_single):+.4f}")
        ok = g.mean() >= 0.0005 and (g > 0).sum() >= 4 and g.mean() > 2 * se
        print(f"  RULE (>= +0.0005, >= 4/5 folds, > 2 SE): {'PASS' if ok else 'FAIL'}")
    print("  blend found per outer fold (stability):")
    for f, r in enumerate(rows_out):
        print(f"    fold {f}: {r[3]}")

    if not args.smoke:
        sc, wv, wc = search(allr, "ALL rows: the blend to ship IF the rule passes (in-sample, optimistic)",
                            show_pairs=True)
        print(f"  in-sample: current {fast:.4f} -> found {sc:.4f};  weights on sd-scaled scores:")
        for i in np.argsort(-wv):
            if wv[i] > 1e-9:
                print(f"    {wv[i]:.3f}  {names[i]}")
        if wc > 1e-9:
            print(f"    {wc:.3f}  CURRENT blend")
    print(f"\ndone in {time.time() - t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
