"""Export a sample of training series for the signal viewer (viz/viewer.html).

Writes viz/data.js with:
  - N sampled series (half with a break, half without; hard cases over-sampled),
    values as float16 base64, plus history length, tau_index and per-series stats
  - before/after ratios around the break (+-200 points) for break series
  - the best model's out-of-fold score per online step, as a percentile across all
    series alive at that step (what TS-AUC actually ranks)
  - an event study over ALL break series: per-point statistics aligned on tau,
    against no-break series aligned at a pseudo-tau drawn from the same distribution

Usage (about 30-60 s):
    python viz/export_viz.py            # 400 series
    python viz/export_viz.py --n 800    # more series, bigger file
"""
import argparse
import base64
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "Data")
OOF = os.path.join(ROOT, "cv_out", "oof_aug2_df_bo_nbm_bs24_ep25.npz")
OUT = os.path.join(ROOT, "viz", "data.js")

EV_LO, EV_HI = -300, 300   # event-study window (offsets from tau)
RATIO_W = 200              # before/after window for break-type ratios


def ac1(x):
    x = x - x.mean()
    d = (x * x).sum()
    return float((x[1:] * x[:-1]).sum() / d) if d > 0 else 0.0


def kurt(x):
    x = x - x.mean()
    v = (x * x).mean()
    return float((x ** 4).mean() / v ** 2 - 3) if v > 0 else 0.0


def seg_stats(x):
    d = np.diff(x)
    return {
        "var": float(x.var()),
        "dvar": float(d.var() / 2),
        "ac1": ac1(x),
        "kurt": kurt(x),
        "tail": float((np.abs(x - x.mean()) > 3 * max(x.std(), 1e-9)).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    print("reading parquet ...")
    X = pd.read_parquet(os.path.join(DATA, "X_train.parquet")).reset_index()
    X = X.sort_values(["id", "time"])
    yi = pd.read_parquet(os.path.join(DATA, "y_train_index.parquet"))
    ids = X["id"].to_numpy()
    vals = X["value"].to_numpy(np.float32)
    per = X["period"].to_numpy()
    starts = np.r_[0, np.flatnonzero(np.diff(ids)) + 1]
    ends = np.r_[starts[1:], len(ids)]
    sid_list = ids[starts]

    series = {}
    for s, e, i in zip(starts, ends, sid_list):
        v = vals[s:e]
        h = int((per[s:e] == 1).sum())
        series[int(i)] = (v, h)
    tau_idx = yi["tau_index"].reindex(sid_list).to_numpy()

    # ---- per-series stats (history) for stratified sampling ---------------------------
    print("per-series stats ...")
    meta = []
    for i, ti in zip(sid_list, tau_idx):
        v, h = series[int(i)]
        hs = v[:h]
        meta.append((int(i), h, len(v) - h, int(ti), ac1(hs), kurt(hs)))
    meta = pd.DataFrame(meta, columns=["id", "hlen", "olen", "tau", "ac1", "hkurt"])
    meta["brk"] = meta.tau >= 0
    # pseudo break for no-break series: same relative position distribution as real breaks,
    # so both groups pass the same selection rule (otherwise online length leaks the label)
    rel_pool = (meta.tau / meta.olen)[meta.brk].to_numpy()
    rel = rng.choice(rel_pool, len(meta))
    meta["ptau"] = np.where(meta.brk, meta.tau, np.floor(rel * meta.olen).astype(int))

    def pick(df, k):
        hard = df[(df.ac1.abs() > 0.3) | (df.hkurt > 3)]
        easy = df.drop(hard.index)
        kh = min(len(hard), int(round(k * 0.4)))
        out = list(rng.choice(hard.id.to_numpy(), kh, replace=False))
        out += list(rng.choice(easy.id.to_numpy(), k - kh, replace=False))
        return out

    # require at least 30 online points after the (pseudo) break, identically for both groups
    ok = meta.olen - meta.ptau >= 30
    bdf = meta[meta.brk & ok]
    ndf = meta[~meta.brk & ok]
    chosen_b = pick(bdf, a.n // 2)
    # match each break pick to the unused no-break series with the nearest online length
    # (same hard/easy stratum), so neither length nor hardness hints at the label
    pool = ndf.assign(hard=(ndf.ac1.abs() > 0.3) | (ndf.hkurt > 3)).set_index("id")
    used, chosen_n = set(), []
    for i in chosen_b:
        r = meta.loc[meta.id == i].iloc[0]
        hard = abs(r.ac1) > 0.3 or r.hkurt > 3
        c = pool[(pool.hard == hard) & ~pool.index.isin(used)]
        j = int((c.olen - r.olen).abs().idxmin())
        used.add(j)
        chosen_n.append(j)
    chosen = chosen_b + chosen_n
    rng.shuffle(chosen)
    cm = meta.set_index("id").loc[chosen]
    print(f"  median online length: break {cm[cm.brk].olen.median():.0f}, "
          f"no break {cm[~cm.brk].olen.median():.0f}; median history: "
          f"{cm[cm.brk].hlen.median():.0f} vs {cm[~cm.brk].hlen.median():.0f}")

    # ---- model OOF score as per-step percentile ----------------------------------------
    oof_by_id = {}
    if os.path.exists(OOF):
        print("OOF scores ...")
        z = np.load(OOF)
        df = pd.DataFrame({"sid": z["sid"], "step": z["step"], "s": z["oof"], "y": z["y"]})
        # sanity: label must equal step >= tau_index for the id mapping to be right
        t = pd.Series(tau_idx, index=sid_list).reindex(df.sid).to_numpy()
        ok = ((df.step.to_numpy() >= t) & (t >= 0)).astype(int) == df.y.to_numpy()
        print(f"  OOF id mapping agreement: {ok.mean():.4f}")
        if ok.mean() > 0.999:
            df["pct"] = df.groupby("step").s.rank(pct=True)
            sel = df[df.sid.isin(chosen)]
            for i, g in sel.groupby("sid"):
                oof_by_id[int(i)] = [g.step.tolist(), np.round(g.pct.to_numpy(), 3).tolist()]
        else:
            print("  mapping disagrees -> OOF overlay skipped")

    # ---- pack chosen series --------------------------------------------------------------
    print("packing ...")
    out_series = []
    blobs = []
    off = 0
    m = meta.set_index("id")
    for i in chosen:
        v, h = series[i]
        r = m.loc[i]
        rec = {"id": i, "h": h, "n": len(v), "tau": int(r.tau), "off": off,
               "hac1": round(float(r.ac1), 3), "hkurt": round(float(r.hkurt), 2)}
        if r.tau >= 0:
            b = h + int(r.tau)
            pre = v[max(0, b - RATIO_W):b].astype(np.float64)
            post = v[b:b + RATIO_W].astype(np.float64)
            sp, sq = seg_stats(pre), seg_stats(post)
            rec["ratio"] = {
                "var": round(np.log(sq["var"] / sp["var"]), 3),
                "dvar": round(np.log(sq["dvar"] / sp["dvar"]), 3),
                "ac1": round(sq["ac1"] - sp["ac1"], 3),
                "kurt": round(sq["kurt"] - sp["kurt"], 2),
                "mean": round(float(post.mean() - pre.mean()), 3),
                "npost": len(post),
            }
        if i in oof_by_id:
            rec["oof"] = oof_by_id[i]
        out_series.append(rec)
        blobs.append(v.astype(np.float16).tobytes())
        off += len(v)
    blob = base64.b64encode(b"".join(blobs)).decode()

    # ---- event study over ALL break series -----------------------------------------------
    print("event study ...")
    L = EV_HI - EV_LO
    CLIP = 25.0  # clip squared terms at |x| = 5 so a few exploding series cannot dominate

    def accum(pairs):
        mats = {k: np.full((len(pairs), L), np.nan, np.float32) for k in ("x2", "d2", "lag", "absx")}
        for r, (i, b) in enumerate(pairs):
            v, _ = series[i]
            v = v.astype(np.float64)
            lo, hi = max(1, b + EV_LO), min(len(v), b + EV_HI)
            if hi <= lo:
                continue
            x = v[lo:hi]
            xp = v[lo - 1:hi - 1]
            sl = slice(lo - (b + EV_LO), hi - (b + EV_LO))
            mats["x2"][r, sl] = x * x
            mats["d2"][r, sl] = (x - xp) ** 2 / 2
            mats["lag"][r, sl] = x * xp
            mats["absx"][r, sl] = np.abs(x)
        cnt = np.isfinite(mats["x2"]).sum(0)
        rnd = lambda a: np.round(a, 4).tolist()
        return {
            "x2": rnd(np.nanmean(np.minimum(mats["x2"], CLIP), 0)),
            "x2med": rnd(np.nanmedian(mats["x2"], 0)),
            "d2": rnd(np.nanmean(np.minimum(mats["d2"], CLIP), 0)),
            "d2med": rnd(np.nanmedian(mats["d2"], 0)),
            "lag": rnd(np.nanmean(np.clip(mats["lag"], -CLIP, CLIP), 0)),
            "tail": rnd(np.nanmean(mats["absx"] > 3, 0, where=np.isfinite(mats["absx"]))),
            "absx": rnd(np.nanmean(np.minimum(mats["absx"], 5), 0)),
            "count": cnt.astype(int).tolist(),
        }

    brk_rows = meta[meta.brk]
    ev_b = accum([(int(i), int(h + t)) for i, h, t in zip(brk_rows.id, brk_rows.hlen, brk_rows.tau)])
    nb_rows = meta[~meta.brk]
    ev_n = accum([(int(i), int(h + t)) for i, h, t in zip(nb_rows.id, nb_rows.hlen, nb_rows.ptau)])

    # ---- per-series before/after change at the break (or pseudo-break), all series ----
    # Signed averages above cancel two-sided changes (some series get calmer, some wilder);
    # these distributions show the spread, and the AUC says how separable each one is.
    print("before/after distributions ...")
    rows = []
    for i, h, o, t, brk in zip(meta.id, meta.hlen, meta.olen, meta.ptau, meta.brk):
        if o - t < 30:
            continue
        v, _ = series[int(i)]
        b = h + int(t)
        pre = v[max(0, b - RATIO_W):b].astype(np.float64)
        post = v[b:b + RATIO_W].astype(np.float64)
        sp, sq = seg_stats(pre), seg_stats(post)
        rows.append((bool(brk), np.log(sq["var"] / sp["var"]), np.log(sq["dvar"] / sp["dvar"]),
                     sq["ac1"] - sp["ac1"], sq["kurt"] - sp["kurt"], post.mean() - pre.mean()))
    D = pd.DataFrame(rows, columns=["brk", "var", "dvar", "ac1", "kurt", "mean"])

    def auc(pos, neg):
        r = pd.Series(np.r_[pos, neg]).rank().to_numpy()
        return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))

    dists = {}
    for k, (lo, hi) in {"var": (-2, 2), "dvar": (-2, 2), "ac1": (-0.6, 0.6),
                        "kurt": (-6, 6), "mean": (-1, 1)}.items():
        edges = np.linspace(lo, hi, 61)
        hb = np.histogram(np.clip(D[D.brk][k], lo, hi), edges)[0] / D.brk.sum()
        hn = np.histogram(np.clip(D[~D.brk][k], lo, hi), edges)[0] / (~D.brk).sum()
        dists[k] = {"x": np.round(edges[:-1], 4).tolist(), "brk": np.round(hb, 5).tolist(),
                    "nobrk": np.round(hn, 5).tolist(),
                    "auc_abs": round(auc(D[D.brk][k].abs(), D[~D.brk][k].abs()), 3),
                    "auc_signed": round(auc(D[D.brk][k], D[~D.brk][k]), 3),
                    "up_brk": round(float((D[D.brk][k] > 0).mean()), 3),
                    "up_nobrk": round(float((D[~D.brk][k] > 0).mean()), 3)}
        print(f"  {k:5s} AUC |stat| {dists[k]['auc_abs']:.3f}  signed {dists[k]['auc_signed']:.3f}")

    payload = {
        "series": out_series,
        "event": {"lo": EV_LO, "hi": EV_HI, "brk": ev_b, "nobrk": ev_n,
                  "n_brk": int(len(brk_rows)), "n_nobrk": int(len(nb_rows))},
        "dists": dists, "dist_n": [int(D.brk.sum()), int((~D.brk).sum())], "ratio_w": RATIO_W,
        "has_oof": bool(oof_by_id),
    }
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("window.VIZ_META = ")
        json.dump(payload, f, separators=(",", ":"), default=lambda o: o.item())
        f.write(";\nwindow.VIZ_BLOB = \"" + blob + "\";\n")
    print(f"wrote {OUT}  ({os.path.getsize(OUT) / 1e6:.1f} MB, {len(out_series)} series)")


if __name__ == "__main__":
    main()
