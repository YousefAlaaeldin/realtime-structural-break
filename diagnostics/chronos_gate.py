"""Gate for Chronos-2 as a surprise stream (signal only; platform checks are separate).

Every K online steps, Chronos-2 forecasts the next K points (21 quantiles) from the last CTX points of
history + online so far. Each new point's PIT under that forecast -> normal score w (linear tail extension beyond
the 1% / 99% quantiles). w goes through the same BOCPD posterior as sbrt/nbm.py (own empirical-Bayes prior, fitted
on series 0-1999), plus the forecast-spread drift log(IQR80_t / IQR80_0). Evaluated on held-out series 5000-6999
against bo_/vol_ and the NBM features.

    python diagnostics/chronos_gate.py [K] [CTX]
"""
import os
import sys
import time

import numpy as np
import torch
from scipy.special import ndtri

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sbrt import bocpd as BO      # noqa: E402
from sbrt import nbm as NB        # noqa: E402
from sbrt import volstate as VS   # noqa: E402

K = int(sys.argv[1]) if len(sys.argv) > 1 else 16
CTX = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
MODEL = os.path.join("data", "hf", "chronos-2")
NAMES = ["ch_logodds", "ch_tsb", "ch_lvr", "ch_mu", "ch_lsd", "ch_lsd_mean100"]


def pit_scores(x, q, levels):
    """Normal scores of values x (n,) under forecast quantiles q (n, Q) at increasing levels."""
    zl = ndtri(levels)
    out = np.empty(len(x))
    for i in range(len(x)):
        qi = np.maximum.accumulate(q[i])
        if x[i] <= qi[0]:
            sc = max((qi[1] - qi[0]) / (zl[1] - zl[0]), 1e-6)
            out[i] = zl[0] - (qi[0] - x[i]) / sc
        elif x[i] >= qi[-1]:
            sc = max((qi[-1] - qi[-2]) / (zl[-1] - zl[-2]), 1e-6)
            out[i] = zl[-1] + (x[i] - qi[-1]) / sc
        else:
            out[i] = np.interp(x[i], qi, zl)
    return np.clip(out, -8, 8)


def chronos_streams(pipe, items, batch=256):
    levels = np.array(pipe.quantiles)
    i10, i90 = int(np.argmin(np.abs(levels - 0.1))), int(np.argmin(np.abs(levels - 0.9)))
    jobs = []      # (series, checkpoint)
    for s, (h, o) in enumerate(items):
        for c in range(0, len(o), K):
            jobs.append((s, c))
    W = [np.zeros(len(o)) for _, o in items]
    D = [np.zeros(len(o)) for _, o in items]
    iqr0 = np.zeros(len(items))
    for b in range(0, len(jobs), batch):
        part = jobs[b:b + batch]
        ctxs = [torch.tensor(np.r_[items[s][0], items[s][1][:c]][-CTX:], dtype=torch.float32)[None] for s, c in part]
        preds = pipe.predict(ctxs, prediction_length=K, batch_size=batch)
        for (s, c), p in zip(part, preds):
            q = p[0].float().cpu().numpy().T          # (K, Q)
            o = items[s][1]
            m = min(K, len(o) - c)
            W[s][c:c + m] = pit_scores(o[c:c + m], q[:m], levels)
            iqr = np.log(np.maximum(q[:m, i90] - q[:m, i10], 1e-6))
            if c == 0:
                iqr0[s] = iqr[0]
            D[s][c:c + m] = iqr - iqr0[s]
    return W, D


def main():
    from joblib import Parallel, delayed
    from scipy.stats import rankdata

    from chronos import Chronos2Pipeline
    from sbrt.data import load_train
    from sbrt.metric import ts_auc

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = Chronos2Pipeline.from_pretrained(MODEL, device_map=dev, dtype=torch.float32)
    S_all, tau_all = load_train()
    t0 = time.time()
    fit_items = [(s[1], s[2]) for s in S_all[:2000]]
    Wf, _ = chronos_streams(pipe, fit_items)
    print(f"K {K} CTX {CTX} | Chronos on 2000 prior-fit series: {time.time() - t0:.0f}s")
    prior, nfit = NB.fit_prior_w(Wf, tau_all[:2000])
    nob = np.concatenate([w for w, t in zip(Wf, tau_all[:2000]) if t < 0])
    print(f"EB prior from {nfit} breaks: kappa0 {prior[0]:.0f}, alpha0 {prior[1]:.2f}, beta0 {prior[2]:.2f} "
          f"(mean post-break var {prior[2] / (prior[1] - 1):.2f}); no-break stream mean {nob.mean():+.3f} var {nob.var():.3f}")

    S, tau = S_all[5000:7000], tau_all[5000:7000]
    items = [(s[1], s[2]) for s in S]
    t1 = time.time()
    W, D = chronos_streams(pipe, items)
    print(f"Chronos on 2000 held-out series: {time.time() - t1:.0f}s")
    CH = np.concatenate(Parallel(n_jobs=8, backend="threading")(
        delayed(NB.nb_features)(w, d, prior, 3) for w, d in zip(W, D)))
    st, _ = NB.streams(items)
    nb_prior, _ = NB.fit_prior_w([q[1] for q in NB.streams([(s[1], s[2]) for s in S_all[:2000]])[0]], tau_all[:2000])
    NBF = np.concatenate([NB.nb_features(q[1], q[2], nb_prior, 3) for q in st])
    Bo = np.concatenate(Parallel(n_jobs=8, backend="threading")(delayed(BO.run_series_bocpd)(h, o, 3) for h, o in items))
    V = np.concatenate([VS.run_series_vol(h, o)[::3] for h, o in items])
    step = np.concatenate([np.arange(0, len(o), 3) for _, o in items])
    sid = np.repeat(np.arange(len(S)), [(len(o) + 2) // 3 for _, o in items])
    y = ((tau[sid] >= 0) & (step >= tau[sid])).astype(np.int8)
    vr = np.full(len(S), np.nan)
    for i, ((h, o), t) in enumerate(zip(items, tau)):
        if t >= 0 and len(o) - t >= 30:
            vr[i] = BO.whitened_online(h, o)[0][t:t + 300].var()
    same = (vr > 0.85) & (vr < 1.15)
    in_slice = (tau[sid] < 0) | same[sid]
    buckets = ((0, 100), (100, 300), (300, 700), (700, 10000))

    def row(nm, x, extra=""):
        r = [ts_auc(x, y, step)] + [ts_auc(x[(step >= lo) & (step < hi)], y[(step >= lo) & (step < hi)],
                                          step[(step >= lo) & (step < hi)]) for lo, hi in buckets]
        r.append(ts_auc(x[in_slice], y[in_slice], step[in_slice]))
        print(f"  {nm:20s} " + "  ".join(f"{v:7.4f}" for v in r) + extra)

    print("\nsingle-feature TS-AUC                all     0-100  100-300  300-700    700+   var-same")
    for c in (0, 5):
        row(BO.FEATURE_NAMES[c], Bo[:, c])
    row("vol_scanmax", V[:, VS.FEATURE_NAMES.index("vol_scanmax")])
    row("nb_logodds", NBF[:, 0])
    rng = np.random.default_rng(0)
    sel = rng.choice(len(y), min(len(y), 150000), replace=False)
    Z = np.column_stack([Bo[sel], V[sel], NBF[sel]])
    names = list(BO.FEATURE_NAMES) + list(VS.FEATURE_NAMES) + list(NB.FEATURE_NAMES)
    R = np.column_stack([rankdata(Z[:, j]) for j in range(Z.shape[1])])
    R = (R - R.mean(0)) / (R.std(0) + 1e-12)
    for c, nm in enumerate(NAMES):
        r = rankdata(CH[sel, c])
        r = (r - r.mean()) / (r.std() + 1e-12)
        cc = np.abs(R.T @ r) / len(r)
        j = int(np.argmax(cc))
        row(nm, CH[:, c], f"   {cc[j]:.2f} {names[j]}")
    np.savez(os.path.join("diagnostics", f"chronos_gate_K{K}_C{CTX}.npz"), CH=CH, NBF=NBF, step=step, y=y, sid=sid)


if __name__ == "__main__":
    main()
