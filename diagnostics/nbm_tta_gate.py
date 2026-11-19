"""NBM test-time adaptation (TTA) gate: fine-tune the shared NBM on each series' OWN history before streaming its
online points (owner's idea 2026-10-15, HANDOFF section 40).

Why: the NBM (sbrt/nbm.py) is ONE generating equation x_t = m(h_t) + s(h_t) eps_t shared by all series (it adapts to a
series only through its hidden state and 6 history descriptors); the per-series equations we fit (AR(10), GARCH) are
simple. A per-series rich equation would overfit 1,000-5,000 points, but a short, shrunk fine-tune of the shared one
may fit each series' own dynamics better, i.e. cleaner errors on no-break series, whose wandering online variance is
the main error source (section 15.2).

Arms (each on the SAME pretrained NBM, bs24_ep25 = the adopted config, trained on histories 0-4499):
  base     no adaptation; whitening reference = all in-history errors from row 51 (= the NBM sweep gate)
  baseR    no adaptation; reference = only the LAST R history rows (R = max(300, 30% of the history)): the control
           for the reference change below
  head_s   per-series fine-tune of the output layer (64 -> 2: location and log scale, 130 parameters) on history rows
  head_w   before the last R; L2 shrinkage to the pretrained layer (strong / weak); Adam; reference = the last R rows,
           which the adaptation never saw (in-sample errors after a fine-tune are too small, so an in-sample reference
           would make every no-break online stream look surprising)
  full     (optional, --arms ...,full; slow) all 30k parameters fine-tuned per series the same way
The GRU trunk is frozen in the head arms, so the hidden states are unchanged; only how they map to location / scale
adapts. Production cost of a head arm: about 50-100 small gradient steps on 130 parameters per series at inference
(the NBM history pass already exists), cheap in numpy.

Scoring = the round-1/2/3 NBM gate: prior refitted per arm on the fit series' post-break streams (0..n_fit-1),
single-feature TS-AUC of nb_logodds on held-out series 5000-6999 (all, by step range, variance-unchanged slice),
nb_lvr, nb_lsd. Extra diagnostics per arm: held-out history NLL (last R rows: does the adaptation generalise?),
no-break calibration (variance of the whitened online stream on no-break eval series, should be about 1),
and the PAIRED difference of nb_logodds vs base and vs baseR with a series-bootstrap SE.

Decision rule (written before any run): a TTA arm passes only if, on BOTH pretraining seeds, its nb_logodds beats
the better of base / baseR by >= +0.003 and > 2 bootstrap SE, its held-out NLL improves, and the no-break whitened
variance stays within 1.00 +- 0.05. A pass then needs a numpy port of the adaptation inside NbmStream and a paired
cv.py --nbm run (usual rule) before anything ships.

    python diagnostics/nbm_tta_gate.py                         # full run (GPU), seeds 0,1, arms base,baseR,head_s,head_w
    python diagnostics/nbm_tta_gate.py --arms base,baseR,head_s,head_w,full --n-full 1000     # + the slow full arm
    python diagnostics/nbm_tta_gate.py --n-train 300 --n-fit 300 --n-eval 200 --epochs 1 --seeds 0 --out <txt>  # smoke
"""
import argparse
import copy
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sbrt import bocpd as BO      # noqa: E402
from sbrt import evidence as EV   # noqa: E402
from sbrt import nbm as NB        # noqa: E402

CFG = dict(hidden=64, lr=2e-3, epochs=25, batch=24, wd=1e-4, loss_from=NB.LOSS_FROM)   # adopted bs24_ep25
ARMS = {   # name -> (adapt, l2 shrinkage, steps, lr, reference)
    "base": (None, 0.0, 0, 0.0, "all"),
    "baseR": (None, 0.0, 0, 0.0, "last"),
    "head_s": ("head", 1.0, 100, 1e-2, "last"),
    "head_w": ("head", 0.05, 100, 1e-2, "last"),
    "full": ("full", 1.0, 30, 1e-3, "last"),
}


def r_len(nh):
    return max(300, int(0.3 * nh))


class Log:
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")

    def __call__(self, s=""):
        print(s, flush=True)
        self.f.write(s + "\n")
        self.f.flush()


def trunk(net, x):
    """Hidden features feeding the NBM's last layer (B, L, H) and the pretrained outputs."""
    a = net.inp(x)
    o, _ = net.gru(a)
    z = net.out[1](net.out[0](o))
    return z


def batches_of(built, size):
    order = np.argsort([len(b[1]) for b in built])
    return [order[k:k + size] for k in range(0, len(order), size)]


def masks(built, b, L, dev, torch):
    """Targets (row p predicts e[p+1]), adaptation mask (targets in [LOSS_FROM+1, nh-R)) and held-out mask
    (targets in [nh-R, nh)), padded to L."""
    Y = np.zeros((len(b), L), np.float32)
    Ma = np.zeros((len(b), L), np.float32)
    Mh = np.zeros((len(b), L), np.float32)
    for r, i in enumerate(b):
        X, e, nh = built[i]
        n = len(e) - 1
        Y[r, :n] = e[1:]
        R = r_len(nh)
        q0, q1 = NB.LOSS_FROM + 1, nh - R          # target index range for adaptation
        if q1 > q0:
            Ma[r, q0 - 1:q1 - 1] = 1.0
        Mh[r, max(nh - R, 1) - 1:nh - 1] = 1.0
    return (torch.as_tensor(Y, device=dev), torch.as_tensor(Ma, device=dev), torch.as_tensor(Mh, device=dev))


def head_tta(torch, z, Y, Ma, W0, b0, nu, lam, steps, lr):
    """Per-series fine-tune of the last layer, all series of the batch in parallel (independent parameters)."""
    B = z.shape[0]
    dW = torch.zeros((B,) + tuple(W0.shape), device=z.device, requires_grad=True)
    db = torch.zeros((B,) + tuple(b0.shape), device=z.device, requires_grad=True)
    opt = torch.optim.Adam([dW, db], lr=lr)
    cnt = Ma.sum(1).clamp(min=1.0)
    for _ in range(steps):
        r = torch.einsum("blh,bkh->blk", z, W0[None] + dW) + (b0[None] + db)[:, None, :]
        nll = NB.t_nll(torch, Y, r[..., 0], r[..., 1].clamp(-4, 4), nu)
        loss = ((nll * Ma).sum(1) / cnt).sum() + 0.5 * lam * ((dW ** 2).sum((1, 2)) + (db ** 2).sum(1)).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        r = torch.einsum("blh,bkh->blk", z, W0[None] + dW) + (b0[None] + db)[:, None, :]
        return r[..., 0], r[..., 1].clamp(-4, 4)


def full_tta(torch, net, x, Y, Ma, lam, steps, lr):
    """Per-series fine-tune of all parameters (one series at a time; x is (1, L, CIN))."""
    m_net = copy.deepcopy(net)
    m_net.train()
    p0 = [p.detach().clone() for p in m_net.parameters()]
    opt = torch.optim.Adam(m_net.parameters(), lr=lr)
    cnt = Ma.sum().clamp(min=1.0)
    for _ in range(steps):
        m, ls = m_net(x)
        loss = (NB.t_nll(torch, Y, m, ls, m_net.nu()) * Ma).sum() / cnt
        loss = loss + 0.5 * lam * sum(((p - q) ** 2).sum() for p, q in zip(m_net.parameters(), p0))
        opt.zero_grad()
        loss.backward()
        opt.step()
    m_net.eval()
    with torch.no_grad():
        m, ls = m_net(x)
    return m, ls, m_net.nu()


def stream(e, nh, m, ls, ref_kind):
    """NbmStream's whitened online stream and scale drift from one series' predictions (m[p], ls[p] predict
    e[p+1]); returns the cumulative sums used by NB.features_from_sums and the whitened online values."""
    mp = np.r_[0.0, m[:len(e) - 1]]
    sp = np.r_[0.0, ls[:len(e) - 1]]
    v = (e - mp) * np.exp(-sp)
    lo = NB.LOSS_FROM + 1 if ref_kind == "all" else max(nh - r_len(nh), NB.LOSS_FROM + 1)
    ref = np.sort(v[lo:nh])
    w = np.clip(np.array([EV._whiten(float(q), ref) for q in v[nh:]]), -8, 8)
    d = sp[nh:] - sp[nh - 1]
    return (np.r_[0.0, np.cumsum(w)][None], np.r_[0.0, np.cumsum(w * w)][None], np.r_[0.0, np.cumsum(d)]), w


def run_arm(arm, net, built, torch, dev, n_full_limit=None):
    """Sums for every series in `built`, plus held-out NLL (base and arm) totals."""
    adapt, lam, steps, lr, ref_kind = ARMS[arm]
    out, ws = [None] * len(built), [None] * len(built)
    nll_h = np.zeros(2)
    cnt_h = 0.0
    W0, b0 = net.out[2].weight.detach(), net.out[2].bias.detach()
    nu = net.nu().detach()
    for b in batches_of(built, 64):
        x = NB._pad([built[i][0] for i in b], dev, torch)
        Y, Ma, Mh = masks(built, b, x.shape[1], dev, torch)
        with torch.no_grad():
            z = trunk(net, x)
            r0 = torch.einsum("blh,kh->blk", z, W0) + b0
            m0, ls0 = r0[..., 0], r0[..., 1].clamp(-4, 4)
        if adapt == "head":
            m, ls = head_tta(torch, z, Y, Ma, W0, b0, nu, lam, steps, lr)

        elif adapt == "full":
            ms, lss = [], []
            for r in range(len(b)):
                mr, lr_, nu_r = full_tta(torch, net, x[r:r + 1], Y[r:r + 1], Ma[r:r + 1], lam, steps, lr)
                with torch.no_grad():
                    nll_h[1] += float((NB.t_nll(torch, Y[r:r + 1], mr, lr_, nu_r) * Mh[r:r + 1]).sum())
                ms.append(mr)
                lss.append(lr_)
            m, ls = torch.cat(ms), torch.cat(lss)
        else:
            m, ls = m0, ls0
        with torch.no_grad():
            nll_h[0] += float((NB.t_nll(torch, Y, m0, ls0, nu) * Mh).sum())
            if adapt != "full":
                nll_h[1] += float((NB.t_nll(torch, Y, m, ls, nu) * Mh).sum())
            cnt_h += float(Mh.sum())
        m, ls = m.detach().cpu().numpy().astype(np.float64), ls.detach().cpu().numpy().astype(np.float64)
        del x, z, Y, Ma, Mh
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        for r, i in enumerate(b):
            _, e, nh = built[i]
            out[i], ws[i] = stream(e, nh, m[r], ls[r], ref_kind)
    return out, ws, nll_h / max(cnt_h, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", type=str, default="base,baseR,head_s,head_w")
    ap.add_argument("--seeds", type=str, default="0,1", help="pretraining seeds of the shared NBM")
    ap.add_argument("--n-train", type=int, default=4500)
    ap.add_argument("--n-fit", type=int, default=5000, help="series 0..N-1 for the prior fit")
    ap.add_argument("--n-eval", type=int, default=2000, help="held-out series from 5000 on")
    ap.add_argument("--n-full", type=int, default=0, help="full arm only: prior-fit series 0..N-1 (0 = n-fit; slow)")
    ap.add_argument("--epochs", type=int, default=0, help="override pretraining epochs (smoke tests)")
    ap.add_argument("--boot", type=int, default=50, help="series-bootstrap draws for the paired SE")
    ap.add_argument("--out", type=str, default=os.path.join("cv_out", "nbm_tta_gate.txt"))
    args = ap.parse_args()
    arms = [a for a in args.arms.split(",") if a]
    bad = [a for a in arms if a not in ARMS]
    if bad or "base" not in arms or "baseR" not in arms:
        sys.exit(f"arms must include base and baseR; unknown {bad}; known {list(ARMS)}")
    import torch
    from sbrt.data import load_train
    from sbrt.metric import ts_auc

    log = Log(args.out)
    t_all = time.time()
    S_all, tau_all = load_train()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ntr = args.n_train
    nva = max(1, min(500, ntr // 9))
    cfg = dict(CFG, epochs=args.epochs or CFG["epochs"])
    log(f"NBM TTA gate | pretrain histories 0-{ntr - nva - 1} (valid {ntr - nva}-{ntr - 1}) | prior fit 0-{args.n_fit - 1} | "
        f"eval {5000}-{5000 + args.n_eval - 1} | cfg {cfg} | arms {arms} | device {dev}")
    for a in arms:
        log(f"  {a:7s} adapt={ARMS[a][0]} l2={ARMS[a][1]} steps={ARMS[a][2]} lr={ARMS[a][3]} reference={ARMS[a][4]}")

    tr = NB.prep_histories([s[1] for s in S_all[:ntr - nva]])
    va = NB.prep_histories([s[1] for s in S_all[ntr - nva:ntr]])
    fit_b = [NB.build_seq(S_all[i][1], S_all[i][2]) for i in range(args.n_fit)]
    S, tau = S_all[5000:5000 + args.n_eval], tau_all[5000:5000 + args.n_eval]
    ev_b = [NB.build_seq(s[1], s[2]) for s in S]
    step = np.concatenate([np.arange(0, len(s[2]), 3) for s in S])
    sid = np.repeat(np.arange(len(S)), [(len(s[2]) + 2) // 3 for s in S])
    y = ((tau[sid] >= 0) & (step >= tau[sid])).astype(np.int8)
    vr = np.full(len(S), np.nan)
    for i, (s, t) in enumerate(zip(S, tau)):
        if t >= 0 and len(s[2]) - t >= 30:
            vr[i] = BO.whitened_online(s[1], s[2])[0][t:t + 300].var()
    in_slice = (tau[sid] < 0) | ((vr > 0.85) & (vr < 1.15))[sid]
    nobreak = np.flatnonzero(tau < 0)
    log(f"inputs prepared ({time.time() - t_all:.0f}s)")

    def auc(x, lo=0, hi=10 ** 6, mask=None):
        mk = (step >= lo) & (step < hi)
        if mask is not None:
            mk &= mask
        return ts_auc(x[mk], y[mk], step[mk])

    def boot_diff(a, b, rng):
        """Paired TS-AUC difference a - b and its series-bootstrap SE."""
        d0 = auc(a) - auc(b)
        starts = np.r_[0, np.cumsum([(len(s[2]) + 2) // 3 for s in S])]
        ds = []
        for _ in range(args.boot):
            pick = rng.integers(0, len(S), len(S))
            idx = np.concatenate([np.arange(starts[i], starts[i + 1]) for i in pick])
            # bootstrap rows keep their own step; duplicated series are separate rows (as in a resampled panel)
            ds.append(ts_auc(a[idx], y[idx], step[idx]) - ts_auc(b[idx], y[idx], step[idx]))
        return d0, float(np.std(ds, ddof=1))

    _, NBM_ = NB._torch()
    summary = {}
    for seed in [int(s) for s in args.seeds.split(",") if s]:
        t0 = time.time()
        state = NB.train_model(tr, va, epochs=cfg["epochs"], batch=cfg["batch"], lr=cfg["lr"], seed=seed, verbose=False,
                               hidden=cfg["hidden"], wd=cfg["wd"], loss_from=cfg["loss_from"], prepared=True)
        net = NBM_(cfg["hidden"]).to(dev)
        net.load_state_dict(state)
        net.eval()
        log(f"\nseed {seed}: shared NBM pretrained in {time.time() - t0:.0f}s")
        feats = {}
        for arm in arms:
            t1 = time.time()
            fb, eb = fit_b, ev_b
            if arm == "full" and args.n_full:
                fb = fit_b[:args.n_full]           # prior from fewer series; eval stays complete
            sf, _, _ = run_arm(arm, net, fb, torch, dev)
            prior, nfit = NB.fit_prior_w([np.diff(q[0][0]) for q in sf], tau_all[:len(fb)])
            se, ws, nll = run_arm(arm, net, eb, torch, dev)
            Fm = np.concatenate([NB.features_from_sums(q, prior, 3) for q in se])
            feats[arm] = Fm
            wnb = np.concatenate([ws[i] for i in nobreak])
            r = dict(nb_logodds=auc(Fm[:, 0]), lo_0_100=auc(Fm[:, 0], 0, 100), lo_100_300=auc(Fm[:, 0], 100, 300),
                     lo_300_700=auc(Fm[:, 0], 300, 700), lo_700p=auc(Fm[:, 0], 700), varsame=auc(Fm[:, 0], mask=in_slice),
                     nb_lvr=auc(Fm[:, 2]), nb_lsd=auc(Fm[:, 4]), nll_base=nll[0], nll_arm=nll[1],
                     nb_var=float(wnb.var()), nb_mean=float(wnb.mean()))
            summary.setdefault(arm, []).append(r)
            log(f"  {arm:7s} nb_logodds {r['nb_logodds']:.4f}  (0-100 {r['lo_0_100']:.4f}, 100-300 {r['lo_100_300']:.4f}, "
                f"300-700 {r['lo_300_700']:.4f}, 700+ {r['lo_700p']:.4f}, var-same {r['varsame']:.4f})  nb_lvr {r['nb_lvr']:.4f}  "
                f"nb_lsd {r['nb_lsd']:.4f} | held-out history NLL {r['nll_base']:.4f} -> {r['nll_arm']:.4f} | no-break "
                f"whitened mean {r['nb_mean']:+.3f} var {r['nb_var']:.3f} | prior from {nfit} breaks "
                f"(k0 {prior[0]:.0f}, a0 {prior[1]:.2f}, b0 {prior[2]:.2f}) | {time.time() - t1:.0f}s")
        rng = np.random.default_rng(seed)
        for arm in arms:
            if arm in ("base", "baseR"):
                continue
            for ref in ("base", "baseR"):
                d, s_ = boot_diff(feats[arm][:, 0], feats[ref][:, 0], rng)
                summary[arm][-1][f"d_{ref}"] = (d, s_)
                log(f"  paired {arm} - {ref}: {d:+.4f} +- {s_:.4f} (series bootstrap, {args.boot} draws)")
        d, s_ = boot_diff(feats["baseR"][:, 0], feats["base"][:, 0], rng)
        log(f"  paired baseR - base (reference-window effect alone): {d:+.4f} +- {s_:.4f}")
        del net
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    log("\nVERDICT (rule written before the run: on every seed, >= +0.003 and > 2 SE over the better of base/baseR, "
        "held-out NLL improves, no-break whitened variance in 1.00 +- 0.05)")
    for arm in arms:
        if arm in ("base", "baseR"):
            continue
        ok = True
        for k, r in enumerate(summary[arm]):
            better = "base" if summary["base"][k]["nb_logodds"] >= summary["baseR"][k]["nb_logodds"] else "baseR"
            d, s_ = r[f"d_{better}"]
            ok &= (d >= 0.003 and d > 2 * s_ and r["nll_arm"] < r["nll_base"] and abs(r["nb_var"] - 1.0) <= 0.05)
        log(f"  {arm:7s} {'PASS -> numpy port + paired cv.py --nbm run' if ok else 'FAIL'}")
    log(f"\ntotal {time.time() - t_all:.0f}s; log {args.out}")


if __name__ == "__main__":
    main()
