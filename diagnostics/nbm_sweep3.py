"""NBM sweep, round 3: the DESIGN of the normal-behaviour model, not its training knobs (HANDOFF section 37).

Rounds 1-2 (diagnostics/nbm_sweep.py) showed the training knobs plateau: once the model gets enough updates, batch,
lr, width and epochs are interchangeable within seed noise (about +-0.004-0.006 on the gate). What was never varied:
the network (layers, cell type), the prediction target and the inputs. This script varies exactly those, from the
adopted configuration bs24_ep25 (GRU width 64, batch 24, lr 2e-3, 25 epochs, wd 1e-4, loss from 50), with
EVERY configuration run on 3 seeds (round 2 showed single-seed leaders fall back by about 0.006).

Designs (one change at a time from `ref`):
  ref      the adopted model, re-run here on 3 seeds (the comparison baseline; must land near 0.629-0.632)
  l2       2-layer GRU
  lstm     1-layer LSTM instead of the GRU
  tcn      causal dilated temporal conv net (8 residual blocks, dilations 1..128, receptive field 511 points)
           instead of the recurrent cell: finite memory, so it cannot "absorb" a break into a long-lived state
  k4       4-step-ahead target: the model predicts e_{t+4} from data up to t, and the online stream compares each
           residual with the forecast made 4 points earlier, so the 3 newest points (possibly post-break) have not
           yet moved the forecast it is judged against (the model adapts to a break more slowly)
  multi4   heads for e_{t+1} .. e_{t+4} trained jointly (loss averaged); the stream uses the 1-step head
           (multi-horizon targets as a regulariser of the representation)
  xin      5 extra input channels, computed identically in the history and online (so no-break streams stay
           calibrated): log EWMA of min(e^2, 25) at half-lives 5 / 20 / 100, the normal score of z's rank in the
           series' own history, and the roughness min(|z_t - z_{t-1}|, 8) - 1
Plus, from the same runs at no extra cost: `<name>_ens` = the 3 seeds' nb_logodds averaged (seed ensemble).

Scoring = the round-1/2 gate: pretrain on histories of series 0-4499 (validation 4500-4999), prior on 0-4999's breaks,
single-feature TS-AUC of nb_logodds on held-out series 5000-6999 (all, by step range, variance-unchanged slice), plus
nb_lvr and nb_lsd. Evaluation in torch float32 on the GPU (same stream as sbrt/nbm.py: whitened by the series'
in-history errors, BOCPD posterior with an EB prior). valid NLL is NOT comparable across k4 / multi4 / the rest.

Decision rule (written before any run): a design is a candidate only if its 3-seed mean nb_logodds beats ref's
3-seed mean by >= +0.003 AND by > 2 standard errors (SE from the two sets of seed values). A candidate then needs a
numba runtime port (only ref's GRU exists in sbrt/nbm.py) and a paired cv.py --nbm run before anything ships.
Combinations (e.g. k4 + xin) only after single changes pass.

    python diagnostics/nbm_sweep3.py --list
    python diagnostics/nbm_sweep3.py                     # all (resumable; results in diagnostics/nbm_sweep3.csv)
    python diagnostics/nbm_sweep3.py --only ref,k4       # some designs (all 3 seeds each)
    python diagnostics/nbm_sweep3.py --n-train 300 --n-fit 300 --n-eval 200 --epochs 1 --seeds 0 --out <csv>     # smoke test
"""
import argparse
import csv
import math
import os
import sys
import time

import numpy as np
from scipy.signal import lfilter
from scipy.special import ndtri

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sbrt import bocpd as BO      # noqa: E402
from sbrt import evidence as EV   # noqa: E402
from sbrt import nbm as NB        # noqa: E402

REF = dict(cell="gru", layers=1, horizon=1, multi=1, xin=0, hidden=64, lr=2e-3, epochs=25, batch=24, wd=1e-4,
           loss_from=50)
DESIGNS = {
    "ref": {},
    "l2": dict(layers=2),
    "lstm": dict(cell="lstm"),
    "tcn": dict(cell="tcn"),
    "k4": dict(horizon=4),
    "multi4": dict(multi=4),
    "xin": dict(xin=1),
}
SEEDS = (0, 1, 2)
N_XIN = 5
TCN_DIL = (1, 2, 4, 8, 16, 32, 64, 128)
FIELDS = ["name", "seed", "cell", "layers", "horizon", "multi", "xin", "params", "train_s", "valid_nll",
          "nb_logodds", "lo_0_100", "lo_100_300", "lo_300_700", "lo_700p", "lo_varsame", "nb_lvr", "nb_lsd"]


# ---------------------------------------------------------------------------------------------
# inputs
def extra_channels(hist, online, X):
    """The 5 `xin` channels for the rows of NB.build_seq (rows = time steps from AR on)."""
    z = np.r_[np.asarray(hist, float), np.asarray(online, float)]
    mu, sd = float(np.mean(hist)), float(np.std(hist)) + 1e-12
    z = (z - mu) / sd
    nh = len(hist)
    zr = z[NB.AR:]
    e2 = X[:, 2].astype(np.float64) + 1.0                     # min(e^2, 25)
    cols = []
    for hl in (5, 20, 100):
        a = 1.0 - 0.5 ** (1.0 / hl)                           # ew_t = (1 - a) ew_{t-1} + a x_t, ew_{-1} = 1
        ew = lfilter([a], [1.0, -(1.0 - a)], e2, zi=[1.0 - a])[0]
        cols.append(np.clip(np.log(np.maximum(ew, 1e-6)), -4, 4))
    srt = np.sort(z[:nh])
    rk = (np.searchsorted(srt, zr, side="right") + 0.5) / (nh + 1.0)
    cols.append(np.clip(ndtri(np.clip(rk, 1e-6, 1 - 1e-6)), -4, 4))
    cols.append(np.minimum(np.abs(np.diff(z)[NB.AR - 1:]), 8.0) - 1.0)
    return np.column_stack(cols).astype(np.float32)


def build(hist, online, xin):
    X, e, nh_e = NB.build_seq(hist, online)
    if xin:
        X = np.hstack([X, extra_channels(hist, online, X)])
    return X, e, nh_e


def prep(hists, cfg):
    """(inputs, targets (L, M)) for pretraining; targets[p, j] = e[p + horizon + j] (horizon 1 and multi M, or
    horizon K and M = 1)."""
    K, M = cfg["horizon"], cfg["multi"]
    out = []
    for h in hists:
        X, e, _ = build(h, np.zeros(0), cfg["xin"])
        span = K + M - 1
        L = len(e) - span
        Y = np.column_stack([e[K + j: K + j + L] for j in range(M)]).astype(np.float32)
        out.append((X[:L], Y))
    return out


# ---------------------------------------------------------------------------------------------
# networks
def make_net(torch, cfg):
    nn = torch.nn
    H, M = cfg["hidden"], cfg["multi"]
    cin = NB.CIN + (N_XIN if cfg["xin"] else 0)

    class Causal(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.d = d
            self.conv = nn.Conv1d(H, H, 3, dilation=d)

        def forward(self, x):                                    # x (B, H, L)
            y = self.conv(nn.functional.pad(x, (2 * self.d, 0)))
            return x + nn.functional.gelu(y)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Sequential(nn.Linear(cin, H), nn.GELU())
            if cfg["cell"] == "gru":
                self.rnn = nn.GRU(H, H, num_layers=cfg["layers"], batch_first=True)
            elif cfg["cell"] == "lstm":
                self.rnn = nn.LSTM(H, H, num_layers=cfg["layers"], batch_first=True)
            else:
                self.tcn = nn.Sequential(*[Causal(d) for d in TCN_DIL])
            self.out = nn.Sequential(nn.Linear(H, H), nn.GELU(), nn.Linear(H, 2 * M))
            self.lnu = nn.Parameter(torch.tensor(math.log(5.0 - 2.0)))

        def forward(self, x):
            a = self.inp(x)
            if cfg["cell"] == "tcn":
                o = self.tcn(a.transpose(1, 2)).transpose(1, 2)
            else:
                o, _ = self.rnn(a)
            r = self.out(o).view(x.shape[0], x.shape[1], M, 2)
            return r[..., 0], r[..., 1].clamp(-4, 4)            # (B, L, M) each

        def nu(self):
            return 2.0 + torch.exp(self.lnu).clamp(0.1, 60.0)

    return Net()


def train(tr, va, cfg, seed, dev, torch):
    """Same optimiser, schedule, masking and best-epoch rule as sbrt/nbm.train_model."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = make_net(torch, cfg).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    B = cfg["batch"]
    steps = cfg["epochs"] * ((len(tr) + B - 1) // B)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=steps, pct_start=0.1)
    lf = cfg["loss_from"]

    def batches(D, shuffle):
        order = np.argsort([len(d[1]) for d in D])
        bs = [order[i:i + B] for i in range(0, len(order), B)]
        if shuffle:
            rng.shuffle(bs)
        for b in bs:
            x = NB._pad([D[i][0] for i in b], dev, torch)
            y = NB._pad([D[i][1] for i in b], dev, torch)
            msk = torch.zeros(y.shape[:2], device=dev)
            for k, i in enumerate(b):
                msk[k, lf:len(D[i][1])] = 1.0
            yield x, y, msk
            del x, y, msk
            if dev.type == "cuda":
                torch.cuda.empty_cache()

    def loss_of(x, y, msk):
        m, ls = net(x)
        nll = NB.t_nll(torch, y, m, ls, net.nu()).mean(-1)      # average over the M horizons
        return (nll * msk).sum(), msk.sum()

    def evaluate():
        net.eval()
        tot, cnt = 0.0, 0.0
        with torch.no_grad():
            for x, y, msk in batches(va, False):
                a, c = loss_of(x, y, msk)
                tot += float(a)
                cnt += float(c)
        net.train()
        return tot / cnt

    best, best_state = 1e9, None
    for _ in range(cfg["epochs"]):
        for x, y, msk in batches(tr, True):
            a, c = loss_of(x, y, msk)
            loss = a / c
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
        v = evaluate()
        if v < best:
            best, best_state = v, {k: t.detach().clone() for k, t in net.state_dict().items()}
    net.load_state_dict(best_state)
    net.eval()
    return net, best


def stream_sums(net, built, cfg, torch, dev, batch=64):
    """The NBM stream (whitened prediction errors, log-scale drift since the history's end) as cumulative sums,
    exactly as sbrt/nbm.NbmStream but for any horizon: the residual e_q is judged against the forecast made at row
    q - K (K = horizon; the multi-head model uses its 1-step head)."""
    K = cfg["horizon"]
    out = [None] * len(built)
    order = np.argsort([len(b[1]) for b in built])
    with torch.no_grad():
        for k in range(0, len(order), batch):
            b = order[k:k + batch]
            x = NB._pad([built[i][0] for i in b], dev, torch)
            m, ls = net(x)
            m = m[..., 0].cpu().numpy().astype(np.float64)
            ls = ls[..., 0].cpu().numpy().astype(np.float64)
            del x
            if dev.type == "cuda":
                torch.cuda.empty_cache()
            for r, i in enumerate(b):
                X, e, nh = built[i]
                mp = np.r_[np.zeros(K), m[r, :len(e) - K]]
                sp = np.r_[np.zeros(K), ls[r, :len(e) - K]]
                v = (e - mp) * np.exp(-sp)
                ref = np.sort(v[NB.LOSS_FROM + K:nh])
                w = np.clip(np.array([EV._whiten(float(q), ref) for q in v[nh:]]), -8, 8)
                d = sp[nh:] - sp[nh - 1]
                out[i] = (np.r_[0.0, np.cumsum(w)][None], np.r_[0.0, np.cumsum(w * w)][None], np.r_[0.0, np.cumsum(d)])
    return out


# ---------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=str, default="")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--seeds", type=str, default=",".join(map(str, SEEDS)))
    ap.add_argument("--n-train", type=int, default=4500, help="pretraining histories (smoke tests: fewer)")
    ap.add_argument("--n-eval", type=int, default=2000, help="held-out evaluation series from 5000 on")
    ap.add_argument("--n-fit", type=int, default=5000, help="series 0..N-1 for the prior fit (smoke tests: fewer)")
    ap.add_argument("--epochs", type=int, default=0, help="override epochs (smoke tests)")
    ap.add_argument("--out", type=str, default=os.path.join("diagnostics", "nbm_sweep3.csv"))
    args = ap.parse_args()
    if args.list:
        for k, v in DESIGNS.items():
            print(f"  {k:8s} {dict(REF, **v)}")
        return
    import torch
    from sbrt.data import load_train
    from sbrt.metric import ts_auc

    seeds = [int(s) for s in args.seeds.split(",") if s]
    names = [n for n in (args.only.split(",") if args.only else DESIGNS) if n]
    bad = [n for n in names if n not in DESIGNS]
    if bad:
        sys.exit(f"unknown designs {bad}; see --list")
    done = set()
    if os.path.exists(args.out):
        with open(args.out, newline="") as fh:
            done = {(r["name"], int(r["seed"])) for r in csv.DictReader(fh)}
    todo = [(n, s) for n in names for s in seeds if (n, s) not in done]
    feat_dir = os.path.splitext(args.out)[0] + "_feats"
    os.makedirs(feat_dir, exist_ok=True)

    S_all, tau_all = load_train()
    ntr = args.n_train
    nva = max(1, min(500, ntr // 9))
    fit_idx = np.arange(0, args.n_fit)
    S, tau = S_all[5000:5000 + args.n_eval], tau_all[5000:5000 + args.n_eval]
    step = np.concatenate([np.arange(0, len(s[2]), 3) for s in S])
    sid = np.repeat(np.arange(len(S)), [(len(s[2]) + 2) // 3 for s in S])
    y = ((tau[sid] >= 0) & (step >= tau[sid])).astype(np.int8)
    vr = np.full(len(S), np.nan)
    for i, (s, t) in enumerate(zip(S, tau)):
        if t >= 0 and len(s[2]) - t >= 30:
            vr[i] = BO.whitened_online(s[1], s[2])[0][t:t + 300].var()
    in_slice = (tau[sid] < 0) | ((vr > 0.85) & (vr < 1.15))[sid]

    def auc(x, lo=0, hi=10 ** 6, mask=None):
        mk = (step >= lo) & (step < hi)
        if mask is not None:
            mk &= mask
        return ts_auc(x[mk], y[mk], step[mk])

    def metrics(Fm):
        return dict(nb_logodds=round(auc(Fm[:, 0]), 4), lo_0_100=round(auc(Fm[:, 0], 0, 100), 4),
                    lo_100_300=round(auc(Fm[:, 0], 100, 300), 4), lo_300_700=round(auc(Fm[:, 0], 300, 700), 4),
                    lo_700p=round(auc(Fm[:, 0], 700), 4), lo_varsame=round(auc(Fm[:, 0], mask=in_slice), 4),
                    nb_lvr=round(auc(Fm[:, 2]), 4), nb_lsd=round(auc(Fm[:, 4]), 4))

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"round-3 NBM design sweep on {dev}: {len(todo)} runs to do ({len(done)} done); pretraining "
          f"{ntr - nva} + {nva} histories, evaluation on {len(S)} series", flush=True)
    cache = {}
    new_file = not os.path.exists(args.out)
    for name, seed in todo:
        cfg = dict(REF, **DESIGNS[name])
        if args.epochs:
            cfg["epochs"] = args.epochs
        key = (cfg["xin"], cfg["horizon"], cfg["multi"])
        if key not in cache:                                    # inputs depend only on xin / targets on horizon+multi
            t0 = time.time()
            cache.clear()
            tr = prep([s[1] for s in S_all[:ntr - nva]], cfg)
            va = prep([s[1] for s in S_all[ntr - nva:ntr]], cfg)
            fit_b = [build(S_all[i][1], S_all[i][2], cfg["xin"]) for i in fit_idx]
            ev_b = [build(s[1], s[2], cfg["xin"]) for s in S]
            cache[key] = (tr, va, fit_b, ev_b)
            print(f"  inputs for xin={cfg['xin']} horizon={cfg['horizon']} multi={cfg['multi']} "
                  f"prepared in {time.time() - t0:.0f}s", flush=True)
        tr, va, fit_b, ev_b = cache[key]
        t1 = time.time()
        net, vnll = train(tr, va, cfg, seed, dev, torch)
        t_train = time.time() - t1
        params = sum(p.numel() for p in net.parameters())
        sf = stream_sums(net, fit_b, cfg, torch, dev)
        prior, _ = NB.fit_prior_w([np.diff(q[0][0]) for q in sf], tau_all[fit_idx])
        se = stream_sums(net, ev_b, cfg, torch, dev)
        Fm = np.concatenate([NB.features_from_sums(q, prior, 3) for q in se])
        np.save(os.path.join(feat_dir, f"{name}_s{seed}.npy"), Fm[:, 0].astype(np.float32))
        del net, sf, se
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        r = dict(name=name, seed=seed, cell=cfg["cell"], layers=cfg["layers"], horizon=cfg["horizon"],
                 multi=cfg["multi"], xin=cfg["xin"], params=params, train_s=round(t_train), valid_nll=round(vnll, 5),
                 **metrics(Fm))
        with open(args.out, "a", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=FIELDS)
            if new_file:
                wr.writeheader()
                new_file = False
            wr.writerow(r)
        print(f"{name:8s} seed {seed}  params {params:7,d}  train {t_train:4.0f}s  valid NLL {vnll:.4f}  "
              f"nb_logodds {r['nb_logodds']:.4f}  (100-300 {r['lo_100_300']:.4f}, 300-700 {r['lo_300_700']:.4f}, "
              f"700+ {r['lo_700p']:.4f}, var-same {r['lo_varsame']:.4f})  | {time.time() - t1:.0f}s", flush=True)

    # summary: per design, mean over seeds, paired-free comparison with ref, and the seed ensemble
    with open(args.out, newline="") as fh:
        rows = list(csv.DictReader(fh))
    by = {}
    for r in rows:
        by.setdefault(r["name"], []).append(r)
    cols = ["nb_logodds", "lo_0_100", "lo_100_300", "lo_300_700", "lo_700p", "lo_varsame", "nb_lsd"]
    print(f"\nround 3, mean over seeds (n = seeds done)       " + "  ".join(f"{c[:10]:>10s}" for c in cols))
    ref = np.array([float(r["nb_logodds"]) for r in by.get("ref", [])])
    for name in DESIGNS:
        if name not in by:
            continue
        v = {c: np.array([float(r[c]) for r in by[name]]) for c in cols}
        line = f"  {name:8s} n={len(by[name])}  params {int(by[name][0]['params']):7,d}            " + \
               "  ".join(f"{v[c].mean():10.4f}" for c in cols)
        x = v["nb_logodds"]
        if name != "ref" and len(ref) > 1 and len(x) > 1:
            d = x.mean() - ref.mean()
            se_ = math.sqrt(x.var(ddof=1) / len(x) + ref.var(ddof=1) / len(ref))
            ok = d >= 0.003 and d > 2 * se_
            line += f"   vs ref {d:+.4f} +- {se_:.4f}  -> {'CANDIDATE' if ok else 'no'}"
        elif len(x) > 1:
            line += f"   seed sd {x.std(ddof=1):.4f}"
        print(line)
        fs = [os.path.join(feat_dir, f"{name}_s{int(r['seed'])}.npy") for r in by[name]]
        if len(fs) > 1 and all(os.path.exists(f) for f in fs):
            ens = np.mean([np.load(f).astype(np.float64) for f in fs], axis=0)
            if len(ens) == len(y):
                print(f"  {name + '_ens':12s} ({len(fs)}-seed average of nb_logodds)          {auc(ens):10.4f}"
                      f"  {auc(ens, 0, 100):10.4f}  {auc(ens, 100, 300):10.4f}  {auc(ens, 300, 700):10.4f}"
                      f"  {auc(ens, 700):10.4f}  {auc(ens, mask=in_slice):10.4f}")
    print("\nRule: CANDIDATE = 3-seed mean beats ref by >= +0.003 and > 2 SE. A candidate needs a numba runtime port "
          "and a paired cv.py --nbm run; seed ensembles cost N x the NBM inference time.")


if __name__ == "__main__":
    main()
