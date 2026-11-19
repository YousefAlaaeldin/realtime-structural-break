"""Does entropy rise WHILE a break happens and fall back afterwards? (stage 0: event study, break position known)

Hypothesis (owner, 2026-10-15): at a breakpoint the series is "in transition", so a rolling entropy rises; after the
break it settles again. If true, a transient entropy bump marks breaks.

What section 20 (info_oracle.py) already tested: permutation entropy of a segment STARTING at tau vs the history ->
no signal. That is the "after" level. The open part is the transient while a trailing window straddles tau.

Method (every series, causal trailing windows of W points, W = 30 / 50 / 100):
  pe3_z   permutation entropy, order 3, of the series z (6 ordinal patterns)
  pe3_e   the same on the AR(10) residual e (as info_oracle / the kernel)
  bin_e   Shannon entropy of e over the history's 10 decile bins (value entropy)
Each rolling entropy is z-scored with the mean and std of the SAME rolling entropy over the series' own history,
so "dev = 2" means "2 history-sds above what this series normally shows".
Break series are aligned at tau (k = t - tau, window ending at online point tau + k; it straddles tau for
0 <= k < W - 1). Controls: no-break series aligned at a pseudo-tau drawn from the break series' taus (same
position distribution). Profiles of mean dev +- SE for k in [-200, 300].

Per-series summaries (all in history-sd units):
  pre      mean dev, k in [-W, -1]          (window entirely before the break)
  peak     mean dev, k in [W/4, 3W/4]       (window about half pre, half post)
  plateau  mean dev, k in [2W, 3W]          (window entirely after the break)
  transient = peak - plateau  (the owner's claim: > 0 for breaks, ~0 for controls)
  excess    = peak - pre
  shift     = plateau - pre    (the "after" level change section 20 found weak)
  bump      = peak - (pre + plateau) / 2   THE TEST STATISTIC. A plain step (entropy moves from one level to another)
              puts the straddle window about half way, so bump ~ 0 and transient > 0 without any rise. The owner's
              claim is "up during the transition, down after": the straddle sits ABOVE the line from before to after.
              (Smoke run lesson: bin_e showed transient > 0 only because its plateau dropped; its peak was below pre.)
              Second lesson (1,500-series plot): a step is not a straight line either. While the window straddles tau
              the mixture keeps entropy near the OLD level until most of the window is post-break (concavity), so a
              falling step also gives bump > 0. Hence:
  rise      = peak - max(pre, plateau)     THE TEST STATISTIC: above BOTH sides, which no monotone step produces.
Reported: mean +- SE for breaks and controls, the difference in units of the control sd, and AUC break vs control
(signed and |.|), on all series and on the VARIANCE-MATCHED slice (|log var ratio of e over the first 300 post points
vs history| < 0.1, same rule for controls at the pseudo-tau).

Note on mechanics: a window that straddles two regimes holds a mixture, and a mixture has higher entropy than the
average of its parts (concavity; the excess is the Jensen-Shannon divergence). So some bump is expected even for an
instantaneous switch; the question is whether it is big enough to see in 30-100 points.

KILL CRITERIA (written before the run):
  K1  no rise: for every stat and W, mean rise(break) - mean rise(control) < +0.1 control-sd  -> stop.
  K2  rise too weak to use: best signed AUC of rise (break > control) < 0.55 on all series AND < 0.53 on the
      variance-matched slice -> stop.
  Read the plot as well: the hypothesis predicts the red/orange curves go UP in the shaded zone and come back down.
  (revised after the 600-series smoke run, before any full run; the first version tested transient/excess, which a
  plain step passes)
  Otherwise -> stage 1 (streaming running-max feature, single-feature TS-AUC gate, incremental check vs the blend).

Run:  python diagnostics/entropy_transition.py [n_series] [threads]   (default all 10,000 series, 8 processes)
Out:  cv_out/entropy_transition.txt, diagnostics/entropy_transition.png
"""
import os
import sys
import time

import numpy as np
from scipy.signal import lfilter
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
os.chdir(HERE)

AR = 10
WS = (30, 50, 100)
STATS = ("pe3_z", "pe3_e", "bin_e")
KMIN, KMAX = -200, 300
KS = np.arange(KMIN, KMAX + 1)
VAR_SLICE = 0.1


def _ord3(x):
    """Ordinal pattern code of (x[t-2], x[t-1], x[t]) at every t >= 2 (8 codes, 6 occur without ties); -1 before."""
    a = (x[:-2] < x[1:-1]).astype(np.int64)
    b = (x[1:-1] < x[2:]).astype(np.int64)
    c = (x[:-2] < x[2:]).astype(np.int64)
    return np.r_[-1, -1, 4 * a + 2 * b + c], 6


def _rolling_entropy(sym, ncat, W):
    """Normalised Shannon entropy of the symbols in the trailing window [t-W+1, t]; NaN where the window has a -1."""
    K = int(sym.max()) + 1
    oh = np.zeros((len(sym) + 1, K), np.int32)
    ok = sym >= 0
    oh[1:][ok, sym[ok]] = 1
    cs = np.cumsum(oh, axis=0)
    bad = np.r_[0, np.cumsum(~ok)]
    H = np.full(len(sym), np.nan)
    t = np.arange(W - 1, len(sym))
    C = (cs[t + 1] - cs[t + 1 - W]).astype(np.float64)
    valid = (bad[t + 1] - bad[t + 1 - W]) == 0
    p = C / W
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -np.nansum(np.where(p > 0, p * np.log(p), 0.0), axis=1) / np.log(ncat)
    H[t[valid]] = h[valid]
    return H


def series_profile(h, o, t, seed, pseudo_taus):
    """Returns (label, tau_used, var_matched flag, profiles [len(STATS), len(WS), len(KS)] float32) or None."""
    h = np.asarray(h, np.float64)
    o = np.asarray(o, np.float64)
    nh, no = len(h), len(o)
    mu, sd = h.mean(), h.std() + 1e-12
    z = np.r_[(h - mu) / sd, (o - mu) / sd]
    zh = z[:nh]
    Xl = np.column_stack([zh[AR - 1 - k: nh - 1 - k] for k in range(AR)])
    coef = np.linalg.lstsq(Xl, zh[AR:], rcond=None)[0]
    e = lfilter(np.r_[1.0, -coef], [1.0], z)
    se = max(float(e[2 * AR:nh].std()), 1e-8)
    e = np.clip(e / se, -10, 10)
    e[:2 * AR] = np.nan

    if t >= 0:
        lab, tau = 1, t
    else:
        rng = np.random.default_rng(seed)
        cand = pseudo_taus[pseudo_taus < no]
        if len(cand) == 0:
            return None
        lab, tau = 0, int(rng.choice(cand))

    # variance-matched flag: first <= 300 points from tau vs history residual variance
    post = e[nh + tau: nh + min(no, tau + 300)]
    if len(post) < 30:
        vm = False
    else:
        vm = abs(np.log(np.nanvar(post) / np.nanvar(e[2 * AR:nh]))) < VAR_SLICE

    syms = {}
    syms["pe3_z"] = _ord3(z)
    se_sym, _ = _ord3(np.nan_to_num(e))
    se_sym[:2 * AR + 2] = -1
    syms["pe3_e"] = (se_sym, 6)
    edges = np.quantile(e[2 * AR:nh], np.linspace(0.1, 0.9, 9))
    bs = np.searchsorted(edges, np.nan_to_num(e)).astype(np.int64)
    bs[:2 * AR] = -1
    syms["bin_e"] = (bs, 10)

    prof = np.full((len(STATS), len(WS), len(KS)), np.nan, np.float32)
    gidx = nh + tau + KS                         # global index of the window's last point
    inside = (gidx >= 0) & (gidx < nh + no)
    for a, st in enumerate(STATS):
        sym, ncat = syms[st]
        for b, W in enumerate(WS):
            H = _rolling_entropy(sym, ncat, W)
            hh = H[:nh]
            hh = hh[np.isfinite(hh)]
            if len(hh) < 50:
                continue
            m, s = hh.mean(), max(hh.std(), 1e-6)
            prof[a, b, inside] = (H[gidx[inside]] - m) / s
    return lab, tau, vm, prof


# ---------------------------------------------------------------------------------------------
def _auc(s, y):
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _ms(x):
    x = x[np.isfinite(x)]
    return (x.mean(), x.std() / np.sqrt(max(len(x), 1)), x.std(), len(x)) if len(x) else (np.nan,) * 3 + (0,)


def zone(P, k0, k1):
    """Mean of profiles over offsets k in [k0, k1] (NaN-aware), per series."""
    j = (KS >= k0) & (KS <= k1)
    return np.nanmean(P[:, j], axis=1)


def main():
    import warnings
    from joblib import Parallel, delayed

    from sbrt.data import load_train

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    n_series = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    threads = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    os.makedirs("cv_out", exist_ok=True)
    log = open(os.path.join("cv_out", "entropy_transition.txt"), "w", encoding="utf-8")

    def say(s=""):
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    S, tau = load_train()
    if n_series:
        S, tau = S[:n_series], tau[:n_series]
    pseudo = np.asarray(tau[tau >= 0], np.int64)
    t0 = time.time()
    res = Parallel(n_jobs=threads, backend="loky", batch_size=32)(
        delayed(series_profile)(s[1], s[2], int(tau[i]), 7000 + i, pseudo) for i, s in enumerate(S))
    res = [r for r in res if r is not None]
    y = np.array([r[0] for r in res])
    vm = np.array([r[2] for r in res])
    P = np.stack([r[3] for r in res])           # [n, stat, W, k]
    say(f"entropy profiles for {len(res)} series ({y.sum()} breaks, {len(y) - y.sum()} controls) in "
        f"{time.time() - t0:.0f}s; variance-matched: {vm[y == 1].sum()} breaks, {vm[y == 0].sum()} controls")
    say("units: history-sds of the series' own rolling entropy.  rise = peak - max(pre, plateau) (hypothesis: > 0)")
    say("")

    verdict = {"k1_bump": False, "best_auc_all": 0.0, "best_auc_vm": 0.0}
    for a, st in enumerate(STATS):
        for b, W in enumerate(WS):
            Q = P[:, a, b, :].astype(np.float64)
            pre = zone(Q, -W, -1)
            peak = zone(Q, W // 4, (3 * W) // 4)
            plat = zone(Q, 2 * W, 3 * W)
            summ = {"pre": pre, "peak": peak, "plateau": plat,
                    "transient": peak - plat, "excess": peak - pre, "shift": plat - pre,
                    "bump": peak - 0.5 * (pre + plat), "rise": peak - np.maximum(pre, plat)}
            say(f"== {st}  W={W}")
            say(f"   {'summary':10s} {'breaks mean+-SE':>18s} {'controls mean+-SE':>19s} {'diff/ctrl-sd':>12s}"
                f" {'AUC':>6s} {'AUC|.|':>7s} {'AUC vm':>7s} {'|.|vm':>6s}")
            for nm, v in summ.items():
                m1, s1, _, _ = _ms(v[y == 1])
                m0, s0, sd0, _ = _ms(v[y == 0])
                d = (m1 - m0) / sd0 if sd0 > 0 else np.nan
                A, Aa = _auc(v, y), _auc(np.abs(v), y)
                Av, Aav = _auc(v[vm], y[vm]), _auc(np.abs(v[vm]), y[vm])
                say(f"   {nm:10s} {m1:+8.3f} +- {s1:5.3f}    {m0:+8.3f} +- {s0:5.3f} {d:+12.3f}"
                    f" {A:6.3f} {Aa:7.3f} {Av:7.3f} {Aav:6.3f}")
                if nm == "rise":
                    if d >= 0.1:
                        verdict["k1_bump"] = True
                    verdict["best_auc_all"] = max(verdict["best_auc_all"], A)
                    verdict["best_auc_vm"] = max(verdict["best_auc_vm"], Av)
            say("")

    say("VERDICT against the kill criteria written in the docstring:")
    say(f"  K1 (rise of breaks exceeds controls by >= 0.1 control-sd somewhere): "
        f"{'bump seen' if verdict['k1_bump'] else 'NO bump -> stop'}")
    k2 = verdict["best_auc_all"] >= 0.55 or verdict["best_auc_vm"] >= 0.53
    say(f"  K2 best signed AUC of rise: all {verdict['best_auc_all']:.3f} (need >= 0.55), "
        f"variance-matched {verdict['best_auc_vm']:.3f} (need >= 0.53): {'usable' if k2 else 'too weak -> stop'}")
    say(f"  => {'PASS: build stage 1 (streaming running-max gate)' if verdict['k1_bump'] and k2 else 'FAIL: hypothesis not useful here'}")
    say("  (with few series these numbers are noisy; read the full 10,000-series run)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(len(STATS), len(WS), figsize=(15, 10), sharex=True)
        groups = (("breaks", y == 1, "C3"), ("breaks, var-matched", (y == 1) & vm, "C1"), ("controls", y == 0, "C0"))
        for a, st in enumerate(STATS):
            for b, W in enumerate(WS):
                g = ax[a, b]
                for lab, msk, col in groups:
                    Q = P[msk, a, b, :].astype(np.float64)
                    m = np.nanmean(Q, axis=0)
                    se = np.nanstd(Q, axis=0) / np.sqrt(np.maximum(np.isfinite(Q).sum(0), 1))
                    g.plot(KS, m, color=col, lw=1.2, label=lab)
                    g.fill_between(KS, m - 2 * se, m + 2 * se, color=col, alpha=0.2)
                g.axvspan(0, W - 1, color="grey", alpha=0.12)
                g.axhline(0, color="k", lw=0.5)
                g.set_title(f"{st}  W={W}  (shaded: window straddles tau)", fontsize=9)
                if b == 0:
                    g.set_ylabel("dev (history sds)")
                if a == len(STATS) - 1:
                    g.set_xlabel("k = t - tau")
        ax[0, 0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join("diagnostics", "entropy_transition.png"), dpi=110)
        say("plot: diagnostics/entropy_transition.png")
    except Exception as ex:  # plotting is optional
        say(f"(plot skipped: {ex})")


if __name__ == "__main__":
    main()
