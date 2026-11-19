"""Small-break ("more sensitive") variants of two production detectors (cv.py --sens, HANDOFF section 34).

The production detectors are tuned toward large breaks:
  - sbrt/evidence.py tests post-break alternatives no closer to "no change" than variance x0.8 / x1.25, mean +-0.3,
    dependence +-0.2, Student-t nu = 6;
  - sbrt/bocpd.py integrates the post-break variance over an empirical-Bayes Inverse-Gamma prior with mean ~1.23 and
    sd ~1.2 (fitted on all training breaks, so dominated by the big, easy ones).
Most of the blend's misses are breaks that move the variance little (HANDOFF section 3). Both variants below use
the SAME kernels, streams and hazard as production; only the alternatives / the prior change:

  ev  6 columns ev2_*: the evidence kernel with 14 SMALL alternatives (same count as production, so the numba
      kernel and its state layout are reused unchanged):
         variance  s in 0.85, 0.9, 0.95, 1.05, 1.1, 1.15
         mean      m in -0.2, -0.1, 0.1, 0.2
         dependence r in -0.1, 0.1
         tails     nu in 10, 20
      columns: model-averaged log odds, log Bayes factor, posterior share of each group (= the --ev compact set).
  bo  6 columns bn_*: the BOCPD posterior with a NARROW post-break prior, variance ~ Inverse-Gamma with mean 1.1 and
      sd 0.15 (covers ratios ~0.85-1.4; kappa0 kept at production's 490); columns logodds, logbf, lvr for both
      streams w / wu (tsb and mu would nearly duplicate production's bo_ columns).
No labels are used (the narrow prior is set by hand, not fitted).

Self-test:  python -m sbrt.sensdet
"""
import math

import numpy as np

from sbrt import bocpd as BO
from sbrt import evidence as EV

VERSION = 1   # bump when the alternatives or the prior change (part of the cache names)

FINE = ([("variance", 0, s) for s in (0.85, 0.9, 0.95, 1.05, 1.1, 1.15)]
        + [("mean", 1, m) for m in (-0.2, -0.1, 0.1, 0.2)]
        + [("dependence", 2, r) for r in (-0.1, 0.1)]
        + [("tails", 3, nu) for nu in (10.0, 20.0)])
assert len(FINE) == EV.NA, "the evidence kernel's state layout needs the production alternative count"
F_KIND = np.array([a[1] for a in FINE], dtype=np.int64)
F_PAR = np.array([a[2] for a in FINE], dtype=np.float64)
F_CONST = np.array([EV._const(a[1], a[2]) for a in FINE])
F_GRP = np.array([EV.GROUPS.index(a[0]) for a in FINE], dtype=np.int64)

_M, _SD = 1.1, 0.15
_A0 = _M * _M / (_SD * _SD) + 2.0
NARROW = (BO.KAPPA0, _A0, _M * (_A0 - 1.0))
BO_KEEP = [BO.FEATURE_NAMES.index(f"bo_{v}_{s}") for v in BO.VARS for s in ("logodds", "logbf", "lvr")]

EV_NAMES = ["ev2_" + nm[3:] for nm in EV.COMPACT]
BO_NAMES = ["bn_" + BO.FEATURE_NAMES[j][3:] for j in BO_KEEP]


def names(mode):
    return (EV_NAMES if mode in ("ev", "both") else []) + (BO_NAMES if mode in ("bo", "both") else [])


def run_ev(hist, online, alts=None):
    """(T, 6) small-alternative evidence columns (alts=None -> FINE; alts='prod' -> production, for the self-test)."""
    Q, eref, ST = EV.make_ev_params(np.asarray(hist, dtype=np.float64))
    online = np.asarray(online, dtype=np.float64)
    out = np.zeros((len(online), EV.N_EV))
    if alts == "prod":
        EV._ev_run(online, Q, eref, EV.HAZARD, EV.KIND, EV.PAR, EV.CONST, EV.GRP, ST, out)
    else:
        EV._ev_run(online, Q, eref, EV.HAZARD, F_KIND, F_PAR, F_CONST, F_GRP, ST, out)
    return out[:, :EV.N_COMPACT]


def run_series_sens(hist, online, sub, mode):
    """Rows for online steps 0, sub, 2 sub, ... (the rows cv.py keeps)."""
    cols = []
    if mode in ("ev", "both"):
        cols.append(run_ev(hist, online)[::sub])
    if mode in ("bo", "both"):
        cols.append(BO.run_series_bocpd(hist, online, sub, prior=NARROW)[:, BO_KEEP])
    return np.hstack(cols)


class SensStream:
    """One series' streaming state for use inside infer() (production would need this; not wired into main.py)."""

    def __init__(self, hist, mode):
        self.mode = mode
        h = np.asarray(hist, dtype=np.float64)
        if mode in ("ev", "both"):
            self.Q, self.eref, self.ST = EV.make_ev_params(h)
            self.eo = np.zeros(EV.N_EV)
        if mode in ("bo", "both"):
            self.bo = BO.BoStream(h)
            self.bo.prior = np.array(NARROW, dtype=np.float64)

    def push(self, x):
        parts = []
        if self.mode in ("ev", "both"):
            EV.ev_step(float(x), self.Q, self.eref, EV.HAZARD, F_KIND, F_PAR, F_CONST, F_GRP, self.ST, self.eo)
            parts.append(self.eo[:EV.N_COMPACT])
        if self.mode in ("bo", "both"):
            parts.append(self.bo.push(x)[BO_KEEP])
        return np.concatenate(parts)


def _selftest(n_series=2000, offset=5000):
    import os
    import sys
    import time
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from sbrt.data import load_train
    from sbrt.metric import ts_auc

    print(f"narrow prior: kappa0 {NARROW[0]:.0f}, alpha0 {NARROW[1]:.2f}, beta0 {NARROW[2]:.2f} "
          f"(variance mean {NARROW[2] / (NARROW[1] - 1):.3f}, sd {NARROW[2] / (NARROW[1] - 1) / math.sqrt(NARROW[1] - 2):.3f}); "
          f"production: alpha0 {BO.ALPHA0:.2f}, beta0 {BO.BETA0:.2f} (mean {BO.BETA0 / (BO.ALPHA0 - 1):.3f}, "
          f"sd {BO.BETA0 / (BO.ALPHA0 - 1) / math.sqrt(BO.ALPHA0 - 2):.3f})")
    S, tau = load_train()
    S, tau = S[offset:offset + n_series], tau[offset:offset + n_series]
    rng = np.random.default_rng(0)
    for i in rng.choice(n_series, 20, replace=False):
        _, h, o = S[i]
        assert np.array_equal(run_ev(h, o, "prod"), EV.run_series_ev(h, o)[:, :EV.N_COMPACT]), "wrapper != evidence.py"
        B = run_series_sens(h, o, 1, "both")
        st = SensStream(h, "both")
        Sm = np.array([st.push(x).copy() for x in o])
        assert np.allclose(B, Sm, rtol=0, atol=1e-12), "streaming != batch"
        assert np.isfinite(B).all()
        assert np.array_equal(run_series_sens(h, o, 3, "both"), B[::3]), "sub-sampled rows != full rows"
        if len(o) > 20:
            o2 = o.copy()
            o2[15:] = rng.standard_normal(len(o) - 15) * 3
            assert np.array_equal(run_series_sens(h, o2, 1, "both")[:15], B[:15]), "future points changed earlier rows"
    print("wrapper == evidence.py on production alternatives, streaming == batch, causal, finite: OK")

    t0 = time.time()
    mats = [run_series_sens(s[1], s[2], 3, "both") for s in S]
    ref_e = [EV.run_series_ev(s[1], s[2])[::3, :2] for s in S]
    ref_b = [BO.run_series_bocpd(s[1], s[2], 3)[:, [BO.FEATURE_NAMES.index("bo_w_logodds"),
                                                     BO.FEATURE_NAMES.index("bo_wu_logodds")]] for s in S]
    print(f"features for {n_series} held-out series in {time.time() - t0:.1f}s")
    E = np.concatenate(mats)
    R = np.hstack([np.concatenate(ref_e), np.concatenate(ref_b)])
    step = np.concatenate([np.arange(0, len(s[2]), 3) for s in S])
    sid = np.repeat(np.arange(n_series), [len(m) for m in mats])
    y = ((tau[sid] >= 0) & (step >= tau[sid])).astype(np.int8)
    print("single-feature TS-AUC (0.5 = no signal), new vs production counterpart:")
    for j, nm in enumerate(names("both")):
        print(f"  {nm:20s} {ts_auc(E[:, j], y, step):.4f}")
    for j, nm in enumerate(["ev_mix_logodds", "ev_logbf", "bo_w_logodds", "bo_wu_logodds"]):
        print(f"  {nm:20s} {ts_auc(R[:, j], y, step):.4f}   (production)")
    for a, b in (("ev2_mix_logodds", 0), ("bn_w_logodds", 2), ("bn_wu_logodds", 3)):
        rk = lambda v: np.argsort(np.argsort(v))
        c = np.corrcoef(rk(E[:, names("both").index(a)]), rk(R[:, b]))[0, 1]
        print(f"  rank correlation {a} vs production: {c:.3f}")
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    _selftest()
