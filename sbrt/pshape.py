"""Shape of the break-location posterior (HANDOFF section 31): how CONCENTRATED the evidence about WHERE the break
happened is, next to the existing odds (logodds) and mean location (tsb).

At online step t the one-break posterior (bocpd.py) weighs every candidate break time j given a break by t:
q_j ∝ pi_j exp(LR_j). Three numbers per stream:
    ent   entropy of q (nats): low = the evidence points at one location
    kl    KL(q || prior over j given a break by t): how much the data sharpened the belief about WHERE
    sdl   posterior sd of log(1 + time since the break)
Streams: bo_w, bo_wu (bocpd.py streams, fixed EB prior) -> 6 columns; nb (the NBM stream with the fold's prior)
-> 3 columns. Computed from the same cumulative sums the existing features use, so a streaming version is one extra
call per point next to BoStream / NbmStream (not built yet; inference is not wired).

cv.py --pshape appends them (bo_ columns always; nb_ columns with --nbm). Gate: diagnostics/posterior_entropy_gate.py.
"""
import math

import numpy as np
from numba import njit

from sbrt import bocpd as BO

VERSION = 1
BO_NAMES = [f"bo_{v}_{s}" for v in ("w", "wu") for s in ("ent", "kl", "sdl")]
NB_NAMES = ["nb_ent", "nb_kl", "nb_sdl"]


@njit(cache=True, nogil=True)
def post_ent(C1, C2, k, n, prior, LOGH, LP, buf, out, o):
    """bocpd._posterior plus the shape of q_j: out[o..o+4] = logodds, tsb, ent, kl, sdl (logodds/tsb equal bocpd's)."""
    k0, a0, b0 = prior[0], prior[1], prior[2]
    base = a0 * math.log(b0) - math.lgamma(a0)
    mx = -1e300
    mx0 = -1e300
    for j in range(n):
        m = n - j
        S1 = C1[k, n] - C1[k, j]
        S2 = C2[k, n] - C2[k, j]
        kn = k0 + m
        an = a0 + 0.5 * m
        bn = b0 + 0.5 * (S2 - S1 * S1 / kn)
        if bn < 1e-12:
            bn = 1e-12
        lr = 0.5 * math.log(k0 / kn) + base - an * math.log(bn) + math.lgamma(an) + 0.5 * S2
        p0 = LP[j] + LOGH[j]
        buf[j] = p0 + lr
        if buf[j] > mx:
            mx = buf[j]
        if p0 > mx0:
            mx0 = p0
    s = 0.0
    s0 = 0.0
    ea = 0.0
    el = 0.0
    el2 = 0.0
    sp0 = 0.0
    for j in range(n):
        wgt = math.exp(buf[j] - mx)
        s += wgt
        p0 = LP[j] + LOGH[j]
        s0 += math.exp(p0 - mx0)
        ea += wgt * (buf[j] - mx)
        sp0 += wgt * p0
        x = math.log(1.0 + (n - j))
        el += wgt * x
        el2 += wgt * x * x
    lse = mx + math.log(s)
    lse0 = mx0 + math.log(s0)
    out[o] = lse - LP[n]
    mean_l = el / s
    out[o + 1] = mean_l
    ent = math.log(s) - ea / s                       # -sum q log q
    out[o + 2] = ent
    out[o + 3] = -ent - (sp0 / s - lse0)             # KL(q || q0) = sum q log q - sum q log q0
    v = el2 / s - mean_l * mean_l
    out[o + 4] = math.sqrt(v if v > 0 else 0.0)


@njit(cache=True, nogil=True)
def rows_from_sums(C1, C2, nk, n_pts, prior, LOGH, LP, buf, sub, out):
    """out[r, 5k .. 5k+4] for streams k < nk at online steps 0, sub, 2 sub, ..."""
    r = 0
    for t in range(0, n_pts, sub):
        for k in range(nk):
            post_ent(C1, C2, k, t + 1, prior, LOGH, LP, buf, out[r], 5 * k)
        r += 1


def bo_shape(h, o, sub=1):
    """6 columns (BO_NAMES) at online steps 0, sub, ... from the bocpd.py streams of one series."""
    Q, refs, ST, C1, C2 = BO.make_bo_params(h)
    o = np.asarray(o, np.float64)
    BO._bo_sums(o, Q, refs, C1, C2, ST, np.array(BO.WIDE), BO.LOGH, BO.LP, np.zeros(BO.CAP + 1), np.zeros(BO.N_BO))
    n = len(o)
    full = np.zeros(((n + sub - 1) // sub, 10))
    rows_from_sums(C1, C2, 2, n, np.array((BO.KAPPA0, BO.ALPHA0, BO.BETA0)), BO.LOGH, BO.LP, np.zeros(n + 1), sub, full)
    return full[:, [2, 3, 4, 7, 8, 9]]


def nb_shape(sums, prior, sub=1):
    """3 columns (NB_NAMES) from the NBM sums (nbm.run_stream) and the fold's prior."""
    C1, C2 = sums[0], sums[1]
    n = C1.shape[1] - 1
    full = np.zeros(((n + sub - 1) // sub, 5))
    rows_from_sums(C1, C2, 1, n, np.asarray(prior, np.float64), BO.LOGH, BO.LP, np.zeros(n + 1), sub, full)
    return full[:, 2:5]
