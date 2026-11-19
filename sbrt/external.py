"""Extra REAL training series from the 2025 offline edition (ADIA Lab Structural Break Challenge).

The 2025 data: about 10,000 labelled series, each split at a KNOWN boundary into period 0 (before) and
period 1 (after); label True = the generating process changed at the boundary. The organisers' FAQ for the
Real-Time edition allows using it ("Yes, but it is not officially encouraged"; you must accept the old
competition's rules to download it).

Turning one 2025 series into one Real-Time-style series (exact labels, real values only):

    draw an online length n ~ U{10..999} and a break position t ~ U{0..n-1}   (the Real-Time distributions)
    history = the last H points of period 0 before position L0 - t,   H ~ U{1000..5000} capped by what exists
    online  = period 0 [L0 - t :]  ++  period 1 [: n - t]
    label   = break at online index t if the 2025 label is True, otherwise no break anywhere

Break and no-break series are cut with exactly the same geometry, so history length, online length and
the pre-boundary share carry no label information. A draw that does not fit (period 0 shorter than
hmin + t, or period 1 shorter than n - t) is redrawn; series that never fit are skipped.

Leakage guard: `ext2025.py` checks whether any 2025 series shares raw segments with a Real-Time training
series (two fingerprints: exact affine-invariant difference ratios, and the coarse up/down pattern that
survives float32 storage and rounding) and writes `ext2025_match.npz`. A 2025 series matched to one
Real-Time series gets it as its PARENT, so it only trains in folds where its twin is a training series (same
rule as the re-cut variants); a series matched to several Real-Time series gets parent -2 and is dropped;
unmatched series have parent -1 and train in every fold. cv.py / gru_cv.py refuse to use the data before
that check has run.

Self-test (builds a fake 2025 dataset out of the Real-Time series and checks round trips):
    python -m sbrt.external
"""
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DIR = os.path.join(ROOT, "data", "ext2025")
MATCH_FILE = "ext2025_match.npz"
H_MAX = 5000
O_MIN, O_MAX = 10, 999


def load_2025(d=DEFAULT_DIR):
    """List of (id, period0, period1, label) from X_train.parquet / y_train.parquet in folder d (cached)."""
    cache = os.path.join(d, "ext2025_cache.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return list(z["series"])
    import pandas as pd
    X = pd.read_parquet(os.path.join(d, "X_train.parquet"))
    y = pd.read_parquet(os.path.join(d, "y_train.parquet"))
    y = y.iloc[:, 0] if isinstance(y, pd.DataFrame) else y
    if not isinstance(X.index, pd.MultiIndex):
        X = X.set_index(["id", "time"])
    X = X.sort_index()
    ids = X.index.get_level_values(0).to_numpy()
    vals = X["value"].to_numpy(dtype=np.float64)
    per = X["period"].to_numpy()
    bounds = np.flatnonzero(ids[1:] != ids[:-1]) + 1
    starts, ends = np.r_[0, bounds], np.r_[bounds, len(ids)]
    lab = y.astype(bool)
    out = []
    for s, e in zip(starts, ends):
        p = per[s:e]
        if not (np.all(np.diff(p) >= 0) and p.min() == 0 and p.max() == 1):
            continue                                    # needs a clean 0...0 1...1 split
        n0 = int((p == 0).sum())
        out.append((ids[s], vals[s:s + n0].copy(), vals[s + n0:e].copy(), bool(lab.loc[ids[s]])))
    arr = np.empty(len(out), dtype=object)
    for i, q in enumerate(out):
        arr[i] = q
    np.savez(cache, series=arr)
    return out


def to_online(pre, post, label, rng, hmin=1000, tries=50):
    """One Real-Time-style cut of a 2025 series: (hist, online, tau) or None if nothing fits."""
    L0, L1 = len(pre), len(post)
    for _ in range(tries):
        n = int(rng.integers(O_MIN, O_MAX + 1))
        t = int(rng.integers(0, n))
        avail = L0 - t
        if avail < hmin or n - t > L1:
            continue
        H = min(int(rng.integers(1000, H_MAX + 1)), avail)
        H = max(H, hmin)
        hist = pre[avail - H:avail]
        online = np.concatenate([pre[avail:], post[:n - t]])
        mu, sd = hist.mean(), hist.std()
        if not np.isfinite(sd) or sd <= 0:
            return None
        return (hist - mu) / sd, (online - mu) / sd, (t if label else -1)
    return None


def load_match(d=DEFAULT_DIR):
    """Real-Time parent index of every 2025 series (-1 = no shared segments); None if not checked yet."""
    p = os.path.join(d, MATCH_FILE)
    if not os.path.exists(p):
        return None
    return np.load(p)["parent"]


def make_external(d=DEFAULT_DIR, n_per=1, seed=0, hmin=1000, limit=0):
    """Extra training series: list of (parent, hist, online, tau). Deterministic for a given seed."""
    E = load_2025(d)
    parent = load_match(d)
    if parent is None:
        raise SystemExit(f"run `python ext2025.py --dir {d}` first: it checks the 2025 data against the Real-Time "
                         f"data (and for shared series, which would leak across CV folds) and writes {MATCH_FILE}")
    assert len(parent) == len(E), "match file does not belong to this 2025 data: rerun ext2025.py"
    hmin = max(hmin, 200)        # the GRU warms up on the last 100 history points; AR fits need a few hundred
    rng = np.random.default_rng(seed)
    out = []
    for i, (_, pre, post, lab) in enumerate(E[:limit] if limit else E):
        if parent[i] == -2:      # shares segments with more than one Real-Time series: dropped (cannot be fold-safe)
            continue
        for _ in range(n_per):
            r = to_online(pre, post, lab, rng, hmin)
            if r is not None:
                out.append((int(parent[i]), r[0], r[1], r[2]))
    return out


# ------------------------------------------------------------------------------------------------
# affine-invariant fingerprints (shared segments survive any rescaling / re-centering)

def kgram_hashes(x, k=8, res=1e-4, pos=False):
    """Hashes of k consecutive log-ratios of successive differences (invariant to x -> a*x + b, a > 0).
    pos=True also returns the start position of every kept window."""
    d = np.diff(np.asarray(x, dtype=np.float64))
    if len(d) < k + 1:
        return (np.zeros(0, np.int64), np.zeros(0, np.int64)) if pos else np.zeros(0, np.int64)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = d[1:] / d[:-1]
    ok = np.isfinite(r) & (np.abs(d[:-1]) > 1e-12) & (np.abs(d[1:]) > 1e-12)
    q = np.where(ok, np.round(np.sign(r) * np.log1p(np.abs(r)) / res), 0).astype(np.int64)
    m = len(q) - k + 1
    h = np.zeros(m, np.int64)
    good = np.ones(m, bool)
    with np.errstate(over="ignore"):
        for j in range(k):
            h = h * np.int64(1_000_003) + q[j:j + m]
            good &= ok[j:j + m]
    return (h[good], np.flatnonzero(good)) if pos else h[good]


def sign_hashes(x, k=64, tiny=1e-3, pos=False):
    """Coarse fingerprints: the up/down pattern of k consecutive differences as one 64-bit word.
    Survives any increasing transform, float32 storage and rounding to a tick, unless a difference is
    tiny (windows with a difference below `tiny` x the median |difference| are skipped). Low-complexity
    windows (fewer than 16 or more than 48 sign changes, e.g. alternating or long runs) are skipped too:
    unrelated series share them by chance."""
    d = np.diff(np.asarray(x, dtype=np.float64))
    m = len(d) - k + 1
    if m <= 0:
        return (np.zeros(0, np.int64), np.zeros(0, np.int64)) if pos else np.zeros(0, np.int64)
    small = np.abs(d) <= tiny * max(np.median(np.abs(d)), 1e-300)
    bad = np.r_[0, np.cumsum(small)]
    good = (bad[k:k + m] - bad[:m]) == 0
    b = (d > 0).astype(np.uint64)
    ch = np.r_[0, np.cumsum(b[1:] != b[:-1])]          # sign changes inside each window
    nch = ch[k - 1:k - 1 + m] - ch[:m]
    good &= (nch >= 16) & (nch <= 48)
    h = np.zeros(m, np.uint64)
    for j in range(k):
        h |= b[j:j + m] << np.uint64(j)
    return (h[good].view(np.int64), np.flatnonzero(good)) if pos else h[good].view(np.int64)


def _selftest():
    import pandas as pd
    import tempfile
    from sbrt.data import load_train
    S, tau = load_train()
    rng = np.random.default_rng(1)
    sub = rng.choice(len(S), 400, replace=False)
    rows, labs = [], {}
    for j, i in enumerate(sub):                         # fake 2025 data: boundary = break (or a random cut)
        _, h, o = S[i]
        s = np.concatenate([h, o])
        b = len(h) + (tau[i] if tau[i] >= 0 else int(rng.integers(0, len(o))))
        a, c = 3.7, -12.0                               # a different affine scale, as raw 2025 values would have
        rows.append(pd.DataFrame({"id": j, "time": np.arange(len(s)), "value": a * s + c,
                                  "period": (np.arange(len(s)) >= b).astype(int)}))
        labs[j] = bool(tau[i] >= 0)
    with tempfile.TemporaryDirectory() as d:
        pd.concat(rows).set_index(["id", "time"]).to_parquet(os.path.join(d, "X_train.parquet"))
        pd.Series(labs, name="structural_breakpoint").rename_axis("id").to_frame().to_parquet(
            os.path.join(d, "y_train.parquet"))
        E = load_2025(d)
        assert len(E) == 400
        for j, (_, pre, post, lab) in enumerate(E):
            i = sub[j]
            assert lab == (tau[i] >= 0)
        # fingerprints find the twins
        H = {}
        for i in sub[:50]:
            for v in kgram_hashes(np.concatenate([S[i][1], S[i][2]])):
                H[int(v)] = int(i)
        found = 0
        for j in range(50):
            hs = kgram_hashes(np.concatenate([E[j][1], E[j][2]]))
            hit = [H[int(v)] for v in hs[:200] if int(v) in H]
            found += bool(hit) and max(set(hit), key=hit.count) == sub[j]
        print(f"fingerprints: {found}/50 fake 2025 series matched to their Real-Time twin")
        assert found >= 48
        # cuts: exact labels, geometry independent of the label, z-scored history
        rng2 = np.random.default_rng(0)
        n_ok, hb, hn = 0, [], []
        for _, p0, p1, lab in E:
            r = to_online(p0, p1, lab, rng2)
            if r is None:
                continue
            h, o, t = r
            n_ok += 1
            assert abs(h.mean()) < 1e-9 and abs(h.std() - 1) < 1e-9
            assert O_MIN <= len(o) <= O_MAX and 1000 <= len(h) <= H_MAX
            assert (t >= 0) == lab and t < len(o)
            # the 2025 boundary must sit exactly at online index t (break) / the same cut geometry (no break)
            tt = t if lab else None
            if tt is not None:
                a = len(p0) - tt
                raw_h = p0[a - len(h):a]
                want = (np.concatenate([p0[a:], p1[:len(o) - tt]]) - raw_h.mean()) / raw_h.std()
                assert np.allclose(o, want)
                hb.append(len(h))
            else:
                hn.append(len(h))
        print(f"cuts: {n_ok}/400 series fit; median history break {np.median(hb):.0f} vs no-break {np.median(hn):.0f}")
        print("self-test OK")


if __name__ == "__main__":
    _selftest()
