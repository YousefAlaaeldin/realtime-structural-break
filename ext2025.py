"""Check the 2025 offline-edition data against the Real-Time data before training on it. No model training.

Run:    python ext2025.py                          # data in data\\ext2025 (X_train.parquet, y_train.parquet)
        python ext2025.py --dir D:\\somewhere\\data
Output: cv_out/ext2025_check.txt and <dir>/ext2025_match.npz (needed by cv.py / gru_cv.py --ext)

Four questions, answered side by side with the Real-Time training data:
  1. Same kind of series?   lengths, label rate, heavy tails, autocorrelation, skew of the pre-boundary part
  2. Same kind of breaks?   the "oracle" table of eda.py / synth_calibrate.py: with the break position known,
                            how well each single statistic of the L points after it separates break from
                            no-break series (two-sided AUC; 0.5 = invisible). Similar rows = same break generator.
  3. Shared series?         two fingerprints of every series (exact difference ratios; up/down patterns that
                            survive float32 storage and rounding); a 2025 series sharing segments with ONE
                            Real-Time series is tied to it as its parent (never trains in a fold where its twin
                            is validated); one sharing with several is dropped. Also tells whether both
                            editions were cut from the same pool.
  4. How many fit?          how many 2025 series can be cut into Real-Time shape (history >= 1000 points),
                            how many of those are new, and whether the cut lengths give the label away.
"""
import argparse
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.stats import kurtosis, rankdata, skew

from sbrt import external as X
from sbrt.data import load_train

ap = argparse.ArgumentParser()
ap.add_argument("--dir", type=str, default=X.DEFAULT_DIR, help="folder with the 2025 X_train.parquet / y_train.parquet")
ap.add_argument("--n-sample", type=int, default=3000, help="series per dataset for the descriptive tables")
ap.add_argument("--min-hits", type=int, default=5, help="shared exact fingerprints (at one offset) needed to call two series twins")
ap.add_argument("--min-hits-coarse", type=int, default=50,
                help="shared up/down-pattern windows (at one offset) needed; chance matches give up to about 8, real copies thousands")
args = ap.parse_args()
warnings.filterwarnings("ignore")
np.seterr(all="ignore")


class Tee:
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")

    def write(self, s):
        sys.__stdout__.write(s)
        self.f.write(s)

    def flush(self):
        sys.__stdout__.flush()
        self.f.flush()


os.makedirs("cv_out", exist_ok=True)
sys.stdout = Tee(os.path.join("cv_out", "ext2025_check.txt"))
for f in ("X_train.parquet", "y_train.parquet"):
    if not os.path.exists(os.path.join(args.dir, f)):
        sys.exit(f"{os.path.join(args.dir, f)} not found. Download the 2025 'ADIA Lab Structural Break Challenge' data "
                 f"(accept that competition's rules on the hub first) and put X_train.parquet and y_train.parquet there.")

t0 = time.time()
S, tau = load_train()
E = X.load_2025(args.dir)
print(f"Real-Time training series: {len(S):,}  |  2025 series: {len(E):,}  (loaded in {time.time() - t0:.0f}s)")
rng = np.random.default_rng(0)


def pct(a, q=(5, 25, 50, 75, 95)):
    return " ".join(f"{v:8.2f}" for v in np.nanpercentile(a, q))


# ---------------------------------------------------------------------------------------------- 1
print("\n== 1. series: percentiles 5/25/50/75/95 ==")
nh = np.array([len(s[1]) for s in S]); no = np.array([len(s[2]) for s in S])
L0 = np.array([len(e[1]) for e in E]); L1 = np.array([len(e[2]) for e in E]); lab = np.array([e[3] for e in E])
print(f"{'RT history length':28s} {pct(nh)}")
print(f"{'RT online length':28s} {pct(no)}")
print(f"{'2025 period-0 length':28s} {pct(L0)}")
print(f"{'2025 period-1 length':28s} {pct(L1)}")
print(f"break rate: Real-Time {np.mean(tau >= 0):.3f}  |  2025 {lab.mean():.3f}")


def desc(x):
    x = np.asarray(x, float)
    return kurtosis(x), skew(x), np.corrcoef(x[:-1], x[1:])[0, 1], np.corrcoef(np.abs(x[:-1] - x.mean()), np.abs(x[1:] - x.mean()))[0, 1]


iS = rng.choice(len(S), min(args.n_sample, len(S)), replace=False)
okE = np.flatnonzero(L0 >= 200)                 # descriptive stats need a reasonably long period 0
iE = rng.choice(okE, min(args.n_sample, len(okE)), replace=False)
dS = np.array([desc(S[i][1]) for i in iS])
dE = np.array([desc(E[i][1]) for i in iE])
for j, nm in enumerate(["excess kurtosis", "skew", "lag-1 autocorrelation", "vol clustering (|x| lag-1)"]):
    print(f"{nm:28s} RT  {pct(dS[:, j])}")
    print(f"{'':28s} 2025{pct(dE[:, j])}")
print(f"share kurtosis > 3:           RT {np.mean(dS[:, 0] > 3):.3f} | 2025 {np.mean(dE[:, 0] > 3):.3f}")
print(f"share |rho1| > 0.3:           RT {np.mean(np.abs(dS[:, 2]) > 0.3):.3f} | 2025 {np.mean(np.abs(dE[:, 2]) > 0.3):.3f}")
print(f"share rho1 > 0.9:             RT {np.mean(dS[:, 2] > 0.9):.3f} | 2025 {np.mean(dE[:, 2] > 0.9):.3f}")

# ---------------------------------------------------------------------------------------------- 2
# same statistics as synth_calibrate.py (copied so this script has no side effects on import)
LS = (30, 100, 300)


def auc(a, b):
    r = rankdata(np.r_[a, b])
    return (r[: len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))


def stat_vec(h, z):
    c = lambda x, k: np.corrcoef(x[:-k], x[k:])[0, 1]
    return {
        "mean (|.|)": z.mean() * np.sqrt(len(z)),
        "log var": np.log(z.var() + 1e-9),
        "mean |z|": np.abs(z).mean() - np.abs(h).mean(),
        "ac1": c(z, 1) - c(h, 1),
        "ac2": c(z, 2) - c(h, 2),
        "skew": skew(z) - skew(h),
        "kurt": kurtosis(z) - kurtosis(h),
        "tail>2": (np.abs(z) > 2).mean() - (np.abs(h) > 2).mean(),
        "vol clustering": c(np.abs(z), 1) - c(np.abs(h), 1),
        "log var(diff)": np.log(np.diff(z).var() + 1e-9) - np.log(np.diff(h).var() + 1e-9),
    }


def oracle(items):
    items = list(items)
    table = {}
    for L in LS:
        bk, nbk = [], []
        for h, o, t in items:
            mu, sd = h.mean(), h.std() + 1e-12
            hz, oz = (h - mu) / sd, (o - mu) / sd
            if t >= 0 and len(o) - t >= L:
                bk.append(stat_vec(hz, oz[t:t + L]))
            elif t < 0 and len(o) >= L:
                nbk.append(stat_vec(hz, oz[:L]))
        if len(bk) < 30 or len(nbk) < 30:
            continue
        for key in bk[0]:
            a = np.array([d[key] for d in bk]); b = np.array([d[key] for d in nbk])
            a, b = a[np.isfinite(a)], b[np.isfinite(b)]
            med = np.median(b)
            table.setdefault(key, {})[L] = auc(np.abs(a - med), np.abs(b - med))
    return table


print("\n== 2. break signature: oracle single-statistic AUC with the break position known (0.5 = invisible) ==")
real = oracle((s[1], s[2], int(t)) for s, t in zip(S, tau))
ext = oracle((E[i][1], E[i][2], 0 if E[i][3] else -1) for i in range(len(E)))
print("%-16s %21s   %21s" % ("", "Real-Time", "2025 (break at boundary)"))
print("%-16s %6s %6s %6s    %6s %6s %6s" % ("statistic", "L=30", "L=100", "L=300", "L=30", "L=100", "L=300"))
for key in real:
    r, e = real[key], ext.get(key, {})
    print("%-16s %6.3f %6.3f %6.3f    %6s %6s %6s" % (key, r.get(30, np.nan), r.get(100, np.nan), r.get(300, np.nan),
                                                   *(f"{e[L]:.3f}" if L in e else "  -" for L in LS)))
avg = lambda T, L: np.mean([row[L] for row in T.values() if L in row])
print("%-16s %6.3f %6.3f %6.3f    %6.3f %6.3f %6.3f" % ("AVERAGE", *(avg(real, L) for L in LS), *(avg(ext, L) for L in LS)))
dist = sum(abs(avg(real, L) - avg(ext, L)) for L in LS)
print(f"distance between the AVERAGE rows: {dist:.4f}   (synthetic breaks at the best scale were 0.016 away)")

# ---------------------------------------------------------------------------------------------- 3
print("\n== 3. shared series (fingerprints of every series against every Real-Time series) ==")
import pandas as pd  # noqa: E402
raw = pd.read_parquet(os.path.join(args.dir, "X_train.parquet"), columns=["value"])["value"]
sv = raw.to_numpy()[:200_000]
dec = next((k for k in range(0, 9) if np.allclose(sv * 10 ** k, np.round(sv * 10 ** k), rtol=0, atol=1e-6 * 10 ** k)), None)
print(f"2025 values stored as {raw.dtype}; float32-exact share {np.mean(sv.astype(np.float32).astype(np.float64) == sv):.3f}; "
      f"decimals: {dec if dec is not None else 'more than 8'}")
del raw, sv


def index(fn):
    hs, owner, ps = [], [], []
    for i, s in enumerate(S):
        h, p = fn(np.concatenate([s[1], s[2]]), pos=True)
        hs.append(h); owner.append(np.full(len(h), i, np.int64)); ps.append(p)
    hs = np.concatenate(hs); owner = np.concatenate(owner); ps = np.concatenate(ps)
    o = np.argsort(hs, kind="stable")
    return hs[o], owner[o], ps[o]


def twins(q, qpos, hs, owner, ps):
    """{Real-Time series: shared windows at ONE consistent offset} for the windows q of one 2025 series.
    A real copy shares its windows at a single shift; chance collisions do not line up. A window hash that
    occurs more than once in the index is looked up once (its first occurrence), which is enough here."""
    if not len(q):
        return {}
    k = np.minimum(np.searchsorted(hs, q), len(hs) - 1)
    m = hs[k] == q
    if not m.any():
        return {}
    who = owner[k[m]]
    delta = ps[k[m]] - qpos[m]
    key, c = np.unique(who * 100_000 + (delta + 50_000), return_counts=True)
    out = {}
    for kk, cc in zip(key.tolist(), c.tolist()):
        w = kk // 100_000
        out[w] = max(out.get(w, 0), cc)
    return out


t0 = time.time()
passes = [("fine (difference ratios, 8 points)", X.kgram_hashes), ("coarse (up/down pattern, 64 points)", X.sign_hashes)]
idx = [index(fn) for _, fn in passes]
parent = np.full(len(E), -1, np.int64)
hits_of = np.zeros(len(E), np.int64)
n_by_pass = [0, 0]
for j, e in enumerate(E):
    x = np.concatenate([e[1], e[2]])
    found = {}
    for pi, ((_, fn), (hs, owner, ps)) in enumerate(zip(passes, idx)):
        q, qp = fn(x, pos=True)
        need = args.min_hits if pi == 0 else args.min_hits_coarse
        tw = {k: c for k, c in twins(q, qp, hs, owner, ps).items() if c >= need}
        n_by_pass[pi] += bool(tw)
        for k, c in tw.items():
            found[k] = max(found.get(k, 0), c)
    if len(found) == 1:
        (parent[j], hits_of[j]), = found.items()
    elif len(found) > 1:
        parent[j], hits_of[j] = -2, max(found.values())
n1, nm = int((parent >= 0).sum()), int((parent == -2).sum())
print(f"indexed {len(idx[0][0]):,} + {len(idx[1][0]):,} Real-Time windows ({time.time() - t0:.0f}s); series with a twin: "
      + ", ".join(f"{nm_}: {c:,}" for (nm_, _), c in zip(passes, n_by_pass)))
print(f"2025 series with exactly one Real-Time twin: {n1:,} (train only where the twin trains); with several: {nm:,} (dropped); "
      f"with none: {int((parent == -1).sum()):,}")
if n1 + nm and n_by_pass[1] > 1.2 * max(n_by_pass[0], 1):
    print("   NOTE: the coarse pass finds many more twins than the fine one -> the 2025 values are rounded/stored differently;"
          " the union of both passes is used.")
if n1:
    tw = parent[parent >= 0]
    print(f"   distinct Real-Time twins: {len(np.unique(tw)):,}; median shared windows {np.median(hits_of[parent >= 0]):.0f}")
    print(f"   label agreement between twins (break vs no break): {np.mean((tau[tw] >= 0) == lab[parent >= 0]):.3f}")
np.savez(os.path.join(args.dir, X.MATCH_FILE), parent=parent, hits=hits_of)
print(f"saved {os.path.join(args.dir, X.MATCH_FILE)}")

# ---------------------------------------------------------------------------------------------- 4
print("\n== 4. how many 2025 series fit the Real-Time shape (one cut each), and do the cuts give the label away? ==")
print("   (AUC of break vs no-break cuts on each length; about 0.5 = no label information)")
for hmin in (1000, 700, 500):
    r2 = np.random.default_rng(0)
    rows = []
    for j, e in enumerate(E):
        if parent[j] == -2:
            continue
        c = X.to_online(e[1], e[2], e[3], r2, hmin)
        if c is not None:
            rows.append((e[3], len(c[0]), len(c[1]), parent[j] < 0))
    R = np.array(rows, dtype=float)
    if not len(R):
        print(f"history >= {hmin:4d}: nothing fits")
        continue
    b = R[:, 0] == 1
    a_h = auc(R[b, 1], R[~b, 1]) if b.any() and (~b).any() else np.nan
    a_o = auc(R[b, 2], R[~b, 2]) if b.any() and (~b).any() else np.nan
    print(f"history >= {hmin:4d}: {len(R):,} fit ({len(R) / len(E):.0%}), of which {int(R[:, 3].sum()):,} are NEW series "
          f"(no Real-Time twin); break share {b.mean():.3f}; AUC history length {a_h:.3f}, online length {a_o:.3f}")

print("\nHOW TO READ THIS")
print(" - Section 2 matters most: if the 2025 AVERAGE row is within about 0.01 of the Real-Time row at every L, the breaks")
print("   look alike and the data should behave like more real training series (doubling the data was worth about +0.015).")
print("   If the 2025 breaks are much clearer, it can still help but may need --ext-rel < 1.")
print(" - Section 3: series with a twin are near-copies of Real-Time series and add little that is new (like extra re-cuts,")
print("   where --aug 4 gave only +0.0007 over --aug 2). The real gain depends on the NEW series count in section 4.")
print(" - Section 4: the HISTORY-length AUC must stay within about 0.45-0.55 (the model sees history length; outside that the")
print("   cuts leak the label). Online length is not a model input: a deviation there only changes how many break rows each")
print("   step gets, which the per-step weights re-balance.")
print(" - Then run cv.py / gru_cv.py with --ext 1 (HANDOFF.md section 15.4); --ext-hmin 700 if few series fit at 1000.")
