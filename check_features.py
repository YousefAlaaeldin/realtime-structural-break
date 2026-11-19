"""Self-check for the streaming feature kernel. Run this BEFORE cv.py.

Run:  python check_features.py

Checks
  1. streaming == batch (the code path used at inference gives the same numbers as training)
  2. causality: changing FUTURE points never changes features at earlier steps
  3. all features finite, and how extreme the largest values get
  4. calibration: on no-break series, the standardised 'full_*' features should be O(1);
     on break series (100 points after the break) they should be larger -> mini detectability table
  5. speed, and an extrapolation to the cloud workload (16,000 series, ~504 online steps each)
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from sbrt.data import load_train
from sbrt import features as F

S, tau = load_train()
rng = np.random.default_rng(0)
ok = True


def verdict(name, passed, extra=""):
    global ok
    ok &= bool(passed)
    print(f"[{'PASS' if passed else 'FAIL'}] {name} {extra}")


print(f"increment series K = {F.K}, features = {F.N_FEAT}")
i, h, o = S[0]
t0 = time.time()
F.run_series(h, o)
print(f"first call (includes numba compile): {time.time() - t0:.1f}s")

# 1. stream == batch
worst = 0.0
for idx in rng.choice(len(S), 6, replace=False):
    _, h, o = S[idx]
    M = F.run_series(h, o)
    st = F.Stream(h)
    M2 = np.array([st.push(x).copy() for x in o])
    worst = max(worst, float(np.abs(M - M2).max()))
verdict("streaming == batch", worst == 0.0, f"(max abs diff {worst:.2e})")

# 2. causality
worst = 0.0
for idx in rng.choice(np.flatnonzero([len(s[2]) > 120 for s in S]), 6, replace=False):
    _, h, o = S[idx]
    cut = int(rng.integers(20, len(o) - 50))
    o2 = o.copy()
    o2[cut + 1:] = rng.standard_normal(len(o) - cut - 1) * 5.0
    A = F.run_series(h, o)[: cut + 1]
    B = F.run_series(h, o2)[: cut + 1]
    worst = max(worst, float(np.abs(A - B).max()))
verdict("causal (future points do not change earlier features)", worst == 0.0, f"(max abs diff {worst:.2e})")

# 3. finiteness and magnitude
mats = []
nonfin = 0
for idx in rng.choice(len(S), 300, replace=False):
    _, h, o = S[idx]
    M = F.run_series(h, o)
    if not np.isfinite(M).all():
        nonfin += 1
    mats.append((idx, M))
verdict("all features finite on 300 random series", nonfin == 0, f"({nonfin} series with NaN/inf)")
allm = np.concatenate([m for _, m in mats])
mx = np.abs(allm).max(axis=0)
big = [(F.FEATURE_NAMES[j], float(mx[j])) for j in np.argsort(-mx)[:6]]
print("largest |feature| seen:", [(n, round(v, 1)) for n, v in big])

# 4. calibration / mini detectability table
full_idx = [F.FEATURE_NAMES.index(f"full_{n}") for n in F.INC_NAMES]
rows_null, rows_brk = [], []
for idx, M in mats:
    t = tau[idx]
    if t < 0 and len(M) > 150:
        rows_null.append(np.abs(M[150, full_idx]))
    elif t >= 0 and t + 100 < len(M):
        rows_brk.append(np.abs(M[t + 100, full_idx]))
print(f"\nmean |full_*| feature: no-break series at step 150 (n={len(rows_null)}) vs break series 100 steps after the break (n={len(rows_brk)})")
if rows_null and rows_brk:
    mn, mb = np.mean(rows_null, axis=0), np.mean(rows_brk, axis=0)
    print("%-10s %8s %8s %8s" % ("increment", "null", "break", "ratio"))
    for j in np.argsort(-(mb / np.maximum(mn, 1e-9))):
        print("%-10s %8.2f %8.2f %8.2f" % (F.INC_NAMES[j], mn[j], mb[j], mb[j] / max(mn[j], 1e-9)))
    print("null values near 0.8 mean the standardisation is well calibrated (|N(0,1)| has mean 0.80);")
    print("much larger null values would mean the long-run-variance scaling is too small for those series.")
    bad = float(np.median(mn))
    verdict("null calibration (median over increments within 0.5-1.6)", 0.5 < bad < 1.6, f"(median {bad:.2f})")

# 5. speed
t0 = time.time()
sel = rng.choice(len(S), 200, replace=False)
for idx in sel:
    F.run_series(S[idx][1], S[idx][2])
dt = (time.time() - t0) / 200
print(f"\nbatch features: {dt * 1000:.1f} ms/series (single thread)")
t0 = time.time()
steps = 0
for idx in rng.choice(len(S), 15, replace=False):
    _, h, o = S[idx]
    st = F.Stream(h)
    for x in o:
        st.push(x)
    steps += len(o)
dts = (time.time() - t0) / steps
print(f"streaming features: {dts * 1e6:.1f} us per online step (single process, includes per-series setup)")
est = dts * 16000 * 504 / 60
print(f"estimated feature time for the cloud workload (16,000 series x ~504 steps): {est:.1f} min single process "
      f"(+ model prediction time per step; budget is 15 h = 900 min)")
verdict("feature time well under budget", est < 300)

print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED - do not trust cv.py results until fixed")
