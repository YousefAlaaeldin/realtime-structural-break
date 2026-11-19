import sys, time
sys.path.insert(0, ".")
import numpy as np
from sbrt.data import load_train
from sbrt import features as F

S, tau = load_train()
print("N_FEAT", F.N_FEAT)
i, h, o = S[0]
t = time.time()
M = F.run_series(h, o)
print("compile+run", round(time.time() - t, 2), "s", M.shape, "finite:", np.isfinite(M).all())

st = F.Stream(h)
M2 = np.array([st.push(x).copy() for x in o])
print("stream==batch max diff", np.abs(M - M2).max())

t = time.time()
for s in S[:200]:
    F.run_series(s[1], s[2])
dt = time.time() - t
print("200 series", round(dt, 2), "s ->", round(dt / 200 * 1000, 1), "ms/series; x16000 =",
      round(dt / 200 * 16000 / 60, 1), "min single core")

bad = 0
for s in S[:300]:
    if not np.isfinite(F.run_series(s[1], s[2])).all():
        bad += 1
print("series with non-finite features (of 300):", bad)
