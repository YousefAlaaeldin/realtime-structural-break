import sys, os
sys.path.insert(0, ".")
import numpy as np
from sbrt import features as F
from sbrt.metric import ts_auc

z = np.load("data/feat_v1.npz")
X, y, sid, step = z["X"], z["y"], z["sid"], z["step"]
sub = (sid % 4 == 0)  # quarter of series for speed
X, y, step = X[sub], y[sub], step[sub]
rows = []
for i, name in enumerate(F.FEATURE_NAMES):
    a = ts_auc(X[:, i], y, step)
    b = ts_auc(np.abs(X[:, i]), y, step)
    rows.append((name, a, b))
rows.sort(key=lambda r: -max(abs(r[1] - 0.5), abs(r[2] - 0.5)))
print("%-18s %8s %8s" % ("feature", "AUC", "AUC|abs|"))
for name, a, b in rows[:30]:
    print("%-18s %8.4f %8.4f" % (name, a, b))
