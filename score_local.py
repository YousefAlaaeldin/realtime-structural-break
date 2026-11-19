"""Score prediction/prediction.parquet (written by `crunch test`) with TS-AUC on the 100 local test series.

Run:  python score_local.py
Note: 100 series is a small sample -- expect this number to move by a few hundredths versus the cloud
score. Trust cv.py (10,000 series, cross-validated) for the level, this for 'does the pipeline work'.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd

from sbrt.metric import ts_auc

pred = pd.read_parquet(os.path.join("prediction", "prediction.parquet"))
y = pd.read_parquet(os.path.join("data", "y_test.reduced.parquet"))["target"]
print("prediction rows:", len(pred), "| label rows:", len(y), "| series:", pred.index.get_level_values("id").nunique())

df = pred.join(y, how="inner")
if len(df) != len(y):
    print(f"WARNING: only {len(df)} of {len(y)} label rows matched a prediction (index mismatch?)")
col = "prediction" if "prediction" in df.columns else df.columns[0]
df["step"] = df.groupby(level="id").cumcount()
print("prediction range:", float(df[col].min()), "to", float(df[col].max()),
      "| distinct values:", df[col].nunique())

score = ts_auc(df[col].to_numpy(), df["target"].to_numpy(), df["step"].to_numpy())
print(f"\nLOCAL TS-AUC (100 series): {score:.4f}")

# a rough sense of the noise: bootstrap over series
ids = df.index.get_level_values("id").unique().to_numpy()
rng = np.random.default_rng(0)
groups = {i: g for i, g in df.groupby(level="id")}
boot = []
for _ in range(40):
    pick = rng.choice(ids, len(ids), replace=True)
    sub = pd.concat([groups[i] for i in pick])
    sub_step = np.concatenate([np.arange(len(groups[i])) for i in pick])
    boot.append(ts_auc(sub[col].to_numpy(), sub["target"].to_numpy(), sub_step))
print(f"bootstrap over series: mean {np.mean(boot):.4f}, std {np.std(boot):.4f}  (so roughly +/- {2 * np.std(boot):.3f})")
