"""Load the Crunch parquet files into per-series numpy arrays and cache them."""
import os
import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CACHE = os.path.join(DATA, "train_cache.npz")


def _split(X):
    """X has MultiIndex (id, time) and columns value, period (1 = history, 2 = online)."""
    ids = X.index.get_level_values("id").to_numpy()
    vals = X["value"].to_numpy(dtype=np.float64)
    per = X["period"].to_numpy()
    # rows are sorted by id then time
    bounds = np.flatnonzero(np.diff(ids)) + 1
    starts = np.r_[0, bounds]
    ends = np.r_[bounds, len(ids)]
    out = []
    for s, e in zip(starts, ends):
        p = per[s:e]
        nh = int((p == 1).sum())
        out.append((int(ids[s]), vals[s:s + nh].copy(), vals[s + nh:e].copy()))
    return out


def load_train():
    """Returns list of (id, hist, online) and array tau_index (-1 = no break)."""
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        return list(z["series"]), z["tau"]
    X = pd.read_parquet(os.path.join(DATA, "X_train.parquet"))
    yi = pd.read_parquet(os.path.join(DATA, "y_train_index.parquet"))
    series = _split(X)
    tau = yi.loc[[s[0] for s in series], "tau_index"].to_numpy()
    arr = np.empty(len(series), dtype=object)
    for i, s in enumerate(series):
        arr[i] = s
    np.savez(CACHE, series=arr, tau=tau)
    return series, tau


def load_test_reduced():
    X = pd.read_parquet(os.path.join(DATA, "X_test.reduced.parquet"))
    y = pd.read_parquet(os.path.join(DATA, "y_test.reduced.parquet"))["target"]
    return _split(X), y
