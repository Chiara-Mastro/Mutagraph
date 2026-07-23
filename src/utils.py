import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from src.CONFIG import EPS


def choose_device(device_arg: str):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def ensure_dir(path: str) :
    os.makedirs(path, exist_ok=True)


def normalize_status(value: object):
    return str(value).strip().lower().replace(" ", "_")

def pick_column(df, preferred, candidates, label):
    """Pick a column name from a preferred name or candidate aliases."""
    if preferred is not None:
        if preferred not in df.columns:
            raise ValueError(f"Requested {label} column '{preferred}' not found. Available columns: {list(df.columns)}")
        return preferred
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"Could not infer {label} column. Tried {list(candidates)}. "
        f"Available columns: {list(df.columns)}"
    )


def clip_prob(p: np.ndarray | pd.Series | float):
    return np.clip(p, EPS, 1.0 - EPS)


def logit(p: np.ndarray | float):
    p = clip_prob(p)
    return np.log(p / (1.0 - p))


def sigmoid(x: np.ndarray | float) :
    return 1.0 / (1.0 + np.exp(-x))


def weighted_mean(x: np.ndarray, w: np.ndarray):
    mask = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not np.any(mask):
        return float("nan")
    return float(np.sum(w[mask] * x[mask]) / np.sum(w[mask]))


def safe_divide(num: float, den: float, default: float = np.nan):
    return float(num / den) if den and np.isfinite(den) else default


def make_bins(series: pd.Series, mode: str = "auto", n_bins: int = 5) :
    """Create stable categorical bins for diagnostics."""
    x = pd.to_numeric(series, errors="coerce")
    if x.notna().sum() == 0:
        return pd.Series(["missing"] * len(series), index=series.index, dtype="object")

    if mode == "integer":
        out = x.astype("Int64").astype(str)
        out[x.isna()] = "missing"
        return out

    # If small integer support, keep exact values.
    non_null = x.dropna()
    unique_vals = np.sort(non_null.unique())
    if len(unique_vals) <= 8 and np.allclose(unique_vals, np.round(unique_vals)):
        out = x.astype("Int64").astype(str)
        out[x.isna()] = "missing"
        return out

    try:
        binned = pd.qcut(x, q=min(n_bins, x.nunique()), duplicates="drop")
        out = binned.astype(str)
        out[x.isna()] = "missing"
        return out
    except Exception:
        binned = pd.cut(x, bins=min(n_bins, max(1, x.nunique())), duplicates="drop")
        out = binned.astype(str)
        out[x.isna()] = "missing"
        return out

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)