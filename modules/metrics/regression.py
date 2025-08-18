# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any, Tuple
import numpy as np

def _safe_np(a) -> np.ndarray:
    return np.asarray(a, dtype=np.float64)

def mae_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    err = _safe_np(y_pred) - _safe_np(y_true)
    mae_vec = np.abs(err).mean(axis=0)
    rmse_vec = np.sqrt((err ** 2).mean(axis=0))
    mae_macro = float(mae_vec.mean())
    rmse_macro = float(rmse_vec.mean())
    return (mae_macro, mae_vec), (rmse_macro, rmse_vec)

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, np.ndarray]:
    y_true = _safe_np(y_true); y_pred = _safe_np(y_pred)
    num = ((y_true - y_pred) ** 2).sum(axis=0)
    den = ((y_true - y_true.mean(axis=0)) ** 2).sum(axis=0)
    # jeśli wariancja 0 -> R^2 = 0 (defensive)
    r2_vec = 1.0 - np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    r2_macro = float(r2_vec.mean())
    return r2_macro, r2_vec

def _rankdata(a: np.ndarray) -> np.ndarray:
    """Ranks with average for ties (Spearman-friendly), pure NumPy."""
    order = a.argsort(kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(a) + 1, dtype=np.float64)
    # average ties
    # find groups of equal values in sorted order
    sorted_a = a[order]
    i = 0
    while i < len(sorted_a):
        j = i
        while j + 1 < len(sorted_a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        if j > i:
            avg = (i + j + 2) / 2.0  # average of ranks i+1..j+1
            ranks[order[i:j+1]] = avg
        i = j + 1
    return ranks

def spearman_rho(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, np.ndarray]:
    y_true = _safe_np(y_true); y_pred = _safe_np(y_pred)
    n, d = y_true.shape
    rhos = np.zeros(d, dtype=np.float64)
    for k in range(d):
        rt = _rankdata(y_true[:, k])
        rp = _rankdata(y_pred[:, k])
        rt_c = rt - rt.mean()
        rp_c = rp - rp.mean()
        num = (rt_c * rp_c).sum()
        den = np.sqrt((rt_c ** 2).sum() * (rp_c ** 2).sum())
        rhos[k] = num / den if den > 0 else 0.0
    return float(rhos.mean()), rhos

def compute_all_metrics(y_true, y_pred) -> Dict[str, Any]:
    y_true = _safe_np(y_true); y_pred = _safe_np(y_pred)
    (mae_macro, mae_vec), (rmse_macro, rmse_vec) = mae_rmse(y_true, y_pred)
    r2_macro, r2_vec = r2_score(y_true, y_pred)
    sp_macro, sp_vec = spearman_rho(y_true, y_pred)
    return {
        "mae": mae_macro,
        "rmse": rmse_macro,
        "r2": r2_macro,
        "spearman": sp_macro,
        "mae_per_label": mae_vec.tolist(),
        "rmse_per_label": rmse_vec.tolist(),
        "r2_per_label": r2_vec.tolist(),
        "spearman_per_label": sp_vec.tolist(),
    }
