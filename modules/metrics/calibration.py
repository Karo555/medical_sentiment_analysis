# modules/metrics/calibration.py
# -*- coding: utf-8 -*-
"""
Calibration utilities for bounded regression (targets and predictions in [0,1]).
Implements reliability curves and Expected Calibration Error (ECE) per label.

Definitions (per label k):
- Split predictions into bins B_1..B_M (uniform or quantile).
- For each bin b:
    p̄_b = mean(pred ∈ b)               # average predicted value in the bin
    ȳ_b = mean(true  ∈ b)               # average true value for examples in the bin
    w_b  = |b| / N                       # bin weight
- ECE_k   = Σ_b w_b * |p̄_b - ȳ_b|
- RMCE_k  = sqrt( Σ_b w_b * (p̄_b - ȳ_b)^2 )

This module computes per-label curves and aggregates (macro over labels).

Usage (N×D arrays):
    from modules.metrics.calibration import compute_calibration_summary
    summary = compute_calibration_summary(y_true, y_pred, n_bins=15, strategy="uniform")
    # summary["ece_macro"], summary["rmce_macro"], summary["per_label"][k]["curve"]

Notes:
- Inputs are defensively clamped to [0,1].
- Labels with no samples in a bin simply skip that bin (weight 0).
- Quantile binning guarantees (near) equal counts per bin (except ties).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Literal, Optional, Tuple
import numpy as np


BinStrategy = Literal["uniform", "quantile"]


@dataclass
class ReliabilityCurve:
    """Container for a single label's reliability data."""
    bin_edges: np.ndarray          # shape [M+1]
    bin_centers: np.ndarray        # shape [M]
    counts: np.ndarray             # shape [M]
    mean_pred: np.ndarray          # shape [M]
    mean_true: np.ndarray          # shape [M]
    ece: float
    rmce: float


def _clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x.astype(np.float64, copy=False), 0.0, 1.0)


def _uniform_edges(n_bins: int) -> np.ndarray:
    # edges from 0.0 to 1.0 inclusive
    return np.linspace(0.0, 1.0, num=n_bins + 1, dtype=np.float64)


def _quantile_edges(pred: np.ndarray, n_bins: int) -> np.ndarray:
    # robust quantiles with unique edges, always include 0.0 and 1.0
    qs = np.linspace(0.0, 1.0, num=n_bins + 1, dtype=np.float64)
    edges = np.quantile(pred, qs, method="linear")
    # enforce monotonic, unique and bounds [0,1]
    edges = np.clip(edges, 0.0, 1.0)
    # guard against identical edges (all preds equal) → fall back to uniform
    if np.unique(edges).size < 2:
        return _uniform_edges(n_bins)
    # ensure strictly increasing by small eps if necessary
    eps = 1e-12
    for i in range(1, edges.size):
        if edges[i] <= edges[i - 1]:
            edges[i] = min(1.0, edges[i - 1] + eps)
    edges[0] = 0.0
    edges[-1] = 1.0
    return edges


def _bin_indices(pred: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """
    Assigns each prediction to a bin index in [0, M-1].
    Rightmost edge is inclusive.
    """
    # searchsorted places x in [edges[i], edges[i+1])
    idx = np.searchsorted(edges, pred, side="right") - 1
    idx = np.clip(idx, 0, edges.size - 2)
    # Ensure exactly 1.0 falls into last bin
    idx[pred >= edges[-1]] = edges.size - 2
    return idx


def reliability_curve_for_label(
    y_true_k: np.ndarray,
    y_pred_k: np.ndarray,
    n_bins: int = 15,
    strategy: BinStrategy = "uniform",
) -> ReliabilityCurve:
    """
    Compute reliability curve and calibration errors for one label (length N vectors).
    """
    y_true_k = _clamp01(np.asarray(y_true_k))
    y_pred_k = _clamp01(np.asarray(y_pred_k))
    assert y_true_k.shape == y_pred_k.shape, "y_true and y_pred must have the same shape for a single label."

    # Choose bin edges
    if strategy == "uniform":
        edges = _uniform_edges(n_bins)
    elif strategy == "quantile":
        edges = _quantile_edges(y_pred_k, n_bins)
    else:
        raise ValueError(f"Unknown binning strategy: {strategy}")

    M = edges.size - 1
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx = _bin_indices(y_pred_k, edges)

    mean_pred = np.zeros(M, dtype=np.float64)
    mean_true = np.zeros(M, dtype=np.float64)
    counts = np.zeros(M, dtype=np.int64)

    # Aggregate per bin
    for b in range(M):
        mask = (idx == b)
        cnt = int(mask.sum())
        counts[b] = cnt
        if cnt > 0:
            mean_pred[b] = float(y_pred_k[mask].mean())
            mean_true[b] = float(y_true_k[mask].mean())
        else:
            mean_pred[b] = np.nan
            mean_true[b] = np.nan

    # Weights (ignore empty bins)
    N = float(y_true_k.shape[0])
    non_empty = counts > 0
    w = counts[non_empty].astype(np.float64) / max(N, 1.0)
    diff = np.abs(mean_pred[non_empty] - mean_true[non_empty])
    ece = float(np.sum(w * diff))
    rmce = float(np.sqrt(np.sum(w * (diff ** 2))))

    # For plotting convenience, replace NaNs with 0 in curves (counts indicate empties anyway)
    mean_pred_plot = np.where(np.isnan(mean_pred), 0.0, mean_pred)
    mean_true_plot = np.where(np.isnan(mean_true), 0.0, mean_true)

    return ReliabilityCurve(
        bin_edges=edges,
        bin_centers=centers,
        counts=counts,
        mean_pred=mean_pred_plot,
        mean_true=mean_true_plot,
        ece=ece,
        rmce=rmce,
    )


def compute_calibration_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 15,
    strategy: BinStrategy = "uniform",
) -> Dict[str, Any]:
    """
    Compute calibration for all labels.
    Inputs:
        y_true: (N, D) float in [0,1]
        y_pred: (N, D) float in [0,1]
    Returns dict with macro metrics and per-label curves.
    """
    y_true = _clamp01(np.asarray(y_true))
    y_pred = _clamp01(np.asarray(y_pred))
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    if y_true.ndim != 2:
        raise ValueError("Expected 2D arrays (N, D).")

    N, D = y_true.shape
    per_label: list[Dict[str, Any]] = []
    eces = np.zeros(D, dtype=np.float64)
    rmces = np.zeros(D, dtype=np.float64)

    for k in range(D):
        rc = reliability_curve_for_label(y_true[:, k], y_pred[:, k], n_bins=n_bins, strategy=strategy)
        eces[k] = rc.ece
        rmces[k] = rc.rmce
        per_label.append({
            "label_index": int(k),
            "curve": {
                "bin_edges": rc.bin_edges.tolist(),
                "bin_centers": rc.bin_centers.tolist(),
                "counts": rc.counts.tolist(),
                "mean_pred": rc.mean_pred.tolist(),
                "mean_true": rc.mean_true.tolist(),
            },
            "ece": rc.ece,
            "rmce": rc.rmce,
        })

    summary: Dict[str, Any] = {
        "n_samples": int(N),
        "n_labels": int(D),
        "n_bins": int(n_bins),
        "strategy": strategy,
        "ece_macro": float(eces.mean()),
        "ece_per_label": eces.tolist(),
        "rmce_macro": float(rmces.mean()),
        "rmce_per_label": rmces.tolist(),
        "per_label": per_label,
    }
    return summary


def save_calibration_summary(path: str, summary: Dict[str, Any]) -> None:
    """Save summary as JSON."""
    import json, os
    os.makedirs(str(__import__("os").path.dirname(path) or "."), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


# -------- Convenience wrapper for quick ECE only --------

def expected_calibration_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 15,
    strategy: BinStrategy = "uniform",
    reduce: Literal["macro", "none"] = "macro",
) -> np.ndarray | float:
    """
    Quick ECE computation. If reduce="macro" → float (mean across labels); else returns per-label array [D].
    """
    summary = compute_calibration_summary(y_true, y_pred, n_bins=n_bins, strategy=strategy)
    if reduce == "none":
        return np.asarray(summary["ece_per_label"], dtype=np.float64)
    return float(summary["ece_macro"])