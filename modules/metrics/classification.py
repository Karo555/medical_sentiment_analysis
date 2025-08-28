# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def _safe_np(a) -> np.ndarray:
    return np.asarray(a, dtype=np.float64)

def binary_accuracy(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> Tuple[float, np.ndarray]:
    """Compute accuracy for binary multi-label classification."""
    y_true_int = _safe_np(y_true).astype(np.int32)
    y_pred_bin = (_safe_np(y_pred) >= threshold).astype(np.int32)
    
    # Per-label accuracy
    acc_vec = np.zeros(y_true_int.shape[1], dtype=np.float64)
    for i in range(y_true_int.shape[1]):
        acc_vec[i] = accuracy_score(y_true_int[:, i], y_pred_bin[:, i])
    
    # Macro accuracy
    macro_acc = float(acc_vec.mean())
    return macro_acc, acc_vec


def binary_f1_score(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> Tuple[float, np.ndarray]:
    """Compute F1-score for binary multi-label classification."""
    y_true_int = _safe_np(y_true).astype(np.int32)
    y_pred_bin = (_safe_np(y_pred) >= threshold).astype(np.int32)
    
    # Per-label F1
    f1_vec = np.zeros(y_true_int.shape[1], dtype=np.float64)
    for i in range(y_true_int.shape[1]):
        # Handle edge cases where there are no positive samples
        if y_true_int[:, i].sum() == 0 and y_pred_bin[:, i].sum() == 0:
            f1_vec[i] = 1.0  # Perfect when both are all zeros
        elif y_true_int[:, i].sum() == 0 or y_pred_bin[:, i].sum() == 0:
            f1_vec[i] = 0.0  # Zero when one is all zeros and other is not
        else:
            f1_vec[i] = f1_score(y_true_int[:, i], y_pred_bin[:, i], average='binary', zero_division=0)
    
    # Macro F1
    macro_f1 = float(f1_vec.mean())
    return macro_f1, f1_vec


def binary_precision_recall(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> Tuple[Tuple[float, np.ndarray], Tuple[float, np.ndarray]]:
    """Compute precision and recall for binary multi-label classification."""
    y_true_int = _safe_np(y_true).astype(np.int32)
    y_pred_bin = (_safe_np(y_pred) >= threshold).astype(np.int32)
    
    # Per-label precision and recall
    prec_vec = np.zeros(y_true_int.shape[1], dtype=np.float64)
    rec_vec = np.zeros(y_true_int.shape[1], dtype=np.float64)
    
    for i in range(y_true_int.shape[1]):
        prec_vec[i] = precision_score(y_true_int[:, i], y_pred_bin[:, i], average='binary', zero_division=0)
        rec_vec[i] = recall_score(y_true_int[:, i], y_pred_bin[:, i], average='binary', zero_division=0)
    
    # Macro averages
    macro_prec = float(prec_vec.mean())
    macro_rec = float(rec_vec.mean())
    return (macro_prec, prec_vec), (macro_rec, rec_vec)


def compute_all_metrics(y_true, y_pred) -> Dict[str, Any]:
    y_true = _safe_np(y_true); y_pred = _safe_np(y_pred)
    
    # Binary classification metrics
    binary_acc_macro, binary_acc_vec = binary_accuracy(y_true, y_pred)
    binary_f1_macro, binary_f1_vec = binary_f1_score(y_true, y_pred)
    (binary_prec_macro, binary_prec_vec), (binary_rec_macro, binary_rec_vec) = binary_precision_recall(y_true, y_pred)
    
    return {
        # Primary binary classification metrics
        "accuracy": binary_acc_macro,
        "f1_score": binary_f1_macro,
        "precision": binary_prec_macro,
        "recall": binary_rec_macro,
        "accuracy_per_label": binary_acc_vec.tolist(),
        "f1_score_per_label": binary_f1_vec.tolist(),
        "precision_per_label": binary_prec_vec.tolist(),
        "recall_per_label": binary_rec_vec.tolist(),
    }
