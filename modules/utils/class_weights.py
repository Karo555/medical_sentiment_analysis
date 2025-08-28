#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for computing class weights for imbalanced multi-label classification.
"""
from __future__ import annotations
import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional


def load_jsonl_labels(file_path: Path) -> np.ndarray:
    """Load labels from JSONL file."""
    labels = []
    with file_path.open('r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            labels.append(data['labels'])
    return np.array(labels, dtype=np.float32)


def compute_class_weights_per_label(
    y_true: np.ndarray,
    method: str = "balanced",
    smooth_factor: float = 1.0
) -> np.ndarray:
    """
    Compute class weights for multi-label binary classification.
    
    Args:
        y_true: Binary labels array of shape (n_samples, n_labels)
        method: Method for computing weights:
            - "balanced": n_samples / (2 * n_positives_per_label)
            - "inverse": 1 / positive_ratio_per_label  
            - "sqrt_inv": 1 / sqrt(positive_ratio_per_label)
        smooth_factor: Smoothing factor to prevent extreme weights
        
    Returns:
        Class weights tensor of shape (n_labels,) for positive class weights
    """
    n_samples, n_labels = y_true.shape
    pos_counts = np.sum(y_true, axis=0)  # Count of positive samples per label
    neg_counts = n_samples - pos_counts   # Count of negative samples per label
    
    # Handle edge cases: labels with 0 positive samples
    pos_counts_smooth = np.maximum(pos_counts, smooth_factor)
    pos_ratios = pos_counts_smooth / n_samples
    
    if method == "balanced":
        # Standard scikit-learn balanced weighting
        weights = n_samples / (2.0 * pos_counts_smooth)
    elif method == "inverse":
        # Inverse frequency weighting
        weights = 1.0 / pos_ratios
    elif method == "sqrt_inv":
        # Square root inverse frequency (less extreme)
        weights = 1.0 / np.sqrt(pos_ratios)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Cap extreme weights to prevent numerical instability
    max_weight = n_samples / smooth_factor  # Reasonable upper bound
    weights = np.minimum(weights, max_weight)
    
    return torch.tensor(weights, dtype=torch.float32)


def compute_class_weights_from_files(
    train_file: Path,
    method: str = "balanced",
    smooth_factor: float = 1.0,
    save_path: Optional[Path] = None
) -> Dict[str, torch.Tensor]:
    """
    Compute class weights from training data file.
    
    Args:
        train_file: Path to training JSONL file
        method: Weighting method
        smooth_factor: Smoothing factor
        save_path: Optional path to save weights
        
    Returns:
        Dictionary with computed weights and statistics
    """
    print(f"Computing class weights from: {train_file}")
    
    # Load training labels
    y_train = load_jsonl_labels(train_file)
    n_samples, n_labels = y_train.shape
    
    print(f"Training set: {n_samples} samples, {n_labels} labels")
    
    # Compute weights
    pos_weights = compute_class_weights_per_label(y_train, method, smooth_factor)
    
    # Compute statistics
    pos_counts = np.sum(y_train, axis=0)
    neg_counts = n_samples - pos_counts
    pos_ratios = pos_counts / n_samples
    imbalance_ratios = np.maximum(neg_counts, pos_counts) / np.maximum(np.minimum(neg_counts, pos_counts), 1)
    
    results = {
        'pos_weights': pos_weights,
        'method': method,
        'smooth_factor': smooth_factor,
        'n_samples': n_samples,
        'n_labels': n_labels,
        'statistics': {
            'pos_counts': pos_counts.tolist(),
            'neg_counts': neg_counts.tolist(),
            'pos_ratios': pos_ratios.tolist(),
            'imbalance_ratios': imbalance_ratios.tolist(),
            'weight_values': pos_weights.tolist()
        }
    }
    
    # Print summary
    print(f"\nClass Weights Summary ({method} method):")
    print(f"  Min weight: {pos_weights.min():.3f}")
    print(f"  Max weight: {pos_weights.max():.3f}")
    print(f"  Mean weight: {pos_weights.mean():.3f}")
    print(f"  Median weight: {pos_weights.median():.3f}")
    
    # Show top 5 most weighted labels
    sorted_indices = torch.argsort(pos_weights, descending=True)
    print(f"\nTop 5 highest weights (most imbalanced):")
    for i in range(min(5, len(sorted_indices))):
        idx = sorted_indices[i].item()
        weight = pos_weights[idx].item()
        pos_ratio = pos_ratios[idx]
        print(f"  Label {idx}: weight={weight:.2f}, pos_ratio={pos_ratio:.3f}")
    
    # Save if requested
    if save_path:
        save_data = {
            'pos_weights': pos_weights.tolist(),
            'method': method,
            'smooth_factor': smooth_factor,
            'n_samples': n_samples,
            'n_labels': n_labels,
            'statistics': results['statistics']
        }
        with save_path.open('w') as f:
            json.dump(save_data, f, indent=2)
        print(f"\nWeights saved to: {save_path}")
    
    return results


def load_class_weights(weights_file: Union[Path, str]) -> torch.Tensor:
    """Load pre-computed class weights from file."""
    weights_path = Path(weights_file)
    with weights_path.open('r') as f:
        data = json.load(f)
    return torch.tensor(data['pos_weights'], dtype=torch.float32)


def compare_weighting_methods(
    train_file: Path,
    methods: List[str] = ["balanced", "inverse", "sqrt_inv"],
    smooth_factor: float = 1.0
) -> Dict[str, Dict]:
    """Compare different class weighting methods."""
    
    print("Comparing class weighting methods...")
    y_train = load_jsonl_labels(train_file)
    
    results = {}
    for method in methods:
        print(f"\n--- {method.upper()} METHOD ---")
        weights_info = compute_class_weights_from_files(
            train_file, method, smooth_factor
        )
        results[method] = weights_info
    
    # Summary comparison
    print(f"\n{'='*60}")
    print("WEIGHTING METHODS COMPARISON")
    print(f"{'='*60}")
    print(f"{'Method':<12} {'Min':<8} {'Max':<8} {'Mean':<8} {'Median':<8}")
    print(f"{'-'*60}")
    
    for method, info in results.items():
        weights = info['pos_weights']
        print(f"{method:<12} {weights.min():.2f}   {weights.max():.2f}   "
              f"{weights.mean():.2f}   {weights.median():.2f}")
    
    return results


if __name__ == "__main__":
    # Example usage
    train_file = Path("data/processed/encoder/train.jsonl")
    
    if train_file.exists():
        # Compute and save balanced weights
        weights_dir = Path("artifacts/class_weights")
        weights_dir.mkdir(parents=True, exist_ok=True)
        
        compute_class_weights_from_files(
            train_file,
            method="balanced",
            save_path=weights_dir / "balanced_weights.json"
        )
        
        # Compare methods
        compare_weighting_methods(train_file)
    else:
        print(f"Training file not found: {train_file}")