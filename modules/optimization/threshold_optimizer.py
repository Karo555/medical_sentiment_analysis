#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Threshold optimization for multi-label binary classification.
Implements various strategies to optimize decision thresholds per label.
"""
from __future__ import annotations
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union, Callable
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import precision_recall_curve, roc_curve
import json
from pathlib import Path
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class ThresholdOptimizationConfig:
    """Configuration for threshold optimization."""
    metric: str = "f1_score"  # f1_score, precision, recall, accuracy, f1_macro
    search_strategy: str = "grid"  # grid, precision_recall_curve, roc_curve
    search_range: Tuple[float, float] = (0.1, 0.9)
    n_thresholds: int = 81  # For grid search
    min_positive_samples: int = 5  # Minimum positive samples needed for optimization
    default_threshold: float = 0.5  # Fallback threshold for labels with insufficient data
    class_weight_adjustment: bool = True  # Adjust thresholds based on class weights


class ThresholdOptimizer:
    """
    Multi-label threshold optimizer for binary classification tasks.
    Optimizes decision thresholds per label to maximize specified metrics.
    """
    
    def __init__(self, config: ThresholdOptimizationConfig = None):
        self.config = config or ThresholdOptimizationConfig()
        self.optimal_thresholds_ = None
        self.optimization_stats_ = None
    
    def _compute_metric(self, y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
        """Compute specified metric for binary predictions."""
        if len(np.unique(y_pred)) <= 1:
            return 0.0  # No positive predictions or all positive
        
        try:
            if metric == "f1_score":
                return f1_score(y_true, y_pred, zero_division=0.0)
            elif metric == "precision":
                return precision_score(y_true, y_pred, zero_division=0.0)
            elif metric == "recall":
                return recall_score(y_true, y_pred, zero_division=0.0)
            elif metric == "accuracy":
                return accuracy_score(y_true, y_pred)
            else:
                raise ValueError(f"Unknown metric: {metric}")
        except Exception:
            return 0.0
    
    def _grid_search_threshold(
        self, 
        y_true: np.ndarray, 
        y_proba: np.ndarray, 
        metric: str
    ) -> Tuple[float, float]:
        """Grid search for optimal threshold."""
        thresholds = np.linspace(
            self.config.search_range[0], 
            self.config.search_range[1], 
            self.config.n_thresholds
        )
        
        best_threshold = self.config.default_threshold
        best_score = 0.0
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            score = self._compute_metric(y_true, y_pred, metric)
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        return best_threshold, best_score
    
    def _precision_recall_curve_threshold(
        self, 
        y_true: np.ndarray, 
        y_proba: np.ndarray, 
        metric: str
    ) -> Tuple[float, float]:
        """Find optimal threshold using precision-recall curve."""
        try:
            precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
            
            if metric == "f1_score":
                # Compute F1 scores for each threshold
                f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
                best_idx = np.argmax(f1_scores)
                best_score = f1_scores[best_idx]
            elif metric == "precision":
                best_idx = np.argmax(precisions)
                best_score = precisions[best_idx]
            elif metric == "recall":
                best_idx = np.argmax(recalls)
                best_score = recalls[best_idx]
            else:
                return self._grid_search_threshold(y_true, y_proba, metric)
            
            if best_idx < len(thresholds):
                best_threshold = thresholds[best_idx]
            else:
                best_threshold = self.config.default_threshold
                
            return best_threshold, best_score
            
        except Exception:
            return self._grid_search_threshold(y_true, y_proba, metric)
    
    def _adjust_threshold_for_class_weights(
        self, 
        threshold: float, 
        class_weight: Optional[float] = None
    ) -> float:
        """Adjust threshold based on class weights."""
        if not self.config.class_weight_adjustment or class_weight is None:
            return threshold
        
        # Lower threshold for higher class weights (more imbalanced classes)
        if class_weight > 1.0:
            adjustment_factor = 1.0 / np.sqrt(class_weight)
            adjusted_threshold = threshold * adjustment_factor
            # Ensure adjusted threshold is within reasonable bounds
            adjusted_threshold = np.clip(adjusted_threshold, 0.1, 0.8)
            return adjusted_threshold
        
        return threshold
    
    def optimize_thresholds(
        self, 
        y_true: np.ndarray, 
        y_proba: np.ndarray,
        label_names: Optional[List[str]] = None,
        class_weights: Optional[np.ndarray] = None
    ) -> Dict[str, Union[np.ndarray, Dict]]:
        """
        Optimize thresholds for multi-label classification.
        
        Args:
            y_true: True binary labels, shape (n_samples, n_labels)
            y_proba: Predicted probabilities, shape (n_samples, n_labels)
            label_names: Optional list of label names
            class_weights: Optional class weights for threshold adjustment
            
        Returns:
            Dictionary containing optimal thresholds and optimization statistics
        """
        n_samples, n_labels = y_true.shape
        
        if label_names is None:
            label_names = [f"label_{i}" for i in range(n_labels)]
        
        optimal_thresholds = np.full(n_labels, self.config.default_threshold)
        optimization_stats = {}
        
        print(f"Optimizing thresholds for {n_labels} labels using {self.config.metric} metric")
        print(f"Strategy: {self.config.search_strategy}")
        
        for label_idx in tqdm(range(n_labels), desc="Optimizing thresholds"):
            label_name = label_names[label_idx]
            
            y_true_label = y_true[:, label_idx]
            y_proba_label = y_proba[:, label_idx]
            
            # Check if we have sufficient positive samples
            n_positive = np.sum(y_true_label)
            if n_positive < self.config.min_positive_samples:
                print(f"  {label_name}: Insufficient positive samples ({n_positive}), using default threshold")
                optimal_thresholds[label_idx] = self.config.default_threshold
                optimization_stats[label_name] = {
                    'threshold': self.config.default_threshold,
                    'score': 0.0,
                    'n_positive': int(n_positive),
                    'status': 'insufficient_data'
                }
                continue
            
            # Optimize threshold
            if self.config.search_strategy == "grid":
                best_threshold, best_score = self._grid_search_threshold(
                    y_true_label, y_proba_label, self.config.metric
                )
            elif self.config.search_strategy == "precision_recall_curve":
                best_threshold, best_score = self._precision_recall_curve_threshold(
                    y_true_label, y_proba_label, self.config.metric
                )
            else:
                best_threshold, best_score = self._grid_search_threshold(
                    y_true_label, y_proba_label, self.config.metric
                )
            
            # Adjust threshold based on class weights
            if class_weights is not None and label_idx < len(class_weights):
                class_weight = float(class_weights[label_idx])
                adjusted_threshold = self._adjust_threshold_for_class_weights(
                    best_threshold, class_weight
                )
            else:
                adjusted_threshold = best_threshold
                class_weight = 1.0
            
            optimal_thresholds[label_idx] = adjusted_threshold
            
            # Compute final score with adjusted threshold
            y_pred_adjusted = (y_proba_label >= adjusted_threshold).astype(int)
            final_score = self._compute_metric(y_true_label, y_pred_adjusted, self.config.metric)
            
            optimization_stats[label_name] = {
                'threshold': float(adjusted_threshold),
                'original_threshold': float(best_threshold),
                'score': float(final_score),
                'n_positive': int(n_positive),
                'class_weight': float(class_weight) if class_weights is not None else 1.0,
                'status': 'optimized'
            }
            
            print(f"  {label_name}: threshold={adjusted_threshold:.3f}, {self.config.metric}={final_score:.3f}")
        
        self.optimal_thresholds_ = optimal_thresholds
        self.optimization_stats_ = optimization_stats
        
        return {
            'optimal_thresholds': optimal_thresholds,
            'optimization_stats': optimization_stats,
            'config': {
                'metric': self.config.metric,
                'search_strategy': self.config.search_strategy,
                'n_thresholds': self.config.n_thresholds,
                'search_range': self.config.search_range
            }
        }
    
    def apply_thresholds(
        self, 
        y_proba: np.ndarray, 
        thresholds: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Apply optimized thresholds to probability predictions.
        
        Args:
            y_proba: Predicted probabilities, shape (n_samples, n_labels)
            thresholds: Optional custom thresholds, uses optimized if None
            
        Returns:
            Binary predictions with optimized thresholds applied
        """
        if thresholds is None:
            if self.optimal_thresholds_ is None:
                raise ValueError("No thresholds available. Run optimize_thresholds first.")
            thresholds = self.optimal_thresholds_
        
        # Ensure thresholds match the number of labels
        if len(thresholds) != y_proba.shape[1]:
            raise ValueError(f"Threshold count ({len(thresholds)}) doesn't match labels ({y_proba.shape[1]})")
        
        # Apply thresholds per label
        y_pred = np.zeros_like(y_proba, dtype=int)
        for label_idx in range(y_proba.shape[1]):
            y_pred[:, label_idx] = (y_proba[:, label_idx] >= thresholds[label_idx]).astype(int)
        
        return y_pred
    
    def save_thresholds(self, filepath: Union[str, Path]):
        """Save optimized thresholds and stats to JSON file."""
        if self.optimal_thresholds_ is None or self.optimization_stats_ is None:
            raise ValueError("No optimization results to save. Run optimize_thresholds first.")
        
        filepath = Path(filepath)
        
        save_data = {
            'optimal_thresholds': self.optimal_thresholds_.tolist(),
            'optimization_stats': self.optimization_stats_,
            'config': {
                'metric': self.config.metric,
                'search_strategy': self.config.search_strategy,
                'n_thresholds': self.config.n_thresholds,
                'search_range': self.config.search_range,
                'min_positive_samples': self.config.min_positive_samples,
                'default_threshold': self.config.default_threshold,
                'class_weight_adjustment': self.config.class_weight_adjustment
            }
        }
        
        with filepath.open('w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"Thresholds saved to: {filepath}")
    
    @classmethod
    def load_thresholds(cls, filepath: Union[str, Path]) -> Dict:
        """Load optimized thresholds from JSON file."""
        filepath = Path(filepath)
        
        with filepath.open('r', encoding='utf-8') as f:
            data = json.load(f)
        
        return {
            'optimal_thresholds': np.array(data['optimal_thresholds']),
            'optimization_stats': data['optimization_stats'],
            'config': data.get('config', {})
        }


def compare_threshold_strategies(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    label_names: Optional[List[str]] = None,
    class_weights: Optional[np.ndarray] = None
) -> Dict[str, Dict]:
    """
    Compare different threshold optimization strategies.
    
    Returns results for multiple optimization approaches.
    """
    strategies = [
        ("f1_score_grid", ThresholdOptimizationConfig(metric="f1_score", search_strategy="grid")),
        ("f1_score_pr_curve", ThresholdOptimizationConfig(metric="f1_score", search_strategy="precision_recall_curve")),
        ("precision_grid", ThresholdOptimizationConfig(metric="precision", search_strategy="grid")),
        ("recall_grid", ThresholdOptimizationConfig(metric="recall", search_strategy="grid")),
    ]
    
    results = {}
    
    for strategy_name, config in strategies:
        print(f"\n--- Testing strategy: {strategy_name} ---")
        
        optimizer = ThresholdOptimizer(config)
        result = optimizer.optimize_thresholds(y_true, y_proba, label_names, class_weights)
        
        # Evaluate performance with optimized thresholds
        y_pred_optimized = optimizer.apply_thresholds(y_proba)
        
        # Compute overall metrics
        overall_metrics = {}
        for metric_name in ["f1_score", "precision", "recall", "accuracy"]:
            if metric_name == "f1_score":
                overall_metrics[metric_name] = f1_score(y_true, y_pred_optimized, average='macro', zero_division=0.0)
            elif metric_name == "precision":
                overall_metrics[metric_name] = precision_score(y_true, y_pred_optimized, average='macro', zero_division=0.0)
            elif metric_name == "recall":
                overall_metrics[metric_name] = recall_score(y_true, y_pred_optimized, average='macro', zero_division=0.0)
            elif metric_name == "accuracy":
                # For multi-label, use sample-wise accuracy
                overall_metrics[metric_name] = np.mean(np.all(y_true == y_pred_optimized, axis=1))
        
        results[strategy_name] = {
            'thresholds': result['optimal_thresholds'],
            'optimization_stats': result['optimization_stats'],
            'overall_metrics': overall_metrics,
            'config': result['config']
        }
        
        print(f"Overall F1: {overall_metrics['f1_score']:.4f}, "
              f"Precision: {overall_metrics['precision']:.4f}, "
              f"Recall: {overall_metrics['recall']:.4f}")
    
    return results


if __name__ == "__main__":
    # Example usage and testing
    np.random.seed(42)
    
    # Generate sample data
    n_samples, n_labels = 1000, 5
    y_true = np.random.randint(0, 2, (n_samples, n_labels))
    y_proba = np.random.beta(2, 5, (n_samples, n_labels))  # Skewed probabilities
    
    label_names = [f"emotion_{i}" for i in range(n_labels)]
    
    # Test basic optimization
    print("Testing basic threshold optimization...")
    optimizer = ThresholdOptimizer()
    results = optimizer.optimize_thresholds(y_true, y_proba, label_names)
    
    print("\nOptimal thresholds:", results['optimal_thresholds'])
    
    # Test with predictions
    y_pred_default = (y_proba >= 0.5).astype(int)
    y_pred_optimized = optimizer.apply_thresholds(y_proba)
    
    f1_default = f1_score(y_true, y_pred_default, average='macro', zero_division=0.0)
    f1_optimized = f1_score(y_true, y_pred_optimized, average='macro', zero_division=0.0)
    
    print(f"\nF1 Score - Default (0.5): {f1_default:.4f}")
    print(f"F1 Score - Optimized: {f1_optimized:.4f}")
    print(f"Improvement: {((f1_optimized - f1_default) / f1_default) * 100:.1f}%")