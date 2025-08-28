#!/usr/bin/env python3

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional

from modules.optimization.threshold_optimizer import ThresholdOptimizer, ThresholdOptimizationConfig, compare_threshold_strategies
from modules.data.datasets import EncoderDataset
from modules.models.encoder_classifier import EncoderClassifier
from modules.metrics.classification import compute_all_metrics
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


def load_model_and_data(checkpoint_path: str, data_path: str) -> tuple:
    """Load trained model and dataset."""
    print(f"Loading model from: {checkpoint_path}")
    print(f"Loading data from: {data_path}")
    
    # Load model
    model = EncoderClassifier.from_pretrained(checkpoint_path)
    model.eval()
    
    # Load dataset
    dataset = EncoderDataset.from_jsonl(
        data_path=data_path,
        tokenizer=model.tokenizer,
        max_length=512
    )
    
    return model, dataset


def get_model_predictions(model: EncoderClassifier, dataset: EncoderDataset) -> tuple:
    """Get model predictions and ground truth labels."""
    print("Getting model predictions...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    all_probs = []
    all_labels = []
    
    # Use DataLoader for batch processing
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].cpu().numpy()
            
            # Get model outputs (logits)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Convert logits to probabilities
            probabilities = torch.sigmoid(logits).cpu().numpy()
            
            all_probs.append(probabilities)
            all_labels.append(labels)
    
    # Concatenate all batches
    y_proba = np.vstack(all_probs)
    y_true = np.vstack(all_labels)
    
    print(f"Predictions shape: {y_proba.shape}")
    print(f"Labels shape: {y_true.shape}")
    
    return y_true, y_proba


def evaluate_with_thresholds(y_true: np.ndarray, y_proba: np.ndarray, 
                           thresholds: np.ndarray, label_names: List[str]) -> Dict:
    """Evaluate model performance with given thresholds."""
    # Apply thresholds
    y_pred = np.zeros_like(y_proba, dtype=int)
    for i in range(len(thresholds)):
        y_pred[:, i] = (y_proba[:, i] >= thresholds[i]).astype(int)
    
    # Compute overall metrics
    overall_metrics = {
        'accuracy': np.mean(np.all(y_true == y_pred, axis=1)),  # Sample-wise accuracy
        'f1_score': f1_score(y_true, y_pred, average='macro', zero_division=0.0),
        'precision': precision_score(y_true, y_pred, average='macro', zero_division=0.0),
        'recall': recall_score(y_true, y_pred, average='macro', zero_division=0.0)
    }
    
    # Compute per-label metrics
    per_label_metrics = {}
    for i, label_name in enumerate(label_names):
        per_label_metrics[label_name] = {
            'threshold': float(thresholds[i]),
            'accuracy': accuracy_score(y_true[:, i], y_pred[:, i]),
            'f1_score': f1_score(y_true[:, i], y_pred[:, i], zero_division=0.0),
            'precision': precision_score(y_true[:, i], y_pred[:, i], zero_division=0.0),
            'recall': recall_score(y_true[:, i], y_pred[:, i], zero_division=0.0),
            'n_positive': int(np.sum(y_true[:, i])),
            'n_predicted': int(np.sum(y_pred[:, i]))
        }
    
    return {
        'overall_metrics': overall_metrics,
        'per_label_metrics': per_label_metrics
    }


def main():
    parser = argparse.ArgumentParser(description="Optimize classification thresholds")
    parser.add_argument('--checkpoint', required=True, help='Path to trained model checkpoint')
    parser.add_argument('--data', required=True, help='Path to evaluation data (JSONL)')
    parser.add_argument('--output-dir', required=True, help='Output directory for results')
    parser.add_argument('--metric', default='f1_score', choices=['f1_score', 'precision', 'recall', 'accuracy'],
                       help='Metric to optimize for')
    parser.add_argument('--strategy', default='grid', choices=['grid', 'precision_recall_curve'],
                       help='Optimization strategy')
    parser.add_argument('--compare-strategies', action='store_true',
                       help='Compare multiple optimization strategies')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model and data
    model, dataset = load_model_and_data(args.checkpoint, args.data)
    
    # Get predictions
    y_true, y_proba = get_model_predictions(model, dataset)
    
    # Get label names (18D emotion labels)
    label_names = [
        "positive", "negative", "happiness", "delight", "inspiring", "calm",
        "surprise", "compassion", "fear", "sadness", "disgust", "anger",
        "ironic", "political", "interesting", "understandable", "offensive", "funny"
    ]
    
    print(f"\nEvaluating with {len(label_names)} labels")
    
    # Evaluate with default thresholds (0.5)
    default_thresholds = np.full(len(label_names), 0.5)
    default_results = evaluate_with_thresholds(y_true, y_proba, default_thresholds, label_names)
    
    print("\n" + "="*60)
    print("BASELINE PERFORMANCE (0.5 thresholds)")
    print("="*60)
    print(f"Overall Accuracy: {default_results['overall_metrics']['accuracy']:.4f}")
    print(f"Overall F1-Score: {default_results['overall_metrics']['f1_score']:.4f}")
    print(f"Overall Precision: {default_results['overall_metrics']['precision']:.4f}")
    print(f"Overall Recall: {default_results['overall_metrics']['recall']:.4f}")
    
    if args.compare_strategies:
        print("\n" + "="*60)
        print("COMPARING THRESHOLD OPTIMIZATION STRATEGIES")
        print("="*60)
        
        # Compare multiple strategies
        strategy_results = compare_threshold_strategies(
            y_true, y_proba, label_names
        )
        
        # Save strategy comparison
        comparison_path = output_dir / "strategy_comparison.json"
        with comparison_path.open('w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = {}
            for strategy, result in strategy_results.items():
                serializable_results[strategy] = {
                    'thresholds': result['thresholds'].tolist(),
                    'optimization_stats': result['optimization_stats'],
                    'overall_metrics': result['overall_metrics'],
                    'config': result['config']
                }
            json.dump(serializable_results, f, indent=2)
        
        print(f"Strategy comparison saved to: {comparison_path}")
        
        # Find best strategy
        best_strategy = None
        best_f1 = 0.0
        for strategy, result in strategy_results.items():
            f1 = result['overall_metrics']['f1_score']
            if f1 > best_f1:
                best_f1 = f1
                best_strategy = strategy
        
        print(f"\nBest strategy: {best_strategy} (F1: {best_f1:.4f})")
        optimized_thresholds = strategy_results[best_strategy]['thresholds']
        
    else:
        print(f"\n" + "="*60)
        print(f"OPTIMIZING THRESHOLDS FOR {args.metric.upper()}")
        print("="*60)
        
        # Single strategy optimization
        config = ThresholdOptimizationConfig(
            metric=args.metric,
            search_strategy=args.strategy
        )
        
        optimizer = ThresholdOptimizer(config)
        optimization_results = optimizer.optimize_thresholds(
            y_true, y_proba, label_names
        )
        
        optimized_thresholds = optimization_results['optimal_thresholds']
        
        # Save optimization results
        threshold_path = output_dir / f"optimized_thresholds_{args.metric}_{args.strategy}.json"
        optimizer.save_thresholds(threshold_path)
    
    # Evaluate with optimized thresholds
    optimized_results = evaluate_with_thresholds(y_true, y_proba, optimized_thresholds, label_names)
    
    print("\n" + "="*60)
    print("OPTIMIZED THRESHOLD PERFORMANCE")
    print("="*60)
    print(f"Overall Accuracy: {optimized_results['overall_metrics']['accuracy']:.4f}")
    print(f"Overall F1-Score: {optimized_results['overall_metrics']['f1_score']:.4f}")
    print(f"Overall Precision: {optimized_results['overall_metrics']['precision']:.4f}")
    print(f"Overall Recall: {optimized_results['overall_metrics']['recall']:.4f}")
    
    # Calculate improvements
    print("\n" + "="*60)
    print("IMPROVEMENT ANALYSIS")
    print("="*60)
    
    accuracy_improvement = ((optimized_results['overall_metrics']['accuracy'] - 
                           default_results['overall_metrics']['accuracy']) / 
                          default_results['overall_metrics']['accuracy']) * 100
    
    f1_improvement = ((optimized_results['overall_metrics']['f1_score'] - 
                      default_results['overall_metrics']['f1_score']) / 
                     max(default_results['overall_metrics']['f1_score'], 1e-8)) * 100
    
    precision_improvement = ((optimized_results['overall_metrics']['precision'] - 
                            default_results['overall_metrics']['precision']) / 
                           max(default_results['overall_metrics']['precision'], 1e-8)) * 100
    
    recall_improvement = ((optimized_results['overall_metrics']['recall'] - 
                         default_results['overall_metrics']['recall']) / 
                        max(default_results['overall_metrics']['recall'], 1e-8)) * 100
    
    print(f"Accuracy: {accuracy_improvement:+.1f}%")
    print(f"F1-Score: {f1_improvement:+.1f}%")
    print(f"Precision: {precision_improvement:+.1f}%")
    print(f"Recall: {recall_improvement:+.1f}%")
    
    # Per-label improvements
    print("\n" + "="*40)
    print("PER-LABEL IMPROVEMENTS")
    print("="*40)
    
    improved_labels = []
    for label in label_names:
        default_f1 = default_results['per_label_metrics'][label]['f1_score']
        optimized_f1 = optimized_results['per_label_metrics'][label]['f1_score']
        
        if optimized_f1 > default_f1:
            improvement = ((optimized_f1 - default_f1) / max(default_f1, 1e-8)) * 100
            improved_labels.append((label, improvement, optimized_f1))
            threshold = optimized_results['per_label_metrics'][label]['threshold']
            print(f"{label:>15}: F1={optimized_f1:.3f} (threshold={threshold:.3f}) +{improvement:.1f}%")
    
    # Save detailed results
    results_summary = {
        'default_performance': default_results,
        'optimized_performance': optimized_results,
        'optimized_thresholds': optimized_thresholds.tolist(),
        'improvements': {
            'accuracy': accuracy_improvement,
            'f1_score': f1_improvement,
            'precision': precision_improvement,
            'recall': recall_improvement
        },
        'improved_labels': improved_labels,
        'optimization_config': {
            'metric': args.metric,
            'strategy': args.strategy
        }
    }
    
    results_path = output_dir / "threshold_optimization_results.json"
    with results_path.open('w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_path}")
    print(f"Labels with improved F1-scores: {len(improved_labels)}/{len(label_names)}")


if __name__ == "__main__":
    main()