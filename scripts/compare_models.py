#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare performance between different encoder models.
Creates side-by-side visualizations and statistical comparisons.
"""
from __future__ import annotations
import argparse
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def load_model_results(checkpoint_path: Path, split: str = 'val') -> Dict[str, Any]:
    """Load model evaluation results"""
    results_file = checkpoint_path / f'eval_results_{split}.json'
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    with open(results_file, 'r') as f:
        return json.load(f)

def load_model_summary(checkpoint_path: Path) -> Dict[str, Any]:
    """Load model performance summary"""
    summary_file = checkpoint_path / 'performance_summary.json'
    if not summary_file.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_file}")
    
    with open(summary_file, 'r') as f:
        return json.load(f)

def compare_overall_metrics(models_data: Dict[str, Dict], output_dir: Path) -> None:
    """Compare overall performance metrics between models"""
    model_names = list(models_data.keys())
    metrics = ['R² Score', 'MAE', 'RMSE', 'Spearman ρ']
    
    # Extract overall performance data
    comparison_data = {}
    for metric in metrics:
        comparison_data[metric] = []
        for model in model_names:
            summary = models_data[model]['summary']
            comparison_data[metric].append(summary['Overall Performance'][metric])
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'plum']
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        bars = ax.bar(model_names, comparison_data[metric], color=colors[i], alpha=0.8, edgecolor='black')
        
        # Add value labels on bars
        for bar, value in zip(bars, comparison_data[metric]):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(comparison_data[metric]) * 0.02,
                   f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title(f'{metric} Comparison', fontsize=14, fontweight='bold')
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Highlight better performance
        best_idx = np.argmax(comparison_data[metric]) if metric in ['R² Score', 'Spearman ρ'] else np.argmin(comparison_data[metric])
        bars[best_idx].set_color('gold')
        bars[best_idx].set_alpha(1.0)
    
    plt.suptitle('Overall Model Performance Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'overall_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def compare_per_label_performance(models_data: Dict[str, Dict], label_names: List[str], output_dir: Path) -> None:
    """Compare per-label performance between models"""
    model_names = list(models_data.keys())
    
    # Prepare data for comparison
    r2_data = {}
    mae_data = {}
    spearman_data = {}
    
    for model in model_names:
        summary = models_data[model]['summary']
        per_label = summary['Per-Label Performance']
        
        r2_data[model] = [per_label[label]['R² Score'] for label in label_names]
        mae_data[model] = [per_label[label]['MAE'] for label in label_names]
        spearman_data[model] = [per_label[label]['Spearman ρ'] for label in label_names]
    
    # Create comparison plots
    fig, axes = plt.subplots(3, 1, figsize=(16, 18))
    
    # R² Score comparison
    ax1 = axes[0]
    x = np.arange(len(label_names))
    width = 0.35
    
    for i, model in enumerate(model_names):
        offset = (i - len(model_names)/2 + 0.5) * width
        bars = ax1.bar(x + offset, r2_data[model], width, label=model, alpha=0.8)
        
    ax1.set_xlabel('Emotion Labels')
    ax1.set_ylabel('R² Score')
    ax1.set_title('R² Score per Emotion Label - Model Comparison', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(label_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # MAE comparison
    ax2 = axes[1]
    for i, model in enumerate(model_names):
        offset = (i - len(model_names)/2 + 0.5) * width
        bars = ax2.bar(x + offset, mae_data[model], width, label=model, alpha=0.8)
        
    ax2.set_xlabel('Emotion Labels')
    ax2.set_ylabel('MAE')
    ax2.set_title('MAE per Emotion Label - Model Comparison', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(label_names, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Spearman correlation comparison
    ax3 = axes[2]
    for i, model in enumerate(model_names):
        offset = (i - len(model_names)/2 + 0.5) * width
        bars = ax3.bar(x + offset, spearman_data[model], width, label=model, alpha=0.8)
        
    ax3.set_xlabel('Emotion Labels')
    ax3.set_ylabel('Spearman ρ')
    ax3.set_title('Spearman Correlation per Emotion Label - Model Comparison', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(label_names, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'per_label_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def statistical_significance_tests(models_data: Dict[str, Dict], output_dir: Path) -> None:
    """Perform statistical significance tests between models"""
    model_names = list(models_data.keys())
    if len(model_names) != 2:
        print("Statistical tests require exactly 2 models")
        return
    
    model1_name, model2_name = model_names
    
    # Get predictions and labels
    model1_preds = np.array(models_data[model1_name]['results']['predictions'])
    model1_labels = np.array(models_data[model1_name]['results']['labels'])
    model2_preds = np.array(models_data[model2_name]['results']['predictions'])
    model2_labels = np.array(models_data[model2_name]['results']['labels'])
    
    # Ensure same labels (should be the same anyway)
    assert np.array_equal(model1_labels, model2_labels), "Labels don't match between models"
    
    labels = model1_labels
    n_labels = labels.shape[1]
    
    # Calculate per-sample errors
    model1_errors = np.abs(model1_preds - labels)
    model2_errors = np.abs(model2_preds - labels)
    
    # Statistical tests
    test_results = {}
    
    # Overall comparison (mean absolute error per sample)
    model1_mae_per_sample = model1_errors.mean(axis=1)
    model2_mae_per_sample = model2_errors.mean(axis=1)
    
    # Paired t-test for overall MAE
    overall_t_stat, overall_p_val = stats.ttest_rel(model1_mae_per_sample, model2_mae_per_sample)
    
    test_results['overall'] = {
        'model1_mean_mae': float(model1_mae_per_sample.mean()),
        'model2_mean_mae': float(model2_mae_per_sample.mean()),
        't_statistic': float(overall_t_stat),
        'p_value': float(overall_p_val),
        'significant': float(overall_p_val) < 0.05
    }
    
    # Per-label comparisons
    test_results['per_label'] = {}
    for i in range(n_labels):
        t_stat, p_val = stats.ttest_rel(model1_errors[:, i], model2_errors[:, i])
        test_results['per_label'][f'label_{i}'] = {
            'model1_mean_error': float(model1_errors[:, i].mean()),
            'model2_mean_error': float(model2_errors[:, i].mean()),
            't_statistic': float(t_stat),
            'p_value': float(p_val),
            'significant': float(p_val) < 0.05
        }
    
    # Wilcoxon signed-rank test (non-parametric)
    wilcoxon_stat, wilcoxon_p = stats.wilcoxon(model1_mae_per_sample, model2_mae_per_sample, 
                                               alternative='two-sided')
    
    test_results['wilcoxon'] = {
        'statistic': float(wilcoxon_stat),
        'p_value': float(wilcoxon_p),
        'significant': float(wilcoxon_p) < 0.05
    }
    
    # Save results
    with open(output_dir / 'significance_tests.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    
    # Create summary
    print(f"\nStatistical Significance Tests: {model1_name} vs {model2_name}")
    print("="*60)
    print(f"Overall MAE - {model1_name}: {test_results['overall']['model1_mean_mae']:.4f}")
    print(f"Overall MAE - {model2_name}: {test_results['overall']['model2_mean_mae']:.4f}")
    print(f"Paired t-test p-value: {test_results['overall']['p_value']:.6f}")
    print(f"Statistically significant: {'Yes' if test_results['overall']['significant'] else 'No'}")
    print(f"Wilcoxon test p-value: {test_results['wilcoxon']['p_value']:.6f}")
    
    significant_labels = sum(1 for label_data in test_results['per_label'].values() if label_data['significant'])
    print(f"Significantly different labels: {significant_labels}/{n_labels}")

def create_performance_heatmap(models_data: Dict[str, Dict], label_names: List[str], output_dir: Path) -> None:
    """Create heatmap showing performance across models and labels"""
    model_names = list(models_data.keys())
    
    # Create R² score matrix
    r2_matrix = []
    for model in model_names:
        summary = models_data[model]['summary']
        per_label = summary['Per-Label Performance']
        r2_scores = [per_label[label]['R² Score'] for label in label_names]
        r2_matrix.append(r2_scores)
    
    r2_matrix = np.array(r2_matrix)
    
    # Create heatmap
    plt.figure(figsize=(20, 6))
    
    # Custom colormap: red for negative, white around 0, green for positive
    cmap = sns.diverging_palette(10, 150, as_cmap=True, center='light')
    
    sns.heatmap(r2_matrix, 
                xticklabels=label_names,
                yticklabels=model_names,
                annot=True, 
                fmt='.3f',
                cmap=cmap,
                center=0,
                cbar_kws={'label': 'R² Score'},
                square=False)
    
    plt.title('R² Score Heatmap: Models vs Emotion Labels', fontsize=14, fontweight='bold')
    plt.xlabel('Emotion Labels')
    plt.ylabel('Models')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_comparison_report(models_data: Dict[str, Dict], label_names: List[str], output_dir: Path) -> None:
    """Create comprehensive comparison report"""
    model_names = list(models_data.keys())
    
    report = {
        'comparison_summary': {
            'models_compared': model_names,
            'evaluation_split': 'val',
            'number_of_labels': len(label_names),
            'total_samples': len(models_data[model_names[0]]['results']['predictions'])
        },
        'overall_performance': {},
        'per_label_winners': {},
        'performance_analysis': {}
    }
    
    # Overall performance comparison
    for model in model_names:
        summary = models_data[model]['summary']['Overall Performance']
        report['overall_performance'][model] = {
            'R² Score': summary['R² Score'],
            'MAE': summary['MAE'],
            'RMSE': summary['RMSE'],
            'Spearman ρ': summary['Spearman ρ']
        }
    
    # Determine winners for each metric
    metrics = ['R² Score', 'MAE', 'RMSE', 'Spearman ρ']
    winners = {}
    
    for metric in metrics:
        values = [report['overall_performance'][model][metric] for model in model_names]
        if metric in ['R² Score', 'Spearman ρ']:  # Higher is better
            best_idx = np.argmax(values)
        else:  # Lower is better
            best_idx = np.argmin(values)
        winners[metric] = {
            'winner': model_names[best_idx],
            'value': values[best_idx],
            'margin': abs(max(values) - min(values))
        }
    
    report['metric_winners'] = winners
    
    # Per-label analysis
    label_winners = {}
    for label in label_names:
        r2_scores = []
        for model in model_names:
            summary = models_data[model]['summary']
            r2_scores.append(summary['Per-Label Performance'][label]['R² Score'])
        
        best_idx = np.argmax(r2_scores)
        label_winners[label] = {
            'winner': model_names[best_idx],
            'r2_score': r2_scores[best_idx],
            'margin': max(r2_scores) - min(r2_scores)
        }
    
    report['per_label_winners'] = label_winners
    
    # Performance analysis
    # Count wins per model
    overall_wins = {}
    label_wins = {}
    
    for model in model_names:
        overall_wins[model] = sum(1 for winner_info in winners.values() if winner_info['winner'] == model)
        label_wins[model] = sum(1 for winner_info in label_winners.values() if winner_info['winner'] == model)
    
    report['performance_analysis'] = {
        'overall_metric_wins': overall_wins,
        'per_label_wins': label_wins,
        'total_labels': len(label_names)
    }
    
    # Save report
    with open(output_dir / 'model_comparison_report.json', 'w') as f:
        json.dump(report, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Compare performance between encoder models')
    parser.add_argument('--checkpoints', nargs='+', required=True, help='Paths to model checkpoint directories')
    parser.add_argument('--model-names', nargs='+', help='Custom names for models (default: use directory names)')
    parser.add_argument('--split', default='val', choices=['val', 'test'], help='Dataset split to compare')
    parser.add_argument('--label-names', default='schema/label_names.json', help='Path to label names file')
    parser.add_argument('--output-dir', default='comparison_results', help='Output directory for comparison results')
    
    args = parser.parse_args()
    
    # Validate inputs
    checkpoint_paths = [Path(cp) for cp in args.checkpoints]
    for cp in checkpoint_paths:
        if not cp.exists():
            raise ValueError(f"Checkpoint path does not exist: {cp}")
    
    # Model names
    if args.model_names:
        if len(args.model_names) != len(checkpoint_paths):
            raise ValueError("Number of model names must match number of checkpoints")
        model_names = args.model_names
    else:
        model_names = [cp.name for cp in checkpoint_paths]
    
    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load label names
    with open(args.label_names, 'r') as f:
        label_names = json.load(f)
    
    # Load model data
    print("Loading model results...")
    models_data = {}
    
    for model_name, checkpoint_path in zip(model_names, checkpoint_paths):
        print(f"  Loading {model_name} from {checkpoint_path}")
        try:
            models_data[model_name] = {
                'results': load_model_results(checkpoint_path, args.split),
                'summary': load_model_summary(checkpoint_path)
            }
        except FileNotFoundError as e:
            print(f"  Error loading {model_name}: {e}")
            continue
    
    if len(models_data) < 2:
        print("Need at least 2 models for comparison")
        return
    
    print(f"\nComparing {len(models_data)} models: {list(models_data.keys())}")
    print(f"Output directory: {output_dir}")
    
    # Create comparisons
    compare_overall_metrics(models_data, output_dir)
    compare_per_label_performance(models_data, label_names, output_dir)
    create_performance_heatmap(models_data, label_names, output_dir)
    create_comparison_report(models_data, label_names, output_dir)
    
    # Statistical tests (if exactly 2 models)
    if len(models_data) == 2:
        statistical_significance_tests(models_data, output_dir)
    
    print("\nComparison complete! Generated files:")
    print(f"  - {output_dir}/overall_comparison.png")
    print(f"  - {output_dir}/per_label_comparison.png")
    print(f"  - {output_dir}/performance_heatmap.png")
    print(f"  - {output_dir}/model_comparison_report.json")
    
    if len(models_data) == 2:
        print(f"  - {output_dir}/significance_tests.json")

if __name__ == '__main__':
    main()