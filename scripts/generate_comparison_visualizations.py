#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate comprehensive comparison visualizations between baseline and fine-tuned models.

This script creates multiple visualization types to compare model performance:
1. Overall metrics comparison (bar charts)
2. Per-label performance comparison (heatmaps)
3. Improvement scatter plots
4. Performance distribution comparisons
5. Summary dashboard

Usage:
  python scripts/generate_comparison_visualizations.py --comparison-data comparison_results/xlmr_base_val.json
"""
from __future__ import annotations
import argparse
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_comparison_data(path: Path) -> Dict[str, Any]:
    """Load comparison results from JSON file."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def load_label_names(label_path: str = 'schema/label_names.json') -> List[str]:
    """Load emotion label names"""
    try:
        with open(label_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        # Fallback - generate generic label names
        return [f"emotion_{i+1}" for i in range(21)]

def plot_overall_comparison(comparison_data: Dict[str, Any], output_dir: Path) -> None:
    """Create overall metrics comparison bar chart."""
    metrics = comparison_data['metric_improvements']
    
    # Prepare data
    metric_names = list(metrics.keys())
    baseline_values = [metrics[m]['baseline'] for m in metric_names]
    finetuned_values = [metrics[m]['finetuned'] for m in metric_names]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Bar plot comparison
    x_pos = np.arange(len(metric_names))
    width = 0.35
    
    bars1 = ax1.bar(x_pos - width/2, baseline_values, width, 
                   label='Baseline', alpha=0.8, color='lightcoral')
    bars2 = ax1.bar(x_pos + width/2, finetuned_values, width,
                   label='Fine-tuned', alpha=0.8, color='lightblue')
    
    ax1.set_xlabel('Metrics')
    ax1.set_ylabel('Values')
    ax1.set_title(f'Overall Performance Comparison\n{comparison_data["model_name"]}')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([m.upper() for m in metric_names])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars1 + bars2, baseline_values + finetuned_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + abs(height) * 0.01,
               f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Improvement percentage plot
    percent_changes = [metrics[m]['percent_change'] for m in metric_names]
    colors = ['green' if pc > 0 else 'red' for pc in percent_changes]
    
    bars3 = ax2.bar(metric_names, percent_changes, alpha=0.7, color=colors)
    ax2.set_xlabel('Metrics')
    ax2.set_ylabel('Improvement (%)')
    ax2.set_title('Percentage Improvement from Baseline')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars3, percent_changes):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., 
               height + (5 if height >= 0 else -10),
               f'{value:+.1f}%', ha='center', va='bottom' if height >= 0 else 'top', 
               fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'overall_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_per_label_comparison(comparison_data: Dict[str, Any], label_names: List[str], output_dir: Path) -> None:
    """Create per-label performance comparison heatmap."""
    per_label_data = comparison_data.get('per_label_improvements', {})
    
    if not per_label_data:
        print("No per-label data available for comparison")
        return
    
    # Prepare data for heatmap
    metrics = ['mae', 'rmse', 'r2', 'spearman']
    available_metrics = [m for m in metrics if m in per_label_data]
    
    if not available_metrics:
        return
    
    # Create comparison matrices
    n_labels = len(label_names)
    n_metrics = len(available_metrics)
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()
    
    for i, metric in enumerate(available_metrics):
        if i >= 4:  # Only plot first 4 metrics
            break
            
        metric_data = per_label_data[metric]
        baseline_vals = np.array(metric_data['baseline'])
        finetuned_vals = np.array(metric_data['finetuned'])
        improvements = np.array(metric_data['improvements'])
        
        # Create comparison matrix (baseline vs finetuned)
        comparison_matrix = np.column_stack([baseline_vals, finetuned_vals, improvements])
        
        # Plot heatmap
        ax = axes[i]
        sns.heatmap(comparison_matrix.T, 
                   xticklabels=label_names,
                   yticklabels=['Baseline', 'Fine-tuned', 'Improvement'],
                   annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                   ax=ax, cbar_kws={'shrink': 0.8})
        
        ax.set_title(f'{metric.upper()} Comparison Across Labels')
        ax.set_xlabel('Emotion Labels')
        
        # Rotate x-axis labels
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'per_label_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_performance_heatmap(comparison_data: Dict[str, Any], label_names: List[str], output_dir: Path) -> None:
    """Create performance heatmap showing improvements across all dimensions."""
    per_label_data = comparison_data.get('per_label_improvements', {})
    
    if not per_label_data:
        return
    
    # Create improvement matrix
    metrics = ['mae', 'rmse', 'r2', 'spearman']
    available_metrics = [m for m in metrics if m in per_label_data]
    
    if not available_metrics:
        return
    
    improvement_matrix = []
    for metric in available_metrics:
        improvements = per_label_data[metric]['improvements']
        improvement_matrix.append(improvements)
    
    improvement_matrix = np.array(improvement_matrix)
    
    # Create the heatmap
    fig, ax = plt.subplots(figsize=(16, 8))
    
    sns.heatmap(improvement_matrix, 
               xticklabels=label_names,
               yticklabels=[m.upper() for m in available_metrics],
               annot=True, fmt='.3f', 
               cmap='RdYlBu_r', center=0,
               ax=ax, cbar_kws={'label': 'Improvement'})
    
    ax.set_title(f'Performance Improvement Heatmap\n{comparison_data["model_name"]}')
    ax.set_xlabel('Emotion Labels')
    ax.set_ylabel('Metrics')
    
    # Rotate x-axis labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_improvement_scatter(comparison_data: Dict[str, Any], label_names: List[str], output_dir: Path) -> None:
    """Create scatter plots showing baseline vs fine-tuned performance."""
    per_label_data = comparison_data.get('per_label_improvements', {})
    
    if not per_label_data:
        return
    
    metrics = ['mae', 'rmse', 'r2', 'spearman']
    available_metrics = [m for m in metrics if m in per_label_data]
    
    if len(available_metrics) == 0:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, metric in enumerate(available_metrics[:4]):  # Plot up to 4 metrics
        metric_data = per_label_data[metric]
        baseline_vals = np.array(metric_data['baseline'])
        finetuned_vals = np.array(metric_data['finetuned'])
        
        ax = axes[i]
        
        # Create scatter plot
        scatter = ax.scatter(baseline_vals, finetuned_vals, 
                           alpha=0.7, s=60, c=range(len(baseline_vals)), 
                           cmap='tab20')
        
        # Add diagonal line (no improvement)
        min_val = min(baseline_vals.min(), finetuned_vals.min())
        max_val = max(baseline_vals.max(), finetuned_vals.max())
        ax.plot([min_val, max_val], [min_val, max_val], 
               'k--', alpha=0.5, label='No improvement')
        
        # Add quadrant indicators
        if metric in ['mae', 'rmse']:  # Lower is better
            ax.axhline(y=baseline_vals.mean(), color='red', alpha=0.3, linestyle=':', label='Baseline mean')
            ax.axvline(x=baseline_vals.mean(), color='red', alpha=0.3, linestyle=':')
        else:  # Higher is better
            ax.axhline(y=baseline_vals.mean(), color='green', alpha=0.3, linestyle=':', label='Baseline mean')
            ax.axvline(x=baseline_vals.mean(), color='green', alpha=0.3, linestyle=':')
        
        ax.set_xlabel(f'Baseline {metric.upper()}')
        ax.set_ylabel(f'Fine-tuned {metric.upper()}')
        ax.set_title(f'{metric.upper()}: Baseline vs Fine-tuned')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Annotate points with label names (for extreme points)
        for j, (baseline_val, finetuned_val) in enumerate(zip(baseline_vals, finetuned_vals)):
            # Annotate top 3 most improved and 3 most degraded points
            improvement = (finetuned_val - baseline_val) if metric in ['r2', 'spearman'] else (baseline_val - finetuned_val)
            if j < 3 or improvement in sorted([
                (finetuned_vals[k] - baseline_vals[k]) if metric in ['r2', 'spearman'] else (baseline_vals[k] - finetuned_vals[k])
                for k in range(len(baseline_vals))
            ], reverse=True)[:3]:
                ax.annotate(label_names[j] if j < len(label_names) else f'L{j}', 
                          (baseline_val, finetuned_val), 
                          xytext=(5, 5), textcoords='offset points',
                          fontsize=8, alpha=0.7)
    
    # Hide unused subplots
    for i in range(len(available_metrics), 4):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'improvement_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_dashboard(comparison_data: Dict[str, Any], output_dir: Path) -> None:
    """Create a comprehensive summary dashboard."""
    fig = plt.figure(figsize=(20, 12))
    
    # Main title
    fig.suptitle(f'Model Performance Comparison Dashboard\n{comparison_data["model_name"]}', 
                fontsize=20, fontweight='bold')
    
    # Create grid layout
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # 1. Overall metrics summary (top-left)
    ax1 = fig.add_subplot(gs[0, :2])
    metrics = comparison_data['metric_improvements']
    metric_names = list(metrics.keys())
    improvements = [metrics[m]['percent_change'] for m in metric_names]
    
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    bars = ax1.barh(metric_names, improvements, color=colors, alpha=0.7)
    ax1.set_xlabel('Improvement (%)')
    ax1.set_title('Overall Performance Improvements')
    ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, improvements):
        width = bar.get_width()
        ax1.text(width + (5 if width >= 0 else -5), bar.get_y() + bar.get_height()/2,
               f'{value:+.1f}%', ha='left' if width >= 0 else 'right', va='center', 
               fontweight='bold')
    
    # 2. Performance summary stats (top-right)
    ax2 = fig.add_subplot(gs[0, 2:])
    ax2.axis('off')
    
    # Calculate summary statistics
    total_metrics = len(metric_names)
    improved_metrics = sum(1 for m in metric_names if metrics[m]['absolute_change'] > 0)
    avg_improvement = np.mean([abs(metrics[m]['percent_change']) for m in metric_names])
    
    summary_text = f"""
    PERFORMANCE SUMMARY
    
    Total Metrics: {total_metrics}
    Improved Metrics: {improved_metrics}
    Success Rate: {improved_metrics/total_metrics*100:.1f}%
    
    Average Improvement: {avg_improvement:.1f}%
    
    Baseline Samples: {comparison_data.get('baseline_samples', 'N/A')}
    Fine-tuned Samples: {comparison_data.get('finetuned_samples', 'N/A')}
    """
    
    ax2.text(0.1, 0.9, summary_text, transform=ax2.transAxes, 
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # 3. Per-label improvement distribution (middle)
    per_label_data = comparison_data.get('per_label_improvements', {})
    if per_label_data:
        ax3 = fig.add_subplot(gs[1, :])
        
        # Use R² improvements for visualization
        if 'r2' in per_label_data:
            improvements = per_label_data['r2']['improvements']
            ax3.hist(improvements, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
            ax3.axvline(np.mean(improvements), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(improvements):.3f}')
            ax3.axvline(np.median(improvements), color='orange', linestyle='--',
                       label=f'Median: {np.median(improvements):.3f}')
            ax3.set_xlabel('R² Improvement')
            ax3.set_ylabel('Number of Labels')
            ax3.set_title('Distribution of R² Improvements Across Labels')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
    
    # 4. Best and worst performing labels (bottom)
    if per_label_data and 'r2' in per_label_data:
        label_names = load_label_names()
        r2_data = per_label_data['r2']
        improvements = r2_data['improvements']
        
        # Get top 5 best and worst
        sorted_indices = np.argsort(improvements)
        worst_5 = sorted_indices[:5]
        best_5 = sorted_indices[-5:]
        
        ax4 = fig.add_subplot(gs[2, :2])
        ax5 = fig.add_subplot(gs[2, 2:])
        
        # Worst performing
        worst_labels = [label_names[i] if i < len(label_names) else f'L{i}' for i in worst_5]
        worst_values = [improvements[i] for i in worst_5]
        ax4.barh(worst_labels, worst_values, color='lightcoral', alpha=0.7)
        ax4.set_xlabel('R² Improvement')
        ax4.set_title('5 Least Improved Labels')
        ax4.grid(True, alpha=0.3)
        
        # Best performing
        best_labels = [label_names[i] if i < len(label_names) else f'L{i}' for i in best_5]
        best_values = [improvements[i] for i in best_5]
        ax5.barh(best_labels, best_values, color='lightgreen', alpha=0.7)
        ax5.set_xlabel('R² Improvement')
        ax5.set_title('5 Most Improved Labels')
        ax5.grid(True, alpha=0.3)
    
    plt.savefig(output_dir / 'summary_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_comparison_report(comparison_data: Dict[str, Any], output_dir: Path) -> None:
    """Generate a detailed comparison report."""
    report_data = {
        'model_name': comparison_data['model_name'],
        'summary': {
            'baseline_file': comparison_data['baseline_file'],
            'finetuned_file': comparison_data['finetuned_file'],
            'baseline_samples': comparison_data.get('baseline_samples', 'N/A'),
            'finetuned_samples': comparison_data.get('finetuned_samples', 'N/A')
        },
        'overall_improvements': {},
        'per_label_analysis': {}
    }
    
    # Overall improvements
    metrics = comparison_data['metric_improvements']
    for metric, data in metrics.items():
        report_data['overall_improvements'][metric] = {
            'baseline_value': data['baseline'],
            'finetuned_value': data['finetuned'],
            'absolute_improvement': data['absolute_change'],
            'percentage_improvement': data['percent_change'],
            'improved': data['absolute_change'] > 0
        }
    
    # Per-label analysis
    per_label_data = comparison_data.get('per_label_improvements', {})
    if per_label_data:
        label_names = load_label_names()
        
        for metric, data in per_label_data.items():
            improvements = np.array(data['improvements'])
            report_data['per_label_analysis'][metric] = {
                'mean_improvement': float(np.mean(improvements)),
                'median_improvement': float(np.median(improvements)),
                'std_improvement': float(np.std(improvements)),
                'min_improvement': float(np.min(improvements)),
                'max_improvement': float(np.max(improvements)),
                'labels_improved': int(np.sum(improvements > 0)),
                'total_labels': len(improvements),
                'improvement_rate': float(np.sum(improvements > 0) / len(improvements))
            }
            
            # Best and worst labels
            sorted_indices = np.argsort(improvements)
            report_data['per_label_analysis'][metric]['worst_labels'] = [
                {
                    'label': label_names[i] if i < len(label_names) else f'label_{i}',
                    'improvement': float(improvements[i])
                } for i in sorted_indices[:5]
            ]
            report_data['per_label_analysis'][metric]['best_labels'] = [
                {
                    'label': label_names[i] if i < len(label_names) else f'label_{i}',
                    'improvement': float(improvements[i])
                } for i in sorted_indices[-5:]
            ]
    
    # Save report
    with open(output_dir / 'comparison_report.json', 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser(description='Generate comparison visualizations')
    parser.add_argument('--comparison-data', required=True, 
                       help='Path to comparison results JSON file')
    parser.add_argument('--label-names', default='schema/label_names.json',
                       help='Path to label names file')
    parser.add_argument('--output-dir', 
                       help='Output directory for visualizations (default: same as comparison data)')
    
    args = parser.parse_args()
    
    # Load data
    comparison_path = Path(args.comparison_data)
    if not comparison_path.exists():
        raise FileNotFoundError(f"Comparison data file not found: {comparison_path}")
    
    comparison_data = load_comparison_data(comparison_path)
    label_names = load_label_names(args.label_names)
    
    # Output directory
    output_dir = Path(args.output_dir) if args.output_dir else comparison_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating comparison visualizations for {comparison_data['model_name']}")
    print(f"Output directory: {output_dir}")
    
    # Generate all visualizations
    plot_overall_comparison(comparison_data, output_dir)
    plot_per_label_comparison(comparison_data, label_names, output_dir)
    plot_performance_heatmap(comparison_data, label_names, output_dir)
    plot_improvement_scatter(comparison_data, label_names, output_dir)
    create_summary_dashboard(comparison_data, output_dir)
    generate_comparison_report(comparison_data, output_dir)
    
    print("\nGenerated visualizations:")
    print(f"  - {output_dir}/overall_comparison.png")
    print(f"  - {output_dir}/per_label_comparison.png") 
    print(f"  - {output_dir}/performance_heatmap.png")
    print(f"  - {output_dir}/improvement_scatter.png")
    print(f"  - {output_dir}/summary_dashboard.png")
    print(f"  - {output_dir}/comparison_report.json")

if __name__ == '__main__':
    main()