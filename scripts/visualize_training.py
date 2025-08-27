#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training visualization script for sentiment analysis results.
Creates comprehensive plots for training progress and model performance.
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

def load_training_logs(log_path: str) -> Dict[str, Any]:
    """Load training logs from trainer_state.json"""
    with open(log_path, 'r') as f:
        state = json.load(f)
    return state

def load_label_names(label_path: str) -> List[str]:
    """Load emotion label names"""
    with open(label_path, 'r') as f:
        return json.load(f)

def plot_training_curves(log_history: List[Dict], output_dir: Path) -> None:
    """Plot training and validation loss curves"""
    plt.style.use('seaborn-v0_8')
    
    train_losses = []
    val_losses = []
    epochs = []
    
    for entry in log_history:
        if 'loss' in entry:
            train_losses.append(entry['loss'])
            epochs.append(entry['epoch'])
        if 'eval_loss' in entry:
            val_losses.append(entry['eval_loss'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Training loss
    if train_losses:
        ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss (MSE)')
        ax1.set_title('Training Loss Over Time')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
    
    # Validation metrics
    val_epochs = []
    val_losses_clean = []
    val_r2 = []
    val_spearman = []
    
    for entry in log_history:
        if 'eval_loss' in entry:
            val_epochs.append(entry['epoch'])
            val_losses_clean.append(entry['eval_loss'])
            val_r2.append(entry.get('eval_r2', 0))
            val_spearman.append(entry.get('eval_spearman', 0))
    
    if val_losses_clean:
        ax2_twin = ax2.twinx()
        
        line1 = ax2.plot(val_epochs, val_losses_clean, 'r-', label='Validation Loss', linewidth=2)
        line2 = ax2_twin.plot(val_epochs, val_r2, 'g-', label='R² Score', linewidth=2)
        line3 = ax2_twin.plot(val_epochs, val_spearman, 'orange', linestyle='--', label='Spearman ρ', linewidth=2)
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Validation Loss', color='r')
        ax2_twin.set_ylabel('Score', color='g')
        ax2.set_title('Validation Metrics Over Time')
        ax2.grid(True, alpha=0.3)
        
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc='center right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_per_label_performance(log_history: List[Dict], label_names: List[str], output_dir: Path) -> None:
    """Plot per-label performance metrics"""
    final_eval = None
    for entry in reversed(log_history):
        if 'eval_r2_per_label' in entry:
            final_eval = entry
            break
    
    if not final_eval:
        print("No per-label evaluation metrics found")
        return
    
    metrics = {
        'R² Score': final_eval['eval_r2_per_label'],
        'MAE': final_eval['eval_mae_per_label'],
        'RMSE': final_eval['eval_rmse_per_label'],
        'Spearman ρ': final_eval['eval_spearman_per_label']
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'plum']
    
    for i, (metric_name, values) in enumerate(metrics.items()):
        ax = axes[i]
        
        # Create bar plot
        bars = ax.bar(range(len(label_names)), values, color=colors[i], alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(values) * 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=8, rotation=0)
        
        ax.set_xlabel('Emotion Labels')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name} per Emotion Label')
        ax.set_xticks(range(len(label_names)))
        ax.set_xticklabels(label_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add horizontal line for mean
        mean_val = np.mean(values)
        ax.axhline(y=mean_val, color='red', linestyle='--', alpha=0.8, 
                  label=f'Mean: {mean_val:.3f}')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'per_label_performance.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_metric_distribution(log_history: List[Dict], output_dir: Path) -> None:
    """Plot distribution of per-label metrics"""
    final_eval = None
    for entry in reversed(log_history):
        if 'eval_r2_per_label' in entry:
            final_eval = entry
            break
    
    if not final_eval:
        return
    
    metrics_data = {
        'R² Score': final_eval['eval_r2_per_label'],
        'MAE': final_eval['eval_mae_per_label'],
        'RMSE': final_eval['eval_rmse_per_label'],
        'Spearman ρ': final_eval['eval_spearman_per_label']
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, (metric_name, values) in enumerate(metrics_data.items()):
        ax = axes[i]
        
        # Histogram
        ax.hist(values, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(np.mean(values), color='red', linestyle='--', 
                  label=f'Mean: {np.mean(values):.3f}')
        ax.axvline(np.median(values), color='orange', linestyle='--', 
                  label=f'Median: {np.median(values):.3f}')
        
        ax.set_xlabel(metric_name)
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution of {metric_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metric_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_performance_summary(log_history: List[Dict], label_names: List[str], output_path: Path) -> None:
    """Create a comprehensive performance summary report"""
    final_eval = None
    for entry in reversed(log_history):
        if 'eval_r2' in entry:
            final_eval = entry
            break
    
    if not final_eval:
        print("No evaluation metrics found")
        return
    
    summary = {
        'Overall Performance': {
            'R² Score': final_eval.get('eval_r2', 0),
            'MAE': final_eval.get('eval_mae', 0),
            'RMSE': final_eval.get('eval_rmse', 0),
            'Spearman ρ': final_eval.get('eval_spearman', 0),
            'Validation Loss': final_eval.get('eval_loss', 0)
        },
        'Per-Label Performance': {}
    }
    
    if 'eval_r2_per_label' in final_eval:
        for i, label in enumerate(label_names):
            summary['Per-Label Performance'][label] = {
                'R² Score': final_eval['eval_r2_per_label'][i],
                'MAE': final_eval['eval_mae_per_label'][i],
                'RMSE': final_eval['eval_rmse_per_label'][i],
                'Spearman ρ': final_eval['eval_spearman_per_label'][i]
            }
    
    # Statistics
    if 'eval_r2_per_label' in final_eval:
        r2_values = final_eval['eval_r2_per_label']
        summary['Statistics'] = {
            'Best performing labels (R²)': sorted(
                [(label_names[i], r2_values[i]) for i in range(len(label_names))],
                key=lambda x: x[1], reverse=True
            )[:5],
            'Worst performing labels (R²)': sorted(
                [(label_names[i], r2_values[i]) for i in range(len(label_names))],
                key=lambda x: x[1]
            )[:5],
            'Performance variance (R²)': np.var(r2_values),
            'Performance std (R²)': np.std(r2_values)
        }
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Visualize sentiment analysis training results')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint directory')
    parser.add_argument('--label-names', default='schema/label_names.json', help='Path to label names file')
    parser.add_argument('--output-dir', help='Output directory for plots (default: same as checkpoint)')
    
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")
    
    # Find trainer_state.json - check in checkpoint subdirectories first
    state_file = None
    for subdir in checkpoint_path.iterdir():
        if subdir.is_dir() and subdir.name.startswith('checkpoint-'):
            potential_state = subdir / 'trainer_state.json'
            if potential_state.exists():
                state_file = potential_state
                break
    
    # Fallback to main directory
    if state_file is None:
        state_file = checkpoint_path / 'trainer_state.json'
        if not state_file.exists():
            raise ValueError(f"trainer_state.json not found in {checkpoint_path} or its checkpoint subdirectories")
    
    # Output directory
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_path
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    training_state = load_training_logs(str(state_file))
    label_names = load_label_names(args.label_names)
    
    print(f"Creating visualizations in {output_dir}")
    
    # Create plots
    log_history = training_state.get('log_history', [])
    
    if log_history:
        plot_training_curves(log_history, output_dir)
        plot_per_label_performance(log_history, label_names, output_dir)
        plot_metric_distribution(log_history, output_dir)
        create_performance_summary(log_history, label_names, output_dir / 'performance_summary.json')
        
        print("Visualizations created:")
        print(f"  - {output_dir}/training_curves.png")
        print(f"  - {output_dir}/per_label_performance.png") 
        print(f"  - {output_dir}/metric_distributions.png")
        print(f"  - {output_dir}/performance_summary.json")
    else:
        print("No training history found in trainer_state.json")

if __name__ == '__main__':
    main()