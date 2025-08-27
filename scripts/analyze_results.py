#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive analysis script for sentiment analysis model results.
Performs detailed evaluation including confusion matrices, error analysis, and statistical tests.
"""
from __future__ import annotations
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

def load_predictions_and_labels(checkpoint_path: Path, split: str = 'val') -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load model predictions and true labels"""
    # Try to find evaluation results
    eval_file = checkpoint_path / f'eval_results_{split}.json'
    if eval_file.exists():
        with open(eval_file, 'r') as f:
            results = json.load(f)
        
        predictions = np.array(results.get('predictions', []))
        labels = np.array(results.get('labels', []))
        
        # Create metadata if available
        meta_df = None
        if 'metadata' in results:
            meta_df = pd.DataFrame(results['metadata'])
        
        return predictions, labels, meta_df
    else:
        print(f"Evaluation results not found at {eval_file}")
        print("Please run evaluation first using the eval script")
        return None, None, None

def analyze_prediction_distribution(predictions: np.ndarray, labels: np.ndarray, 
                                  label_names: List[str], output_dir: Path) -> None:
    """Analyze the distribution of predictions vs true labels"""
    n_labels = len(label_names)
    
    fig, axes = plt.subplots(3, 7, figsize=(21, 9))  # 21 emotions, 3x7 grid
    axes = axes.flatten()
    
    for i in range(n_labels):
        ax = axes[i]
        
        # Scatter plot of predictions vs true labels
        ax.scatter(labels[:, i], predictions[:, i], alpha=0.5, s=10)
        
        # Perfect prediction line
        min_val = min(labels[:, i].min(), predictions[:, i].min())
        max_val = max(labels[:, i].max(), predictions[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        
        # Calculate correlation
        corr = np.corrcoef(labels[:, i], predictions[:, i])[0, 1]
        
        ax.set_xlabel('True Values')
        ax.set_ylabel('Predictions')
        ax.set_title(f'{label_names[i]}\nρ={corr:.3f}', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    # Hide extra subplots
    for i in range(n_labels, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'prediction_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_residuals(predictions: np.ndarray, labels: np.ndarray, 
                     label_names: List[str], output_dir: Path) -> None:
    """Analyze residuals for each emotion label"""
    residuals = predictions - labels
    
    fig, axes = plt.subplots(3, 7, figsize=(21, 9))
    axes = axes.flatten()
    
    for i in range(len(label_names)):
        ax = axes[i]
        
        # Residual plot
        ax.scatter(predictions[:, i], residuals[:, i], alpha=0.5, s=10)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.8)
        
        # Statistics
        mean_residual = np.mean(residuals[:, i])
        std_residual = np.std(residuals[:, i])
        
        ax.set_xlabel('Predicted Values')
        ax.set_ylabel('Residuals')
        ax.set_title(f'{label_names[i]}\nμ={mean_residual:.3f}, σ={std_residual:.3f}', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    # Hide extra subplots
    for i in range(len(label_names), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'residual_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_error_heatmap(predictions: np.ndarray, labels: np.ndarray, 
                        label_names: List[str], output_dir: Path) -> None:
    """Create heatmap of errors across emotion labels"""
    errors = np.abs(predictions - labels)
    error_matrix = np.corrcoef(errors.T)
    
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(error_matrix, dtype=bool))
    
    sns.heatmap(error_matrix, mask=mask, annot=True, fmt='.2f', 
                xticklabels=label_names, yticklabels=label_names,
                cmap='coolwarm', center=0, square=True)
    
    plt.title('Error Correlation Matrix Between Emotion Labels')
    plt.tight_layout()
    plt.savefig(output_dir / 'error_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_by_language(predictions: np.ndarray, labels: np.ndarray, 
                       meta_df: pd.DataFrame, label_names: List[str], 
                       output_dir: Path) -> None:
    """Analyze performance by language if metadata available"""
    if meta_df is None or 'lang' not in meta_df.columns:
        print("Language metadata not available, skipping language analysis")
        return
    
    languages = meta_df['lang'].unique()
    
    fig, axes = plt.subplots(len(languages), 1, figsize=(12, 6*len(languages)))
    if len(languages) == 1:
        axes = [axes]
    
    language_stats = {}
    
    for i, lang in enumerate(languages):
        lang_mask = meta_df['lang'] == lang
        lang_pred = predictions[lang_mask]
        lang_true = labels[lang_mask]
        
        # Calculate metrics for this language
        r2_scores = []
        mae_scores = []
        
        for j in range(len(label_names)):
            # R² score
            ss_res = np.sum((lang_true[:, j] - lang_pred[:, j]) ** 2)
            ss_tot = np.sum((lang_true[:, j] - np.mean(lang_true[:, j])) ** 2)
            r2 = 1 - (ss_res / (ss_tot + 1e-8))
            r2_scores.append(r2)
            
            # MAE
            mae = mean_absolute_error(lang_true[:, j], lang_pred[:, j])
            mae_scores.append(mae)
        
        language_stats[lang] = {
            'samples': len(lang_pred),
            'r2_mean': np.mean(r2_scores),
            'r2_std': np.std(r2_scores),
            'mae_mean': np.mean(mae_scores),
            'mae_std': np.std(mae_scores)
        }
        
        # Plot R² scores for this language
        ax = axes[i]
        bars = ax.bar(range(len(label_names)), r2_scores, alpha=0.7)
        ax.set_title(f'R² Scores by Emotion Label - {lang.upper()} (n={len(lang_pred)})')
        ax.set_xlabel('Emotion Labels')
        ax.set_ylabel('R² Score')
        ax.set_xticks(range(len(label_names)))
        ax.set_xticklabels(label_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add mean line
        mean_r2 = np.mean(r2_scores)
        ax.axhline(y=mean_r2, color='red', linestyle='--', 
                  label=f'Mean: {mean_r2:.3f}')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_by_language.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save language statistics
    with open(output_dir / 'language_statistics.json', 'w') as f:
        json.dump(language_stats, f, indent=2)

def analyze_by_persona(predictions: np.ndarray, labels: np.ndarray, 
                      meta_df: pd.DataFrame, label_names: List[str], 
                      output_dir: Path) -> None:
    """Analyze performance by persona if metadata available"""
    if meta_df is None or 'persona_id' not in meta_df.columns:
        print("Persona metadata not available, skipping persona analysis")
        return
    
    personas = meta_df['persona_id'].unique()
    persona_stats = {}
    
    for persona in personas:
        persona_mask = meta_df['persona_id'] == persona
        persona_pred = predictions[persona_mask]
        persona_true = labels[persona_mask]
        
        if len(persona_pred) == 0:
            continue
        
        # Calculate overall metrics for this persona
        r2_scores = []
        mae_scores = []
        
        for j in range(len(label_names)):
            # R² score
            ss_res = np.sum((persona_true[:, j] - persona_pred[:, j]) ** 2)
            ss_tot = np.sum((persona_true[:, j] - np.mean(persona_true[:, j])) ** 2)
            r2 = 1 - (ss_res / (ss_tot + 1e-8))
            r2_scores.append(r2)
            
            # MAE
            mae = mean_absolute_error(persona_true[:, j], persona_pred[:, j])
            mae_scores.append(mae)
        
        persona_stats[persona] = {
            'samples': len(persona_pred),
            'r2_mean': np.mean(r2_scores),
            'r2_std': np.std(r2_scores),
            'mae_mean': np.mean(mae_scores),
            'mae_std': np.std(mae_scores),
            'r2_per_label': r2_scores,
            'mae_per_label': mae_scores
        }
    
    # Create visualization of persona performance
    persona_names = list(persona_stats.keys())
    mean_r2s = [persona_stats[p]['r2_mean'] for p in persona_names]
    
    plt.figure(figsize=(12, 8))
    bars = plt.bar(range(len(persona_names)), mean_r2s, alpha=0.7)
    plt.xlabel('Personas')
    plt.ylabel('Mean R² Score')
    plt.title('Mean Performance by Persona')
    plt.xticks(range(len(persona_names)), persona_names, rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add sample size annotations
    for i, (bar, persona) in enumerate(zip(bars, persona_names)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'n={persona_stats[persona]["samples"]}', 
                ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_by_persona.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save persona statistics
    with open(output_dir / 'persona_statistics.json', 'w') as f:
        json.dump(persona_stats, f, indent=2)

def statistical_tests(predictions: np.ndarray, labels: np.ndarray, 
                     label_names: List[str], output_dir: Path) -> None:
    """Perform statistical tests on model performance"""
    results = {}
    
    for i, label in enumerate(label_names):
        true_vals = labels[:, i]
        pred_vals = predictions[:, i]
        
        # Normality tests
        _, shapiro_p = stats.shapiro((pred_vals - true_vals)[:5000])  # Limit for computational efficiency
        _, ks_p = stats.kstest(pred_vals - true_vals, 'norm')
        
        # Correlation tests
        pearson_r, pearson_p = stats.pearsonr(true_vals, pred_vals)
        spearman_r, spearman_p = stats.spearmanr(true_vals, pred_vals)
        
        # Mean difference test (paired t-test)
        _, ttest_p = stats.ttest_rel(true_vals, pred_vals)
        
        results[label] = {
            'shapiro_p': float(shapiro_p),
            'ks_p': float(ks_p),
            'pearson_r': float(pearson_r),
            'pearson_p': float(pearson_p),
            'spearman_r': float(spearman_r),
            'spearman_p': float(spearman_p),
            'ttest_p': float(ttest_p),
            'rmse': float(np.sqrt(mean_squared_error(true_vals, pred_vals))),
            'mae': float(mean_absolute_error(true_vals, pred_vals))
        }
    
    with open(output_dir / 'statistical_tests.json', 'w') as f:
        json.dump(results, f, indent=2)

def create_comprehensive_report(predictions: np.ndarray, labels: np.ndarray, 
                               meta_df: pd.DataFrame, label_names: List[str], 
                               output_dir: Path) -> None:
    """Create a comprehensive analysis report"""
    
    report = {
        'model_evaluation_summary': {
            'total_samples': len(predictions),
            'number_of_labels': len(label_names),
            'label_names': label_names
        },
        'overall_performance': {},
        'per_label_analysis': {},
        'data_characteristics': {}
    }
    
    # Overall performance
    overall_r2 = []
    overall_mae = []
    overall_rmse = []
    
    for i in range(len(label_names)):
        true_vals = labels[:, i]
        pred_vals = predictions[:, i]
        
        # R² score
        ss_res = np.sum((true_vals - pred_vals) ** 2)
        ss_tot = np.sum((true_vals - np.mean(true_vals)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        overall_r2.append(r2)
        
        # MAE and RMSE
        mae = mean_absolute_error(true_vals, pred_vals)
        rmse = np.sqrt(mean_squared_error(true_vals, pred_vals))
        overall_mae.append(mae)
        overall_rmse.append(rmse)
    
    report['overall_performance'] = {
        'mean_r2': float(np.mean(overall_r2)),
        'std_r2': float(np.std(overall_r2)),
        'mean_mae': float(np.mean(overall_mae)),
        'std_mae': float(np.std(overall_mae)),
        'mean_rmse': float(np.mean(overall_rmse)),
        'std_rmse': float(np.std(overall_rmse))
    }
    
    # Per-label analysis
    for i, label in enumerate(label_names):
        report['per_label_analysis'][label] = {
            'r2': float(overall_r2[i]),
            'mae': float(overall_mae[i]),
            'rmse': float(overall_rmse[i]),
            'true_mean': float(np.mean(labels[:, i])),
            'true_std': float(np.std(labels[:, i])),
            'pred_mean': float(np.mean(predictions[:, i])),
            'pred_std': float(np.std(predictions[:, i]))
        }
    
    # Data characteristics
    report['data_characteristics'] = {
        'label_value_ranges': {
            label: {
                'min': float(labels[:, i].min()),
                'max': float(labels[:, i].max()),
                'mean': float(labels[:, i].mean()),
                'std': float(labels[:, i].std())
            }
            for i, label in enumerate(label_names)
        }
    }
    
    if meta_df is not None:
        if 'lang' in meta_df.columns:
            report['data_characteristics']['language_distribution'] = meta_df['lang'].value_counts().to_dict()
        if 'persona_id' in meta_df.columns:
            report['data_characteristics']['persona_distribution'] = meta_df['persona_id'].value_counts().to_dict()
    
    with open(output_dir / 'comprehensive_analysis_report.json', 'w') as f:
        json.dump(report, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Analyze sentiment analysis model results')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint directory')
    parser.add_argument('--split', default='val', choices=['val', 'test'], help='Dataset split to analyze')
    parser.add_argument('--label-names', default='schema/label_names.json', help='Path to label names file')
    parser.add_argument('--output-dir', help='Output directory for analysis (default: same as checkpoint)')
    
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")
    
    # Output directory
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_path / 'analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    predictions, labels, meta_df = load_predictions_and_labels(checkpoint_path, args.split)
    
    if predictions is None:
        print("Cannot proceed without evaluation results. Please run evaluation first.")
        return
    
    with open(args.label_names, 'r') as f:
        label_names = json.load(f)
    
    print(f"Analyzing {len(predictions)} samples with {len(label_names)} emotion labels")
    print(f"Output directory: {output_dir}")
    
    # Perform analyses
    analyze_prediction_distribution(predictions, labels, label_names, output_dir)
    analyze_residuals(predictions, labels, label_names, output_dir)
    create_error_heatmap(predictions, labels, label_names, output_dir)
    statistical_tests(predictions, labels, label_names, output_dir)
    create_comprehensive_report(predictions, labels, meta_df, label_names, output_dir)
    
    if meta_df is not None:
        analyze_by_language(predictions, labels, meta_df, label_names, output_dir)
        analyze_by_persona(predictions, labels, meta_df, label_names, output_dir)
    
    print("\nAnalysis complete! Generated files:")
    print(f"  - {output_dir}/prediction_scatter.png")
    print(f"  - {output_dir}/residual_analysis.png")
    print(f"  - {output_dir}/error_correlation_heatmap.png")
    print(f"  - {output_dir}/statistical_tests.json")
    print(f"  - {output_dir}/comprehensive_analysis_report.json")
    
    if meta_df is not None:
        if 'lang' in meta_df.columns:
            print(f"  - {output_dir}/performance_by_language.png")
            print(f"  - {output_dir}/language_statistics.json")
        if 'persona_id' in meta_df.columns:
            print(f"  - {output_dir}/performance_by_persona.png")
            print(f"  - {output_dir}/persona_statistics.json")

if __name__ == '__main__':
    main()