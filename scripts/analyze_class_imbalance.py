#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze class imbalance in the medical sentiment analysis dataset.
Computes statistics and creates visualizations for label distributions.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import Counter

def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """Load JSONL file and return list of records."""
    data = []
    with file_path.open('r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def load_label_names(label_names_path: Path) -> List[str]:
    """Load emotion label names."""
    with label_names_path.open('r', encoding='utf-8') as f:
        return json.load(f)

def compute_imbalance_stats(labels_array: np.ndarray, label_names: List[str]) -> Dict[str, Any]:
    """Compute detailed imbalance statistics for each label."""
    n_samples, n_labels = labels_array.shape
    stats = {}
    
    for i, label_name in enumerate(label_names):
        label_col = labels_array[:, i]
        positive_count = int(np.sum(label_col))
        negative_count = int(n_samples - positive_count)
        
        # Basic counts and ratios
        positive_ratio = positive_count / n_samples
        negative_ratio = negative_count / n_samples
        
        # Imbalance ratio (majority class / minority class)
        if positive_count == 0 or negative_count == 0:
            imbalance_ratio = float('inf') if positive_count == 0 else float('inf')
            minority_class = 'positive' if positive_count == 0 else 'negative'
        else:
            imbalance_ratio = max(positive_count, negative_count) / min(positive_count, negative_count)
            minority_class = 'positive' if positive_count < negative_count else 'negative'
        
        # Classification severity
        if imbalance_ratio == float('inf'):
            severity = 'extreme (complete)'
        elif imbalance_ratio > 100:
            severity = 'extreme'
        elif imbalance_ratio > 10:
            severity = 'severe'
        elif imbalance_ratio > 3:
            severity = 'moderate'
        else:
            severity = 'balanced'
        
        stats[label_name] = {
            'index': i,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'positive_ratio': positive_ratio,
            'negative_ratio': negative_ratio,
            'imbalance_ratio': imbalance_ratio,
            'minority_class': minority_class,
            'severity': severity
        }
    
    return stats

def analyze_dataset_split(split_path: Path, split_name: str, label_names: List[str]) -> Dict[str, Any]:
    """Analyze class imbalance for a single dataset split."""
    print(f"Analyzing {split_name} split...")
    
    # Load data
    data = load_jsonl(split_path)
    n_samples = len(data)
    
    # Extract labels
    labels_list = [item['labels'] for item in data]
    labels_array = np.array(labels_list, dtype=np.int32)
    
    # Compute statistics
    stats = compute_imbalance_stats(labels_array, label_names)
    
    # Overall statistics
    total_positive_labels = np.sum(labels_array)
    total_possible_labels = n_samples * len(label_names)
    overall_sparsity = 1 - (total_positive_labels / total_possible_labels)
    
    # Distribution of positive labels per sample
    labels_per_sample = np.sum(labels_array, axis=1)
    
    analysis = {
        'split_name': split_name,
        'n_samples': n_samples,
        'n_labels': len(label_names),
        'total_positive_labels': int(total_positive_labels),
        'total_possible_labels': int(total_possible_labels),
        'overall_sparsity': float(overall_sparsity),
        'labels_per_sample': {
            'mean': float(np.mean(labels_per_sample)),
            'std': float(np.std(labels_per_sample)),
            'min': int(np.min(labels_per_sample)),
            'max': int(np.max(labels_per_sample)),
            'distribution': Counter(labels_per_sample.tolist())
        },
        'per_label_stats': stats
    }
    
    return analysis

def create_imbalance_visualizations(analysis_results: Dict[str, Dict], label_names: List[str], output_dir: Path):
    """Create comprehensive visualizations of class imbalance."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. Imbalance ratio heatmap across splits
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # Collect data for plotting
    splits = list(analysis_results.keys())
    imbalance_ratios = []
    positive_ratios = []
    
    for split in splits:
        split_imbalance = []
        split_positive = []
        for label_name in label_names:
            ratio = analysis_results[split]['per_label_stats'][label_name]['imbalance_ratio']
            pos_ratio = analysis_results[split]['per_label_stats'][label_name]['positive_ratio']
            # Cap extreme values for visualization
            capped_ratio = min(ratio, 100) if ratio != float('inf') else 100
            split_imbalance.append(capped_ratio)
            split_positive.append(pos_ratio)
        imbalance_ratios.append(split_imbalance)
        positive_ratios.append(split_positive)
    
    # Imbalance ratio heatmap
    im1 = axes[0, 0].imshow(imbalance_ratios, cmap='Reds', aspect='auto')
    axes[0, 0].set_title('Imbalance Ratio by Split and Label', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Emotion Labels')
    axes[0, 0].set_ylabel('Dataset Splits')
    axes[0, 0].set_xticks(range(len(label_names)))
    axes[0, 0].set_xticklabels(label_names, rotation=45, ha='right')
    axes[0, 0].set_yticks(range(len(splits)))
    axes[0, 0].set_yticklabels(splits)
    plt.colorbar(im1, ax=axes[0, 0], label='Imbalance Ratio (capped at 100)')
    
    # Positive class ratio heatmap
    im2 = axes[0, 1].imshow(positive_ratios, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    axes[0, 1].set_title('Positive Class Ratio by Split and Label', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Emotion Labels')
    axes[0, 1].set_ylabel('Dataset Splits')
    axes[0, 1].set_xticks(range(len(label_names)))
    axes[0, 1].set_xticklabels(label_names, rotation=45, ha='right')
    axes[0, 1].set_yticks(range(len(splits)))
    axes[0, 1].set_yticklabels(splits)
    plt.colorbar(im2, ax=axes[0, 1], label='Positive Class Ratio')
    
    # 2. Distribution of labels per sample (train split)
    train_analysis = analysis_results['train']
    labels_dist = train_analysis['labels_per_sample']['distribution']
    labels_counts = sorted(labels_dist.items())
    
    axes[1, 0].bar([x[0] for x in labels_counts], [x[1] for x in labels_counts], 
                   alpha=0.7, edgecolor='black')
    axes[1, 0].set_title('Distribution of Positive Labels per Sample (Train)', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Number of Positive Labels')
    axes[1, 0].set_ylabel('Number of Samples')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 3. Imbalance severity distribution
    severity_counts = Counter()
    for split in splits:
        for label_name in label_names:
            severity = analysis_results[split]['per_label_stats'][label_name]['severity']
            severity_counts[severity] += 1
    
    severity_order = ['balanced', 'moderate', 'severe', 'extreme', 'extreme (complete)']
    severity_colors = ['green', 'yellow', 'orange', 'red', 'darkred']
    
    severities = [severity_counts.get(sev, 0) for sev in severity_order]
    bars = axes[1, 1].bar(severity_order, severities, color=severity_colors, alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('Imbalance Severity Distribution', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Severity Level')
    axes[1, 1].set_ylabel('Number of Label-Split Combinations')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, severities):
        if value > 0:
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                           str(value), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'class_imbalance_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Detailed per-label imbalance chart (train split)
    train_stats = train_analysis['per_label_stats']
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # Positive ratios bar chart
    pos_ratios = [train_stats[label]['positive_ratio'] for label in label_names]
    colors = ['red' if ratio < 0.1 else 'orange' if ratio < 0.2 else 'yellow' if ratio < 0.4 else 'green' 
              for ratio in pos_ratios]
    
    bars1 = ax1.bar(range(len(label_names)), pos_ratios, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_title('Positive Class Ratio by Emotion Label (Train Split)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Emotion Labels')
    ax1.set_ylabel('Positive Class Ratio')
    ax1.set_xticks(range(len(label_names)))
    ax1.set_xticklabels(label_names, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Balanced (0.5)')
    ax1.legend()
    
    # Add ratio labels on bars
    for i, (bar, ratio) in enumerate(zip(bars1, pos_ratios)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{ratio:.3f}', ha='center', va='bottom', fontsize=9, rotation=0)
    
    # Imbalance ratios (log scale)
    imb_ratios = [train_stats[label]['imbalance_ratio'] for label in label_names]
    # Cap infinite values for visualization
    imb_ratios_capped = [min(ratio, 1000) if ratio != float('inf') else 1000 for ratio in imb_ratios]
    
    bars2 = ax2.bar(range(len(label_names)), imb_ratios_capped, 
                    color='red', alpha=0.6, edgecolor='black')
    ax2.set_title('Imbalance Ratio by Emotion Label (Train Split)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Emotion Labels')
    ax2.set_ylabel('Imbalance Ratio (log scale)')
    ax2.set_yscale('log')
    ax2.set_xticks(range(len(label_names)))
    ax2.set_xticklabels(label_names, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=1, color='green', linestyle='--', alpha=0.7, label='Balanced (1:1)')
    ax2.axhline(y=3, color='yellow', linestyle='--', alpha=0.7, label='Moderate (3:1)')
    ax2.axhline(y=10, color='orange', linestyle='--', alpha=0.7, label='Severe (10:1)')
    ax2.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='Extreme (100:1)')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'per_label_imbalance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to {output_dir}")

def generate_imbalance_report(analysis_results: Dict[str, Dict], label_names: List[str], output_path: Path):
    """Generate a comprehensive imbalance analysis report."""
    
    report = {
        'analysis_summary': {
            'dataset_splits': list(analysis_results.keys()),
            'total_emotion_labels': len(label_names),
            'label_names': label_names
        },
        'overall_findings': {},
        'per_split_analysis': analysis_results,
        'recommendations': []
    }
    
    # Aggregate findings across splits
    all_imbalance_ratios = []
    severely_imbalanced_labels = set()
    
    for split_name, split_data in analysis_results.items():
        split_ratios = []
        for label_name in label_names:
            ratio = split_data['per_label_stats'][label_name]['imbalance_ratio']
            if ratio != float('inf'):
                split_ratios.append(ratio)
                all_imbalance_ratios.append(ratio)
            
            # Track severely imbalanced labels
            severity = split_data['per_label_stats'][label_name]['severity']
            if severity in ['severe', 'extreme', 'extreme (complete)']:
                severely_imbalanced_labels.add(label_name)
    
    # Overall statistics
    if all_imbalance_ratios:
        report['overall_findings'] = {
            'mean_imbalance_ratio': float(np.mean(all_imbalance_ratios)),
            'median_imbalance_ratio': float(np.median(all_imbalance_ratios)),
            'max_imbalance_ratio': float(np.max(all_imbalance_ratios)),
            'severely_imbalanced_labels': sorted(list(severely_imbalanced_labels)),
            'proportion_severely_imbalanced': len(severely_imbalanced_labels) / len(label_names)
        }
    
    # Generate recommendations
    recommendations = []
    
    if len(severely_imbalanced_labels) > len(label_names) * 0.5:
        recommendations.append({
            'priority': 'high',
            'issue': 'Majority of labels are severely imbalanced',
            'recommendation': 'Consider class balancing techniques: class weights, SMOTE, or focal loss',
            'technical_details': 'Use class_weight="balanced" in BCEWithLogitsLoss or implement focal loss'
        })
    
    if report['overall_findings'].get('mean_imbalance_ratio', 0) > 10:
        recommendations.append({
            'priority': 'high',
            'issue': 'High average imbalance ratio across labels',
            'recommendation': 'Implement class-sensitive evaluation metrics and training strategies',
            'technical_details': 'Use macro-averaged F1, precision-recall curves, and stratified sampling'
        })
    
    # Check for extremely sparse labels
    sparse_labels = []
    for label_name in label_names:
        train_ratio = analysis_results['train']['per_label_stats'][label_name]['positive_ratio']
        if train_ratio < 0.05:  # Less than 5% positive
            sparse_labels.append((label_name, train_ratio))
    
    if sparse_labels:
        recommendations.append({
            'priority': 'medium',
            'issue': f'Extremely sparse labels: {len(sparse_labels)} labels with <5% positive samples',
            'sparse_labels': sparse_labels,
            'recommendation': 'Consider label subset selection or specialized sampling strategies',
            'technical_details': 'Use stratified train-test split and consider label-specific thresholds'
        })
    
    # Sparsity check
    train_sparsity = analysis_results['train']['overall_sparsity']
    if train_sparsity > 0.8:
        recommendations.append({
            'priority': 'medium',
            'issue': f'High overall sparsity ({train_sparsity:.1%} of possible labels are 0)',
            'recommendation': 'Multi-label dataset is very sparse - consider specialized architectures',
            'technical_details': 'Use attention mechanisms or hierarchical labeling approaches'
        })
    
    report['recommendations'] = recommendations
    
    # Save report
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    return report

def print_summary(analysis_results: Dict[str, Dict], label_names: List[str]):
    """Print a concise summary of the imbalance analysis."""
    
    print("\n" + "="*80)
    print("🔍 CLASS IMBALANCE ANALYSIS SUMMARY")
    print("="*80)
    
    for split_name, split_data in analysis_results.items():
        print(f"\n📊 {split_name.upper()} SPLIT:")
        print(f"  Samples: {split_data['n_samples']:,}")
        print(f"  Overall sparsity: {split_data['overall_sparsity']:.1%}")
        print(f"  Avg labels per sample: {split_data['labels_per_sample']['mean']:.2f} ± {split_data['labels_per_sample']['std']:.2f}")
        
        # Count severity levels
        severity_counts = Counter()
        for label_name in label_names:
            severity = split_data['per_label_stats'][label_name]['severity']
            severity_counts[severity] += 1
        
        print(f"  Imbalance severity distribution:")
        for severity in ['balanced', 'moderate', 'severe', 'extreme', 'extreme (complete)']:
            count = severity_counts.get(severity, 0)
            if count > 0:
                print(f"    {severity}: {count} labels")
        
        # Most and least imbalanced
        ratios_by_label = []
        for label_name in label_names:
            ratio = split_data['per_label_stats'][label_name]['imbalance_ratio']
            if ratio != float('inf'):
                ratios_by_label.append((label_name, ratio))
        
        ratios_by_label.sort(key=lambda x: x[1])
        
        print(f"  Most balanced: {ratios_by_label[0][0]} ({ratios_by_label[0][1]:.2f}:1)")
        print(f"  Most imbalanced: {ratios_by_label[-1][0]} ({ratios_by_label[-1][1]:.1f}:1)")

def main():
    # Paths
    data_dir = Path("data/processed/encoder")
    schema_dir = Path("schema")
    output_dir = Path("analysis/class_imbalance")
    
    # Load label names
    label_names = load_label_names(schema_dir / "label_names.json")
    
    # Analyze all splits
    analysis_results = {}
    for split in ['train', 'val', 'test']:
        split_path = data_dir / f"{split}.jsonl"
        if split_path.exists():
            analysis_results[split] = analyze_dataset_split(split_path, split, label_names)
    
    # Generate outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create visualizations
    create_imbalance_visualizations(analysis_results, label_names, output_dir)
    
    # Generate detailed report
    report = generate_imbalance_report(analysis_results, label_names, output_dir / "imbalance_report.json")
    
    # Print summary
    print_summary(analysis_results, label_names)
    
    # Print key recommendations
    if report['recommendations']:
        print(f"\n⚠️  KEY RECOMMENDATIONS:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"{i}. [{rec['priority'].upper()}] {rec['issue']}")
            print(f"   → {rec['recommendation']}")
            if 'technical_details' in rec:
                print(f"   💻 {rec['technical_details']}")
    
    print(f"\n📁 Full analysis saved to: {output_dir}")
    print(f"📊 Visualizations: {output_dir}/class_imbalance_overview.png")
    print(f"📊 Per-label details: {output_dir}/per_label_imbalance.png")
    print(f"📄 Detailed report: {output_dir}/imbalance_report.json")

if __name__ == "__main__":
    main()