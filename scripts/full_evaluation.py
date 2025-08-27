#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full evaluation pipeline for sentiment analysis models.
Runs evaluation, visualization, and analysis in one command.
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

def run_command(cmd: List[str], description: str, cwd: str = ".") -> bool:
    """Run a command and return success status"""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"COMMAND: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, cwd=cwd, check=True, capture_output=False)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed with exit code {e.returncode}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Full evaluation pipeline for sentiment analysis models')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint directory')
    parser.add_argument('--config', required=True, help='Path to experiment config file')
    parser.add_argument('--split', default='val', choices=['val', 'test'], help='Dataset split to evaluate')
    parser.add_argument('--label-names', default='schema/label_names.json', help='Path to label names file')
    parser.add_argument('--skip-eval', action='store_true', help='Skip evaluation step (use existing results)')
    parser.add_argument('--skip-visualize', action='store_true', help='Skip visualization step')
    parser.add_argument('--skip-analyze', action='store_true', help='Skip analysis step')
    
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint path does not exist: {checkpoint_path}")
        sys.exit(1)
    
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config path does not exist: {config_path}")
        sys.exit(1)
    
    # Track success of each step
    steps_completed = []
    
    # Step 1: Model Evaluation
    if not args.skip_eval:
        eval_cmd = [
            'python', 'scripts/eval_encoder.py',
            '--config', str(config_path),
            '--split', args.split,
            '--checkpoint', str(checkpoint_path)
        ]
        if run_command(eval_cmd, f"Model evaluation on {args.split} set"):
            steps_completed.append("Evaluation")
    else:
        print("Skipping evaluation step")
        steps_completed.append("Evaluation (skipped)")
    
    # Step 2: Training Visualization
    if not args.skip_visualize:
        viz_cmd = [
            'python', 'scripts/visualize_training.py',
            '--checkpoint', str(checkpoint_path),
            '--label-names', args.label_names
        ]
        if run_command(viz_cmd, "Training visualization"):
            steps_completed.append("Visualization")
    else:
        print("Skipping visualization step")
        steps_completed.append("Visualization (skipped)")
    
    # Step 3: Results Analysis
    if not args.skip_analyze:
        analysis_cmd = [
            'python', 'scripts/analyze_results.py',
            '--checkpoint', str(checkpoint_path),
            '--split', args.split,
            '--label-names', args.label_names
        ]
        if run_command(analysis_cmd, "Results analysis"):
            steps_completed.append("Analysis")
    else:
        print("Skipping analysis step")
        steps_completed.append("Analysis (skipped)")
    
    # Summary
    print(f"\n{'='*60}")
    print("EVALUATION PIPELINE SUMMARY")
    print(f"{'='*60}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Config: {config_path}")
    print(f"Split: {args.split}")
    print("\nCompleted steps:")
    for step in steps_completed:
        print(f"  ✓ {step}")
    
    # Output locations
    print(f"\nOutput locations:")
    print(f"  - Evaluation metrics: {checkpoint_path}/eval_{args.split}/")
    print(f"  - Training plots: {checkpoint_path}/")
    print(f"  - Analysis results: {checkpoint_path}/analysis/")
    
    print("\nKey files generated:")
    print("  Training visualizations:")
    print("    - training_curves.png")
    print("    - per_label_performance.png") 
    print("    - metric_distributions.png")
    print("    - performance_summary.json")
    print("  Analysis results:")
    print("    - prediction_scatter.png")
    print("    - residual_analysis.png")
    print("    - error_correlation_heatmap.png")
    print("    - performance_by_language.png")
    print("    - performance_by_persona.png")
    print("    - comprehensive_analysis_report.json")

if __name__ == '__main__':
    main()