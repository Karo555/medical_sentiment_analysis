#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete LLM workflow: evaluation, training, visualization, and comparison.
This script orchestrates the entire process for LLM fine-tuning experiments.
"""
from __future__ import annotations
import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, Any


def run_command(cmd: list, description: str, timeout: int = 3600) -> bool:
    """Run a shell command with error handling."""
    print(f"\n=== {description} ===")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd, 
            check=True, 
            capture_output=True, 
            text=True,
            timeout=timeout
        )
        print("✓ Success")
        if result.stdout:
            print(f"Output:\n{result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed with return code {e.returncode}")
        if e.stdout:
            print(f"Stdout:\n{e.stdout}")
        if e.stderr:
            print(f"Stderr:\n{e.stderr}")
        return False
    except subprocess.TimeoutExpired:
        print(f"✗ Command timed out after {timeout} seconds")
        return False


def ensure_dependencies():
    """Ensure required dependencies are installed."""
    print("=== Checking Dependencies ===")
    
    # Check if we can import required modules
    try:
        import transformers
        import torch
        import peft
        print("✓ Core dependencies available")
        return True
    except ImportError as e:
        print(f"✗ Missing dependencies: {e}")
        print("Please install with: uv sync")
        return False


def run_baseline_evaluation(config_path: str, output_dir: Path) -> bool:
    """Run baseline evaluation of the pre-trained model."""
    cmd = [
        "python", "scripts/eval_llm.py",
        "--config", config_path,
        "--split", "val",
        "--max-samples", "50",  # Limit samples for faster evaluation
        "--output-dir", str(output_dir / "eval_baseline")
    ]
    
    return run_command(cmd, "Baseline Evaluation", timeout=1800)  # 30 minutes


def run_training(config_path: str) -> bool:
    """Run LLM fine-tuning."""
    cmd = [
        "python", "scripts/train_llm.py",
        "--config", config_path
    ]
    
    return run_command(cmd, "LLM Fine-tuning", timeout=7200)  # 2 hours


def run_post_training_evaluation(config_path: str, checkpoint_dir: Path, output_dir: Path) -> bool:
    """Run evaluation of the fine-tuned model."""
    cmd = [
        "python", "scripts/eval_llm.py",
        "--config", config_path,
        "--split", "val",
        "--checkpoint", str(checkpoint_dir),
        "--max-samples", "50",
        "--output-dir", str(output_dir / "eval_trained")
    ]
    
    return run_command(cmd, "Post-training Evaluation", timeout=1800)  # 30 minutes


def run_visualization(checkpoint_dir: Path, output_dir: Path) -> bool:
    """Create training visualizations."""
    cmd = [
        "python", "scripts/visualize_llm_training.py",
        "--checkpoint", str(checkpoint_dir),
        "--output-dir", str(output_dir / "visualization")
    ]
    
    return run_command(cmd, "Training Visualization", timeout=300)  # 5 minutes


def compare_results(baseline_dir: Path, trained_dir: Path, output_dir: Path) -> Dict[str, Any]:
    """Compare baseline vs trained model results."""
    print("\n=== Comparing Results ===")
    
    comparison = {
        "baseline": {},
        "trained": {},
        "improvement": {}
    }
    
    # Load baseline results
    baseline_metrics_path = baseline_dir / "eval_metrics.json"
    if baseline_metrics_path.exists():
        with baseline_metrics_path.open("r") as f:
            comparison["baseline"] = json.load(f)
    
    # Load trained results
    trained_metrics_path = trained_dir / "eval_metrics.json"
    if trained_metrics_path.exists():
        with trained_metrics_path.open("r") as f:
            comparison["trained"] = json.load(f)
    
    # Calculate improvements
    if comparison["baseline"] and comparison["trained"]:
        for metric in ["generation_success_rate", "r2", "mae", "mse"]:
            if metric in comparison["baseline"] and metric in comparison["trained"]:
                baseline_val = comparison["baseline"][metric]
                trained_val = comparison["trained"][metric]
                
                if metric == "mae" or metric == "mse":
                    # Lower is better for error metrics
                    improvement = ((baseline_val - trained_val) / baseline_val) * 100
                else:
                    # Higher is better for success rate and R²
                    improvement = ((trained_val - baseline_val) / baseline_val) * 100
                
                comparison["improvement"][metric] = improvement
    
    # Save comparison
    comparison_path = output_dir / "model_comparison.json"
    with comparison_path.open("w") as f:
        json.dump(comparison, f, indent=2)
    
    # Print summary
    print("Performance Comparison:")
    if comparison["improvement"]:
        for metric, improvement in comparison["improvement"].items():
            direction = "↑" if improvement > 0 else "↓"
            print(f"  {metric}: {improvement:+.2f}% {direction}")
    
    print(f"Detailed comparison saved to: {comparison_path}")
    return comparison


def main():
    """Main workflow function."""
    parser = argparse.ArgumentParser(description="Complete LLM training workflow")
    parser.add_argument("--config", required=True, help="Path to experiment configuration")
    parser.add_argument("--skip-baseline", action="store_true", help="Skip baseline evaluation")
    parser.add_argument("--skip-training", action="store_true", help="Skip training (use existing checkpoint)")
    parser.add_argument("--skip-visualization", action="store_true", help="Skip visualization")
    parser.add_argument("--quick-eval", action="store_true", help="Use fewer samples for faster evaluation")
    args = parser.parse_args()
    
    # Load configuration to get output directory
    import yaml
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    
    output_dir = Path(cfg["training"]["output_dir"])
    checkpoint_dir = output_dir
    
    print(f"=== LLM Complete Workflow ===")
    print(f"Config: {args.config}")
    print(f"Output: {output_dir}")
    print(f"Quick eval: {args.quick_eval}")
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check dependencies
    if not ensure_dependencies():
        print("Please install dependencies and try again.")
        return 1
    
    # Track timing
    start_time = time.time()
    workflow_log = {
        "config": args.config,
        "start_time": start_time,
        "steps_completed": [],
        "steps_failed": [],
    }
    
    # Step 1: Baseline evaluation
    if not args.skip_baseline:
        print("\n🔍 Step 1: Baseline Evaluation")
        if run_baseline_evaluation(args.config, output_dir):
            workflow_log["steps_completed"].append("baseline_evaluation")
        else:
            workflow_log["steps_failed"].append("baseline_evaluation")
            print("Warning: Baseline evaluation failed. Continuing with training...")
    
    # Step 2: Fine-tuning
    if not args.skip_training:
        print("\n🚀 Step 2: Fine-tuning")
        if run_training(args.config):
            workflow_log["steps_completed"].append("training")
        else:
            workflow_log["steps_failed"].append("training")
            print("Error: Training failed. Cannot continue.")
            return 1
    
    # Step 3: Post-training evaluation
    print("\n📊 Step 3: Post-training Evaluation")
    if run_post_training_evaluation(args.config, checkpoint_dir, output_dir):
        workflow_log["steps_completed"].append("post_training_evaluation")
    else:
        workflow_log["steps_failed"].append("post_training_evaluation")
        print("Warning: Post-training evaluation failed.")
    
    # Step 4: Visualization
    if not args.skip_visualization:
        print("\n📈 Step 4: Training Visualization")
        if run_visualization(checkpoint_dir, output_dir):
            workflow_log["steps_completed"].append("visualization")
        else:
            workflow_log["steps_failed"].append("visualization")
            print("Warning: Visualization failed.")
    
    # Step 5: Comparison
    print("\n🔄 Step 5: Results Comparison")
    baseline_dir = output_dir / "eval_baseline"
    trained_dir = output_dir / "eval_trained"
    
    if baseline_dir.exists() and trained_dir.exists():
        comparison = compare_results(baseline_dir, trained_dir, output_dir)
        workflow_log["comparison"] = comparison
        workflow_log["steps_completed"].append("comparison")
    else:
        print("Warning: Cannot compare results - missing evaluation directories.")
        workflow_log["steps_failed"].append("comparison")
    
    # Finalize workflow log
    end_time = time.time()
    workflow_log["end_time"] = end_time
    workflow_log["total_duration_minutes"] = (end_time - start_time) / 60
    
    # Save workflow log
    log_path = output_dir / "workflow_log.json"
    with log_path.open("w") as f:
        json.dump(workflow_log, f, indent=2)
    
    # Print final summary
    print(f"\n=== Workflow Complete ===")
    print(f"Duration: {workflow_log['total_duration_minutes']:.1f} minutes")
    print(f"Steps completed: {len(workflow_log['steps_completed'])}")
    print(f"Steps failed: {len(workflow_log['steps_failed'])}")
    print(f"Results saved to: {output_dir}")
    print(f"Workflow log: {log_path}")
    
    return 0 if not workflow_log["steps_failed"] else 1


if __name__ == "__main__":
    exit(main())