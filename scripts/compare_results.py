#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare baseline (pre-trained) vs fine-tuned model results.

This script compares evaluation results between:
1. Baseline models (pre-trained with random regression head)  
2. Fine-tuned models (after training on medical sentiment data)

Usage:
  python scripts/compare_results.py --baseline artifacts/baseline_eval/xlm_roberta_base/baseline_eval_val.json --finetuned artifacts/models/encoder/enc_baseline_xlmr/eval_results_val.json
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np


def load_results(path: Path) -> Dict[str, Any]:
    """Load evaluation results from JSON file."""
    with path.open("r", encoding="utf-8") as f:
        results = json.load(f)
    
    # Try to load companion metrics file if main file lacks detailed metrics
    if "metrics" not in results or not any(k.startswith("per_label_") for k in results.get("metrics", {})):
        # Check for companion metrics.json in same directory
        metrics_path = path.parent / "metrics.json"
        if metrics_path.exists():
            with metrics_path.open("r", encoding="utf-8") as f:
                metrics = json.load(f)
                if "metrics" in results:
                    results["metrics"].update(metrics)
                else:
                    results["metrics"] = metrics
        
        # Also check for eval_val/metrics.json pattern
        eval_metrics_path = path.parent / "eval_val" / "metrics.json"
        if eval_metrics_path.exists():
            with eval_metrics_path.open("r", encoding="utf-8") as f:
                metrics = json.load(f)
                if "metrics" in results:
                    results["metrics"].update(metrics)
                else:
                    results["metrics"] = metrics
    
    # Calculate num_samples if missing
    if "num_samples" not in results:
        if "predictions" in results and isinstance(results["predictions"], list):
            results["num_samples"] = len(results["predictions"])
        elif "labels" in results and isinstance(results["labels"], list):
            results["num_samples"] = len(results["labels"])
        else:
            # Try to infer from preds.jsonl file if it exists in the same directory
            preds_path = path.parent / "preds.jsonl"
            if preds_path.exists():
                try:
                    with preds_path.open("r", encoding="utf-8") as f:
                        results["num_samples"] = sum(1 for line in f)
                except:
                    pass
    
    return results


def extract_metrics(results: Dict[str, Any]) -> Dict[str, float]:
    """Extract key metrics from results dictionary."""
    if "metrics" in results:
        metrics = results["metrics"]
    else:
        # Fallback for different result formats
        metrics = {k: v for k, v in results.items() if isinstance(v, (int, float))}
    
    key_metrics = {}
    for metric in ["mae", "rmse", "r2", "spearman"]:
        if metric in metrics:
            key_metrics[metric] = metrics[metric]
        elif f"macro_{metric}" in metrics:
            key_metrics[metric] = metrics[f"macro_{metric}"]
    
    return key_metrics


def compute_improvements(baseline_metrics: Dict[str, float], 
                        finetuned_metrics: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    """Compute improvement metrics."""
    improvements = {}
    
    for metric in baseline_metrics:
        if metric in finetuned_metrics:
            baseline_val = baseline_metrics[metric]
            finetuned_val = finetuned_metrics[metric]
            
            # For MAE and RMSE: lower is better (improvement = reduction)
            # For R2 and Spearman: higher is better (improvement = increase)
            if metric in ["mae", "rmse"]:
                abs_change = baseline_val - finetuned_val  # Positive = improvement
                pct_change = ((baseline_val - finetuned_val) / abs(baseline_val)) * 100 if baseline_val != 0 else 0
            else:  # r2, spearman
                abs_change = finetuned_val - baseline_val  # Positive = improvement  
                pct_change = ((finetuned_val - baseline_val) / abs(baseline_val)) * 100 if baseline_val != 0 else float('inf') if finetuned_val > 0 else 0
            
            improvements[metric] = {
                "baseline": baseline_val,
                "finetuned": finetuned_val, 
                "absolute_change": abs_change,
                "percent_change": pct_change
            }
    
    return improvements


def format_comparison_table(improvements: Dict[str, Dict[str, float]]) -> str:
    """Format comparison results as a table."""
    rows = []
    rows.append("| Metric | Baseline | Fine-tuned | Abs. Change | % Change |")
    rows.append("|--------|----------|------------|-------------|----------|")
    
    for metric, data in improvements.items():
        baseline = f"{data['baseline']:.4f}"
        finetuned = f"{data['finetuned']:.4f}"
        abs_change = f"{data['absolute_change']:+.4f}"
        
        if abs(data['percent_change']) == float('inf'):
            pct_change = "∞%" if data['percent_change'] > 0 else "-∞%"
        else:
            pct_change = f"{data['percent_change']:+.2f}%"
        
        rows.append(f"| {metric.upper()} | {baseline} | {finetuned} | {abs_change} | {pct_change} |")
    
    return "\n".join(rows)


def compute_per_label_improvements(baseline_results: Dict[str, Any], 
                                 finetuned_results: Dict[str, Any]) -> Dict[str, Any]:
    """Compute per-label improvements if available."""
    per_label_improvements = {}
    
    # Check if both results have per-label metrics
    baseline_metrics = baseline_results.get("metrics", {})
    finetuned_metrics = finetuned_results.get("metrics", {})
    
    for metric_type in ["mae", "rmse", "r2", "spearman"]:
        # Try both formats: "per_label_X" and "X_per_label"
        per_label_keys = [f"per_label_{metric_type}", f"{metric_type}_per_label"]
        baseline_vals = None
        finetuned_vals = None
        
        for key in per_label_keys:
            if key in baseline_metrics:
                baseline_vals = np.array(baseline_metrics[key])
            if key in finetuned_metrics:
                finetuned_vals = np.array(finetuned_metrics[key])
        
        if baseline_vals is not None and finetuned_vals is not None:
            if metric_type in ["mae", "rmse"]:
                improvements = baseline_vals - finetuned_vals  # Lower is better
            else:
                improvements = finetuned_vals - baseline_vals  # Higher is better
            
            per_label_improvements[metric_type] = {
                "baseline": baseline_vals.tolist(),
                "finetuned": finetuned_vals.tolist(), 
                "improvements": improvements.tolist(),
                "avg_improvement": float(np.mean(improvements))
            }
    
    return per_label_improvements


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="Path to baseline evaluation results JSON")
    ap.add_argument("--finetuned", required=True, help="Path to fine-tuned evaluation results JSON")  
    ap.add_argument("--output", default=None, help="Output file for comparison results (optional)")
    ap.add_argument("--model_name", default=None, help="Model name for output (optional)")
    args = ap.parse_args()

    baseline_path = Path(args.baseline)
    finetuned_path = Path(args.finetuned)
    
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline results not found: {baseline_path}")
    if not finetuned_path.exists():
        raise FileNotFoundError(f"Fine-tuned results not found: {finetuned_path}")

    # Load results
    baseline_results = load_results(baseline_path)
    finetuned_results = load_results(finetuned_path)
    
    # Extract metrics
    baseline_metrics = extract_metrics(baseline_results)
    finetuned_metrics = extract_metrics(finetuned_results)
    
    # Compute improvements
    improvements = compute_improvements(baseline_metrics, finetuned_metrics)
    per_label_improvements = compute_per_label_improvements(baseline_results, finetuned_results)
    
    # Create comparison results
    comparison_results = {
        "model_name": args.model_name or "unknown",
        "baseline_file": str(baseline_path),
        "finetuned_file": str(finetuned_path),
        "baseline_samples": baseline_results.get("num_samples", "unknown"),
        "finetuned_samples": finetuned_results.get("num_samples", "unknown"),
        "metric_improvements": improvements,
        "per_label_improvements": per_label_improvements
    }
    
    # Print summary
    print(f"Model Comparison: {args.model_name or 'Unknown Model'}")
    print("=" * 60)
    print(format_comparison_table(improvements))
    print()
    
    # Print per-label summary if available
    if per_label_improvements:
        print("Per-label Average Improvements:")
        print("-" * 40)
        for metric, data in per_label_improvements.items():
            print(f"{metric.upper()}: {data['avg_improvement']:+.4f}")
        print()
    
    # Determine overall assessment
    key_metrics = ["mae", "rmse", "r2", "spearman"]
    improved_count = 0
    total_count = 0
    
    for metric in key_metrics:
        if metric in improvements:
            total_count += 1
            if improvements[metric]["absolute_change"] > 0:
                improved_count += 1
    
    if total_count > 0:
        improvement_ratio = improved_count / total_count
        if improvement_ratio >= 0.75:
            assessment = "Significant improvement"
        elif improvement_ratio >= 0.5:
            assessment = "Moderate improvement"
        elif improvement_ratio >= 0.25:
            assessment = "Mixed results"
        else:
            assessment = "Little to no improvement"
        
        print(f"Overall Assessment: {assessment} ({improved_count}/{total_count} metrics improved)")
    
    # Save results if output path provided
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(comparison_results, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    main()