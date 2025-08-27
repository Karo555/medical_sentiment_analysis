#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization script for LLM training results and metrics.
Creates plots for training curves, loss progression, and generation success rates.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def load_training_logs(checkpoint_dir: Path) -> Optional[pd.DataFrame]:
    """Load training logs from checkpoint directory."""
    
    # Try different possible log locations
    log_files = [
        checkpoint_dir / "trainer_state.json",
        checkpoint_dir / "training_logs.json",
        checkpoint_dir / "logs.json",
    ]
    
    # Also look for HuggingFace logs
    if (checkpoint_dir / "runs").exists():
        for run_dir in (checkpoint_dir / "runs").iterdir():
            if run_dir.is_dir():
                # Look for TensorBoard logs or JSON logs
                for log_file in run_dir.rglob("*.json"):
                    log_files.append(log_file)
    
    # Try to load trainer_state.json first (HuggingFace standard)
    trainer_state_path = checkpoint_dir / "trainer_state.json"
    if trainer_state_path.exists():
        try:
            with trainer_state_path.open("r") as f:
                state = json.load(f)
            
            # Extract log history
            if "log_history" in state and state["log_history"]:
                df = pd.DataFrame(state["log_history"])
                return df
        except Exception as e:
            print(f"Warning: Could not load trainer_state.json: {e}")
    
    # Try other log files
    for log_file in log_files:
        if log_file.exists():
            try:
                with log_file.open("r") as f:
                    logs = json.load(f)
                    if isinstance(logs, list):
                        return pd.DataFrame(logs)
                    elif isinstance(logs, dict) and "log_history" in logs:
                        return pd.DataFrame(logs["log_history"])
            except Exception as e:
                print(f"Warning: Could not load {log_file}: {e}")
                continue
    
    return None


def plot_training_curves(df: pd.DataFrame, output_dir: Path):
    """Create training curve plots."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("LLM Training Curves", fontsize=16)
    
    # Training and validation loss
    if "train_loss" in df.columns:
        axes[0, 0].plot(df["epoch"], df["train_loss"], label="Train Loss", color="blue", alpha=0.7)
    if "eval_loss" in df.columns:
        eval_df = df.dropna(subset=["eval_loss"])
        if not eval_df.empty:
            axes[0, 0].plot(eval_df["epoch"], eval_df["eval_loss"], label="Val Loss", color="red", alpha=0.7)
    
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Training and Validation Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Learning rate schedule
    if "learning_rate" in df.columns:
        lr_df = df.dropna(subset=["learning_rate"])
        if not lr_df.empty:
            axes[0, 1].plot(lr_df["epoch"], lr_df["learning_rate"], color="green", alpha=0.7)
            axes[0, 1].set_xlabel("Epoch")
            axes[0, 1].set_ylabel("Learning Rate")
            axes[0, 1].set_title("Learning Rate Schedule")
            axes[0, 1].grid(True, alpha=0.3)
    
    # Generation success rate (if available)
    gen_cols = [col for col in df.columns if "generation_success_rate" in col]
    if gen_cols:
        for col in gen_cols:
            gen_df = df.dropna(subset=[col])
            if not gen_df.empty:
                axes[1, 0].plot(gen_df["epoch"], gen_df[col], label=col.replace("eval_", ""), alpha=0.7)
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Success Rate")
        axes[1, 0].set_title("Generation Success Rate")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # R² score (if available)
    r2_cols = [col for col in df.columns if "r2" in col]
    if r2_cols:
        for col in r2_cols:
            r2_df = df.dropna(subset=[col])
            if not r2_df.empty:
                axes[1, 1].plot(r2_df["epoch"], r2_df[col], label=col.replace("eval_gen_", ""), alpha=0.7)
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("R² Score")
        axes[1, 1].set_title("Generation R² Score")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Training curves saved to: {output_dir / 'training_curves.png'}")


def plot_loss_progression(df: pd.DataFrame, output_dir: Path):
    """Create detailed loss progression plot."""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Plot training loss with steps
    if "train_loss" in df.columns and "step" in df.columns:
        train_df = df.dropna(subset=["train_loss"])
        ax.plot(train_df["step"], train_df["train_loss"], label="Training Loss", alpha=0.7, color="blue")
    
    # Plot validation loss
    if "eval_loss" in df.columns and "step" in df.columns:
        eval_df = df.dropna(subset=["eval_loss"])
        if not eval_df.empty:
            ax.scatter(eval_df["step"], eval_df["eval_loss"], label="Validation Loss", alpha=0.8, color="red", s=50)
    
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Progression During Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "loss_progression.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Loss progression saved to: {output_dir / 'loss_progression.png'}")


def create_metrics_summary(checkpoint_dir: Path, output_dir: Path):
    """Create summary of final metrics."""
    
    # Load final evaluation metrics
    eval_files = [
        checkpoint_dir / "eval_results_val.json",
        checkpoint_dir / "eval_metrics.json",
    ]
    
    metrics_data = {}
    
    for eval_file in eval_files:
        if eval_file.exists():
            try:
                with eval_file.open("r") as f:
                    metrics = json.load(f)
                    metrics_data.update(metrics)
            except Exception as e:
                print(f"Warning: Could not load {eval_file}: {e}")
    
    if metrics_data:
        # Create metrics summary plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Final Model Performance Summary", fontsize=14)
        
        # Extract key metrics
        key_metrics = ["generation_success_rate", "r2", "mae", "mse"]
        metric_values = []
        metric_names = []
        
        for metric in key_metrics:
            if metric in metrics_data:
                metric_values.append(metrics_data[metric])
                metric_names.append(metric.upper().replace("_", " "))
        
        # Bar plot of key metrics
        if metric_values:
            axes[0, 0].bar(metric_names, metric_values, alpha=0.7)
            axes[0, 0].set_title("Key Performance Metrics")
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Generation examples (if available)
        detailed_results_path = checkpoint_dir / "evaluation" / "val" / "detailed_results.jsonl"
        if detailed_results_path.exists():
            try:
                results = []
                with detailed_results_path.open("r") as f:
                    for line in f:
                        results.append(json.loads(line))
                
                # Success rate by position
                success_by_position = [r["valid"] for r in results[:20]]  # First 20 examples
                axes[0, 1].bar(range(len(success_by_position)), success_by_position, alpha=0.7)
                axes[0, 1].set_title("Generation Success (First 20 samples)")
                axes[0, 1].set_xlabel("Sample Index")
                axes[0, 1].set_ylabel("Success (1) / Failure (0)")
                
            except Exception as e:
                print(f"Warning: Could not load detailed results: {e}")
        
        plt.tight_layout()
        plt.savefig(output_dir / "metrics_summary.png", dpi=150, bbox_inches="tight")
        plt.close()
        
        # Save metrics to text file
        with (output_dir / "final_metrics.txt").open("w") as f:
            f.write("=== Final Model Performance ===\n\n")
            for key, value in metrics_data.items():
                if isinstance(value, (int, float)):
                    f.write(f"{key}: {value:.4f}\n")
                else:
                    f.write(f"{key}: {value}\n")
        
        print(f"Metrics summary saved to: {output_dir / 'metrics_summary.png'}")
        print(f"Final metrics saved to: {output_dir / 'final_metrics.txt'}")


def main():
    """Main visualization function."""
    parser = argparse.ArgumentParser(description="Visualize LLM training results")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint directory")
    parser.add_argument("--output-dir", help="Output directory for plots")
    args = parser.parse_args()
    
    checkpoint_dir = Path(args.checkpoint)
    if not checkpoint_dir.exists():
        print(f"Error: Checkpoint directory does not exist: {checkpoint_dir}")
        return
    
    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = checkpoint_dir / "visualization"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"=== LLM Training Visualization ===")
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"Output: {output_dir}")
    
    # Set style
    plt.style.use("seaborn-v0_8")
    sns.set_palette("husl")
    
    # Load training logs
    print("Loading training logs...")
    df = load_training_logs(checkpoint_dir)
    
    if df is not None:
        print(f"Loaded {len(df)} log entries")
        print("Available columns:", list(df.columns))
        
        # Create training curve plots
        print("Creating training curves...")
        plot_training_curves(df, output_dir)
        
        # Create loss progression plot
        print("Creating loss progression plot...")
        plot_loss_progression(df, output_dir)
        
    else:
        print("Warning: Could not load training logs. Skipping curve plots.")
    
    # Create metrics summary
    print("Creating metrics summary...")
    create_metrics_summary(checkpoint_dir, output_dir)
    
    print(f"\n=== Visualization Complete ===")
    print(f"All plots saved to: {output_dir}")


if __name__ == "__main__":
    main()