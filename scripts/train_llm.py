#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training script for LLM fine-tuning on medical sentiment analysis.
Supports both full fine-tuning and LoRA (Low-Rank Adaptation).

Usage:
  python scripts/train_llm.py --config configs/experiment/llm_baseline.yaml
  python scripts/train_llm.py --config configs/experiment/llm_lora_gemma.yaml
"""
from __future__ import annotations
import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import yaml
from transformers import AutoTokenizer

from modules.data.datasets import LLMJsonlDataset, LLMDatasetConfig  
from modules.data.collate import llm_collate
from modules.training.trainer_llm import (
    load_llm_model,
    apply_lora_to_model, 
    build_llm_trainer,
    prepare_model_for_training
)


def load_yaml(path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed_all(seed: int):
    """Set random seeds for reproducibility."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed) 
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_tokenizer(tokenizer_cfg: Dict[str, Any]) -> AutoTokenizer:
    """Setup tokenizer with proper padding configuration."""
    tok_path = tokenizer_cfg.get("path_or_name")
    use_fast = bool(tokenizer_cfg.get("use_fast", True))
    
    tokenizer = AutoTokenizer.from_pretrained(tok_path, use_fast=use_fast)
    
    # Set padding token if not present
    if tokenizer.pad_token is None:
        # Try to use eos_token as pad_token
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            # Add a new pad token
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    
    # Ensure padding side is correct for causal LM
    tokenizer.padding_side = tokenizer_cfg.get("padding_side", "left")  # "left" for causal LM
    
    return tokenizer


def create_datasets(
    cfg: Dict[str, Any], 
    tokenizer: AutoTokenizer
) -> tuple[LLMJsonlDataset, LLMJsonlDataset]:
    """Create train and validation datasets."""
    
    # Get paths
    paths = cfg["paths"] 
    llm_view_dir = Path(paths["llm_view_dir"])
    train_path = llm_view_dir / cfg["files"]["train"]
    val_path = llm_view_dir / cfg["files"]["val"]
    
    # Dataset configuration
    data_cfg = cfg.get("data", {})
    llm_ds_cfg = LLMDatasetConfig(
        max_length=int(data_cfg.get("max_length", 1024)),
        return_meta=bool(data_cfg.get("return_meta", True)),
        tokenize_target=bool(data_cfg.get("tokenize_target", True)),  # Needed for causal LM
        parse_labels_from_json=bool(data_cfg.get("parse_labels_from_json", True)),  # For metrics
        clamp_labels_to=tuple(data_cfg.get("clamp_labels_to", [0.0, 1.0])),
    )
    
    # Create datasets
    train_ds = LLMJsonlDataset(train_path, tokenizer, cfg=llm_ds_cfg)
    val_ds = LLMJsonlDataset(val_path, tokenizer, cfg=llm_ds_cfg)
    
    print(f"[INFO] Loaded {len(train_ds)} training samples")
    print(f"[INFO] Loaded {len(val_ds)} validation samples")
    
    return train_ds, val_ds


def create_collate_fn(tokenizer: AutoTokenizer, cfg: Dict[str, Any]):
    """Create data collator for the model."""
    collate_cfg = cfg.get("collate", {})
    
    return llm_collate(
        tokenizer=tokenizer,
        pad_to_multiple_of=collate_cfg.get("pad_to_multiple_of", 8),
        join_input_target=bool(collate_cfg.get("join_input_target", True)),  # True for causal LM
        label_pad_id=int(collate_cfg.get("label_pad_id", -100)),
        max_combined_length=collate_cfg.get("max_combined_length", None),
        pass_meta_keys=tuple(collate_cfg.get("pass_meta_keys", ["id", "lang", "persona_id"])),
    )


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train LLM for medical sentiment analysis")
    parser.add_argument("--config", required=True, help="Path to experiment config file")
    parser.add_argument("--resume_from_checkpoint", type=str, help="Resume training from checkpoint")
    args = parser.parse_args()
    
    # Load configuration
    cfg = load_yaml(args.config)
    
    # Set seed for reproducibility
    seed = int(cfg.get("seed", 1337))
    set_seed_all(seed)
    
    print(f"[INFO] Starting LLM training with config: {args.config}")
    print(f"[INFO] Random seed: {seed}")
    
    # Setup tokenizer
    print("[INFO] Setting up tokenizer...")
    tokenizer = setup_tokenizer(cfg["tokenizer"])
    
    # Load model
    print("[INFO] Loading model...")
    model_cfg = cfg["model"]
    model = load_llm_model(
        model_name_or_path=model_cfg["name_or_path"],
        use_flash_attention=bool(model_cfg.get("use_flash_attention", False)),
        torch_dtype=model_cfg.get("torch_dtype", "auto"),
        device_map=model_cfg.get("device_map", "auto"),
        use_4bit=bool(model_cfg.get("use_4bit", False)),
        use_8bit=bool(model_cfg.get("use_8bit", False)),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
    )
    
    # Apply LoRA if specified
    lora_cfg = model_cfg.get("lora")
    if lora_cfg and lora_cfg.get("enabled", False):
        print("[INFO] Applying LoRA...")
        model = apply_lora_to_model(model, lora_cfg)
        model.print_trainable_parameters()
    
    # Resize tokenizer embeddings if needed
    if len(tokenizer) > model.config.vocab_size:
        print(f"[INFO] Resizing model embeddings from {model.config.vocab_size} to {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))
    
    # Prepare model for training
    model = prepare_model_for_training(model, cfg["training"])
    
    # Create datasets
    print("[INFO] Creating datasets...")
    train_ds, val_ds = create_datasets(cfg, tokenizer)
    
    # Create data collator
    print("[INFO] Setting up data collator...")
    data_collator = create_collate_fn(tokenizer, cfg)
    
    # Create trainer
    print("[INFO] Building trainer...")
    trainer = build_llm_trainer(
        model=model,
        tokenizer=tokenizer,
        train_ds=train_ds,
        val_ds=val_ds,
        training_cfg=cfg["training"],
        data_collator=data_collator,
    )
    
    # Start training
    print("[INFO] Starting training...")
    if args.resume_from_checkpoint:
        print(f"[INFO] Resuming from checkpoint: {args.resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        trainer.train()
    
    # Save model and metrics
    output_dir = Path(trainer.args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("[INFO] Evaluating on validation set...")
    eval_results = trainer.evaluate()
    
    # Save evaluation results
    eval_path = output_dir / "eval_results_val.json"
    with eval_path.open("w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)
    
    print("[INFO] Saving model...")
    trainer.save_model(output_dir)
    
    # Save training state
    trainer.save_state()
    
    # If LoRA was used, also save the adapter
    if lora_cfg and lora_cfg.get("enabled", False):
        lora_path = output_dir / "lora_adapter" 
        print(f"[INFO] Saving LoRA adapter to: {lora_path}")
        model.save_pretrained(lora_path)
    
    print(f"[OK] Training finished successfully!")
    print(f"[OK] Model saved to: {output_dir}")
    print(f"[OK] Final validation results:")
    for key, value in eval_results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()