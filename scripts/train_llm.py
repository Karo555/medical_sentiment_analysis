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
import os
import warnings
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import yaml
from transformers import AutoTokenizer
from dotenv import load_dotenv
import huggingface_hub

# Load environment variables from .env file
load_dotenv()

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


def authenticate_huggingface():
    """Authenticate with Hugging Face using token from .env file."""
    hf_token = os.getenv('HUGGINGFACE_HUB_TOKEN') or os.getenv('HF_TOKEN')
    if hf_token:
        print("[INFO] Authenticating with Hugging Face...")
        huggingface_hub.login(token=hf_token)
        print("[INFO] Authentication successful")
    else:
        print("[WARNING] No HUGGINGFACE_HUB_TOKEN or HF_TOKEN found in .env file")
        print("[WARNING] Some models may not be accessible")

def setup_tokenizer(tokenizer_cfg: Dict[str, Any]) -> AutoTokenizer:
    """Setup tokenizer with proper padding configuration."""
    tok_path = tokenizer_cfg.get("path_or_name")
    use_fast = bool(tokenizer_cfg.get("use_fast", True))
    
    print(f"[INFO] Loading tokenizer from: {tok_path}")
    print(f"[INFO] Use fast tokenizer: {use_fast}")
    
    tokenizer = AutoTokenizer.from_pretrained(tok_path, use_fast=use_fast)
    
    print(f"[INFO] Tokenizer loaded. Vocab size: {len(tokenizer)}")
    
    # Set padding token if not present
    if tokenizer.pad_token is None:
        print("[INFO] No pad token found, setting up...")
        # Try to use eos_token as pad_token
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"[INFO] Using EOS token as pad token: {tokenizer.eos_token}")
        else:
            # Add a new pad token
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            print("[INFO] Added new [PAD] token")
    else:
        print(f"[INFO] Pad token already exists: {tokenizer.pad_token}")
    
    # Ensure padding side is correct for causal LM
    tokenizer.padding_side = tokenizer_cfg.get("padding_side", "left")  # "left" for causal LM
    print(f"[INFO] Padding side set to: {tokenizer.padding_side}")
    
    return tokenizer


def create_datasets(
    cfg: Dict[str, Any], 
    tokenizer: AutoTokenizer
) -> tuple[LLMJsonlDataset, LLMJsonlDataset]:
    """Create train and validation datasets."""
    
    print("[INFO] Creating datasets...")
    
    # Get paths
    paths = cfg["paths"] 
    llm_view_dir = Path(paths["llm_view_dir"])
    train_path = llm_view_dir / cfg["files"]["train"]
    val_path = llm_view_dir / cfg["files"]["val"]
    
    print(f"[INFO] Training data path: {train_path}")
    print(f"[INFO] Validation data path: {val_path}")
    
    # Check if files exist
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Validation data not found: {val_path}")
    
    # Dataset configuration
    data_cfg = cfg.get("data", {})
    llm_ds_cfg = LLMDatasetConfig(
        max_length=int(data_cfg.get("max_length", 1024)),
        return_meta=bool(data_cfg.get("return_meta", True)),
        tokenize_target=bool(data_cfg.get("tokenize_target", True)),  # Needed for causal LM
        parse_labels_from_json=bool(data_cfg.get("parse_labels_from_json", True)),  # For metrics
        clamp_labels_to=tuple(data_cfg.get("clamp_labels_to", [0.0, 1.0])),
    )
    
    print(f"[INFO] Dataset config - Max length: {llm_ds_cfg.max_length}")
    print(f"[INFO] Dataset config - Tokenize target: {llm_ds_cfg.tokenize_target}")
    print(f"[INFO] Dataset config - Parse labels from JSON: {llm_ds_cfg.parse_labels_from_json}")
    
    # Create datasets
    print("[INFO] Loading training dataset...")
    train_ds = LLMJsonlDataset(train_path, tokenizer, cfg=llm_ds_cfg)
    print("[INFO] Loading validation dataset...")
    val_ds = LLMJsonlDataset(val_path, tokenizer, cfg=llm_ds_cfg)
    
    print(f"[INFO] ✅ Loaded {len(train_ds)} training samples")
    print(f"[INFO] ✅ Loaded {len(val_ds)} validation samples")
    
    # Sample a few examples for verification
    if len(train_ds) > 0:
        sample = train_ds[0]
        print(f"[INFO] Sample input length: {len(sample['input_ids'])} tokens")
        print(f"[INFO] Sample keys: {list(sample.keys())}")
    
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
    
    # Authenticate with Hugging Face
    authenticate_huggingface()
    
    # Setup tokenizer
    print("[INFO] Setting up tokenizer...")
    tokenizer = setup_tokenizer(cfg["tokenizer"])
    
    # Load model
    print("[INFO] Loading model...")
    model_cfg = cfg["model"]
    print(f"[INFO] Model name: {model_cfg['name_or_path']}")
    print(f"[INFO] Flash attention: {model_cfg.get('use_flash_attention', False)}")
    print(f"[INFO] Torch dtype: {model_cfg.get('torch_dtype', 'auto')}")
    print(f"[INFO] 4-bit quantization: {model_cfg.get('use_4bit', False)}")
    print(f"[INFO] 8-bit quantization: {model_cfg.get('use_8bit', False)}")
    
    model = load_llm_model(
        model_name_or_path=model_cfg["name_or_path"],
        use_flash_attention=bool(model_cfg.get("use_flash_attention", False)),
        torch_dtype=model_cfg.get("torch_dtype", "auto"),
        device_map=model_cfg.get("device_map", "auto"),
        use_4bit=bool(model_cfg.get("use_4bit", False)),
        use_8bit=bool(model_cfg.get("use_8bit", False)),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
    )
    
    print(f"[INFO] ✅ Model loaded successfully")
    print(f"[INFO] Model device: {next(model.parameters()).device}")
    print(f"[INFO] Model dtype: {next(model.parameters()).dtype}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Total parameters: {total_params:,}")
    print(f"[INFO] Trainable parameters: {trainable_params:,}")
    
    # Apply LoRA if specified
    lora_cfg = model_cfg.get("lora")
    if lora_cfg and lora_cfg.get("enabled", False):
        print("[INFO] Applying LoRA configuration...")
        print(f"[INFO] LoRA rank (r): {lora_cfg.get('r', 8)}")
        print(f"[INFO] LoRA alpha: {lora_cfg.get('lora_alpha', 32)}")
        print(f"[INFO] LoRA dropout: {lora_cfg.get('lora_dropout', 0.1)}")
        print(f"[INFO] Target modules: {lora_cfg.get('target_modules', [])}")
        
        model = apply_lora_to_model(model, lora_cfg)
        print("[INFO] ✅ LoRA applied successfully")
        model.print_trainable_parameters()
        
        # Recount trainable parameters after LoRA
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[INFO] Trainable parameters after LoRA: {trainable_params:,}")
        print(f"[INFO] Trainable ratio: {trainable_params/total_params*100:.2f}%")
    else:
        print("[INFO] No LoRA configuration found, using full fine-tuning")
    
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
    training_cfg = cfg["training"]
    print(f"[INFO] Training config summary:")
    print(f"[INFO]   - Epochs: {training_cfg.get('epochs', 3)}")
    print(f"[INFO]   - Learning rate: {training_cfg.get('lr', 1e-4)}")
    print(f"[INFO]   - Train batch size: {training_cfg.get('train_bs', 1)}")
    print(f"[INFO]   - Eval batch size: {training_cfg.get('eval_bs', 1)}")
    print(f"[INFO]   - Gradient accumulation: {training_cfg.get('grad_accum', 8)}")
    print(f"[INFO]   - Warmup ratio: {training_cfg.get('warmup_ratio', 0.03)}")
    print(f"[INFO]   - Evaluation strategy: {training_cfg.get('evaluation_strategy', 'steps')}")
    print(f"[INFO]   - Save strategy: {training_cfg.get('save_strategy', 'steps')}")
    print(f"[INFO]   - Output directory: {training_cfg.get('output_dir')}")
    
    trainer = build_llm_trainer(
        model=model,
        tokenizer=tokenizer,
        train_ds=train_ds,
        val_ds=val_ds,
        training_cfg=cfg["training"],
        data_collator=data_collator,
    )
    
    # Calculate effective batch size
    effective_batch_size = training_cfg.get('train_bs', 1) * training_cfg.get('grad_accum', 8)
    total_steps = (len(train_ds) // effective_batch_size) * training_cfg.get('epochs', 3)
    print(f"[INFO] Effective batch size: {effective_batch_size}")
    print(f"[INFO] Estimated total training steps: {total_steps:,}")
    
    # Start training
    print("=" * 60)
    print("[INFO] 🚀 STARTING TRAINING 🚀")
    print("=" * 60)
    
    import time
    train_start_time = time.time()
    
    if args.resume_from_checkpoint:
        print(f"[INFO] Resuming from checkpoint: {args.resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        trainer.train()
    
    train_end_time = time.time()
    training_duration = train_end_time - train_start_time
    print("=" * 60)
    print(f"[INFO] ✅ TRAINING COMPLETED in {training_duration/3600:.1f} hours ({training_duration/60:.1f} minutes)")
    print("=" * 60)
    
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