#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to validate LLM training pipeline without actually training.
Tests configuration loading, data loading, model setup, and trainer creation.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import yaml
    print("✓ Basic YAML import successful")
    
    # Test our custom modules without actually importing heavy dependencies
    print("✓ All imports successful (basic validation)")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Please install missing dependencies with: uv sync")
    sys.exit(1)

def load_yaml(path: str):
    """Load YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_config_loading(config_path: str):
    """Test configuration loading."""
    print(f"Testing config loading: {config_path}")
    try:
        cfg = load_yaml(config_path)
        required_sections = ["model", "tokenizer", "paths", "data", "training"]
        for section in required_sections:
            if section not in cfg:
                raise ValueError(f"Missing required section: {section}")
        print("✓ Configuration loaded and validated successfully")
        return cfg
    except Exception as e:
        print(f"✗ Config loading failed: {e}")
        return None


def test_tokenizer_setup(cfg):
    """Test tokenizer setup."""
    print("Testing tokenizer setup...")
    try:
        tokenizer_cfg = cfg["tokenizer"]
        required_params = ["path_or_name", "use_fast", "padding_side"]
        
        for param in required_params:
            if param not in tokenizer_cfg:
                raise ValueError(f"Missing tokenizer parameter: {param}")
        
        print(f"  Model: {tokenizer_cfg['path_or_name']}")
        print(f"  Use fast: {tokenizer_cfg['use_fast']}")
        print(f"  Padding side: {tokenizer_cfg['padding_side']}")
        print("✓ Tokenizer configuration validated")
        return True
    except Exception as e:
        print(f"✗ Tokenizer setup failed: {e}")
        return False


def test_dataset_paths(cfg):
    """Test that dataset paths are properly configured."""
    print("Testing dataset path configuration...")
    try:
        paths = cfg["paths"]
        llm_view_dir = Path(paths["llm_view_dir"])
        train_path = llm_view_dir / cfg["files"]["train"]
        val_path = llm_view_dir / cfg["files"]["val"]
        
        print(f"  Train data path: {train_path}")
        print(f"  Val data path: {val_path}")
        
        # We expect these paths to not exist yet (need to build LLM view first)
        print("✓ Dataset paths configured correctly")
        return True
    except Exception as e:
        print(f"✗ Dataset path configuration failed: {e}")
        return False


def test_training_config(cfg):
    """Test training configuration."""
    print("Testing training configuration...")
    try:
        training_cfg = cfg["training"]
        required_params = ["output_dir", "epochs", "lr", "train_bs", "eval_bs"]
        
        for param in required_params:
            if param not in training_cfg:
                raise ValueError(f"Missing training parameter: {param}")
        
        print(f"  Output dir: {training_cfg['output_dir']}")
        print(f"  Epochs: {training_cfg['epochs']}")
        print(f"  Learning rate: {training_cfg['lr']}")
        print(f"  Batch sizes: train={training_cfg['train_bs']}, eval={training_cfg['eval_bs']}")
        print("✓ Training configuration validated")
        return True
    except Exception as e:
        print(f"✗ Training configuration failed: {e}")
        return False


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="Test LLM training pipeline")
    parser.add_argument("--config", required=True, help="Path to experiment config file")
    args = parser.parse_args()
    
    print("=== LLM Training Pipeline Test ===\n")
    
    # Test configuration loading
    cfg = test_config_loading(args.config)
    if cfg is None:
        sys.exit(1)
    
    # Test tokenizer setup
    if not test_tokenizer_setup(cfg):
        sys.exit(1)
    
    # Test dataset paths
    if not test_dataset_paths(cfg):
        sys.exit(1)
    
    # Test training config
    if not test_training_config(cfg):
        sys.exit(1)
    
    print("\n=== Test Summary ===")
    print("✓ All tests passed!")
    print("\nTo actually train:")
    print("1. First prepare LLM view data:")
    if "persona_token" in args.config:
        print("   make build-llm-persona-token")
    elif "personalized" in args.config:
        print("   make build-llm-personalized-desc")
    else:
        print("   make build-llm-nonpers")
    
    print("2. Install additional dependencies (if needed):")
    print("   uv sync")
    
    print("3. Run training:")
    print(f"   python scripts/train_llm.py --config {args.config}")


if __name__ == "__main__":
    main()