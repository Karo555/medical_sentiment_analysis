#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation script for LLM models on medical sentiment analysis.
Supports both pre-trained and fine-tuned model evaluation with generation-based metrics.
"""
from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from dotenv import load_dotenv
import huggingface_hub

# Load environment variables from .env file
load_dotenv()
from peft import PeftModel
import numpy as np
from tqdm import tqdm

from modules.data.datasets import LLMJsonlDataset, LLMDatasetConfig
from modules.training.trainer_llm import parse_emotion_labels_from_generation
from modules.metrics.regression import compute_all_metrics


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

def load_yaml(path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model_for_evaluation(
    model_name_or_path: str,
    lora_adapter_path: Optional[str] = None,
    use_4bit: bool = True,
    torch_dtype: str = "bfloat16",
    device_map: str = "auto",
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model and tokenizer for evaluation."""
    
    # Quantization config
    quantization_config = None
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    
    # Convert torch_dtype
    dtype_mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype_obj = dtype_mapping.get(torch_dtype, torch.bfloat16)
    
    # Load base model
    print(f"Loading model: {model_name_or_path}")
    print(f"  Torch dtype: {torch_dtype_obj}")
    print(f"  Device map: {device_map}")
    print(f"  Quantization: {'4-bit' if quantization_config else 'None'}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype_obj,
        device_map=device_map,
        quantization_config=quantization_config,
        trust_remote_code=True,
    )
    print(f"  ✓ Base model loaded")
    
    # Load LoRA adapter if specified
    if lora_adapter_path:
        print(f"Loading LoRA adapter: {lora_adapter_path}")
        model = PeftModel.from_pretrained(model, lora_adapter_path)
        print(f"  ✓ LoRA adapter loaded")
        model = model.merge_and_unload()  # Merge for inference
        print(f"  ✓ LoRA adapter merged with base model")
    
    # Load tokenizer
    print(f"Loading tokenizer from: {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"  Set pad_token to eos_token: {tokenizer.pad_token}")
    tokenizer.padding_side = "left"
    print(f"  ✓ Tokenizer loaded (vocab size: {len(tokenizer)})")
    
    model.eval()
    print(f"  ✓ Model set to evaluation mode")
    return model, tokenizer


def generate_predictions(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset: LLMJsonlDataset,
    generation_config: Dict[str, Any],
    max_samples: Optional[int] = None,
) -> tuple[List[str], List[str], List[Optional[List[float]]]]:
    """Generate predictions for the dataset."""
    
    all_prompts = []
    all_generated = []
    all_ground_truth = []
    
    n_samples = min(len(dataset), max_samples) if max_samples else len(dataset)
    
    print(f"Generating predictions for {n_samples} samples...")
    print(f"Generation config: max_new_tokens={generation_config.get('max_new_tokens', 100)}, "
          f"temperature={generation_config.get('temperature', 0.1)}, "
          f"do_sample={generation_config.get('do_sample', False)}")
    
    with torch.no_grad():
        for i in tqdm(range(n_samples), desc="Generating", 
                     unit="samples", dynamic_ncols=True):
            item = dataset[i]
            input_text = item["input_text"]
            all_prompts.append(input_text)
            
            # Tokenize input
            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=generation_config.get("max_input_length", 1024),
            ).to(model.device)
            
            # Generate
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=generation_config.get("max_new_tokens", 100),
                    do_sample=generation_config.get("do_sample", False),
                    temperature=generation_config.get("temperature", 0.1),
                    top_p=generation_config.get("top_p", 0.9),
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            # Decode generated text (only the new tokens)
            generated_text = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:], 
                skip_special_tokens=True
            )
            all_generated.append(generated_text)
            
            # Get ground truth labels if available
            if "labels" in item:
                ground_truth = item["labels"].cpu().numpy().tolist()
                all_ground_truth.append(ground_truth)
            else:
                all_ground_truth.append(None)
    
    return all_prompts, all_generated, all_ground_truth


def evaluate_generations(
    generated_texts: List[str],
    ground_truth_labels: List[Optional[List[float]]],
    output_dir: Path,
) -> Dict[str, Any]:
    """Evaluate generated text against ground truth labels."""
    
    # Parse emotion labels from generated JSON
    parsed_labels = parse_emotion_labels_from_generation(generated_texts)
    
    # Calculate success rate
    valid_generations = sum(1 for labels in parsed_labels if labels is not None)
    generation_success_rate = valid_generations / len(parsed_labels)
    
    metrics = {
        "generation_success_rate": generation_success_rate,
        "total_samples": len(generated_texts),
        "valid_generations": valid_generations,
    }
    
    # If we have ground truth, compute regression metrics
    valid_ground_truth = [gt for gt in ground_truth_labels if gt is not None]
    if valid_ground_truth and valid_generations > 0:
        valid_pred = []
        valid_true = []
        
        for pred, true in zip(parsed_labels, ground_truth_labels):
            if pred is not None and true is not None:
                valid_pred.append(pred)
                valid_true.append(true)
        
        if valid_pred:
            pred_array = np.array(valid_pred)
            true_array = np.array(valid_true)
            regression_metrics = compute_all_metrics(true_array, pred_array)
            metrics.update(regression_metrics)
    
    # Save detailed results
    results_path = output_dir / "detailed_results.jsonl"
    with results_path.open("w", encoding="utf-8") as f:
        for i, (gen_text, parsed, gt) in enumerate(zip(generated_texts, parsed_labels, ground_truth_labels)):
            result = {
                "sample_id": i,
                "generated_text": gen_text,
                "parsed_labels": parsed,
                "ground_truth": gt,
                "valid": parsed is not None,
            }
            f.write(json.dumps(result) + "\n")
    
    print(f"Detailed results saved to: {results_path}")
    return metrics


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate LLM on medical sentiment analysis")
    parser.add_argument("--config", required=True, help="Path to experiment config")
    parser.add_argument("--checkpoint", help="Path to model checkpoint (for fine-tuned models)")
    parser.add_argument("--lora-adapter", help="Path to LoRA adapter")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"], help="Data split to evaluate")
    parser.add_argument("--max-samples", type=int, help="Maximum number of samples to evaluate")
    parser.add_argument("--output-dir", help="Output directory for results")
    args = parser.parse_args()
    
    # Load configuration
    cfg = load_yaml(args.config)
    
    print(f"=== LLM Evaluation ===")
    print(f"Config: {args.config}")
    print(f"Split: {args.split}")
    print(f"Max samples: {args.max_samples or 'All'}")
    
    # Authenticate with Hugging Face
    authenticate_huggingface()
    
    # Determine model path
    if args.checkpoint:
        model_path = args.checkpoint
        print(f"Using checkpoint: {model_path}")
    else:
        model_path = cfg["model"]["name_or_path"]
        print(f"Using base model: {model_path}")
    
    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(cfg["training"]["output_dir"]) / "evaluation" / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model and tokenizer
    print(f"\n=== Loading Model ===")
    print(f"Model path: {model_path}")
    print(f"Use 4-bit quantization: {cfg['model'].get('use_4bit', True)}")
    print(f"Torch dtype: {cfg['model'].get('torch_dtype', 'bfloat16')}")
    if args.lora_adapter:
        print(f"LoRA adapter: {args.lora_adapter}")
    
    model, tokenizer = load_model_for_evaluation(
        model_name_or_path=model_path,
        lora_adapter_path=args.lora_adapter,
        use_4bit=cfg["model"].get("use_4bit", True),
        torch_dtype=cfg["model"].get("torch_dtype", "bfloat16"),
    )
    print(f"✓ Model loaded successfully")
    
    # Load dataset
    print(f"\n=== Loading Dataset ===")
    paths = cfg["paths"]
    llm_view_dir = Path(paths["llm_view_dir"])
    data_file = cfg["files"][args.split]
    data_path = llm_view_dir / data_file
    print(f"Data path: {data_path}")
    
    data_cfg = cfg.get("data", {})
    llm_ds_cfg = LLMDatasetConfig(
        max_length=int(data_cfg.get("max_length", 1024)),
        return_meta=bool(data_cfg.get("return_meta", True)),
        tokenize_target=False,  # We don't need tokenized target for evaluation
        parse_labels_from_json=bool(data_cfg.get("parse_labels_from_json", True)),
    )
    print(f"Dataset config - Max length: {llm_ds_cfg.max_length}, Return meta: {llm_ds_cfg.return_meta}")
    
    dataset = LLMJsonlDataset(data_path, tokenizer, cfg=llm_ds_cfg)
    print(f"✓ Loaded {len(dataset)} samples from {data_path}")
    
    if args.max_samples:
        print(f"Will evaluate only first {args.max_samples} samples")
    
    # Generation configuration
    print(f"\n=== Generation Configuration ===")
    generation_config = cfg.get("training", {}).get("eval_generation", {})
    if not generation_config:
        generation_config = {
            "max_new_tokens": 100,
            "do_sample": False,
            "temperature": 0.1,
        }
        print("Using default generation config")
    else:
        print("Using config-specified generation settings")
    
    for key, value in generation_config.items():
        print(f"  {key}: {value}")
    
    # Generate predictions
    print(f"\n=== Starting Generation ===")
    start_time = time.time()
    prompts, generated_texts, ground_truth = generate_predictions(
        model, tokenizer, dataset, generation_config, args.max_samples
    )
    generation_time = time.time() - start_time
    
    print(f"\n✓ Generation completed in {generation_time:.2f} seconds")
    print(f"Average time per sample: {generation_time / len(generated_texts):.3f} seconds")
    
    # Evaluate results
    print(f"\n=== Evaluating Results ===")
    print("Parsing JSON responses and computing metrics...")
    metrics = evaluate_generations(generated_texts, ground_truth, output_dir)
    
    # Add timing metrics
    metrics["generation_time_seconds"] = generation_time
    metrics["avg_time_per_sample"] = generation_time / len(generated_texts)
    
    # Save metrics
    metrics_path = output_dir / "eval_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    
    # Print results
    print(f"\n=== Evaluation Results ===")
    print(f"Generation success rate: {metrics['generation_success_rate']:.3f}")
    if "r2" in metrics:
        print(f"R² score: {metrics['r2']:.3f}")
    if "mae" in metrics:
        print(f"MAE: {metrics['mae']:.3f}")
    if "mse" in metrics:
        print(f"MSE: {metrics['mse']:.3f}")
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()