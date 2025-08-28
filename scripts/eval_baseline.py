#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation of PRE-TRAINED (baseline) encoder models on sentiment analysis task.

This script evaluates untrained transformer models (XLM-RoBERTa, mDeBERTa) 
by adding a regression head and testing on the medical sentiment dataset 
WITHOUT any fine-tuning.

Usage:
  python scripts/eval_baseline.py --model xlm-roberta-base --split val
  python scripts/eval_baseline.py --model microsoft/mdeberta-v3-base --split test
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from modules.models.encoder_regressor import EncoderRegressor
from modules.data.datasets import EncoderJsonlDataset, EncoderDatasetConfig
from modules.data.collate import encoder_collate
from modules.metrics.regression import compute_all_metrics


def load_yaml(path: str) -> Dict[str, Any]:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed_all(seed: int):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def run_inference(model: EncoderRegressor, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    model.eval()
    preds_list, labels_list, metadata = [], [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out["logits"]
        preds_list.append(logits.detach().cpu().numpy())
        labels_list.append(labels.detach().cpu().numpy())
        
        # Extract metadata for analysis
        batch_size = input_ids.shape[0]
        for i in range(batch_size):
            meta_item = {}
            if "id" in batch:
                meta_item["id"] = batch["id"][i] if i < len(batch["id"]) else str(len(metadata))
            if "lang" in batch:
                meta_item["lang"] = batch["lang"][i] if i < len(batch["lang"]) else "unknown"
            if "persona_id" in batch:
                meta_item["persona_id"] = batch["persona_id"][i] if i < len(batch["persona_id"]) else "unknown"
            metadata.append(meta_item)
            
    y_pred = np.concatenate(preds_list, axis=0) if preds_list else np.zeros((0, 21))
    y_true = np.concatenate(labels_list, axis=0) if labels_list else np.zeros((0, 21))
    return y_true, y_pred, metadata


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Model name/path (e.g., xlm-roberta-base, microsoft/mdeberta-v3-base)")
    ap.add_argument("--split", choices=["val", "test"], default="val", help="Which split to evaluate")
    ap.add_argument("--config", default="configs/experiment/enc_baseline.yaml", help="Config for data processing")
    ap.add_argument("--eval_bs", type=int, default=32, help="Evaluation batch size")
    ap.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    ap.add_argument("--output_dir", default=None, help="Output directory (default: artifacts/baseline_eval/{model_name})")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    seed = int(cfg.get("seed", 1337))
    set_seed_all(seed)

    # Data paths
    paths = cfg["paths"]
    enc_view_dir = Path(paths["encoder_view_dir"])
    file_key = args.split
    data_file = enc_view_dir / cfg["files"][file_key]
    if not data_file.is_file():
        raise FileNotFoundError(f"Split file not found: {data_file}")

    # Output directory
    model_safe_name = args.model.replace("/", "_").replace("-", "_")
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path("artifacts/baseline_eval") / model_safe_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Tokenizer - use model's default tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    # Model - create fresh model with regression head (NO checkpoint loading)
    model = EncoderRegressor(
        model_name_or_path=args.model,
        out_dim=21,  # 21D emotion regression
        dropout_prob=0.1,
        use_fast_pooler=True,
    )
    
    # Initialize regression head with small random weights
    torch.nn.init.normal_(model.head.weight, std=0.02)
    torch.nn.init.zeros_(model.head.bias)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Evaluating baseline model: {args.model}")
    print(f"Device: {device}")

    # Dataset / loader
    dcfg = cfg.get("data", {})
    enc_ds_cfg = EncoderDatasetConfig(
        max_length=int(dcfg.get("max_length", 256)),
        label_dim=int(dcfg.get("label_dim", 21)),
        clamp_labels_to=tuple(dcfg.get("clamp_labels_to", [0.0, 1.0])),
        return_meta=True,
    )
    ds = EncoderJsonlDataset(data_file, tokenizer, cfg=enc_ds_cfg)
    collate = encoder_collate(
        tokenizer,
        pad_to_multiple_of=8,
        pass_meta_keys=("id", "lang", "persona_id")
    )

    loader = DataLoader(ds, batch_size=args.eval_bs, shuffle=False, 
                       num_workers=args.num_workers, collate_fn=collate)

    # Inference
    y_true, y_pred, metadata = run_inference(model, loader, device)

    # Metrics
    metrics = compute_all_metrics(y_true, y_pred)

    # Save results
    eval_results = {
        "model": args.model,
        "split": args.split,
        "metrics": metrics,
        "predictions": y_pred.tolist(),
        "labels": y_true.tolist(),
        "metadata": metadata,
        "num_samples": len(y_true)
    }

    # Save detailed results
    results_file = out_dir / f"baseline_eval_{args.split}.json"
    with results_file.open("w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)

    # Save metrics only
    metrics_file = out_dir / f"metrics_{args.split}.json"
    with metrics_file.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Save predictions/labels
    np.savez_compressed(out_dir / f"preds_labels_{args.split}.npz", 
                       y_pred=y_pred, y_true=y_true)

    # Console summary
    print(f"[OK] Baseline eval on {args.split}:")
    print(f" MAE={metrics['mae']:.4f} | RMSE={metrics['rmse']:.4f} | R²={metrics['r2']:.4f} | Spearman={metrics['spearman']:.4f}")
    print(f"Results saved to: {results_file}")

if __name__ == "__main__":
    main()