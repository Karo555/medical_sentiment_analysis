#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training baseline encoder (XLM-R / mDeBERTa) with binary classification head (21D).
Reads YAML config, loads tokenizer (from artifacts or HF), safely resizes embeddings,
builds DataSet/Collate, runs HF Trainer and saves model.

Usage:
  python scripts/train_encoder.py --config configs/experiment/enc_baseline.yaml
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, Any
import json
import yaml
import torch
from transformers import AutoTokenizer
from modules.models.encoder_classifier import EncoderClassifier
from modules.data.datasets import EncoderJsonlDataset, EncoderDatasetConfig
from modules.data.collate import encoder_collate
from modules.training.trainer_encoder import build_trainer
from modules.utils.class_weights import compute_class_weights_from_files

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def set_seed_all(seed: int):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def safe_resize_up_only(model, tokenizer):
    cur = model.backbone.get_input_embeddings().weight.shape[0]
    tgt = len(tokenizer)
    if tgt > cur:
        model.resize_token_embeddings(tgt)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    seed = int(cfg.get("seed", 1337))
    set_seed_all(seed)

    # ścieżki danych
    paths = cfg["paths"]
    enc_view_dir = Path(paths["encoder_view_dir"])
    train_path = enc_view_dir / cfg["files"]["train"]
    val_path = enc_view_dir / cfg["files"]["val"]

    # tokenizer
    tok_cfg = cfg["tokenizer"]
    tok_path = tok_cfg.get("path_or_name")  # artifacts/... lub HF id
    use_fast = bool(tok_cfg.get("use_fast", False))  # dla SP lepiej False
    tokenizer = AutoTokenizer.from_pretrained(tok_path, use_fast=use_fast)

    # class weights (optional)
    class_weights = None
    if cfg["model"].get("use_class_weights", False):
        print("[INFO] Computing class weights for imbalanced dataset...")
        weights_method = cfg["model"].get("class_weights_method", "balanced")
        smooth_factor = float(cfg["model"].get("class_weights_smooth", 1.0))
        
        weights_result = compute_class_weights_from_files(
            train_path, 
            method=weights_method,
            smooth_factor=smooth_factor
        )
        class_weights = weights_result['pos_weights']
        print(f"[INFO] Using {weights_method} class weights: min={class_weights.min():.3f}, max={class_weights.max():.3f}")
    
    # model
    model_name = cfg["model"]["name_or_path"]
    model = EncoderClassifier(
        model_name_or_path=model_name,
        out_dim=int(cfg["model"].get("out_dim", 21)),
        dropout_prob=float(cfg["model"].get("dropout", 0.1)),
        use_fast_pooler=bool(cfg["model"].get("use_fast_pooler", True)),
        use_binary_classification=bool(cfg["model"].get("use_binary_classification", True)),
        class_weights=class_weights,
    )
    # dopasuj embeddings do tokenizer (tylko w górę)
    safe_resize_up_only(model, tokenizer)

    # datasety
    dcfg = cfg.get("data", {})
    enc_ds_cfg = EncoderDatasetConfig(
        max_length=int(dcfg.get("max_length", 256)),
        label_dim=int(dcfg.get("label_dim", 21)),
        clamp_labels_to=tuple(dcfg.get("clamp_labels_to", [0.0, 1.0])),
        return_meta=bool(dcfg.get("return_meta", True)),
    )
    train_ds = EncoderJsonlDataset(train_path, tokenizer, cfg=enc_ds_cfg)
    val_ds = EncoderJsonlDataset(val_path, tokenizer, cfg=enc_ds_cfg)

    # collate
    collate = encoder_collate(
        tokenizer,
        pad_to_multiple_of=cfg.get("collate", {}).get("pad_to_multiple_of", 8),
        pass_meta_keys=tuple(cfg.get("collate", {}).get("pass_meta_keys", ["id","lang","persona_id"]))
    )

    # trainer
    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        train_ds=train_ds,
        val_ds=val_ds,
        training_cfg=cfg["training"],
    )

    # override collate_fn (HF domyślny nie wie o naszych batchach)
    trainer.data_collator = collate

    trainer.train()

    # zapisz metryki i model
    out_dir = Path(trainer.args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = trainer.evaluate()
    (out_dir / "eval_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    trainer.save_model(out_dir)  # zapisuje pytorch_model.bin + config

    print("[OK] Training finished.")
    print(f"[OK] Saved to: {out_dir}")

if __name__ == "__main__":
    main()
