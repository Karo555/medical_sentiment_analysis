#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ewaluacja wytrenowanego ENCODERA (regresja 21D) na {val|test}.

- Ładuje config eksperymentu (ten sam, co do treningu).
- Odtwarza model (EncoderRegressor) i wczytuje wagi z checkpointu (pytorch_model.bin).
- Buduje dataset/dataloader z widoku encodera (data/processed/encoder/*.jsonl).
- Oblicza MAE, RMSE, R², Spearman (macro i per-label).
- Zapisuje wyniki i predykcje do plików.

Użycie:
  python scripts/eval_encoder.py --config configs/experiment/enc_baseline.yaml --split val
  # opcjonalnie inny checkpoint:
  python scripts/eval_encoder.py --config ... --split test --checkpoint artifacts/models/encoder/enc_baseline_xlmr
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


def safe_resize_up_only(model: EncoderRegressor, tokenizer):
    cur = model.backbone.get_input_embeddings().weight.shape[0]
    tgt = len(tokenizer)
    if tgt > cur:
        model.resize_token_embeddings(tgt)


def load_checkpoint_weights(model: EncoderRegressor, ckpt_dir: Path):
    """Load model weights from either pytorch_model.bin or model.safetensors."""
    bin_path = ckpt_dir / "pytorch_model.bin"
    safetensors_path = ckpt_dir / "model.safetensors"
    
    if safetensors_path.is_file():
        # Use safetensors if available
        from safetensors.torch import load_file
        state = load_file(safetensors_path, device="cpu")
        print(f"Loading weights from: {safetensors_path}")
    elif bin_path.is_file():
        # Fallback to pytorch_model.bin
        state = torch.load(bin_path, map_location="cpu")
        print(f"Loading weights from: {bin_path}")
    else:
        raise FileNotFoundError(f"No checkpoint found: tried {bin_path} and {safetensors_path}")
    
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[WARN] Missing keys while loading: {missing[:10]}{'...' if len(missing)>10 else ''}")
    if unexpected:
        print(f"[WARN] Unexpected keys while loading: {unexpected[:10]}{'...' if len(unexpected)>10 else ''}")


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
    ap.add_argument("--config", required=True, help="Ścieżka do configs/experiment/*.yaml (jak w treningu)")
    ap.add_argument("--split", choices=["val", "test"], default="val", help="Który split ewaluować")
    ap.add_argument("--checkpoint", default=None, help="Ścieżka do katalogu checkpointu (domyślnie training.output_dir)")
    ap.add_argument("--eval_bs", type=int, default=None, help="Nadpisz batch size na ewaluacji")
    ap.add_argument("--num_workers", type=int, default=None, help="Nadpisz liczbę workerów DataLoadera")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    seed = int(cfg.get("seed", 1337))
    set_seed_all(seed)

    # ścieżki danych
    paths = cfg["paths"]
    enc_view_dir = Path(paths["encoder_view_dir"])
    file_key = args.split
    data_file = enc_view_dir / cfg["files"][file_key]
    if not data_file.is_file():
        raise FileNotFoundError(f"Split file not found: {data_file}")

    # checkpoint directory
    ckpt_dir = Path(args.checkpoint) if args.checkpoint else Path(cfg["training"]["output_dir"])
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint dir not found: {ckpt_dir}")

    # tokenizer
    tok_cfg = cfg["tokenizer"]
    tok_path = tok_cfg.get("path_or_name")
    use_fast = bool(tok_cfg.get("use_fast", False))  # SP → False
    tokenizer = AutoTokenizer.from_pretrained(tok_path, use_fast=use_fast)

    # model
    model_name = cfg["model"]["name_or_path"]
    model = EncoderRegressor(
        model_name_or_path=model_name,
        out_dim=int(cfg["model"].get("out_dim", 21)),
        dropout_prob=float(cfg["model"].get("dropout", 0.1)),
        use_fast_pooler=bool(cfg["model"].get("use_fast_pooler", True)),
    )
    # dopasuj embeddings do tokenizer (tylko w górę)
    safe_resize_up_only(model, tokenizer)
    # wczytaj wagi
    load_checkpoint_weights(model, ckpt_dir)

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # dataset / loader
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
        pad_to_multiple_of=cfg.get("collate", {}).get("pad_to_multiple_of", 8),
        pass_meta_keys=tuple(cfg.get("collate", {}).get("pass_meta_keys", ["id","lang","persona_id"]))
    )

    eval_bs = int(args.eval_bs or cfg["training"].get("eval_bs", 32))
    num_workers = int(args.num_workers or cfg["training"].get("num_workers", 2))
    loader = DataLoader(ds, batch_size=eval_bs, shuffle=False, num_workers=num_workers, collate_fn=collate)

    # inference
    y_true, y_pred, metadata = run_inference(model, loader, device)

    # metrics
    metrics = compute_all_metrics(y_true, y_pred)

    # zapisy
    # domyślnie zapisujemy do katalogu checkpointu
    out_dir = ckpt_dir / f"eval_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    # metryki
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    # predykcje + prawdy (lekko skompresowane)
    np.savez_compressed(out_dir / "preds_labels.npz", y_pred=y_pred, y_true=y_true)
    
    # Save results in format expected by analysis script
    results_for_analysis = {
        "predictions": y_pred.tolist(),
        "labels": y_true.tolist(),
        "metadata": metadata,
        "metrics": metrics
    }
    results_file = ckpt_dir / f"eval_results_{args.split}.json"
    with results_file.open("w", encoding="utf-8") as f:
        json.dump(results_for_analysis, f, indent=2, ensure_ascii=False)
    
    # (opcjonalnie) jsonl z id i predykcją – pomocne do debug
    preds_jsonl = out_dir / "preds.jsonl"
    with preds_jsonl.open("w", encoding="utf-8") as f:
        for i, row in enumerate(y_pred):
            meta_id = metadata[i].get("id", str(i)) if i < len(metadata) else str(i)
            rec = {"id": meta_id, "pred": row.tolist()}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # konsola – krótkie podsumowanie
    print(f"[OK] Eval on {args.split}:")
    print(f" MAE={metrics['mae']:.4f} | RMSE={metrics['rmse']:.4f} | R²={metrics['r2']:.4f} | Spearman={metrics['spearman']:.4f}")
    print(f"Saved metrics to: {out_dir/'metrics.json'}")
    print(f"Saved preds/labels to: {out_dir/'preds_labels.npz'}")
    print(f"Saved preds jsonl to: {preds_jsonl}")

if __name__ == "__main__":
    main()
