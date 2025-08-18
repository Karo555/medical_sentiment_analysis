# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any
import numpy as np
import torch
from transformers import TrainingArguments, Trainer
from modules.metrics.regression import compute_all_metrics

def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def build_trainer(model, tokenizer, train_ds, val_ds, training_cfg: Dict[str, Any]):
    args = TrainingArguments(
        output_dir=training_cfg.get("output_dir", "artifacts/models/encoder/run"),
        num_train_epochs=float(training_cfg.get("epochs", 3)),
        learning_rate=float(training_cfg.get("lr", 5e-5)),
        weight_decay=float(training_cfg.get("weight_decay", 0.01)),
        per_device_train_batch_size=int(training_cfg.get("train_bs", 16)),
        per_device_eval_batch_size=int(training_cfg.get("eval_bs", 32)),
        gradient_accumulation_steps=int(training_cfg.get("grad_accum", 1)),
        evaluation_strategy=training_cfg.get("evaluation_strategy", "epoch"),
        save_strategy=training_cfg.get("save_strategy", "epoch"),
        load_best_model_at_end=bool(training_cfg.get("load_best_at_end", True)),
        metric_for_best_model=training_cfg.get("metric_for_best_model", "r2"),
        greater_is_better=bool(training_cfg.get("greater_is_better", True)),
        logging_steps=int(training_cfg.get("logging_steps", 50)),
        warmup_ratio=float(training_cfg.get("warmup_ratio", 0.1)),
        fp16=bool(training_cfg.get("fp16", False)),
        bf16=bool(training_cfg.get("bf16", False)),
        report_to=training_cfg.get("report_to", ["none"]),
        save_total_limit=int(training_cfg.get("save_total_limit", 2)),
        dataloader_num_workers=int(training_cfg.get("num_workers", 2)),
        seed=int(training_cfg.get("seed", 1337)),
    )

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        preds = _to_numpy(preds)
        labels = _to_numpy(labels)
        return compute_all_metrics(labels, preds)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    return trainer
