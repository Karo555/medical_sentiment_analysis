from __future__ import annotations
from typing import Dict, Any
import numpy as np
import torch
from transformers import TrainingArguments, Trainer
from modules.metrics.regression import compute_all_metrics
from inspect import signature

def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def build_trainer(model, tokenizer, train_ds, val_ds, training_cfg: Dict[str, Any]):
    # zczytaj strategie i wyrównaj, jeśli trzeba
    eval_strat = str(training_cfg.get("evaluation_strategy", "epoch")).lower()
    save_strat = str(training_cfg.get("save_strategy", "epoch")).lower()
    load_best = bool(training_cfg.get("load_best_at_end", True))

    if load_best:
        if eval_strat == "no" and save_strat != "no":
            # jeśli ktoś zapomniał ustawić evaluation_strategy, dopasuj do save
            eval_strat = save_strat
        if save_strat == "no" and eval_strat != "no":
            save_strat = eval_strat
        if eval_strat == "no" and save_strat == "no":
            # nie da się użyć load_best bez ewaluacji i zapisu → wyłącz
            load_best = False

    base_kwargs = dict(
        output_dir=training_cfg.get("output_dir", "artifacts/models/encoder/run"),
        num_train_epochs=float(training_cfg.get("epochs", 3)),
        learning_rate=float(training_cfg.get("lr", 5e-5)),
        weight_decay=float(training_cfg.get("weight_decay", 0.01)),
        per_device_train_batch_size=int(training_cfg.get("train_bs", 16)),
        per_device_eval_batch_size=int(training_cfg.get("eval_bs", 32)),
        gradient_accumulation_steps=int(training_cfg.get("grad_accum", 1)),
        eval_strategy=eval_strat,
        save_strategy=save_strat,
        load_best_model_at_end=load_best,
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

    # jeśli steps, zadbaj o eval_steps/save_steps
    if eval_strat == "steps":
        base_kwargs["eval_steps"] = int(training_cfg.get("eval_steps", base_kwargs["logging_steps"]))
    if save_strat == "steps":
        base_kwargs["save_steps"] = int(training_cfg.get("save_steps", base_kwargs.get("eval_steps", base_kwargs["logging_steps"])))

    # filtruj po wersji HF
    sig = signature(TrainingArguments.__init__)
    supported = set(sig.parameters.keys())
    kwargs = {k: v for k, v in base_kwargs.items() if k in supported}

    # fallback dla starych HF: evaluate_during_training / save_steps
    if "eval_strategy" not in supported and "evaluate_during_training" in supported:
        kwargs["evaluate_during_training"] = (eval_strat != "no")
        if eval_strat == "steps" and "eval_steps" in supported:
            kwargs["eval_steps"] = int(base_kwargs.get("eval_steps", 50))
    if "save_strategy" not in supported and "save_steps" in supported and save_strat == "steps":
        kwargs["save_steps"] = int(base_kwargs.get("save_steps", base_kwargs.get("logging_steps", 50)))

    # niektóre wersje nie mają tych pól – usuń bez szkody
    for k in ["greater_is_better", "metric_for_best_model", "report_to"]:
        if k not in supported and k in kwargs:
            kwargs.pop(k, None)

    args = TrainingArguments(**kwargs)

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        preds = _to_numpy(preds)
        labels = _to_numpy(labels)
        return compute_all_metrics(labels, preds)

    return Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )