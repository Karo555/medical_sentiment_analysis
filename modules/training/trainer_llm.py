# modules/training/trainer_llm.py
# -*- coding: utf-8 -*-
"""
LLM trainer for supervised fine-tuning (SFT) on medical sentiment analysis.
Supports both causal language modeling (next token prediction) and custom metrics 
for emotion regression evaluation.
"""
from __future__ import annotations
from typing import Dict, Any, Optional, List, Union
import json
import numpy as np
import torch
import torch.nn as nn
from transformers import (
    TrainingArguments, 
    Trainer, 
    PreTrainedModel,
    PreTrainedTokenizerBase,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType
from inspect import signature
from pathlib import Path

from modules.metrics.regression import compute_all_metrics


def _to_numpy(x):
    """Convert tensor to numpy safely."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def load_llm_model(
    model_name_or_path: str,
    use_flash_attention: bool = False,
    torch_dtype: str = "auto",
    device_map: str = "auto",
    use_4bit: bool = False,
    use_8bit: bool = False,
    trust_remote_code: bool = False,
) -> PreTrainedModel:
    """Load LLM model for causal language modeling."""
    
    # Quantization config
    quantization_config = None
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif use_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    
    # Convert torch_dtype string to actual dtype
    if torch_dtype == "auto":
        torch_dtype_obj = None
    else:
        dtype_mapping = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype_obj = dtype_mapping.get(torch_dtype, torch.float16)
    
    # Build model loading kwargs
    model_kwargs = {
        "torch_dtype": torch_dtype_obj,
        "device_map": device_map,
        "quantization_config": quantization_config,
        "trust_remote_code": trust_remote_code,
    }
    
    # Add flash attention only if supported
    if use_flash_attention:
        try:
            # Try with attn_implementation first (newer approach)
            model_kwargs["attn_implementation"] = "flash_attention_2"
        except:
            try:
                # Fallback to use_flash_attention_2 parameter
                model_kwargs["use_flash_attention_2"] = True
            except:
                print("[WARNING] Flash attention requested but not supported, continuing without it")
    
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    
    return model


def apply_lora_to_model(
    model: PreTrainedModel,
    lora_config: Dict[str, Any],
) -> PreTrainedModel:
    """Apply LoRA (Low-Rank Adaptation) to the model."""
    from peft import prepare_model_for_kbit_training
    
    # Check if model is quantized (4-bit or 8-bit)
    is_quantized = getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False)
    
    # Prepare model for k-bit training if using quantization
    if is_quantized:
        print("[INFO] Model is quantized, preparing for k-bit training...")
        model = prepare_model_for_kbit_training(model)
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_config.get("r", 8),
        lora_alpha=lora_config.get("lora_alpha", 32),
        lora_dropout=lora_config.get("lora_dropout", 0.1),
        target_modules=lora_config.get("target_modules", ["q_proj", "v_proj"]),
        bias=lora_config.get("bias", "none"),
        modules_to_save=lora_config.get("modules_to_save", None),
    )
    
    model = get_peft_model(model, peft_config)
    return model


def parse_emotion_labels_from_generation(
    generated_texts: List[str],
    expected_length: int = 21
) -> List[Optional[List[float]]]:
    """
    Parse emotion labels from generated JSON strings.
    Expected format: {"labels": [0.1, 0.2, ...]}
    Returns None for unparseable generations.
    """
    results = []
    for text in generated_texts:
        try:
            # Try to find JSON in the text
            start_idx = text.find('{')
            end_idx = text.rfind('}')
            
            if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
                results.append(None)
                continue
            
            json_str = text[start_idx:end_idx+1]
            data = json.loads(json_str)
            
            if isinstance(data, dict) and "labels" in data:
                labels = data["labels"]
                if isinstance(labels, list) and len(labels) == expected_length:
                    # Convert to float and clamp to [0,1]
                    parsed_labels = []
                    for val in labels:
                        try:
                            float_val = float(val)
                            float_val = max(0.0, min(1.0, float_val))  # Clamp to [0,1]
                            parsed_labels.append(float_val)
                        except:
                            parsed_labels.append(0.0)  # Default fallback
                    results.append(parsed_labels)
                else:
                    results.append(None)
            else:
                results.append(None)
                
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            results.append(None)
    
    return results


class LLMTrainer(Trainer):
    """Custom trainer for LLM fine-tuning with emotion prediction metrics."""
    
    def __init__(self, eval_generation_config: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(**kwargs)
        self.eval_generation_config = eval_generation_config or {
            "max_new_tokens": 100,
            "do_sample": False,
            "temperature": 0.7,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        self.step_count = 0
        self.last_log_time = None
        
    def log(self, logs: Dict[str, float]) -> None:
        """Override log method to add custom logging."""
        # Add step information
        if self.state.global_step != self.step_count:
            self.step_count = self.state.global_step
            
        # Add timing information
        import time
        current_time = time.time()
        if self.last_log_time is not None:
            time_per_step = (current_time - self.last_log_time) / max(1, self.args.logging_steps)
            logs["seconds_per_step"] = time_per_step
            
            # Estimate remaining time
            if hasattr(self.state, 'max_steps') and self.state.max_steps > 0:
                remaining_steps = self.state.max_steps - self.state.global_step
                remaining_time_hours = (remaining_steps * time_per_step) / 3600
                logs["estimated_remaining_hours"] = remaining_time_hours
        
        self.last_log_time = current_time
        
        # Enhanced logging output
        if logs:
            print(f"\n[TRAINING] Step {self.state.global_step:,}")
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    if 'loss' in key.lower():
                        print(f"[TRAINING]   {key}: {value:.4f}")
                    elif 'lr' in key.lower() or 'learning_rate' in key.lower():
                        print(f"[TRAINING]   {key}: {value:.2e}")
                    elif 'time' in key.lower() or 'seconds' in key.lower():
                        print(f"[TRAINING]   {key}: {value:.3f}")
                    elif 'hours' in key.lower():
                        print(f"[TRAINING]   {key}: {value:.2f}h")
                    elif 'epoch' in key.lower():
                        print(f"[TRAINING]   {key}: {value:.2f}")
                    else:
                        print(f"[TRAINING]   {key}: {value:.4f}")
        
        super().log(logs)
        
    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        """Override to add evaluation progress tracking."""
        if self.control.should_evaluate:
            print(f"\n[EVALUATION] Starting evaluation at step {self.state.global_step}")
            
        result = super()._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
        
        if self.control.should_evaluate:
            print(f"[EVALUATION] ✅ Evaluation completed at step {self.state.global_step}")
            if hasattr(self.state, 'log_history') and self.state.log_history:
                latest_eval = self.state.log_history[-1]
                if any(k.startswith('eval_') for k in latest_eval.keys()):
                    print("[EVALUATION] Results:")
                    for key, value in latest_eval.items():
                        if key.startswith('eval_') and isinstance(value, (int, float)):
                            print(f"[EVALUATION]   {key}: {value:.4f}")
                            
        if self.control.should_save:
            print(f"[CHECKPOINT] Model saved at step {self.state.global_step}")
            
        return result
    
    def compute_metrics_for_generation(self, eval_dataset, eval_predictions=None):
        """
        Compute metrics by generating text and parsing emotion labels.
        This is computationally expensive and should be used sparingly.
        """
        if eval_predictions is not None:
            # Standard metrics from loss-based evaluation
            return self.compute_metrics(eval_predictions)
        
        # Generation-based evaluation
        self.model.eval()
        all_generated = []
        all_ground_truth = []
        
        # Generate predictions for a subset of eval data
        eval_sample_size = min(100, len(eval_dataset))  # Limit for performance
        
        for i in range(eval_sample_size):
            item = eval_dataset[i]
            input_text = item.get("input_text", "")
            
            # Tokenize input
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.eval_generation_config.get("max_length", 1024),
            ).to(self.model.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.eval_generation_config["max_new_tokens"],
                    do_sample=self.eval_generation_config.get("do_sample", False),
                    temperature=self.eval_generation_config.get("temperature", 0.7),
                    pad_token_id=self.eval_generation_config["pad_token_id"],
                )
            
            # Decode generated text
            generated_text = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:], 
                skip_special_tokens=True
            )
            all_generated.append(generated_text)
            
            # Get ground truth labels if available
            if "labels" in item:
                ground_truth = item["labels"].cpu().numpy().tolist()
                all_ground_truth.append(ground_truth)
        
        # Parse generated labels
        parsed_labels = parse_emotion_labels_from_generation(all_generated)
        
        # Calculate success rate
        valid_generations = sum(1 for labels in parsed_labels if labels is not None)
        generation_success_rate = valid_generations / len(parsed_labels)
        
        metrics = {"generation_success_rate": generation_success_rate}
        
        # If we have ground truth, compute regression metrics
        if all_ground_truth and valid_generations > 0:
            valid_pred = []
            valid_true = []
            
            for pred, true in zip(parsed_labels, all_ground_truth):
                if pred is not None:
                    valid_pred.append(pred)
                    valid_true.append(true)
            
            if valid_pred:
                pred_array = np.array(valid_pred)
                true_array = np.array(valid_true)
                regression_metrics = compute_all_metrics(true_array, pred_array)
                metrics.update({f"gen_{k}": v for k, v in regression_metrics.items()})
        
        return metrics


def build_llm_trainer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    train_ds,
    val_ds,
    training_cfg: Dict[str, Any],
    data_collator=None,
    compute_metrics_fn=None,
) -> LLMTrainer:
    """Build HuggingFace Trainer for LLM fine-tuning."""
    
    # Handle strategy alignment (copied from encoder trainer)
    eval_strat = str(training_cfg.get("evaluation_strategy", "epoch")).lower()
    save_strat = str(training_cfg.get("save_strategy", "epoch")).lower()
    load_best = bool(training_cfg.get("load_best_at_end", True))

    if load_best:
        if eval_strat == "no" and save_strat != "no":
            eval_strat = save_strat
        if save_strat == "no" and eval_strat != "no":
            save_strat = eval_strat
        if eval_strat == "no" and save_strat == "no":
            load_best = False

    # Build training arguments
    base_kwargs = dict(
        output_dir=training_cfg.get("output_dir", "artifacts/models/llm/run"),
        num_train_epochs=float(training_cfg.get("epochs", 3)),
        learning_rate=float(training_cfg.get("lr", 1e-4)),  # Lower LR for LLMs
        weight_decay=float(training_cfg.get("weight_decay", 0.01)),
        per_device_train_batch_size=int(training_cfg.get("train_bs", 1)),  # Smaller batch for LLMs
        per_device_eval_batch_size=int(training_cfg.get("eval_bs", 1)),
        gradient_accumulation_steps=int(training_cfg.get("grad_accum", 8)),  # Larger accum for LLMs
        eval_strategy=eval_strat,
        save_strategy=save_strat,
        load_best_model_at_end=load_best,
        metric_for_best_model=training_cfg.get("metric_for_best_model", "eval_loss"),
        greater_is_better=bool(training_cfg.get("greater_is_better", False)),  # loss should be minimized
        logging_steps=int(training_cfg.get("logging_steps", 10)),
        warmup_ratio=float(training_cfg.get("warmup_ratio", 0.03)),  # Lower warmup for LLMs
        fp16=bool(training_cfg.get("fp16", False)),
        bf16=bool(training_cfg.get("bf16", True)),  # Prefer bf16 for LLMs
        report_to=training_cfg.get("report_to", ["none"]),
        save_total_limit=int(training_cfg.get("save_total_limit", 2)),
        dataloader_num_workers=int(training_cfg.get("num_workers", 0)),  # Often 0 for LLMs
        seed=int(training_cfg.get("seed", 1337)),
        remove_unused_columns=bool(training_cfg.get("remove_unused_columns", False)),
        prediction_loss_only=bool(training_cfg.get("prediction_loss_only", False)),
    )

    # Handle steps-based evaluation/saving
    if eval_strat == "steps":
        base_kwargs["eval_steps"] = int(training_cfg.get("eval_steps", base_kwargs["logging_steps"]))
    if save_strat == "steps":
        base_kwargs["save_steps"] = int(training_cfg.get("save_steps", base_kwargs.get("eval_steps", base_kwargs["logging_steps"])))

    # Filter supported arguments by HF version
    sig = signature(TrainingArguments.__init__)
    supported = set(sig.parameters.keys())
    kwargs = {k: v for k, v in base_kwargs.items() if k in supported}

    # Backwards compatibility
    if "eval_strategy" not in supported and "evaluate_during_training" in supported:
        kwargs["evaluate_during_training"] = (eval_strat != "no")
        if eval_strat == "steps" and "eval_steps" in supported:
            kwargs["eval_steps"] = int(base_kwargs.get("eval_steps", 50))
    if "save_strategy" not in supported and "save_steps" in supported and save_strat == "steps":
        kwargs["save_steps"] = int(base_kwargs.get("save_steps", base_kwargs.get("logging_steps", 50)))

    # Remove unsupported fields
    for k in ["greater_is_better", "metric_for_best_model", "report_to"]:
        if k not in supported and k in kwargs:
            kwargs.pop(k, None)

    args = TrainingArguments(**kwargs)

    # Default compute_metrics function for loss-based evaluation
    if compute_metrics_fn is None:
        def compute_metrics(eval_pred):
            """Default metrics - just return the loss."""
            if hasattr(eval_pred, 'predictions') and hasattr(eval_pred, 'label_ids'):
                # For generation tasks, we might not have direct predictions
                return {}
            return {}
        compute_metrics_fn = compute_metrics

    # Generation config for evaluation
    eval_generation_config = training_cfg.get("eval_generation", {})
    if eval_generation_config.get("enabled", False):
        eval_generation_config["pad_token_id"] = tokenizer.pad_token_id

    return LLMTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
        eval_generation_config=eval_generation_config if eval_generation_config.get("enabled", False) else None,
    )


def prepare_model_for_training(model: PreTrainedModel, training_cfg: Dict[str, Any]) -> PreTrainedModel:
    """Prepare model for training (gradient checkpointing, etc.)"""
    
    # Enable gradient checkpointing if specified
    if training_cfg.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
    
    # Freeze certain layers if specified
    freeze_config = training_cfg.get("freeze", {})
    if freeze_config.get("embeddings", False):
        if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
            model.model.embed_tokens.requires_grad_(False)
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'wte'):
            model.transformer.wte.requires_grad_(False)
    
    # Freeze encoder layers if specified
    freeze_layers = freeze_config.get("layers", [])
    if freeze_layers:
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = model.model.layers
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            layers = model.transformer.h
        else:
            layers = []
        
        for layer_idx in freeze_layers:
            if 0 <= layer_idx < len(layers):
                layers[layer_idx].requires_grad_(False)
    
    return model