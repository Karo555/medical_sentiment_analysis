# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class EncoderClassifier(nn.Module):
    """
    HF encoder (XLM-R / mDeBERTa) + mean-pooling + dropout + Linear -> 18 (multi-label binary classification).
    Returns dict compatible with Trainer: {"loss", "logits"}.
    """

    def __init__(
        self,
        model_name_or_path: str,
        out_dim: int = 18,
        dropout_prob: float = 0.1,
        use_fast_pooler: bool = True,
        output_hidden_states: bool = False,
        use_binary_classification: bool = True,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name_or_path, output_hidden_states=output_hidden_states)
        self.backbone = AutoModel.from_pretrained(model_name_or_path, config=self.config)
        hidden = getattr(self.config, "hidden_size", 768)
        self.dropout = nn.Dropout(dropout_prob)
        self.head = nn.Linear(hidden, out_dim)
        self.use_binary_classification = use_binary_classification
        
        # Store class weights
        self.class_weights = class_weights
        if self.class_weights is not None:
            self.register_buffer('pos_weight', self.class_weights)
        
        # Choose loss function based on problem type
        if use_binary_classification:
            if class_weights is not None:
                self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weights, reduction="mean")
            else:
                self.loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
        else:
            self.loss_fn = nn.MSELoss(reduction="mean")  # Regression
            
        self.use_fast_pooler = use_fast_pooler

    @staticmethod
    def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # mask: 1 for tokens, 0 for pad
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
        summed = (last_hidden_state * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        return summed / denom

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(out, "last_hidden_state"):
            token_emb = out.last_hidden_state
        else:
            token_emb = out[0]

        pooled = self.mean_pool(token_emb, attention_mask) if self.use_fast_pooler else token_emb[:, 0, :]
        pooled = self.dropout(pooled)
        logits = self.head(pooled)  # [B, D]

        res = {"logits": logits}
        if labels is not None:
            if self.use_binary_classification:
                # For binary classification, convert integer labels to float for BCEWithLogitsLoss
                labels_float = labels.float()
                loss = self.loss_fn(logits, labels_float)
            else:
                # For regression, use labels as is
                loss = self.loss_fn(logits, labels)
            res["loss"] = loss
        return res

    def resize_token_embeddings(self, new_size: int):
        return self.backbone.resize_token_embeddings(new_size)
