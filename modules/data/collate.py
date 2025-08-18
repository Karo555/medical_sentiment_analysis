# modules/data/collate.py
# -*- coding: utf-8 -*-
"""
Collate functions do DataLoaderów:
 - encoder_collate: paduje batch dla modeli ENCODER (regresja 21D),
 - llm_collate:     paduje batch dla LLM; opcjonalnie łączy input+target dla causal LM.

Użycie:
    collate_fn = encoder_collate(tokenizer, pad_to_multiple_of=8)

    # LLM – wariant rozdzielny (trainer sam łączy input i target):
    collate_fn = llm_collate(tokenizer, pad_to_multiple_of=8)

    # LLM – wariant concat_for_causal_lm (tworzy input_ids z [prompt + target] i labels z maską -100 na prompt):
    collate_fn = llm_collate(
        tokenizer,
        pad_to_multiple_of=8,
        join_input_target=True,
        label_pad_id=-100,
        max_combined_length=2048
    )
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizerBase


# --------------------------- Helpers ---------------------------

def _round_up_to_multiple(x: int, m: int) -> int:
    return ((x + m - 1) // m) * m

def _pad_1d(seqs: List[torch.Tensor], pad_val: int, pad_to_multiple_of: Optional[int] = None) -> torch.Tensor:
    """
    Pad listy 1D LongTensor do jednego kształtu [B, T]. Jeśli pad_to_multiple_of podany, wyrówna T w górę.
    """
    if not seqs:
        return torch.empty(0, dtype=torch.long)
    max_len = max(s.numel() for s in seqs)
    if pad_to_multiple_of:
        max_len = _round_up_to_multiple(max_len, pad_to_multiple_of)
    out = []
    for s in seqs:
        if s.numel() < max_len:
            pad_len = max_len - s.numel()
            s = torch.nn.functional.pad(s, (0, pad_len), value=pad_val)
        out.append(s)
    return torch.stack(out, dim=0)

def _pad_mask(seqs: List[torch.Tensor], pad_to_len: int) -> torch.Tensor:
    """
    Pad/extend attention_mask (1s) do [B, pad_to_len] wypełniając zerami.
    """
    out = []
    for s in seqs:
        if s.numel() < pad_to_len:
            pad_len = pad_to_len - s.numel()
            s = torch.nn.functional.pad(s, (0, pad_len), value=0)
        out.append(s)
    return torch.stack(out, dim=0)

def _stack_float(labels: List[torch.Tensor]) -> torch.Tensor:
    return torch.stack(labels, dim=0).to(dtype=torch.float32)


# ---------------------- Encoder collate ------------------------

@dataclass
class EncoderCollateConfig:
    pad_to_multiple_of: Optional[int] = 8
    # meta keys, jeśli mają zostać przekazane dalej (nie są tensorami)
    pass_meta_keys: Sequence[str] = ("id", "lang", "persona_id")


def encoder_collate(
    tokenizer: PreTrainedTokenizerBase,
    pad_to_multiple_of: Optional[int] = 8,
    pass_meta_keys: Sequence[str] = ("id", "lang", "persona_id"),
):
    """
    Zwraca funkcję collate_fn dla DataLoadera encodera.
    Oczekuje elementów batcha z kluczami: input_ids (Long[Ti]), attention_mask (Long[Ti]), labels (Float[21]), + meta.
    """
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        # XLM-R/mDeBERTa mają pad_token_id ustawiony; jeśli nie, awaryjnie przyjmij 0
        pad_id = 0

    def _fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_ids = [b["input_ids"] for b in batch]
        attn = [b["attention_mask"] for b in batch]
        labels = [b["labels"] for b in batch]

        # ustal docelową długość (T) — zgodną z pad_to_multiple_of (jeśli podane)
        max_len = max(x.numel() for x in input_ids) if input_ids else 0
        if pad_to_multiple_of:
            max_len = _round_up_to_multiple(max_len, pad_to_multiple_of)

        input_ids_pad = _pad_1d(input_ids, pad_val=pad_id, pad_to_multiple_of=None)  # najpierw do naturalnego max_len
        if input_ids_pad.size(1) < max_len:
            # dopaduj dodatkowo do wielokrotności
            extra = max_len - input_ids_pad.size(1)
            input_ids_pad = torch.nn.functional.pad(input_ids_pad, (0, extra), value=pad_id)

        attn_pad = _pad_mask(attn, pad_to_len=input_ids_pad.size(1))
        labels_stk = _stack_float(labels)

        out: Dict[str, Any] = {
            "input_ids": input_ids_pad,
            "attention_mask": attn_pad,
            "labels": labels_stk,
        }

        # meta (nie-tensorowe) przekazujemy jako listy
        for k in pass_meta_keys:
            if k in batch[0]:
                out[k] = [b.get(k) for b in batch]
        return out

    return _fn


# ------------------------ LLM collate --------------------------

@dataclass
class LLMCollateConfig:
    pad_to_multiple_of: Optional[int] = 8
    join_input_target: bool = False      # jeśli True -> przygotuj sekwencję [prompt + target] i labels z maską -100 na prompt
    label_pad_id: int = -100
    max_combined_length: Optional[int] = None  # twarde ograniczenie T po złączeniu; None = bez przycinania
    pass_meta_keys: Sequence[str] = ("id", "lang", "persona_id")

def llm_collate(
    tokenizer: PreTrainedTokenizerBase,
    pad_to_multiple_of: Optional[int] = 8,
    join_input_target: bool = False,
    label_pad_id: int = -100,
    max_combined_length: Optional[int] = None,
    pass_meta_keys: Sequence[str] = ("id", "lang", "persona_id"),
):
    """
    Collate dla LLM. Dwa tryby:
     - join_input_target=False: zwraca osobno input_ids/attention_mask (prompt) + surowe stringi input/target (trener łączy sam).
     - join_input_target=True:  zwraca złączone input_ids/attention_mask i labels (CE) z maską -100 dla promptu.
       WYMAGA, by dataset dostarczał *tokenizowany* target jako 'labels_input_ids'/'labels_attention_mask'.
    """
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def _pad_to_len(seq: torch.Tensor, length: int, pad_value: int) -> torch.Tensor:
        if seq.numel() < length:
            seq = torch.nn.functional.pad(seq, (0, length - seq.numel()), value=pad_value)
        return seq

    def _fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        # prompt
        prompt_ids = [b["input_ids"] for b in batch]
        prompt_mask = [b["attention_mask"] for b in batch]

        if not join_input_target:
            # klasyczny wariant: zostaw input zpad-owany, target surowy (JSON) + ewent. parsowane labels
            max_len = max(x.numel() for x in prompt_ids) if prompt_ids else 0
            if pad_to_multiple_of:
                max_len = _round_up_to_multiple(max_len, pad_to_multiple_of)

            input_ids_pad = _pad_1d(prompt_ids, pad_val=pad_id, pad_to_multiple_of=None)
            if input_ids_pad.size(1) < max_len:
                input_ids_pad = torch.nn.functional.pad(input_ids_pad, (0, max_len - input_ids_pad.size(1)), value=pad_id)

            attn_pad = _pad_mask(prompt_mask, pad_to_len=input_ids_pad.size(1))

            out: Dict[str, Any] = {
                "input_ids": input_ids_pad,
                "attention_mask": attn_pad,
                "input_text": [b["input_text"] for b in batch],
                "target_text": [b["target_text"] for b in batch],
            }

            # jeżeli dataset dodał zparsowane wektory etykiet (do metryk), przenieś:
            if "labels" in batch[0]:
                out["labels"] = torch.stack([b["labels"] for b in batch], dim=0).to(dtype=torch.float32)

            for k in pass_meta_keys:
                if k in batch[0]:
                    out[k] = [b.get(k) for b in batch]
            return out

        # --- join_input_target=True ---
        # wymagamy tokenizowanego targetu (labels_input_ids / labels_attention_mask)
        if "labels_input_ids" not in batch[0] or "labels_attention_mask" not in batch[0]:
            raise ValueError("join_input_target=True wymaga, by dataset zwracał 'labels_input_ids' i 'labels_attention_mask'.")

        tgt_ids = [b["labels_input_ids"] for b in batch]
        tgt_mask = [b["labels_attention_mask"] for b in batch]

        # sklej sekwencje
        combined_ids: List[torch.Tensor] = []
        combined_mask: List[torch.Tensor] = []
        combined_lbls: List[torch.Tensor] = []

        for p_ids, p_msk, t_ids, t_msk in zip(prompt_ids, prompt_mask, tgt_ids, tgt_mask):
            ids = torch.cat([p_ids, t_ids], dim=-1)
            msk = torch.cat([p_msk, t_msk], dim=-1)

            # labels: maskuj prompt -100, target = t_ids
            lbl_prompt = torch.full((p_ids.numel(),), fill_value=label_pad_id, dtype=torch.long)
            lbl_target = t_ids.clone().to(dtype=torch.long)
            lbl = torch.cat([lbl_prompt, lbl_target], dim=-1)

            # twarde przycięcie jeśli trzeba
            if max_combined_length is not None and ids.numel() > max_combined_length:
                ids = ids[:max_combined_length]
                msk = msk[:max_combined_length]
                lbl = lbl[:max_combined_length]

            combined_ids.append(ids)
            combined_mask.append(msk)
            combined_lbls.append(lbl)

        # padding do wspólnej długości (i ewentualnie do wielokrotności)
        max_len = max(x.numel() for x in combined_ids) if combined_ids else 0
        if pad_to_multiple_of:
            max_len = _round_up_to_multiple(max_len, pad_to_multiple_of)

        ids_pad = _pad_1d(combined_ids, pad_val=pad_id, pad_to_multiple_of=None)
        if ids_pad.size(1) < max_len:
            ids_pad = torch.nn.functional.pad(ids_pad, (0, max_len - ids_pad.size(1)), value=pad_id)

        mask_pad = _pad_mask(combined_mask, pad_to_len=ids_pad.size(1))
        lbls_pad = []
        for l in combined_lbls:
            if l.numel() < ids_pad.size(1):
                l = torch.nn.functional.pad(l, (0, ids_pad.size(1) - l.numel()), value=label_pad_id)
            lbls_pad.append(l)
        lbls_pad = torch.stack(lbls_pad, dim=0)

        out: Dict[str, Any] = {
            "input_ids": ids_pad,
            "attention_mask": mask_pad,
            "labels": lbls_pad,  # do causal LM
            "input_text": [b["input_text"] for b in batch],
            "target_text": [b["target_text"] for b in batch],
        }
        for k in pass_meta_keys:
            if k in batch[0]:
                out[k] = [b.get(k) for b in batch]
        return out

    return _fn