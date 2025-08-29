# modules/data/datasets.py
# -*- coding: utf-8 -*-
"""
Zestaw prostych Datasetów do wczytywania widoków JSONL zbudowanych przez:
 - scripts/build_encoder_view.py
 - scripts/build_llm_view.py

Kontrakty wejściowe (kolumny w JSONL):
ENCODER VIEW (data/processed/encoder/*.jsonl)
  { "id": str, "input_text": str, "labels": [float x21], "lang": "pl|en", "persona_id": str? }

LLM VIEW (data/processed/llm/*.jsonl)
  { "id": str, "input_text": str, "target_text": str(JSON), "lang": "pl|en", "persona_id": str? }

Uwaga:
- Tokenizacja sekwencji (input_text) jest wykonywana w __getitem__ (on-the-fly).
- Wersja LLM nie scala input + target; zwraca je osobno (tekstowo i/lub ztokenizowane).
  Sklejanie + maskowanie do uczenia CE/LoRA robimy w collate_fn/trainerze.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import json
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


# ============================ I/O Helpers ============================

def read_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Wczytuje plik JSONL do listy dictów."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"JSONL not found: {p}")
    out: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            out.append(json.loads(ln))
    return out


# ============================ Base dataset ============================

@dataclass
class EncoderDatasetConfig:
    text_key: str = "input_text"
    label_key: str = "labels"
    id_key: str = "id"
    lang_key: str = "lang"
    persona_key: str = "persona_id"
    max_length: int = 512
    add_special_tokens: bool = True
    truncation: bool = True
    return_meta: bool = True                # dołącz "id"/"lang"/"persona_id" w itemie
    label_dim: Optional[int] = 18           # weryfikacja długości wektora etykiet (None = bez checku)
    clamp_labels_to: Optional[Tuple[float, float]] = (0.0, 1.0)  # None = bez clampu
    dtype_labels: torch.dtype = torch.float32


class EncoderJsonlDataset(Dataset):
    """
    Dataset pod ENCODER (multi-label binary classification 18D). Zwraca:
      {
        "input_ids": LongTensor [L],
        "attention_mask": LongTensor [L],
        "labels": FloatTensor [D],
        (+ opcjonalnie) "id", "lang", "persona_id"
      }
    """

    def __init__(
        self,
        jsonl_path: Union[str, Path],
        tokenizer: PreTrainedTokenizerBase,
        cfg: Optional[EncoderDatasetConfig] = None,
        filter_langs: Optional[Sequence[str]] = None,
        filter_personas: Optional[Sequence[str]] = None,
    ):
        self.path = Path(jsonl_path)
        self.records = read_jsonl(self.path)
        self.tok = tokenizer
        self.cfg = cfg or EncoderDatasetConfig()

        # Proste filtry (opcjonalnie)
        if filter_langs:
            self.records = [r for r in self.records if r.get(self.cfg.lang_key) in set(filter_langs)]
        if filter_personas:
            self.records = [r for r in self.records if r.get(self.cfg.persona_key) in set(filter_personas)]

        # Sanity check etykiet
        if self.cfg.label_dim is not None:
            bad = [i for i, r in enumerate(self.records) if len(r.get(self.cfg.label_key, [])) != self.cfg.label_dim]
            if bad:
                raise ValueError(
                    f"Found {len(bad)} records with wrong label size; first idx={bad[:5]} "
                    f"(expected {self.cfg.label_dim}, key={self.cfg.label_key})"
                )

    def __len__(self) -> int:
        return len(self.records)

    def _clamp_labels(self, labels: List[float]) -> List[float]:
        if self.cfg.clamp_labels_to is None:
            return labels
        lo, hi = self.cfg.clamp_labels_to
        out = []
        for x in labels:
            try:
                v = float(x)
            except Exception:
                v = 0.0
            v = max(lo, min(hi, v))
            out.append(v)
        return out

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.records[idx]
        text = r[self.cfg.text_key]

        enc = self.tok(
            text,
            add_special_tokens=self.cfg.add_special_tokens,
            truncation=self.cfg.truncation,
            max_length=self.cfg.max_length,
            padding=False,  # padding robimy w collate_fn
            return_tensors=None,
        )
        labels = self._clamp_labels(r[self.cfg.label_key])

        item: Dict[str, Any] = {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(labels, dtype=self.cfg.dtype_labels),
        }

        if self.cfg.return_meta:
            if self.cfg.id_key in r:
                item["id"] = r[self.cfg.id_key]
            if self.cfg.lang_key in r:
                item["lang"] = r[self.cfg.lang_key]
            if self.cfg.persona_key in r:
                item["persona_id"] = r[self.cfg.persona_key]

        return item


# ============================ LLM dataset ============================

@dataclass
class LLMDatasetConfig:
    text_key: str = "input_text"
    target_text_key: str = "target_text"   # JSON string: {"labels":[...]}
    id_key: str = "id"
    lang_key: str = "lang"
    persona_key: str = "persona_id"
    max_length: int = 2048
    add_special_tokens: bool = True
    truncation: bool = True
    return_meta: bool = True
    # Opcjonalna tokenizacja targetu (do SFT). Jeśli True -> zwracamy także "labels_input_ids"/"labels_attention_mask".
    tokenize_target: bool = False
    # Uwaga: łączenie input + target i maskowanie lossu robimy w collate_fn/trainerze.
    clamp_labels_to: Optional[Tuple[float, float]] = (0.0, 1.0)  # gdy parsujemy z JSON do listy
    dtype_labels: torch.dtype = torch.float32
    # Czy równolegle zwracać listę floatów wyjętą z target_text (przydaje się do metryk R^2 na val)
    parse_labels_from_json: bool = True

class LLMJsonlDataset(Dataset):
    """
    Dataset pod LLM SFT. Zwraca:
      - zawsze: "input_ids", "attention_mask" (tokenizacja input_text),
      - zawsze: "input_text" i "target_text" (surowe stringi do późniejszego sklejania),
      - opcjonalnie: "labels_input_ids", "labels_attention_mask" (jeśli cfg.tokenize_target=True),
      - opcjonalnie: "labels" (lista floatów z JSON w target_text, jeśli cfg.parse_labels_from_json=True),
      - opcjonalnie: meta: "id", "lang", "persona_id".
    """

    def __init__(
        self,
        jsonl_path: Union[str, Path],
        tokenizer: PreTrainedTokenizerBase,
        cfg: Optional[LLMDatasetConfig] = None,
        filter_langs: Optional[Sequence[str]] = None,
        filter_personas: Optional[Sequence[str]] = None,
    ):
        self.path = Path(jsonl_path)
        self.records = read_jsonl(self.path)
        self.tok = tokenizer
        self.cfg = cfg or LLMDatasetConfig()

        if filter_langs:
            self.records = [r for r in self.records if r.get(self.cfg.lang_key) in set(filter_langs)]
        if filter_personas:
            self.records = [r for r in self.records if r.get(self.cfg.persona_key) in set(filter_personas)]

    def __len__(self) -> int:
        return len(self.records)

    def _clamp_labels(self, labels: List[float]) -> List[float]:
        if self.cfg.clamp_labels_to is None:
            return labels
        lo, hi = self.cfg.clamp_labels_to
        out = []
        for x in labels:
            try:
                v = float(x)
            except Exception:
                v = 0.0
            v = max(lo, min(hi, v))
            out.append(v)
        return out

    @staticmethod
    def _parse_labels_from_target_json(target_text: str) -> Optional[List[float]]:
        """Oczekuje minimalnego formatu: {"labels":[...]} (bez spacji też ok)."""
        try:
            obj = json.loads(target_text)
            if isinstance(obj, dict) and "labels" in obj:
                labels = obj["labels"]
                if isinstance(labels, list):
                    return [float(x) for x in labels]
        except Exception:
            pass
        return None

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.records[idx]
        input_text: str = r[self.cfg.text_key]
        target_text: str = r[self.cfg.target_text_key]

        # Tokenizacja inputu
        enc_in = self.tok(
            input_text,
            add_special_tokens=self.cfg.add_special_tokens,
            truncation=self.cfg.truncation,
            max_length=self.cfg.max_length,
            padding=False,
            return_tensors=None,
        )
        item: Dict[str, Any] = {
            "input_ids": torch.tensor(enc_in["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc_in["attention_mask"], dtype=torch.long),
            "input_text": input_text,
            "target_text": target_text,
        }

        # Opcjonalnie: tokenizacja targetu (nie łączymy z inputem tutaj)
        if self.cfg.tokenize_target:
            enc_tgt = self.tok(
                target_text,
                add_special_tokens=self.cfg.add_special_tokens,
                truncation=self.cfg.truncation,
                max_length=self.cfg.max_length,
                padding=False,
                return_tensors=None,
            )
            item["labels_input_ids"] = torch.tensor(enc_tgt["input_ids"], dtype=torch.long)
            item["labels_attention_mask"] = torch.tensor(enc_tgt["attention_mask"], dtype=torch.long)

        # Opcjonalnie: parsuj listę floatów z target JSON (do metryk na walidacji)
        if self.cfg.parse_labels_from_json:
            labels_list = self._parse_labels_from_target_json(target_text)
            if labels_list is not None:
                labels_list = self._clamp_labels(labels_list)
                item["labels"] = torch.tensor(labels_list, dtype=self.cfg.dtype_labels)

        if self.cfg.return_meta:
            if self.cfg.id_key in r:
                item["id"] = r[self.cfg.id_key]
            if self.cfg.lang_key in r:
                item["lang"] = r[self.cfg.lang_key]
            if self.cfg.persona_key in r:
                item["persona_id"] = r[self.cfg.persona_key]

        return item