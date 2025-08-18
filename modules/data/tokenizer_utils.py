# modules/data/tokenizer_utils.py
# -*- coding: utf-8 -*-
"""
Narzędzia do rejestracji specjalnych tokenów (lang tags, <persona>..</persona>, <p:{persona_id}>)
w tokenizerach Hugging Face oraz do inicjalizacji embeddingów nowych tokenów.

Użycie (jako CLI):
    python -m modules.data.tokenizer_utils --config configs/tok.yaml

Zależności:
    pip install transformers pyyaml torch
"""

from __future__ import annotations
import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModel
import yaml


# ----------------------------- Helpers / IO ----------------------------------

def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def read_personas(path: str | Path, id_field: str = "id") -> List[str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        # dopusz­czamy format {id: {...}}
        ids = [k for k in data.keys()]
    else:
        ids = [str(x[id_field]) for x in data if isinstance(x, dict) and id_field in x]
    return ids

def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def to_json(obj: Any, path: str | Path):
    ensure_dir(Path(path).parent)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


# ------------------------- Token spec construction ---------------------------

@dataclass
class TokenBuildSpec:
    lang_tags: List[str]
    persona_start: str
    persona_end: str
    extras: List[str]
    persona_token_fmt: str
    persona_ids: List[str]

def build_spec(cfg: Dict[str, Any]) -> TokenBuildSpec:
    sp = cfg["special_tokens"]
    # lang
    lang_tags = list(dict.fromkeys(sp.get("lang_tags", [])))
    # persona block
    pb = sp.get("persona_block", {"start": "<persona>", "end": "</persona>"})
    persona_start = pb.get("start", "<persona>")
    persona_end = pb.get("end", "</persona>")
    # extras
    extras = list(dict.fromkeys(sp.get("extras", [])))
    # per-persona format
    persona_token_fmt = sp.get("persona_token_format", "<p:{persona_id}>")

    # personas
    pers_cfg = cfg.get("personas", {})
    src = pers_cfg.get("source", "from_file")
    if src != "from_file":
        raise SystemExit("[ERROR] personas.source currently only supports 'from_file'")
    personas_file = cfg["paths"]["personas_file"]
    id_field = pers_cfg.get("id_field", "id")
    all_ids = read_personas(personas_file, id_field=id_field)

    # filtry allow/deny
    allow = pers_cfg.get("filters", {}).get("allow_ids") or []
    deny = pers_cfg.get("filters", {}).get("deny_ids") or []
    if allow:
        all_ids = [i for i in all_ids if i in set(allow)]
    if deny:
        all_ids = [i for i in all_ids if i not in set(deny)]

    # sanity pattern
    patt = pers_cfg.get("id_pattern")
    if patt:
        rx = re.compile(patt)
        bad = [i for i in all_ids if not rx.match(i)]
        if bad:
            print(f"[WARN] Personas not matching pattern {patt}: {bad[:5]} (+{max(0,len(bad)-5)} more)")

    return TokenBuildSpec(
        lang_tags=lang_tags,
        persona_start=persona_start,
        persona_end=persona_end,
        extras=extras,
        persona_token_fmt=persona_token_fmt,
        persona_ids=all_ids,
    )

def tokens_from_spec(spec: TokenBuildSpec) -> Dict[str, List[str]]:
    persona_tokens = [spec.persona_token_fmt.format(persona_id=pid) for pid in spec.persona_ids]
    control_tokens = list(dict.fromkeys(spec.lang_tags + [spec.persona_start, spec.persona_end] + spec.extras))
    return {
        "persona_tokens": persona_tokens,
        "control_tokens": control_tokens,
        "all_tokens": list(dict.fromkeys(control_tokens + persona_tokens)),
    }


# ------------------------ Tokenizer manipulation -----------------------------

def add_tokens_to_tokenizer(tokenizer, tokens: List[str], mark_as_special: bool = True) -> List[str]:
    """
    Dodaje tokeny jako additional_special_tokens tak, aby były nierozbijalne.
    Zwraca listę **rzeczywiście dodanych** (pomija kolizje).
    """
    # Usuń duplikaty i te, które już są znane
    to_add = []
    for tok in tokens:
        tok_id = tokenizer.convert_tokens_to_ids(tok)
        if tok_id is None or tok_id == tokenizer.unk_token_id:
            # może i tak być znany, ale nie jako single-token → still add as special
            pass
        # jeżeli już jest w additional_special_tokens, pomijamy
        if tok in (tokenizer.special_tokens_map_extended.get("additional_special_tokens") or []):
            continue
        to_add.append(tok)

    if not to_add:
        return []

    added = tokenizer.add_special_tokens({"additional_special_tokens": to_add})
    if added != len(to_add):
        print(f"[INFO] Requested to add {len(to_add)} tokens; tokenizer reports added={added} (some collisions).")

    # Rzeczywiście dodane jako nowe ID to te, które nie istniały wcześniej – w praktyce
    # nie ma prostego API by odczytać „added list”; przyjmiemy, że to kolejność `to_add`
    # i sprawdzimy, czy teraz są rozpoznawane jako pojedyncze tokeny.
    actually_added = []
    for tok in to_add:
        pieces = tokenizer.tokenize(tok)
        if len(pieces) == 1 and pieces[0] == tok:
            actually_added.append(tok)
    return actually_added

def ensure_single_token(tokenizer, tokens: List[str]) -> List[str]:
    """Zwraca listę tokenów, które **nie** są 1-tok po dodaniu (powinno być puste)."""
    bad = []
    for t in tokens:
        pieces = tokenizer.tokenize(t)
        if not (len(pieces) == 1 and pieces[0] == t):
            bad.append({"token": t, "pieces": pieces})
    return bad

def measure_unk_rate_before_addition(base_tokenizer, tokens: List[str]) -> Dict[str, Any]:
    """
    Jak bardzo dany token byłby zły bez dodawania? Raport:
        {"multi_piece": n, "unk_like": n, "details":[{"token":..., "pieces":[...]}, ...]}
    """
    details = []
    multi, unk_like = 0, 0
    for t in tokens:
        pieces = base_tokenizer.tokenize(t)
        if len(pieces) != 1 or pieces[0] != t:
            multi += 1
        ids = base_tokenizer.convert_tokens_to_ids(pieces)
        if any(i is None or i == base_tokenizer.unk_token_id for i in ids):
            unk_like += 1
        details.append({"token": t, "pieces": pieces})
    return {"multi_piece": multi, "unk_like": unk_like, "total": len(tokens), "details": details[:50]}


# ------------------------ Embedding initialization ---------------------------

def _find_subword_ids_for_persona(tokenizer, persona_id: str, split_rx: re.Pattern) -> List[int]:
    """
    Dzieli 'young_mother' -> ['young','mother'] i próbuje znaleźć sensowne subtokeny
    w bazowym słowniku (warianty: bare, ▁piece, Ġpiece). Zwraca listę ID (bez UNK).
    """
    parts = [p for p in split_rx.split(persona_id) if p]
    cand_tokens: List[str] = []
    for p in parts:
        cand_tokens.extend([p, f"▁{p}", f"Ġ{p}"])

    ids: List[int] = []
    for tok in cand_tokens:
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is not None and tid != tokenizer.unk_token_id and tid >= 0:
            ids.append(tid)
    # dedupe, zachowaj kolejność
    seen = set()
    unique_ids = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            unique_ids.append(i)
    return unique_ids

def _init_vectors_for_new_tokens(
    model,
    tokenizer,
    new_token_list: List[str],
    persona_tokens_set: set[str],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Inicjalizuje wektory dla NOWYCH tokenów zgodnie z configiem.
    Zwraca raport: {token: {"id": int, "init": "avg_subwords"|"random_normal"|"control_random", "norm": float}}
    """
    emb = model.get_input_embeddings()
    weight = emb.weight.data
    device = weight.device
    dim = weight.shape[1]
    report: Dict[str, Any] = {}

    # strategie
    init_cfg = cfg.get("embedding_init", {})
    per_p_cfg = init_cfg.get("persona_tokens", {})
    ctrl_cfg = init_cfg.get("control_tokens", {})
    std_p = float(per_p_cfg.get("random_normal_std", 0.02))
    std_c = float(ctrl_cfg.get("random_normal_std", 0.02))
    split_pat = re.compile(per_p_cfg.get("subword_split_regex", r"[-_]"))

    for tok in new_token_list:
        tok_id = tokenizer.convert_tokens_to_ids(tok)
        if tok_id is None or tok_id < 0:
            continue  # shouldn't happen
        if tok_id < weight.shape[0]:
            # ID istnieje w aktualnym emb. (po resize)
            pass

        if tok in persona_tokens_set and per_p_cfg.get("strategy", "avg_subwords_or_random") == "avg_subwords_or_random":
            # spróbuj złożyć z subtokenów ID persony
            # wydobądź 'persona_id' z '<p:persona_id>'
            m = re.match(r"^<p:(.+)>$", tok)
            base = m.group(1) if m else tok
            sub_ids = _find_subword_ids_for_persona(tokenizer, base, split_pat)

            vecs = []
            for sid in sub_ids:
                if 0 <= sid < weight.shape[0]:
                    vecs.append(weight[sid])
            if vecs:
                avg = torch.stack(vecs, dim=0).mean(dim=0)
                weight[tok_id].copy_(avg)
                report[tok] = {"id": int(tok_id), "init": "avg_subwords", "norm": float(avg.norm().item())}
                continue  # done → kolejny token

            # fallback → random normal
            rnd = torch.randn(dim, device=device) * std_p
            weight[tok_id].copy_(rnd)
            report[tok] = {"id": int(tok_id), "init": "random_normal", "norm": float(rnd.norm().item())}
        else:
            # control tokens albo inna strategia
            rnd = torch.randn(dim, device=device) * std_c
            weight[tok_id].copy_(rnd)
            report[tok] = {"id": int(tok_id), "init": "control_random", "norm": float(rnd.norm().item())}

    return report


# --------------------------- Main registration --------------------------------

def register_for_model(base_model_id: str, save_name: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Rejestruje tokeny dla JEDNEGO modelu:
      - ładuje tokenizer bazowy,
      - raportuje 'unk rate' PRZED dodaniem,
      - dodaje tokeny specjalne,
      - (opcjonalnie) ładuje model, rozszerza embeddingi, inicjalizuje nowe wiersze,
      - zapisuje tokenizer do artifacts/tokenizers/<save_name>,
      - zwraca raport.
    """
    print(f"\n[INFO] Processing model: {base_model_id} -> save as '{save_name}'")

    paths = cfg["paths"]
    out_root = Path(cfg["paths"]["output_root"]) / save_name
    ensure_dir(out_root)
    reports_dir = ensure_dir(cfg["paths"].get("reports_dir", "reports/tokenizer"))

    spec = build_spec(cfg)
    tok_groups = tokens_from_spec(spec)
    persona_tokens = tok_groups["persona_tokens"]
    control_tokens = tok_groups["control_tokens"]
    all_tokens = tok_groups["all_tokens"]

    # 1) Base tokenizer (for before/after comparisons)
    base_tok = AutoTokenizer.from_pretrained(base_model_id, use_fast=False)
    before_report = {}
    if cfg.get("validation", {}).get("measure_unk_rate_if_not_added", True):
        before_report = measure_unk_rate_before_addition(base_tok, all_tokens)

    # 2) Working tokenizer to modify
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=False)

    # Ustaw HF specials jeśli proszone (zachowujemy None = bez zmian)
    hf_spec = cfg["special_tokens"].get("hf_specials", {})
    for k, v in (hf_spec or {}).items():
        if v:
            try:
                setattr(tokenizer, k, v)
            except Exception:
                print(f"[WARN] Could not set hf_specials.{k}='{v}' on tokenizer.")

    # 3) Add as additional_special_tokens (prevents splitting)
    mark_special = cfg.get("backend_hints", {}).get("mark_as_special", True)
    added = add_tokens_to_tokenizer(tokenizer, all_tokens, mark_as_special=mark_special)
    print(f"[OK] Special tokens added (single-token now): {len(added)} / {len(all_tokens)}")

    # 4) Sanity: ensure all intended tokens are truly single-piece now
    if cfg.get("validation", {}).get("enforce_as_single_token", {}).get("enabled", True):
        to_check: List[str] = []
        vcfg = cfg["validation"]["enforce_as_single_token"]
        if vcfg.get("tokens", {}).get("use_lang_tags", True):
            to_check += spec.lang_tags
        if vcfg.get("tokens", {}).get("use_persona_block", True):
            to_check += [spec.persona_start, spec.persona_end]
        if vcfg.get("tokens", {}).get("use_extras", True):
            to_check += spec.extras
        if vcfg.get("tokens", {}).get("use_persona_tokens", True):
            to_check += persona_tokens
        bad = ensure_single_token(tokenizer, to_check)
        if bad:
            sample = bad[:5]
            raise SystemExit(f"[ERROR] Some tokens are not single-piece after addition: {sample}")

   # 5) (Optional) Load model & resize embeddings & initialize
    apply_cfg = cfg.get("apply", {})
    resize_model_embeddings = bool(apply_cfg.get("resize_model_embeddings", True))

    model_report: Dict[str, Any] = {}
    new_token_ids: Dict[str, int] = {}

    if resize_model_embeddings:
        # Load model (no shrink policy)
        model = AutoModel.from_pretrained(base_model_id)
        emb = model.get_input_embeddings()
        orig_vocab = emb.weight.shape[0]

        # Tokens added to the *tokenizer* (token -> id)
        added_vocab_map = tokenizer.get_added_vocab()  # {token: id}
        new_token_ids = {t: int(i) for t, i in added_vocab_map.items()}

        # Determine target embedding size (only grow; never shrink)
        max_added_id = max(new_token_ids.values(), default=-1)
        target_size = max(orig_vocab, max_added_id + 1)
        if target_size > orig_vocab:
            model.resize_token_embeddings(target_size)

        new_vocab = model.get_input_embeddings().weight.shape[0]

        # Initialize ALL added tokens (even if their id < orig_vocab)
        persona_set = set(persona_tokens)
        init_report = _init_vectors_for_new_tokens(
            model=model,
            tokenizer=tokenizer,
            new_token_list=list(new_token_ids.keys()),
            persona_tokens_set=persona_set,
            cfg=cfg,
        )

        model_report = {
            "orig_vocab": int(orig_vocab),
            "new_vocab": int(new_vocab),
            "new_tokens": new_token_ids,  # {token: id}
            "init": init_report,          # per-token init info
        }

        tokenizer_base_size = getattr(tokenizer, "vocab_size", None)
        if tokenizer_base_size is not None and orig_vocab > tokenizer_base_size:
            print(
                f"[WARN] Model embedding size ({orig_vocab}) > tokenizer base vocab ({tokenizer_base_size}). "
                "Model wygląda na wcześniej rozszerzony; nie zmniejszamy embeddingów, inicjalizujemy dodane tokeny."
            )
    else:
        print("[INFO] Skipping model resize/init (apply.resize_token_embeddings=False).")


    # 6) Save tokenizer
    tokenizer.save_pretrained(str(out_root))
    # Dla wygody zapisz też listę naszych tokenów
    to_json({"persona_tokens": persona_tokens, "control_tokens": control_tokens, "all_tokens": all_tokens},
            out_root / "special_tokens_added.json")

    # 7) Registry report
    registry = {
        "base_model_id": base_model_id,
        "save_dir": str(out_root),
        "added_counts": {
            "persona_tokens": len(persona_tokens),
            "control_tokens": len(control_tokens),
            "actually_single_token": len(added),
        },
        "before_unk_report": before_report,
        "model_report": model_report,
    }
    # global report path
    reg_path = Path(cfg["validation"].get("registry_report_file", cfg["paths"].get("reports_dir", "reports/tokenizer") + "/registry.json"))
    # jeśli zapisujemy per-model, dorzuć sufiks
    per_model = {
        "model": base_model_id,
        **registry
    }
    ensure_dir(reg_path.parent)
    # dopisz/połącz istniejący raport
    try:
        if reg_path.exists():
            existing = json.loads(reg_path.read_text(encoding="utf-8"))
            if isinstance(existing, list):
                existing.append(per_model)
                to_json(existing, reg_path)
            else:
                to_json([existing, per_model], reg_path)
        else:
            to_json([per_model], reg_path)
    except Exception:
        to_json([per_model], reg_path)

    print(f"[OK] Tokenizer saved to: {out_root}")
    print(f"[OK] Registry updated: {reg_path}")
    return registry


# ------------------------------- CLI -----------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tok.yaml", help="Ścieżka do configs/tok.yaml")
    ap.add_argument("--models", nargs="*", default=None,
                    help="Nadpisz listę modeli do rejestracji (klucze z sekcji base_tokenizers/save_names), np.: xlm_roberta_base mdeberta_v3_base")
    args = ap.parse_args()

    cfg = load_yaml(args.config)

    base_tokenizers = cfg.get("base_tokenizers", {})
    save_names = cfg.get("save_names", {})
    target_keys = args.models or cfg.get("apply", {}).get("models", list(base_tokenizers.keys()))

    if not target_keys:
        raise SystemExit("[ERROR] Brak listy modeli do rejestracji (apply.models).")

    for key in target_keys:
        if key not in base_tokenizers:
            raise SystemExit(f"[ERROR] Unknown model key '{key}'. Available: {list(base_tokenizers.keys())}")
        base_id = base_tokenizers[key]
        save_name = save_names.get(key, key + "-with-personas")
        register_for_model(base_id, save_name, cfg)

if __name__ == "__main__":
    main()