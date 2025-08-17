#!/usr/bin/env python3
import json, re, argparse
from typing import Dict, Any
import yaml  # pip install pyyaml

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def clamp(x: float, lo=0.0, hi=1.0) -> float:
    try:
        x = float(x)
    except Exception:
        x = lo
    return min(max(x, lo), hi)

def validate_source_record(rec: Dict[str, Any], contract: Dict[str, Any]) -> None:
    required = contract["required_fields"]
    for k in required:
        if k not in rec:
            raise SystemExit(f"[ERROR] Missing required field: {k}")
    lang_allowed = set(contract["constraints"]["lang_allowed"])
    if rec["lang"] not in lang_allowed:
        raise SystemExit(f"[ERROR] lang '{rec['lang']}' not in {lang_allowed}")
    labels = rec["labels"]
    if len(labels) != contract["constraints"]["labels_length"]:
        raise SystemExit(f"[ERROR] labels length != {contract['constraints']['labels_length']}")
    lo, hi = contract["constraints"]["labels_range"]
    bad = [i for i, v in enumerate(labels) if not (lo <= float(v) <= hi)]
    if bad:
        print(f"[WARN] Out-of-range labels at idx {bad} -> clamping per policy")

def render_input_text(cfg: Dict[str, Any], rec: Dict[str, Any], mode: str) -> str:
    ib = cfg["input_build"]
    g = ib["global"]
    tokens = ib["tokens"]
    tmpl = ib["modes"][mode]

    text = rec["text"].strip() if g.get("strip_text", True) else rec["text"]
    lang_tag = tokens["lang_tag_format"].format(lang=rec["lang"])
    persona_token = tokens["persona_token_format"].format(persona_id=rec.get("persona_id", ""))
    persona_desc = rec.get("persona_desc", "")
    rendered = tmpl.format(
        lang_tag=lang_tag,
        persona_token=persona_token,
        persona_block_start=tokens["persona_block"]["start"],
        persona_block_end=tokens["persona_block"]["end"],
        persona_desc=persona_desc,
        text=text,
    )

    if ib["render"].get("collapse_spaces", True) or g.get("normalize_whitespace", True):
        rendered = _normalize_spaces(rendered)

    max_chars = g.get("max_total_chars")
    if max_chars and len(rendered) > max_chars:
        if g.get("truncate_strategy", "end") == "start":
            rendered = rendered[-max_chars:]
        else:
            rendered = rendered[:max_chars]

    if g.get("include_trailing_newline", False):
        rendered = rendered + "\n"
    return rendered

def clamp_labels(cfg: Dict[str, Any], rec: Dict[str, Any]):
    lo, hi = cfg["target"]["clamp_to_range"]
    rec["labels"] = [clamp(x, lo, hi) for x in rec["labels"]]

def maybe_tokenize(text: str, tok_cfg: Dict[str, Any]):
    try:
        from transformers import AutoTokenizer
    except Exception:
        print("[INFO] transformers not installed — skipping tokenization step.")
        return None

    name = tok_cfg.get("pretrained_tokenizer_name", "xlm-roberta-base")
    fast = tok_cfg.get("use_fast", True)
    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=fast)

    # Rejestracja bazowych specjalnych tokenów (jeśli są w configu)
    # Uwaga: to jednorazowy podgląd — w treningu zrób to w kodzie datasetu!
    return tokenizer(
        text,
        max_length=tok_cfg.get("max_length", 256),
        padding=tok_cfg.get("padding", "max_length"),
        truncation=tok_cfg.get("truncation", True),
        return_token_type_ids=tok_cfg.get("return_token_type_ids", False),
        add_special_tokens=tok_cfg.get("add_special_tokens", True),
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data_encoder.yaml")
    ap.add_argument("--sample", default="data/processed/base/sample.jsonl")
    ap.add_argument("--mode", choices=["non_personalized","persona_token","personalized_desc"], default="persona_token")
    ap.add_argument("--tokenize", action="store_true", help="Potokenizuj i pokaż rozmiary/fragmenty")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    # wczytaj rekord
    with open(args.sample, "r", encoding="utf-8") as f:
        rec = json.loads(f.readline())

    # walidacja i clamp
    validate_source_record(rec, cfg["source_record_contract"])
    clamp_labels(cfg, rec)

    # render input_text
    input_text = render_input_text(cfg, rec, args.mode)

    # podsumowanie targetu
    labels = rec["labels"]
    lbl_min, lbl_max = min(labels), max(labels)

    print("=== INPUT TEXT ===")
    print(input_text)
    print("\n=== LABELS (summary) ===")
    print(f"len={len(labels)}, min={lbl_min:.3f}, max={lbl_max:.3f}")

    if args.tokenize:
        print("\n[Tokenization] running…")
        batch = maybe_tokenize(input_text, cfg["tokenization"])
        if batch is not None:
            input_ids = batch["input_ids"]
            attn = batch["attention_mask"]
            print(f"tokens: {len(input_ids)} (max_length={cfg['tokenization'].get('max_length')})")
            print(f"attention_mask sum: {int(sum(attn))}")
            print("input_ids (first 32):", input_ids[:32])
        else:
            print("[Tokenization] skipped.")

    print("\n[OK] Encoder view sample rendered.", flush=True)

if __name__ == "__main__":
    main()