import json, re, sys, argparse, pathlib
from typing import Dict, Any
import yaml 
from jsonschema import validate, ValidationError 

def clamp(x, lo=0.0, hi=1.0):
    return min(max(float(x), lo), hi)

def load_yaml(path): 
    with open(path, "r", encoding="utf-8") as f: 
        return yaml.safe_load(f)

def render_prompt(cfg: Dict[str, Any], rec: Dict[str, Any], mode: str) -> str:
    p = cfg["prompt"]; g = p["global"]; toks = p["tokens"]
    text = rec["text"].strip() if g.get("strip_input_text", True) else rec["text"]
    persona_desc = rec.get("persona_desc", "")
    persona_id = rec.get("persona_id", "")
    lang_tag = toks["lang_tag_format"].format(lang=rec["lang"])
    persona_token = toks["persona_token_format"].format(persona_id=persona_id)
    tmpl = cfg["prompt"]["modes"][mode]
    rendered = tmpl.format(
        system_prefix=cfg.get("prompt", {}).get("system_prefix",""),
        lang_tag=lang_tag,
        persona_block_start=toks["persona_block"]["start"],
        persona_block_end=toks["persona_block"]["end"],
        persona_desc=persona_desc,
        persona_token=persona_token,
        text=text,
    )
    if cfg["prompt"]["render"].get("collapse_empty_lines", True):
        rendered = re.sub(r"\n{3,}", "\n\n", rendered).strip() + ("\n" if g.get("include_trailing_newline", True) else "")
    max_chars = g.get("max_total_chars")
    if max_chars and len(rendered) > max_chars:
        if g.get("truncate_strategy","end") == "start":
            rendered = rendered[-max_chars:]
        else:
            rendered = rendered[:max_chars]
    return rendered

def build_target(cfg: Dict[str, Any], rec: Dict[str, Any]) -> str:
    tcfg = cfg["target"]
    floats = [clamp(x, *tcfg["validation"]["fixup"]["clamp_values_to_range"]) for x in rec["labels"]]
    prec = tcfg["build_from_labels"]["float_precision"]
    arr = [float(f"{x:.{prec}f}") for x in floats]
    obj = {tcfg["json"]["key"]: arr}
    # separators like (",", ":") and ensure_ascii control are informational here; json.dumps approximates
    return json.dumps(obj, ensure_ascii=tcfg["json"]["ensure_ascii"], separators=tuple(tcfg["json"]["separators"]))

def validate_target(cfg: Dict[str, Any], target_text: str) -> None:
    schema = json.loads(cfg["target"]["validation"]["schema"])
    try:
        validate(instance=json.loads(target_text), schema=schema)
    except (ValidationError, json.JSONDecodeError) as e:
        raise SystemExit(f"[ERROR] Target JSON invalid: {e}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data_llm.yaml")
    ap.add_argument("--sample", default="data/processed/base/sample.jsonl")
    ap.add_argument("--mode", choices=["non_personalized","personalized_desc","persona_token","personalized_instruction"], default="personalized_desc")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    rec = None
    with open(args.sample, "r", encoding="utf-8") as f:
        rec = json.loads(f.readline())

    # sanity checks
    base = cfg["source_record_contract"]
    assert all(k in rec for k in base["required_fields"]), "Missing required field in sample record"
    assert rec["lang"] in base["constraints"]["lang_allowed"], "lang not allowed"
    assert len(rec["labels"]) == base["constraints"]["labels_length"], "labels length != 21"

    input_text = render_prompt(cfg, rec, args.mode)
    target_text = build_target(cfg, rec)
    validate_target(cfg, target_text)

    print("=== INPUT TEXT ===")
    print(input_text)
    print("=== TARGET TEXT ===")
    print(target_text)
    print("\n[OK] Sample rendered and validated.")

if __name__ == "__main__":
    main()