#!/usr/bin/env python3
import argparse, json, re, sys, os
from pathlib import Path
from typing import Any, Dict, List

try:
    import yaml
except ImportError:
    print("[ERROR] Missing dependency: pyyaml (pip install pyyaml)", file=sys.stderr)
    sys.exit(1)

def load_yaml(p: str) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_jsonl_first(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        line = f.readline()
        if not line:
            raise SystemExit("[ERROR] Sample JSONL is empty")
        return json.loads(line)

def read_json_if_exists(path: str):
    if Path(path).exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def coerce_float(x):
    try:
        return float(x)
    except Exception:
        return x

def clamp(x: float, lo: float, hi: float) -> float:
    return min(max(float(x), lo), hi)

def validate_record(rec: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    schema = cfg["record_schema"]
    policies = schema["policies"]
    constraints = schema["constraints"]

    # 1) required fields
    for k in schema["required_fields"]:
        if k not in rec:
            if policies["on_missing_field"] == "drop_with_report":
                raise SystemExit(f"[ERROR] Missing required field: {k}")
            elif policies["on_missing_field"] == "error":
                raise SystemExit(f"[ERROR] Missing required field: {k}")
            else:
                print(f"[WARN] Missing field {k}, policy={policies['on_missing_field']}")

    # 2) lang
    if rec["lang"] not in constraints["langs_allowed"]:
        if policies["on_lang_not_allowed"] == "error":
            raise SystemExit(f"[ERROR] lang '{rec['lang']}' not allowed {constraints['langs_allowed']}")
        else:
            print(f"[WARN] lang '{rec['lang']}' not allowed")

    # 3) persona_id pattern
    pid = rec["persona_id"]
    pattern = re.compile(constraints["persona_id_pattern"])
    if not pattern.match(pid):
        if policies["on_bad_persona_id"] == "error":
            raise SystemExit(f"[ERROR] persona_id '{pid}' does not match pattern {constraints['persona_id_pattern']}")
        else:
            print(f"[WARN] bad persona_id '{pid}'")

    # 4) labels length & range (+coercion & clamp policy)
    labels: List[Any] = rec["labels"]
    if len(labels) != constraints["labels_length"]:
        raise SystemExit(f"[ERROR] labels length {len(labels)} != {constraints['labels_length']}")

    labels = [coerce_float(v) for v in labels]
    lo, hi = constraints["labels_range"]
    out_of_range_idx = [i for i, v in enumerate(labels) if not isinstance(v, (float, int)) or v < lo or v > hi]
    if out_of_range_idx:
        if policies["on_out_of_range_label"] == "clamp_and_report":
            print(f"[WARN] labels out of range at idx {out_of_range_idx} -> clamping to [{lo},{hi}]")
            labels = [clamp(v if isinstance(v, (float, int)) else lo, lo, hi) for v in labels]
        else:
            raise SystemExit(f"[ERROR] labels out of range at {out_of_range_idx}")

    rec["labels"] = [float(v) for v in labels]
    return rec

def apply_filters(rec: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    filters = cfg["filters"]
    text = (rec.get("text") or "")
    if filters["normalize_text"]["strip"]:
        text = text.strip()
    if filters["normalize_text"]["collapse_spaces"]:
        text = re.sub(r"\s+", " ", text)
    if filters["normalize_text"]["normalize_quotes"]:
        text = text.replace("“","\"").replace("”","\"").replace("’","'")

    if len(text) < filters["text_min_chars"]:
        raise SystemExit(f"[ERROR] text shorter than text_min_chars ({filters['text_min_chars']})")
    if len(text) > filters["text_max_chars"]:
        print(f"[WARN] text longer than text_max_chars ({filters['text_max_chars']}) -> will be used as-is for base")

def write_preview(rec: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    prev = cfg.get("reports", {}).get("sample_preview", {})
    if not prev.get("enabled", False):
        return
    out = prev.get("output", "reports/base/preview_examples.jsonl")
    Path(os.path.dirname(out)).mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[INFO] Preview written to: {out}")

def check_side_files(cfg: Dict[str, Any]) -> None:
    labels_path = cfg["paths"]["label_names_file"]
    personas_path = cfg["paths"]["personas_file"]
    splits_dir = cfg["paths"]["splits_dir"]
    ext = cfg.get("external_splits", {})

    labels = read_json_if_exists(labels_path)
    if labels is None:
        print(f"[WARN] {labels_path} not found, will fallback to metadata.label_names_fallback")
    else:
        if len(labels) != cfg["record_schema"]["constraints"]["labels_length"]:
            print(f"[WARN] label_names.json has {len(labels)} entries != expected {cfg['record_schema']['constraints']['labels_length']}")

    personas = read_json_if_exists(personas_path)
    if personas is None:
        print(f"[WARN] {personas_path} not found (sanity-check of persona IDs will be limited)")
    else:
        ids = {p.get("id") for p in personas if isinstance(p, dict)}
        print(f"[INFO] Loaded {len(ids)} personas from {personas_path}")

    if ext.get("enabled", False):
        for name in ("train","val","test"):
            p = Path(splits_dir) / ext[name]
            if not p.exists():
                print(f"[WARN] external split file missing: {p}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data_base.yaml")
    ap.add_argument("--sample", default="data/processed/base/sample.jsonl")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    rec = load_jsonl_first(args.sample)

    print("[INFO] Validating side files (labels/personas/splits)…")
    check_side_files(cfg)

    print("[INFO] Validating sample record against data_base contract…")
    rec = validate_record(rec, cfg)

    print("[INFO] Applying base filters/normalization…")
    apply_filters(rec, cfg)

    # quick dedup note (na 1 rekordzie nie ma sensu, ale pokażemy zasady)
    dd = cfg["filters"]["deduplication"]
    print(f"[INFO] Deduplication policy: enabled={dd['enabled']}, scope={dd['scope']}, key={dd['key']}")

    # write preview if enabled
    write_preview(rec, cfg)

    print("\n=== SUMMARY ===")
    print(f"id: {rec['id']}")
    print(f"lang: {rec['lang']}")
    print(f"persona_id: {rec['persona_id']}")
    print(f"text_len: {len(rec['text'])}")
    print(f"labels: len={len(rec['labels'])}, min={min(rec['labels']):.3f}, max={max(rec['labels']):.3f}")
    print("\n[OK] data_base smoke test passed.")

if __name__ == "__main__":
    main()
