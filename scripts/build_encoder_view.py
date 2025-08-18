#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Buduje widok danych dla ENCODERÓW:
 - czyta config configs/data_encoder.yaml,
 - ładuje bazę (data/processed/base/all.jsonl lub base_{train,val,test}.jsonl),
 - wybiera rekordy wg splitów (external lub losowy fallback),
 - renderuje input_text wg trybu (--mode),
 - deduplikuje teksty per split (opcjonalnie),
 - zapisuje {train,val,test}.jsonl do view_dir z kolumnami z configu.

Użycie:
  python scripts/build_encoder_view.py --config configs/data_encoder.yaml --mode persona_token
"""

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, Any, List, Iterable, Tuple
import re

try:
    import yaml
except ImportError:
    raise SystemExit("Missing dependency: pyyaml (pip install pyyaml)")

# ----------------- I/O utils -----------------

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            out.append(json.loads(ln))
    return out

def write_jsonl(records: Iterable[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def load_ids_csv(path: Path) -> List[str]:
    ids = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "id" not in reader.fieldnames:
            raise SystemExit(f"[ERROR] Split file {path} must have a header with column 'id'")
        for row in reader:
            ids.append(str(row["id"]))
    return ids

# ----------------- Text / labels helpers -----------------

def _normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def clamp(x: float, lo=0.0, hi=1.0) -> float:
    try:
        x = float(x)
    except Exception:
        x = lo
    return min(max(x, lo), hi)

def render_input_text(cfg: Dict[str, Any], rec: Dict[str, Any], mode: str) -> str:
    ib = cfg["input_build"]
    g = ib["global"]
    tokens = ib["tokens"]
    modes = ib["modes"]
    if mode not in modes:
        raise SystemExit(f"[ERROR] Unknown mode '{mode}'. Available: {list(modes.keys())}")

    tmpl = modes[mode]
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

def deduplicate_within_split(records: List[Dict[str, Any]], key: str, normalize: Dict[str, bool]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for r in records:
        val = r.get(key, "")
        if normalize.get("strip", True):
            val = val.strip()
        if normalize.get("lower", True):
            val = val.lower()
        if normalize.get("collapse_spaces", True):
            val = _normalize_spaces(val)
        if val in seen:
            continue
        seen.add(val)
        out.append(r)
    return out

# ----------------- Splits handling -----------------

def pick_records_by_ids(base_index: Dict[str, Dict[str, Any]], ids: List[str]) -> List[Dict[str, Any]]:
    out = []
    miss = 0
    for _id in ids:
        rec = base_index.get(_id)
        if rec is None:
            miss += 1
        else:
            out.append(rec)
    if miss:
        print(f"[WARN] {miss} ids from split not found in base; skipped.")
    return out

def random_split(records: List[Dict[str, Any]], ratios: Dict[str, float], seed: int, stratify_by: List[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    rnd = random.Random(seed)
    if not stratify_by:
        shuffled = records[:]
        rnd.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(ratios["train"] * n)
        n_val = int(ratios["val"] * n)
        train = shuffled[:n_train]
        val = shuffled[n_train:n_train+n_val]
        test = shuffled[n_train+n_val:]
        return train, val, test
    # prosty stratyfikowany split po krotce wartości
    buckets: Dict[tuple, List[Dict[str, Any]]] = {}
    for r in records:
        key = tuple(r.get(k) for k in stratify_by)
        buckets.setdefault(key, []).append(r)
    train, val, test = [], [], []
    for key, bucket in buckets.items():
        rnd.shuffle(bucket)
        n = len(bucket)
        n_train = int(ratios["train"] * n)
        n_val = int(ratios["val"] * n)
        train += bucket[:n_train]
        val += bucket[n_train:n_train+n_val]
        test += bucket[n_train+n_val:]
    return train, val, test

# ----------------- Main -----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data_encoder.yaml")
    ap.add_argument("--mode", choices=["non_personalized","persona_token","personalized_desc"], default="persona_token")
    args = ap.parse_args()

    cfg = load_yaml(args.config)

    # Ścieżki/plik
    paths = cfg["paths"]
    processed_base_dir = Path(paths["processed_base_dir"])
    view_dir = Path(paths["view_dir"])
    reports_dir = Path(paths["reports_dir"])
    files_cfg = cfg["files"]

    # Wczytaj bazę: preferuj all.jsonl; jeśli brak – sklej base_{train,val,test}.jsonl
    base_all = processed_base_dir / "all.jsonl"
    base_records: List[Dict[str, Any]] = []
    if base_all.exists():
        base_records = read_jsonl(base_all)
    else:
        collected = []
        for k in ("base_train","base_val","base_test"):
            p = processed_base_dir / files_cfg.get(k, "")
            if p.name and p.exists():
                collected += read_jsonl(p)
        if not collected:
            raise SystemExit("[ERROR] Nie znaleziono ani all.jsonl, ani base_{train,val,test}.jsonl w processed_base_dir.")
        base_records = collected

    # Index po id
    base_index = {r["id"]: r for r in base_records}

    # Splity
    split_cfg = cfg["splits"]
    use_external = bool(split_cfg.get("use_external", True))
    if use_external:
        ext = cfg.get("external_splits", {})
        if not ext or not ext.get("enabled", True):
            raise SystemExit("[ERROR] splits.use_external=True, ale external_splits.enabled nie jest ustawione.")
        split_dir = Path(ext.get("dir", "data/splits"))
        train_ids = load_ids_csv(split_dir / ext["train"])
        val_ids = load_ids_csv(split_dir / ext["val"])
        test_ids = load_ids_csv(split_dir / ext["test"])
        train_records = pick_records_by_ids(base_index, train_ids)
        val_records = pick_records_by_ids(base_index, val_ids)
        test_records = pick_records_by_ids(base_index, test_ids)
    else:
        # losowy fallback
        ratios = split_cfg.get("random_ratios", {"train":0.8,"val":0.1,"test":0.1})
        seed = int(split_cfg.get("random_seed", 1337))
        stratify_by = split_cfg.get("stratify_by", ["lang","persona_id"])
        train_records, val_records, test_records = random_split(base_records, ratios, seed, stratify_by)

    # Filtry: brak opisu persony w trybie wymagającym opisu
    drop_modes = cfg["filters"].get("drop_if_missing_persona_desc_in_modes", [])
    if args.mode in drop_modes:
        def _has_desc(r): return bool(str(r.get("persona_desc","")).strip())
        before = len(train_records); train_records = [r for r in train_records if _has_desc(r)]
        print(f"[INFO] train: drop missing persona_desc -> {before}->{len(train_records)}")
        before = len(val_records);   val_records = [r for r in val_records if _has_desc(r)]
        print(f"[INFO] val:   drop missing persona_desc -> {before}->{len(val_records)}")
        before = len(test_records);  test_records = [r for r in test_records if _has_desc(r)]
        print(f"[INFO] test:  drop missing persona_desc -> {before}->{len(test_records)}")

    # Deduplikacja po tekście (per split)
    dd = cfg["filters"].get("deduplication", {"enabled": False})
    if dd.get("enabled", False) and dd.get("scope","within-split") == "within-split":
        norm = dd.get("normalization", {"lower": True, "strip": True, "collapse_spaces": True})
        train_records = deduplicate_within_split(train_records, dd.get("key","text"), norm)
        val_records   = deduplicate_within_split(val_records,   dd.get("key","text"), norm)
        test_records  = deduplicate_within_split(test_records,  dd.get("key","text"), norm)

    # Render input_text i clamp labels
    def build_view(rec_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for r in rec_list:
            # sanity labels
            labels = r.get("labels", [])
            labels = [clamp(x, *cfg["target"]["clamp_to_range"]) for x in labels] if "target" in cfg else [clamp(x) for x in labels]
            r_render = {
                "id": r["id"],
                "input_text": render_input_text(cfg, r, args.mode),
                "labels": labels,
                "lang": r.get("lang"),
                "persona_id": r.get("persona_id"),
            }
            # ogranicz do kolumn żądanych w materialization.output_columns
            cols = cfg["materialization"].get("output_columns", ["id","input_text","labels","lang","persona_id"])
            r_render = {k: r_render[k] for k in cols if k in r_render}
            out.append(r_render)
        return out

    train_out = build_view(train_records)
    val_out   = build_view(val_records)
    test_out  = build_view(test_records)

    # Zapis
    out_cfg = cfg["materialization"]
    out_format = out_cfg.get("output_format","jsonl")
    if out_format != "jsonl":
        raise SystemExit("[ERROR] Aktualnie wspieramy tylko output_format=jsonl dla widoku encodera.")

    out_files = {
        "train": Path(paths["view_dir"]) / cfg["files"]["train"],
        "val":   Path(paths["view_dir"]) / cfg["files"]["val"],
        "test":  Path(paths["view_dir"]) / cfg["files"]["test"],
    }
    write_jsonl(train_out, out_files["train"])
    write_jsonl(val_out,   out_files["val"])
    write_jsonl(test_out,  out_files["test"])

    print(f"[OK] Encoder view written:\n - {out_files['train']}\n - {out_files['val']}\n - {out_files['test']}")
    print(f"[INFO] rows: train={len(train_out)}, val={len(val_out)}, test={len(test_out)}")

if __name__ == "__main__":
    main()
