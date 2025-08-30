#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Buduje polskie widoki danych LLM używając mapowania z translated_meta.tsv.

Proces:
1. Ładuje dane bazowe (data/processed/base/all.jsonl) z emocjami i personami
2. Ładuje mapowanie opinion_number -> polski tekst z translated_meta.tsv
3. Filtruje rekordy polskie (lang='pl')
4. Zastępuje angielski tekst polskim używając mapowania
5. Generuje prompty używając polskich szablonów z configs/data_llm_pl.yaml
6. Zachowuje te same etykiety emocji (są uniwersalne)
7. Zapisuje {train,val,test}.jsonl do data/processed/llm_pl/

Użycie:
  python scripts/build_llm_view_pl.py --config configs/data_llm_pl.yaml --mode personalized_desc
"""

import argparse
import csv
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

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

def load_translated_meta(path: Path) -> Dict[int, str]:
    """Ładuje translated_meta.tsv i zwraca mapowanie opinion_number -> original_text (polski)"""
    mapping = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            try:
                opinion_number = int(row["opinion_number"])
                original_text = row["original_text"].strip()
                if original_text:
                    mapping[opinion_number] = original_text
            except (ValueError, KeyError) as e:
                print(f"[WARN] Skipping invalid row in translated_meta.tsv: {e}")
                continue
    return mapping

# ----------------- Helpers -----------------

def _normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def extract_opinion_number(opinion_id: str) -> int:
    """Wyciąga numer opinii z ID typu 'op_000001' -> 1"""
    if opinion_id.startswith("op_"):
        try:
            return int(opinion_id[3:])  # usuń 'op_' i skonwertuj na int
        except ValueError:
            pass
    raise ValueError(f"Cannot extract opinion number from ID: {opinion_id}")

def clamp(x: float, lo=0.0, hi=1.0) -> float:
    try:
        x = float(x)
    except Exception:
        x = lo
    return min(max(x, lo), hi)

def maybe_round(values: List[float], decimals: int | None) -> List[float]:
    if decimals is None:
        return values
    return [round(v, decimals) for v in values]

# ----------------- Prompt rendering -----------------

def render_input_text(cfg: Dict[str, Any], rec: Dict[str, Any], mode: str) -> str:
    """
    Składa finalny prompt dla LLM używając polskich szablonów:
      - dokleja system_prefix jeśli włączony,
      - renderuje tokeny: {lang_tag}, {persona_token}, {persona_block_start/end}, {persona_desc}, {text}.
    """
    p = cfg.get("prompt", {})
    g = p.get("global", {})
    tokens = p.get("tokens", {})
    modes = p.get("modes", {})
    if mode not in modes:
        raise SystemExit(f"[ERROR] Unknown mode '{mode}'. Available: {list(modes.keys())}")

    tmpl = modes[mode]

    # pola
    text = rec["text"]
    lang_tag_format = tokens.get("lang_tag_format", "<lang={lang}>")
    persona_token_format = tokens.get("persona_token_format", "<p:{persona_id}>")
    persona_block = tokens.get("persona_block", {"start": "<persona>", "end": "</persona>"})

    lang_tag = lang_tag_format.format(lang=rec["lang"])
    persona_token = persona_token_format.format(persona_id=rec.get("persona_id", ""))
    persona_desc = rec.get("persona_desc", "")

    system_prefix = p.get("system_prefix", "")
    if not g.get("add_system_prefix", True):
        system_prefix = ""

    rendered = tmpl.format(
        system_prefix=system_prefix,
        lang_tag=lang_tag,
        persona_token=persona_token,
        persona_block_start=persona_block.get("start", "<persona>"),
        persona_block_end=persona_block.get("end", "</persona>"),
        persona_desc=persona_desc,
        text=text,
    )

    # Normalizacja whitespace
    if p.get("render", {}).get("collapse_spaces", True) or g.get("normalize_whitespace", True):
        rendered = _normalize_spaces(rendered)

    # Przycięcie długości (opcjonalne)
    max_chars = g.get("max_total_chars")
    if max_chars and len(rendered) > max_chars:
        if g.get("truncate_strategy", "end") == "start":
            rendered = rendered[-max_chars:]
        else:
            rendered = rendered[:max_chars]

    if g.get("include_trailing_newline", False):
        rendered = rendered + "\n"
    return rendered

def build_target_text(cfg: Dict[str, Any], rec: Dict[str, Any]) -> str:
    """Tworzy JSON target_text zgodny z {'labels':[...]} + clamp i rounding wg configu."""
    tcfg = cfg.get("target", {})
    clamp_range = tcfg.get("clamp_to_range", [0.0, 1.0])
    lo, hi = clamp_range[0], clamp_range[1]
    fmt = tcfg.get("json", {})
    decimals = fmt.get("float_precision", 3)

    labels = rec.get("labels", [])
    labels = [clamp(x, lo, hi) for x in labels]
    labels = maybe_round(labels, decimals)

    obj = {"labels": labels}
    # zwarty JSON bez spacji
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

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
    ap.add_argument("--config", default="configs/data_llm_pl.yaml")
    ap.add_argument("--mode", choices=["non_personalized","persona_token","personalized_desc","personalized_instruction"], default="personalized_desc")
    args = ap.parse_args()

    cfg = load_yaml(args.config)

    # Ścieżki/plik
    paths = cfg["paths"]
    processed_dir = Path(paths["processed_dir"])
    view_dir = Path(paths["view_dir"])
    translated_meta_file = Path(paths["translated_meta_file"])
    files_cfg = cfg["files"]

    print(f"[INFO] Loading base data from {processed_dir}")
    print(f"[INFO] Loading Polish text mapping from {translated_meta_file}")

    # Wczytaj bazę: all.jsonl
    base_all = processed_dir / "all.jsonl"
    if not base_all.exists():
        raise SystemExit(f"[ERROR] Base data file not found: {base_all}")
    
    base_records = read_jsonl(base_all)
    print(f"[INFO] Loaded {len(base_records)} base records")

    # Wczytaj mapowanie translated_meta.tsv
    if not translated_meta_file.exists():
        raise SystemExit(f"[ERROR] Translated meta file not found: {translated_meta_file}")
    
    polish_text_mapping = load_translated_meta(translated_meta_file)
    print(f"[INFO] Loaded {len(polish_text_mapping)} Polish text mappings")

    # Filtruj tylko polskie rekordy i zastąp tekst
    polish_records = []
    missing_mapping = 0
    
    for rec in base_records:
        if rec.get("lang") != "pl":
            continue
            
        # Wyciągnij numer opinii z opinion_id
        try:
            opinion_number = extract_opinion_number(rec["opinion_id"])
        except ValueError as e:
            print(f"[WARN] {e}, skipping record")
            continue
            
        # Znajdź polski tekst
        if opinion_number not in polish_text_mapping:
            missing_mapping += 1
            continue
            
        # Skopiuj rekord i zastąp tekst polskim
        new_rec = rec.copy()
        new_rec["text"] = polish_text_mapping[opinion_number]
        polish_records.append(new_rec)
    
    print(f"[INFO] Filtered to {len(polish_records)} Polish records")
    if missing_mapping > 0:
        print(f"[WARN] {missing_mapping} records skipped due to missing Polish text mapping")

    # Index po id
    base_index = {r["id"]: r for r in polish_records}

    # Splity
    split_cfg = cfg.get("splits", {})
    use_external = bool(split_cfg.get("use_external", True))
    if use_external:
        ext = cfg.get("external_splits", {})
        if not ext or not ext.get("enabled", True):
            raise SystemExit("[ERROR] splits.use_external=True, ale external_splits.enabled nie jest ustawione.")
        split_dir = Path(ext.get("dir", "data/splits"))
        train_ids = load_ids_csv(split_dir / ext["train"])
        val_ids = load_ids_csv(split_dir / ext["val"])
        test_ids = load_ids_csv(split_dir / ext["test"])
        def pick(ids: List[str]) -> List[Dict[str, Any]]:
            out, miss = [], 0
            for _id in ids:
                rec = base_index.get(_id)
                if rec is None: 
                    miss += 1
                else: 
                    out.append(rec)
            if miss:
                print(f"[WARN] {miss} ids from split not found in Polish base; skipped.")
            return out
        train_records = pick(train_ids)
        val_records = pick(val_ids)
        test_records = pick(test_ids)
    else:
        # losowy fallback
        ratios = split_cfg.get("random_ratios", {"train":0.8,"val":0.1,"test":0.1})
        seed = int(split_cfg.get("random_seed", 1337))
        stratify_by = split_cfg.get("stratify_by", ["lang","persona_id"])
        train_records, val_records, test_records = random_split(polish_records, ratios, seed, stratify_by)

    # Filtry: drop jeśli tryb wymaga persona_desc
    drop_modes = cfg.get("filters", {}).get("drop_if_missing_persona_desc_in_modes", [])
    if args.mode in drop_modes:
        def _has_desc(r): return bool(str(r.get("persona_desc","")).strip())
        before = len(train_records); train_records = [r for r in train_records if _has_desc(r)]
        print(f"[INFO] train: drop missing persona_desc -> {before}->{len(train_records)}")
        before = len(val_records);   val_records = [r for r in val_records if _has_desc(r)]
        print(f"[INFO] val:   drop missing persona_desc -> {before}->{len(val_records)}")
        before = len(test_records);  test_records = [r for r in test_records if _has_desc(r)]
        print(f"[INFO] test:  drop missing persona_desc -> {before}->{len(test_records)}")

    # Render i materializacja
    def build_view(rec_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for r in rec_list:
            input_text = render_input_text(cfg, r, args.mode)
            target_text = build_target_text(cfg, r)
            row = {
                "id": r["id"],
                "input_text": input_text,
                "target_text": target_text,
                "lang": r.get("lang"),
                "persona_id": r.get("persona_id"),
            }
            # ogranicz do kolumn z configu (lub domyślne)
            cols = cfg.get("materialization", {}).get("output_columns", ["id","input_text","target_text","lang","persona_id"])
            row = {k: row[k] for k in cols if k in row}
            out.append(row)
        return out

    train_out = build_view(train_records)
    val_out   = build_view(val_records)
    test_out  = build_view(test_records)

    # Zapis
    out_files = {
        "train": Path(paths["view_dir"]) / cfg["files"]["train"],
        "val":   Path(paths["view_dir"]) / cfg["files"]["val"],
        "test":  Path(paths["view_dir"]) / cfg["files"]["test"],
    }
    write_jsonl(train_out, out_files["train"])
    write_jsonl(val_out,   out_files["val"])
    write_jsonl(test_out,  out_files["test"])

    print(f"[OK] Polish LLM view written:\\n - {out_files['train']}\\n - {out_files['val']}\\n - {out_files['test']}")
    print(f"[INFO] rows: train={len(train_out)}, val={len(val_out)}, test={len(test_out)}")

if __name__ == "__main__":
    main()