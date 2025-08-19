#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Walidator splitów:
- sprawdza, czy ID z train/val/test istnieją w all.jsonl,
- czy zbiory są rozłączne,
- LOPO: czy persona testowa nie występuje w train/val oraz czy test ma wyłącznie tę personę,
- (opcjonalnie) LOLO: czy test odpowiada folderowi języka; w trybie --strict-lolo wyklucza ten język z train/val.

Użycie:
  python scripts/check_splits.py --all data/processed/base/all.jsonl --splits-dir data/splits
  # opcje:
  #   --no-base / --no-lopo / --no-lolo
  #   --strict-lolo
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            rows.append(json.loads(ln))
    return rows

def load_all_index(all_path: Path):
    """Zwraca:
       - by_id: id -> record
       - persona_by_id: id -> persona_id (lub None)
       - lang_by_id: id -> lang (lub None)
       - opinion_by_id: id -> opinion_id (lub None)
    """
    recs = read_jsonl(all_path)
    by_id = {}
    persona_by_id, lang_by_id, opinion_by_id = {}, {}, {}
    for r in recs:
        rid = r.get("id")
        if not rid:
            # pomiń bez id
            continue
        by_id[rid] = r
        persona_by_id[rid] = r.get("persona_id")
        lang_by_id[rid] = r.get("lang")
        opinion_by_id[rid] = r.get("opinion_id")
    return by_id, persona_by_id, lang_by_id, opinion_by_id

def read_id_csv(path: Path) -> List[str]:
    """
    Czyta plik z ID:
      - format 1 kolumny bez separatora (po 1 ID w linii),
      - CSV z separatorem wśród: , ; \t |
      - z/bez nagłówka 'id'
    Ignoruje puste linie i komentarze zaczynające się od '#'.
    """
    text = path.read_text(encoding="utf-8-sig")  # 'sig' usuwa BOM jeśli jest
    # normalizuj końce linii
    lines = [ln.strip() for ln in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    lines = [ln for ln in lines if ln and not ln.lstrip().startswith("#")]
    if not lines:
        return []

    # spróbuj wykryć separator heurystycznie
    candidates = [",", ";", "\t", "|"]
    sep = None
    # bierz kilka pierwszych niepustych linii do estymacji
    probe = lines[: min(20, len(lines))]
    scores = {c: sum(ln.count(c) for ln in probe) for c in candidates}
    best_sep, best_score = max(scores.items(), key=lambda kv: kv[1])
    if best_score > 0:
        sep = best_sep

    ids: List[str] = []
    if sep is None:
        # traktuj jako jedną kolumnę bez separatora
        # pomiń nagłówek 'id' jeśli jest
        for ln in lines:
            if ln.lower() == "id":
                continue
            ids.append(ln)
        return ids

    # tryb CSV z wykrytym separatorem
    import csv
    reader = csv.reader(lines, delimiter=sep)
    rows = list(reader)
    if not rows:
        return ids

    header = [c.strip().lower() for c in rows[0]]
    has_header = "id" in header
    start = 1 if has_header else 0

    if has_header:
        id_idx = header.index("id")
        for row in rows[start:]:
            if id_idx < len(row) and row[id_idx].strip():
                ids.append(row[id_idx].strip())
        return ids

    # brak nagłówka: jeśli 1 kolumna → bierz ją; jeśli wiele – spróbuj znaleźć kolumnę o nazwie 'id' w pierwszym wierszu
    if len(rows[0]) == 1:
        for row in rows[start:]:
            if row and row[0].strip().lower() != "id":
                ids.append(row[0].strip())
        return ids

    # wiele kolumn, brak nagłówka → bierz pierwszą kolumnę
    for row in rows[start:]:
        if row and row[0].strip():
            val = row[0].strip()
            if val.lower() != "id":
                ids.append(val)
    return ids

def disjoint_check(name: str, a: Set[str], b: Set[str]) -> List[str]:
    inter = sorted(a & b)
    if inter:
        print(f"[ERROR] {name}: zbiory nie są rozłączne; wspólne ID (first 10): {inter[:10]}")
    return inter

def ensure_exists(name: str, ids: List[str], by_id: Dict[str, Any]) -> List[str]:
    missing = [x for x in ids if x not in by_id]
    if missing:
        print(f"[ERROR] {name}: {len(missing)} ID nie występuje w all.jsonl (first 10): {missing[:10]}")
    return missing

def check_base_splits(splits_dir: Path, by_id: Dict[str, Any]) -> bool:
    ok = True
    t = splits_dir / "train_ids.csv"
    v = splits_dir / "val_ids.csv"
    s = splits_dir / "test_ids.csv"
    if not (t.is_file() and v.is_file() and s.is_file()):
        print("[WARN] Pomijam check BASE: brak train_ids.csv/val_ids.csv/test_ids.csv w", splits_dir)
        return True

    train = read_id_csv(t); val = read_id_csv(v); test = read_id_csv(s)
    print(f"[INFO] BASE sizes: train={len(train)} val={len(val)} test={len(test)}")

    # istnieją w all.jsonl
    if ensure_exists("BASE.train", train, by_id): ok = False
    if ensure_exists("BASE.val", val, by_id): ok = False
    if ensure_exists("BASE.test", test, by_id): ok = False

    # rozłączność
    if disjoint_check("BASE train/val", set(train), set(val)): ok = False
    if disjoint_check("BASE train/test", set(train), set(test)): ok = False
    if disjoint_check("BASE val/test", set(val), set(test)): ok = False

    return ok

def check_lopo(splits_dir: Path, by_id: Dict[str, Any], persona_by_id: Dict[str, str]) -> bool:
    ok = True
    lopo_root = splits_dir / "lopo"
    if not lopo_root.is_dir():
        print("[WARN] Pomijam check LOPO: brak katalogu", lopo_root)
        return True

    for persona_dir in sorted([p for p in lopo_root.iterdir() if p.is_dir()]):
        persona = persona_dir.name
        t = persona_dir / "train_ids.csv"
        v = persona_dir / "val_ids.csv"
        s = persona_dir / "test_ids.csv"
        if not (t.is_file() and v.is_file() and s.is_file()):
            print(f"[WARN] LOPO {persona}: brak wymaganych plików, pomijam")
            continue

        train = read_id_csv(t); val = read_id_csv(v); test = read_id_csv(s)
        print(f"[INFO] LOPO[{persona}] sizes: train={len(train)} val={len(val)} test={len(test)}")

        # istnieją w all.jsonl
        if ensure_exists(f"LOPO[{persona}].train", train, by_id): ok = False
        if ensure_exists(f"LOPO[{persona}].val", val, by_id): ok = False
        if ensure_exists(f"LOPO[{persona}].test", test, by_id): ok = False

        # rozłączność lokalna
        if disjoint_check(f"LOPO[{persona}] train/val", set(train), set(val)): ok = False
        if disjoint_check(f"LOPO[{persona}] train/test", set(train), set(test)): ok = False
        if disjoint_check(f"LOPO[{persona}] val/test", set(val), set(test)): ok = False

        # test musi być TYLKO z tej persony
        wrong_test = [rid for rid in test if persona_by_id.get(rid) != persona]
        if wrong_test:
            print(f"[ERROR] LOPO[{persona}]: test zawiera ID innej persony (first 10): {wrong_test[:10]}")
            ok = False

        # train/val nie mogą zawierać tej persony
        leakage_train = [rid for rid in train if persona_by_id.get(rid) == persona]
        leakage_val = [rid for rid in val if persona_by_id.get(rid) == persona]
        if leakage_train:
            print(f"[ERROR] LOPO[{persona}]: person '{persona}' obecny w TRAIN (first 10): {leakage_train[:10]}")
            ok = False
        if leakage_val:
            print(f"[ERROR] LOPO[{persona}]: person '{persona}' obecny w VAL (first 10): {leakage_val[:10]}")
            ok = False

    return ok

def check_lolo(splits_dir: Path, by_id: Dict[str, Any], lang_by_id: Dict[str, str], strict: bool=False) -> bool:
    ok = True
    lolo_root = splits_dir / "lolo"
    if not lolo_root.is_dir():
        print("[WARN] Pomijam check LOLO: brak katalogu", lolo_root)
        return True

    for lang_dir in sorted([p for p in lolo_root.iterdir() if p.is_dir()]):
        lang = lang_dir.name
        t = lang_dir / "train_ids.csv"
        v = lang_dir / "val_ids.csv"
        s = lang_dir / "test_ids.csv"
        if not (t.is_file() and v.is_file() and s.is_file()):
            print(f"[WARN] LOLO {lang}: brak wymaganych plików, pomijam")
            continue

        train = read_id_csv(t); val = read_id_csv(v); test = read_id_csv(s)
        print(f"[INFO] LOLO[{lang}] sizes: train={len(train)} val={len(val)} test={len(test)}")

        # istnieją
        if ensure_exists(f"LOLO[{lang}].train", train, by_id): ok = False
        if ensure_exists(f"LOLO[{lang}].val", val, by_id): ok = False
        if ensure_exists(f"LOLO[{lang}].test", test, by_id): ok = False

        # rozłączność lokalna
        if disjoint_check(f"LOLO[{lang}] train/val", set(train), set(val)): ok = False
        if disjoint_check(f"LOLO[{lang}] train/test", set(train), set(test)): ok = False
        if disjoint_check(f"LOLO[{lang}] val/test", set(val), set(test)): ok = False

        # test MUSI mieć tylko ten język
        wrong_lang_test = [rid for rid in test if lang_by_id.get(rid) != lang]
        if wrong_lang_test:
            print(f"[ERROR] LOLO[{lang}]: test zawiera ID o innym języku (first 10): {wrong_lang_test[:10]}")
            ok = False

        if strict:
            # w trybie strict język z folderu nie może pojawić się w train/val
            leak_tr = [rid for rid in train if lang_by_id.get(rid) == lang]
            leak_va = [rid for rid in val if lang_by_id.get(rid) == lang]
            if leak_tr:
                print(f"[ERROR] LOLO[{lang}]-strict: język '{lang}' obecny w TRAIN (first 10): {leak_tr[:10]}")
                ok = False
            if leak_va:
                print(f"[ERROR] LOLO[{lang}]-strict: język '{lang}' obecny w VAL (first 10): {leak_va[:10]}")
                ok = False

    return ok

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", required=True, help="data/processed/base/all.jsonl")
    ap.add_argument("--splits-dir", default="data/splits", help="Katalog ze splitami (BASE + lopo/ + lolo/)")
    ap.add_argument("--no-base", action="store_true", help="Pomiń check bazowych splitów")
    ap.add_argument("--no-lopo", action="store_true", help="Pomiń check LOPO")
    ap.add_argument("--no-lolo", action="store_true", help="Pomiń check LOLO")
    ap.add_argument("--strict-lolo", action="store_true", help="Wymuszaj pełne LOLO (język testowy nie występuje w train/val)")
    args = ap.parse_args()

    all_path = Path(args.all)
    if not all_path.is_file():
        print("[FATAL] Brak pliku all.jsonl:", all_path); sys.exit(2)

    by_id, persona_by_id, lang_by_id, opinion_by_id = load_all_index(all_path)
    splits_dir = Path(args.splits_dir)

    ok = True
    if not args.no_base:
        ok = check_base_splits(splits_dir, by_id) and ok
    if not args.no_lopo:
        ok = check_lopo(splits_dir, by_id, persona_by_id) and ok
    if not args.no_lolo:
        ok = check_lolo(splits_dir, by_id, lang_by_id, strict=args.strict_lolo) and ok

    if ok:
        print("[OK] Splity wyglądają poprawnie.")
        sys.exit(0)
    else:
        print("[FAIL] Wykryto problemy w splitach.")
        sys.exit(1)

if __name__ == "__main__":
    main()
