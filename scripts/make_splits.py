#!/usr/bin/env python3
# scripts/make_splits.py
# Generuje splity: random (ze stratyfikacją), LOPO, LOLO, K-Fold.
# Wejście: JSONL z polami co najmniej: id, persona_id, lang, text (do filtru długości)
# Wyjście: CSV z jedną kolumną 'id' (nagłówek obowiązkowy), zgodnie z configs/splits.yaml

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import yaml

# scikit-learn do podziałów
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold


# -------------- Narzędzia --------------

def load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def load_jsonl_files(files: List[str], id_field: str) -> pd.DataFrame:
    rows = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise SystemExit(f"[ERROR] JSON decode error in {fp}: {e}")
                if id_field not in obj:
                    raise SystemExit(f"[ERROR] Missing '{id_field}' in record from {fp}")
                rows.append(obj)
    if not rows:
        raise SystemExit("[ERROR] No records loaded from JSONL files.")
    df = pd.DataFrame(rows)
    return df

def write_ids_csv(ids: List[str], out_path: Path, include_header: bool = True, compression: str = "none") -> None:
    ensure_dir(out_path.parent)
    df = pd.DataFrame({"id": ids})
    if compression == "gzip":
        out_path = out_path.with_suffix(out_path.suffix + ".gz")
        df.to_csv(out_path, index=False, header=include_header, compression="gzip")
    else:
        df.to_csv(out_path, index=False, header=include_header)

def combine_strata(df: pd.DataFrame, cols: List[str]) -> Optional[pd.Series]:
    if not cols:
        return None
    for c in cols:
        if c not in df.columns:
            raise SystemExit(f"[ERROR] Missing stratify column '{c}' in data.")
    # Łączymy wartości w jedną krotkę-string (stabilne dla sklearn)
    return df[cols].astype(str).agg("||".join, axis=1)

def check_disjoint(a: set, b: set, name_a: str, name_b: str, on_violation: str = "error"):
    inter = a.intersection(b)
    if inter:
        msg = f"[{name_a} ∩ {name_b}] not disjoint, overlap count={len(inter)}"
        if on_violation == "error":
            raise SystemExit(f"[ERROR] {msg}")
        else:
            print(f"[WARN] {msg}")

def sample_ids(ids: List[str], k: int, seed: int) -> List[str]:
    rng = np.random.RandomState(seed)
    if len(ids) <= k:
        return ids
    idx = rng.choice(len(ids), size=k, replace=False)
    return [ids[i] for i in idx]

def checksum_ids(ids: List[str]) -> str:
    # prosty checksum: suma hashów modulo 1e9+7
    mod = 1_000_000_007
    s = 0
    for i in ids:
        s = (s + (hash(i) & 0xFFFFFFFF)) % mod
    return str(s)

def save_json(obj: Dict, path: Path):
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# -------------- Filtry i sanity checks --------------

def apply_filters(df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    fcfg = cfg.get("filters", {})
    # allow_langs
    allow_langs = fcfg.get("allow_langs")
    if allow_langs:
        before = len(df)
        df = df[df["lang"].isin(allow_langs)].copy()
        print(f"[INFO] Filter lang -> kept {len(df)}/{before}")
    # min_text_chars
    mtc = fcfg.get("min_text_chars")
    if mtc is not None and "text" in df.columns:
        before = len(df)
        df = df[df["text"].astype(str).str.len() >= mtc].copy()
        print(f"[INFO] Filter min_text_chars={mtc} -> kept {len(df)}/{before}")
    # require_persona_desc
    if fcfg.get("require_persona_desc", False):
        before = len(df)
        df = df[df["persona_desc"].astype(str).str.len() > 0].copy()
        print(f"[INFO] Filter require_persona_desc -> kept {len(df)}/{before}")
    return df

def integrity_checks(df: pd.DataFrame, cfg: Dict, id_field: str):
    ic = cfg.get("integrity_checks", {})
    if ic.get("error_on_missing_fields", True):
        required = [id_field, cfg["persona_field"], cfg["lang_field"]]
        for col in required:
            if col not in df.columns:
                raise SystemExit(f"[ERROR] Required field '{col}' missing in dataset.")
    if ic.get("require_unique_ids", True):
        dup = df[id_field].duplicated(keep=False)
        if dup.any():
            d = df.loc[dup, id_field].tolist()[:10]
            raise SystemExit(f"[ERROR] Duplicate ids detected (first 10): {d}")

# -------------- Generatory splitów --------------

def do_random_stratified(df: pd.DataFrame, cfg: Dict, out_cfg: Dict, global_seed: int) -> Dict:
    if not cfg.get("enabled", False):
        return {}
    seed = cfg.get("seed", global_seed)
    ratios = cfg.get("ratios", {"train": 0.8, "val": 0.1, "test": 0.1})
    assert abs(sum(ratios.values()) - 1.0) < 1e-6, "ratios must sum to 1.0"
    strat_cols = cfg.get("stratify_by", ["lang", "persona_id"])
    strat = combine_strata(df, strat_cols)
    # 1) train_val vs test
    test_size = ratios["test"]
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, shuffle=True,
        stratify=strat if strat is not None else None
    )
    # 2) train vs val (relative split within train_val)
    tv = ratios["train"] + ratios["val"]
    val_rel = ratios["val"] / tv if tv > 0 else 0.0
    strat_tv = combine_strata(train_val_df, strat_cols) if strat is not None else None
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_rel, random_state=seed+1, shuffle=True,
        stratify=strat_tv if strat_tv is not None else None
    )
    # Zapis
    out_dir = Path(out_cfg["splits_dir"])
    fn = out_cfg["file_names"]
    include_header = out_cfg.get("include_header", True)
    compression = out_cfg.get("compression", "none")
    if out_cfg.get("overwrite", True):
        ensure_dir(out_dir)
    write_ids_csv(train_df["id"].tolist(), out_dir / fn["train"], include_header, compression)
    write_ids_csv(val_df["id"].tolist(),   out_dir / fn["val"],   include_header, compression)
    write_ids_csv(test_df["id"].tolist(),  out_dir / fn["test"],  include_header, compression)
    print(f"[OK] Random stratified splits written to {out_dir}")
    return {
        "train": train_df["id"].tolist(),
        "val": val_df["id"].tolist(),
        "test": test_df["id"].tolist(),
    }

def do_lopo(df: pd.DataFrame, cfg: Dict, out_cfg: Dict, lopo_dir_name: str, global_seed: int) -> Dict:
    if not cfg.get("enabled", False):
        return {}
    seed = cfg.get("seed", global_seed)
    val_ratio = cfg.get("val_ratio_within_train", 0.1)
    strat_cols = cfg.get("stratify_by", ["lang"])
    min_test_size = cfg.get("min_test_size", 0)
    personas_cfg = cfg.get("personas", "auto")
    if personas_cfg == "auto":
        personas = sorted(df["persona_id"].dropna().unique().tolist())
    else:
        personas = list(personas_cfg)
    root_dir = Path(out_cfg["splits_dir"]) / lopo_dir_name
    include_header = out_cfg.get("include_header", True)
    compression = out_cfg.get("compression", "none")
    results = {}
    for pid in personas:
        test_df = df[df["persona_id"] == pid]
        if len(test_df) < min_test_size:
            if cfg.get("write_empty_folds", False):
                print(f"[WARN] LOPO: persona '{pid}' has only {len(test_df)} records (<{min_test_size}), writing empty fold.")
            else:
                print(f"[INFO] LOPO: skipping '{pid}' (too few test records: {len(test_df)}<{min_test_size}).")
                continue
        train_pool = df[df["persona_id"] != pid]
        if len(train_pool) == 0:
            print(f"[WARN] LOPO: no train pool for persona '{pid}', skipping.")
            continue
        # train/val split in train_pool
        strat = combine_strata(train_pool, strat_cols) if strat_cols else None
        train_df, val_df = train_test_split(
            train_pool, test_size=val_ratio, random_state=seed, shuffle=True,
            stratify=strat if strat is not None and len(set(strat)) > 1 else None
        )
        out_dir = root_dir / pid
        ensure_dir(out_dir)
        write_ids_csv(train_df["id"].tolist(), out_dir / "train_ids.csv", include_header, compression)
        write_ids_csv(val_df["id"].tolist(),   out_dir / "val_ids.csv",   include_header, compression)
        write_ids_csv(test_df["id"].tolist(),  out_dir / "test_ids.csv",  include_header, compression)
        results[pid] = {
            "train": train_df["id"].tolist(),
            "val": val_df["id"].tolist(),
            "test": test_df["id"].tolist(),
        }
        print(f"[OK] LOPO split written for persona='{pid}' -> {out_dir}")
    return results

def do_lolo(df: pd.DataFrame, cfg: Dict, out_cfg: Dict, lolo_dir_name: str, global_seed: int) -> Dict:
    if not cfg.get("enabled", False):
        return {}
    seed = cfg.get("seed", global_seed)
    val_ratio = cfg.get("val_ratio_within_train", 0.1)
    strat_cols = cfg.get("stratify_by", ["persona_id"])
    min_test_size = cfg.get("min_test_size", 0)
    langs_cfg = cfg.get("langs", "auto")
    if langs_cfg == "auto":
        langs = sorted(df["lang"].dropna().unique().tolist())
    else:
        langs = list(langs_cfg)
    root_dir = Path(out_cfg["splits_dir"]) / lolo_dir_name
    include_header = out_cfg.get("include_header", True)
    compression = out_cfg.get("compression", "none")
    results = {}
    for lang in langs:
        test_df = df[df["lang"] == lang]
        if len(test_df) < min_test_size:
            if cfg.get("write_empty_folds", False):
                print(f"[WARN] LOLO: lang '{lang}' has only {len(test_df)} records (<{min_test_size}), writing empty fold.")
            else:
                print(f"[INFO] LOLO: skipping '{lang}' (too few test records: {len(test_df)}<{min_test_size}).")
                continue
        train_pool = df[df["lang"] != lang]
        if len(train_pool) == 0:
            print(f"[WARN] LOLO: no train pool for lang '{lang}', skipping.")
            continue
        strat = combine_strata(train_pool, strat_cols) if strat_cols else None
        train_df, val_df = train_test_split(
            train_pool, test_size=val_ratio, random_state=seed, shuffle=True,
            stratify=strat if strat is not None and len(set(strat)) > 1 else None
        )
        out_dir = root_dir / lang
        ensure_dir(out_dir)
        write_ids_csv(train_df["id"].tolist(), out_dir / "train_ids.csv", include_header, compression)
        write_ids_csv(val_df["id"].tolist(),   out_dir / "val_ids.csv",   include_header, compression)
        write_ids_csv(test_df["id"].tolist(),  out_dir / "test_ids.csv",  include_header, compression)
        results[lang] = {
            "train": train_df["id"].tolist(),
            "val": val_df["id"].tolist(),
            "test": test_df["id"].tolist(),
        }
        print(f"[OK] LOLO split written for lang='{lang}' -> {out_dir}")
    return results

def do_kfold(df: pd.DataFrame, cfg: Dict, out_cfg: Dict, kfold_dir_name: str, global_seed: int) -> Dict:
    if not cfg.get("enabled", False):
        return {}
    k = int(cfg.get("k", 5))
    shuffle = bool(cfg.get("shuffle", True))
    seed = cfg.get("seed", global_seed)
    stratified = bool(cfg.get("stratified", True))
    strat_cols = cfg.get("stratify_by", ["lang", "persona_id"]) if stratified else []
    include_test = bool(cfg.get("include_test", False))
    # generator
    if stratified:
        y = combine_strata(df, strat_cols)
        # jeżeli tylko 1 unikatowa klasa, stratifiedKFold się wywróci → użyj zwykłego KFold
        if y is not None and len(set(y)) > 1:
            splitter = StratifiedKFold(n_splits=k, shuffle=shuffle, random_state=seed)
            y_np = y.to_numpy()
        else:
            print("[WARN] KFold: not enough classes for stratification; falling back to plain KFold.")
            splitter = KFold(n_splits=k, shuffle=shuffle, random_state=seed)
            y_np = None
    else:
        splitter = KFold(n_splits=k, shuffle=shuffle, random_state=seed)
        y_np = None

    out_root = Path(out_cfg["splits_dir"]) / kfold_dir_name.format(k=k, i="{i}")
    include_header = out_cfg.get("include_header", True)
    compression = out_cfg.get("compression", "none")

    idx = np.arange(len(df))
    results = {}
    fold_i = 0
    for train_idx, val_idx in splitter.split(idx, y_np):
        fold_dir = Path(str(out_root).format(i=fold_i))
        ensure_dir(fold_dir)
        train_ids = df.iloc[train_idx]["id"].tolist()
        val_ids = df.iloc[val_idx]["id"].tolist()
        write_ids_csv(train_ids, fold_dir / "train_ids.csv", include_header, compression)
        write_ids_csv(val_ids, fold_dir / "val_ids.csv", include_header, compression)
        if include_test:
            # najprościej: test = val (jeśli ktoś chce, może to zmienić)
            write_ids_csv(val_ids, fold_dir / "test_ids.csv", include_header, compression)
        results[f"fold_{fold_i}"] = {"train": train_ids, "val": val_ids}
        print(f"[OK] KFold split written -> {fold_dir}")
        fold_i += 1
    return results

# -------------- Raporty --------------

def build_distribution(df: pd.DataFrame, group_by: List[str]) -> Dict:
    if not group_by:
        return {}
    g = df.groupby(group_by).size().reset_index(name="count")
    # na dict: { "lang=pl|persona_id=young_mother": count, ... }
    out = {}
    for _, row in g.iterrows():
        key = "|".join(f"{col}={row[col]}" for col in group_by)
        out[str(key)] = int(row["count"])
    return out

def write_reports(all_sets: Dict[str, Dict[str, List[str]]],
                  df: pd.DataFrame,
                  checks_cfg: Dict,
                  splits_dir: Path):
    dist_cfg = checks_cfg.get("distribution_reports", {})
    size_cfg = checks_cfg.get("size_report", {})
    prev_cfg = checks_cfg.get("preview_ids", {})
    print_chksum = checks_cfg.get("print_checksums", False)
    seed = checks_cfg.get("report_seed", 202)

    reports = {}

    # rozmiary
    if size_cfg.get("enabled", True):
        sizes = {}
        for variant, sets in all_sets.items():
            sizes[variant] = {name: len(ids) for name, ids in sets.items()}
        save_json(sizes, Path(size_cfg.get("save_to", "reports/splits/sizes.json")))
        reports["sizes"] = sizes
        print(f"[OK] Size report saved.")

    # dystrybucje
    if dist_cfg.get("enabled", True):
        group_by = dist_cfg.get("group_by", ["lang", "persona_id"])
        dist_all = {}
        for variant, sets in all_sets.items():
            dist_all[variant] = {}
            for name, ids in sets.items():
                sub = df[df["id"].isin(ids)]
                dist_all[variant][name] = build_distribution(sub, group_by)
        save_json(dist_all, Path(dist_cfg.get("save_to", "reports/splits/distributions.json")))
        reports["distributions"] = dist_all
        print(f"[OK] Distribution report saved.")

    # podgląd ID
    if prev_cfg.get("enabled", True):
        per_split = int(prev_cfg.get("per_split", 5))
        prev = {}
        for variant, sets in all_sets.items():
            prev[variant] = {}
            for name, ids in sets.items():
                prev_ids = sample_ids(ids, per_split, seed)
                if print_chksum:
                    prev[variant][name] = {
                        "ids": prev_ids,
                        "checksum": checksum_ids(ids)
                    }
                else:
                    prev[variant][name] = prev_ids
        save_json(prev, Path(prev_cfg.get("save_to", "reports/splits/preview_ids.json")))
        reports["preview_ids"] = prev
        print(f"[OK] Preview IDs report saved.")

    # rozłączność (tylko dla wariantu 'random' – LOPO/LOLO mają rozłączność z definicji)
    dj = checks_cfg.get("disjoint_sets", {})
    if dj.get("enabled", True) and "random" in all_sets:
        sets = all_sets["random"]
        a, b, c = set(sets.get("train", [])), set(sets.get("val", [])), set(sets.get("test", []))
        on_v = dj.get("on_violation", "error")
        check_disjoint(a, b, "train", "val", on_v)
        check_disjoint(a, c, "train", "test", on_v)
        check_disjoint(b, c, "val", "test", on_v)
        print("[OK] Disjointness check passed for random splits.")

# -------------- Main --------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/splits.yaml", help="Ścieżka do pliku konfiguracyjnego splits.yaml")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    global_seed = int(cfg.get("reproducibility", {}).get("global_seed", 1337))
    np.random.seed(global_seed)

    ds = cfg["data_source"]
    files = ds.get("files", [])
    if not files:
        raise SystemExit("[ERROR] data_source.files is empty in splits.yaml")
    id_field = ds.get("id_field", "id")

    # 1) Wczytaj dane
    df = load_jsonl_files(files, id_field=id_field)

    # 2) Filtry i sanity
    df = apply_filters(df, ds)
    integrity_checks(df, ds, id_field=id_field)

    # 3) Konfiguracje wyjścia i integracji
    out_cfg = cfg["output"]
    ih = cfg.get("integration_hints", {})
    lopo_dir_name = ih.get("lopo_dir", "lopo")
    lolo_dir_name = ih.get("lolo_dir", "lolo")
    kfold_dir_name = ih.get("kfold_folder_name_format", "k={k}/fold_{i}")
    ensure_dir(Path(out_cfg["splits_dir"]))

    # 4) Generuj splity
    all_sets = {}

    rnd_sets = do_random_stratified(df, cfg.get("random_stratified", {}), out_cfg, global_seed)
    if rnd_sets:
        all_sets["random"] = rnd_sets

    lopo_sets = do_lopo(df, cfg.get("lopo", {}), out_cfg, lopo_dir_name, global_seed)
    if lopo_sets:
        all_sets["lopo"] = {k: v for k, v in lopo_sets.items()}  # per persona

    lolo_sets = do_lolo(df, cfg.get("lolo", {}), out_cfg, lolo_dir_name, global_seed)
    if lolo_sets:
        all_sets["lolo"] = {k: v for k, v in lolo_sets.items()}  # per lang

    kf_sets = do_kfold(df, cfg.get("kfold", {}), out_cfg, kfold_dir_name, global_seed)
    if kf_sets:
        all_sets["kfold"] = kf_sets

    # 5) Raporty i kontrole
    checks_cfg = cfg.get("checks_and_reports", {})
    write_reports(all_sets, df, checks_cfg, Path(out_cfg["splits_dir"]))

    print("\n[DONE] Splits generated successfully.")

if __name__ == "__main__":
    main()