#!/usr/bin/env python3
"""
Łączy opinions.json + emotion_matrix.json (+ personas.json + label_names.json)
→ zapisuje data/processed/base/all.jsonl w formacie:
{"id","opinion_id","text","persona_id","persona_desc","lang","labels":[21x float 0..1]}

Obsługa kilku wariantów wejścia:
- opinions.json może być listą obiektów, dict-em {id: {...}} albo JSONL.
- łączenie z emotion_matrix po kluczu "opinion_{index+1}" lub po polu ID (parametryzowane).
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import re
import sys

# --- Heurystyczne wykrywanie języka PL/EN (bez ciężkich zależności) ---
PL_CHARS = set("ąćęłńóśżźĄĆĘŁŃÓŚŻŹ")

def detect_lang_heuristic(text: str, default: str = "pl") -> str:
    if not isinstance(text, str) or not text.strip():
        return default
    
    # Strong indicator: Polish diacritics
    if any(ch in PL_CHARS for ch in text):
        return "pl"
    
    # Improved English detection heuristics
    t = text.lower()
    
    # Common English words and patterns
    en_words = [" the ", " and ", " is ", " was ", " with ", " to ", " of ", " in ", " that ", " for ", 
                " a ", " an ", " this ", " will ", " be ", " have ", " has ", " had ", " would ", " could ",
                " should ", " doctor ", " very ", " good ", " great ", " excellent ", " patient ", " visit ",
                " appointment ", " treatment ", " care ", " professional ", " recommend ", " staff "]
    
    # Common Polish words
    pl_words = [" i ", " jest ", " był ", " była ", " oraz ", " że ", " na ", " w ", " z ", " do ", " się ",
                " ma ", " nie ", " jak ", " to ", " po ", " bardzo ", " doktor ", " lekarz ", " wizyta ",
                " pacjent ", " polecam ", " świetny ", " dobry ", " profesjonalny "]
    
    # Count word occurrences
    en_hits = sum(t.count(word) for word in en_words)
    pl_hits = sum(t.count(word) for word in pl_words)
    
    # Additional English patterns
    en_patterns = 0
    if " th" in t:  # "th" is very common in English
        en_patterns += t.count(" th")
    if "'s " in t or "'t " in t or "'ll " in t or "'re " in t:  # English contractions
        en_patterns += 2
    if text.count(".") > 0 and any(word in t for word in [" very ", " really ", " quite "]):
        en_patterns += 1
    
    total_en_score = en_hits + en_patterns
    
    # Decision logic
    if total_en_score > pl_hits and total_en_score > 0:
        return "en"
    elif pl_hits > total_en_score:
        return "pl"
    
    # Fallback: if no clear indicators and text is Latin script without Polish chars, assume English
    # This helps with short English texts that don't contain common words
    if all(ord(c) < 256 for c in text) and len(text.strip()) > 10:
        return "en"
    
    return default

# --- Persona desc renderer ---
def render_persona_desc(p: Dict[str, Any]) -> str:
    name = (p.get("name") or p.get("id") or "").strip()
    parts = [name] if name else []
    if p.get("sensitivity"):
        s = str(p["sensitivity"]).strip().rstrip(".")
        parts.append(f"Sensitivity: {s}.")
    if p.get("values"):
        v = str(p["values"]).strip().rstrip(".")
        parts.append(f"Values: {v}.")
    desc = " ".join(parts).strip()
    return " ".join(desc.split())

def load_json_maybe_jsonl(path: Path) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
    """Wczytuje .json (lista lub dict) albo .jsonl (lista rekordów)."""
    txt = path.read_text(encoding="utf-8")
    # spróbuj klasyczny JSON
    try:
        obj = json.loads(txt)
        return obj
    except json.JSONDecodeError:
        # spróbuj JSONL
        rows: List[Dict[str, Any]] = []
        for line in txt.splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
        return rows

def normalize_opinion_records(op_obj: Union[List[Dict[str, Any]], Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Zwraca listę rekordów opinii. Jeśli wejście to dict {id: {...}}, składa id do rekordu."""
    if isinstance(op_obj, list):
        return op_obj
    elif isinstance(op_obj, dict):
        out = []
        for k, v in op_obj.items():
            if isinstance(v, dict) and "id" not in v:
                v = {**v, "id": k}
            out.append(v)
        return out
    else:
        raise SystemExit("[ERROR] opinions.json must be a JSON array, object, or JSONL")

def build_matrix_key(idx: int, rec: Dict[str, Any], args) -> str:
    """Klucz do emotion_matrix (np. 'opinion_1'). Priorytety:
       1) --matrix_key_field, jeśli obecne w rekordzie
       2) 'opinion_{index+1}' (domyślnie)"""
    if args.matrix_key_field and args.matrix_key_field in rec:
        return str(rec[args.matrix_key_field])
    return f"{args.matrix_key_prefix}{idx+1}"

def coerce_float(x: Any) -> Optional[float]:
    """Toleruje liczby jako str, przecinek dziesiętny i procenty."""
    if x is None:
        return None
    if isinstance(x, str):
        s = x.strip()
        pct = False
        if s.endswith("%"):
            pct = True
            s = s[:-1].strip()
        if "," in s and "." not in s:
            s = s.replace(",", ".")
        try:
            val = float(s)
            if pct:
                val = val / 100.0
            return val
        except Exception:
            return None
    try:
        return float(x)
    except Exception:
        return None

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def norm_key(k: str) -> str:
    """Normalizuje klucze etykiet (case-insensitive, spacje/myślniki → podkreślnik)."""
    return str(k).strip().lower().replace(" ", "_").replace("-", "_")

def get_label_value(label_dict: dict, target_name: str):
    """Pobiera wartość dla etykiety: najpierw exact match, potem po kluczu znormalizowanym."""
    if not isinstance(label_dict, dict):
        return None
    # exact match
    if target_name in label_dict:
        return label_dict[target_name]
    # case/format-insensitive match
    nd = {norm_key(k): v for k, v in label_dict.items()}
    return nd.get(norm_key(target_name))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opinions", required=True, help="Ścieżka do opinions.json (lista, dict lub JSONL)")
    ap.add_argument("--matrix", required=True, help="Ścieżka do emotion_matrix.json (dict opinion_key -> persona -> label->value)")
    ap.add_argument("--personas", required=True, help="Ścieżka do data/personas/personas.json")
    ap.add_argument("--label_names", required=True, help="Ścieżka do schema/label_names.json (kolejność 21 etykiet)")
    ap.add_argument("--out", default="data/processed/base/all.jsonl", help="Ścieżka wyjściowa JSONL")
    # mapowanie pól / join
    ap.add_argument("--id_field", default=None, help="Pole w opinions z ID (jeśli brak, utworzymy syntetyczne op_000001, ...)")
    ap.add_argument("--text_field", default=None, help="Nazwa pola z tekstem opinii (jeśli None, spróbujemy wykryć)")
    ap.add_argument("--lang_field", default=None, help="Pole z językiem ('pl'/'en'); gdy brak, heurystyka")
    ap.add_argument("--matrix_key_field", default=None, help="Pole z kluczem do emotion_matrix (np. 'opinion_id'); jak brak, użyjemy indexu")
    ap.add_argument("--matrix_key_prefix", default="opinion_", help="Prefiks do klucza matrix gdy łączymy po indeksie (np. 'opinion_')")
    ap.add_argument("--default_lang", default="pl", help="Domyślny język, gdy nie można wykryć")
    args = ap.parse_args()

    opinions_path = Path(args.opinions)
    matrix_path = Path(args.matrix)
    personas_path = Path(args.personas)
    labels_path = Path(args.label_names)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) Wczytaj wejścia
    op_raw = load_json_maybe_jsonl(opinions_path)
    ops = normalize_opinion_records(op_raw)
    matrix: Dict[str, Any] = json.loads(matrix_path.read_text(encoding="utf-8"))
    personas_list: List[Dict[str, Any]] = json.loads(personas_path.read_text(encoding="utf-8"))
    label_names_raw: List[str] = json.loads(labels_path.read_text(encoding="utf-8"))
    # kanonizujemy nazwy etykiet do dolnego case'u tylko na potrzeby dopasowania,
    # ale zachowujemy oryginalną kolejność z pliku
    label_names = [str(n) for n in label_names_raw]

    # 2) Indeksy pomocnicze
    personas_by_id = {p["id"]: p for p in personas_list if isinstance(p, dict) and "id" in p}

    # heurystyka pola tekstowego
    text_candidates = [args.text_field, "tekst", "text", "content", "opinia", "review"]
    text_candidates = [c for c in text_candidates if c]
    if not text_candidates:
        probe_keys = set().union(*[set(d.keys()) for d in ops if isinstance(d, dict)])
        for cand in ["tekst", "text", "content", "opinia", "review"]:
            if cand in probe_keys:
                text_candidates.append(cand)
                break
    if not text_candidates:
        raise SystemExit("[ERROR] Nie znalazłem pola z tekstem (spróbuj --text_field).")
    text_field = text_candidates[0]

    # 3) Główna pętla: składamy all.jsonl
    with out_path.open("w", encoding="utf-8") as out_f:
        created = 0
        missing_matrix = 0
        missing_persona = 0
        label_len = len(label_names)

        for idx, rec in enumerate(ops):
            if not isinstance(rec, dict):
                continue

            # Tekst
            text_val = str(rec.get(text_field, "")).strip()
            if not text_val:
                continue

            # Język
            if args.lang_field and args.lang_field in rec and rec[args.lang_field] in ("pl", "en"):
                lang = rec[args.lang_field]
            else:
                lang = detect_lang_heuristic(text_val, default=args.default_lang)

            # Klucz do macierzy emocji (po ID lub indeksie)
            mkey = build_matrix_key(idx, rec, args)
            persona_map: Dict[str, Dict[str, Any]] = matrix.get(mkey)  # może nie istnieć
            if not isinstance(persona_map, dict):
                missing_matrix += 1
                continue

            # Opinion base ID (stabilne dla tej opinii, niezależnie od persony)
            if args.id_field and args.id_field in rec:
                base_id_val = str(rec[args.id_field])
            else:
                base_id_val = f"{idx+1}"
            digits = re.findall(r"\d+", base_id_val)
            opinion_id = f"op_{int(digits[0]):06d}" if digits else f"op_{idx+1:06d}"

            # Dla każdej persony z macierzy zbuduj rekord
            for persona_id, label_dict in persona_map.items():
                # Persona desc
                p = personas_by_id.get(persona_id)
                if p is None:
                    missing_persona += 1
                    persona_desc = persona_id
                else:
                    persona_desc = render_persona_desc(p) or persona_id

                # Wektor etykiet w ustalonej kolejności
                labels_vec: List[float] = []
                for name in label_names:
                    raw = get_label_value(label_dict, name)
                    val = coerce_float(raw)
                    labels_vec.append(clamp01(val if val is not None else 0.0))

                if len(labels_vec) != label_len:
                    continue

                # Unikalny klucz rekordu = opinia × persona
                record_id = f"{opinion_id}__p={persona_id}"

                out_obj = {
                    "id": record_id,
                    "opinion_id": opinion_id,
                    "text": text_val,
                    "persona_id": persona_id,
                    "persona_desc": persona_desc,
                    "lang": lang,
                    "labels": labels_vec,
                }
                out_f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                created += 1

    print(f"[OK] Wygenerowano: {created} rekordów -> {out_path}")
    if 'missing_matrix' in locals() and missing_matrix:
        print(f"[WARN] Opinie bez wpisu w emotion_matrix: {missing_matrix}")
    if 'missing_persona' in locals() and missing_persona:
        print(f"[WARN] Rekordy z nieznaną personą (fallback na id): {missing_persona}")

if __name__ == "__main__":
    main()