#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rysuje reliability plots i liczy ECE/RMCE dla regresji [0,1].

Wejście:
  A) bezpośrednio summary (JSON) z modules.metrics.calibration.compute_calibration_summary
     --summary <path/to/calibration_summary.json>
  B) surowe predykcje z eval_encoder.py:
     --checkpoint <dir> --split {val,test}
     (wczyta <checkpoint>/eval_<split>/preds_labels.npz i policzy summary)

Dodatki:
  - --label-names schema/label_names.json (lista 21 nazw etykiet)
  - filtrowanie: --topk-ece K (rysuje tylko K najgorzej skalibrowanych), albo --labels 0,3,7
  - zapis:
      <outdir>/calibration_summary.json
      <outdir>/calibration_table.csv
      <outdir>/plots/label_{idx}_{slug}.png
      <outdir>/plots/overview.png

Użycie:
  python scripts/plot_calibration.py --checkpoint artifacts/models/encoder/enc_baseline_xlmr --split val --label-names schema/label_names.json
  # albo
  python scripts/plot_calibration.py --summary artifacts/models/encoder/enc_baseline_xlmr/eval_val/calibration_summary.json --label-names schema/label_names.json
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt

from modules.metrics.calibration import (
    compute_calibration_summary,
    save_calibration_summary,
)

# ------------------------- IO helpers -------------------------

def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))

def load_label_names(path: Optional[str | Path], d: int) -> List[str]:
    if not path:
        return [f"L{i}" for i in range(d)]
    obj = load_json(path)
    if isinstance(obj, dict):
        # dopuszczamy {"labels":[...]}
        names = obj.get("labels") or obj.get("label_names")
    else:
        names = obj
    if not isinstance(names, list):
        return [f"L{i}" for i in range(d)]
    # dociąć/padnąć do d
    names = [str(x) for x in names][:d]
    while len(names) < d:
        names.append(f"L{len(names)}")
    return names

def slugify(s: str) -> str:
    import re
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9\-_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "label"


# ---------------------- Plotting helpers ----------------------

def plot_reliability_one(
    ax: plt.Axes,
    centers: np.ndarray,
    mean_pred: np.ndarray,
    mean_true: np.ndarray,
    counts: np.ndarray,
    title: str = "",
):
    """Wykres krzywej reliability + diagonal."""
    # diagonal
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0)
    # krzywa
    ax.plot(centers, mean_true, marker="o", linewidth=2.0)
    # opcjonalnie słupki liczności jako drugi osi Y (subtelnie)
    ax2 = ax.twinx()
    ax2.bar(centers, counts / max(counts.sum(), 1), width=centers[1] - centers[0] if len(centers) > 1 else 0.05, alpha=0.15)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Avg predicted (bin)")
    ax.set_ylabel("Avg true (bin)")
    ax.set_title(title, fontsize=10)
    # schludniej
    ax.grid(alpha=0.2)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def save_table_csv(path: Path, ece: List[float], rmce: List[float], names: List[str]):
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label_idx", "label_name", "ECE", "RMCE"])
        for i, (e, r) in enumerate(zip(ece, rmce)):
            w.writerow([i, names[i] if i < len(names) else f"L{i}", f"{e:.6f}", f"{r:.6f}"])


# ----------------------------- Main -----------------------------

def main():
    ap = argparse.ArgumentParser()
    # tryb A: summary.json
    ap.add_argument("--summary", default=None, help="Ścieżka do istniejącego calibration_summary.json")
    # tryb B: checkpoint + split → policz summary z preds_labels.npz
    ap.add_argument("--checkpoint", default=None, help="Katalog checkpointu (z eval_<split>/preds_labels.npz)")
    ap.add_argument("--split", choices=["val", "test"], default="val")
    # wspólne
    ap.add_argument("--label-names", default=None, help="schema/label_names.json (opcjonalnie)")
    ap.add_argument("--outdir", default=None, help="Katalog wyjściowy; domyślnie <checkpoint>/eval_<split>/calibration lub dirname(summary)")
    ap.add_argument("--bins", type=int, default=15)
    ap.add_argument("--strategy", choices=["uniform", "quantile"], default="uniform")
    ap.add_argument("--topk-ece", type=int, default=None, help="Rysuj tylko K najgorszych etykiet wg ECE")
    ap.add_argument("--labels", default=None, help="Konkretne indeksy etykiet, np. '0,3,7'")
    ap.add_argument("--dpi", type=int, default=140)
    ap.add_argument("--figsize", default="3,3", help="Szerokość,wysokość pojedynczego wykresu w calach, np. '3,3'")
    args = ap.parse_args()

    # Ustal ścieżki
    summary_obj: Dict[str, Any] | None = None
    if args.summary:
        summary_path = Path(args.summary)
        if not summary_path.is_file():
            raise FileNotFoundError(f"Summary not found: {summary_path}")
        summary_obj = load_json(summary_path)
        if args.outdir:
            outdir = Path(args.outdir)
        else:
            outdir = summary_path.parent
    else:
        # checkpoint mode
        if not args.checkpoint:
            raise SystemExit("Podaj --summary lub (--checkpoint i --split).")
        ckpt = Path(args.checkpoint)
        npz_path = ckpt / f"eval_{args.split}" / "preds_labels.npz"
        if not npz_path.is_file():
            raise FileNotFoundError(f"File not found: {npz_path}")
        data = np.load(npz_path)
        y_pred = data["y_pred"]
        y_true = data["y_true"]
        summary_obj = compute_calibration_summary(y_true, y_pred, n_bins=args.bins, strategy=args.strategy)
        # domyślny outdir
        outdir = ckpt / f"eval_{args.split}" / "calibration"
        outdir.mkdir(parents=True, exist_ok=True)
        save_calibration_summary(outdir / "calibration_summary.json", summary_obj)

    # label names
    d = int(summary_obj["n_labels"])
    names = load_label_names(args.label_names, d)

    # wybór etykiet do rysowania
    selected_idx: List[int]
    if args.labels:
        selected_idx = [int(x) for x in str(args.labels).split(",") if str(x).strip().isdigit()]
        selected_idx = [i for i in selected_idx if 0 <= i < d]
    elif args.topk_ece:
        # wybierz K najgorszych wg ECE
        ece_arr = np.asarray(summary_obj["ece_per_label"], dtype=np.float64)
        order = np.argsort(-ece_arr)  # malejąco
        k = min(int(args.topk_ece), d)
        selected_idx = order[:k].tolist()
    else:
        selected_idx = list(range(d))

    # parametry rysowania
    try:
        w, h = [float(x) for x in args.figsize.split(",")]
    except Exception:
        w, h = 3.0, 3.0
    plots_dir = Path(outdir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # tabelka CSV
    save_table_csv(Path(outdir) / "calibration_table.csv", summary_obj["ece_per_label"], summary_obj["rmce_per_label"], names)

    # per-label plots
    for i in selected_idx:
        pl = summary_obj["per_label"][i]
        centers = np.asarray(pl["curve"]["bin_centers"], dtype=np.float64)
        mean_pred = np.asarray(pl["curve"]["mean_pred"], dtype=np.float64)
        mean_true = np.asarray(pl["curve"]["mean_true"], dtype=np.float64)
        counts = np.asarray(pl["curve"]["counts"], dtype=np.int64)

        fig, ax = plt.subplots(figsize=(w, h))
        title = f"[{i}] {names[i]} | ECE={pl['ece']:.3f}, RMCE={pl['rmce']:.3f}"
        plot_reliability_one(ax, centers, mean_pred, mean_true, counts, title=title)
        fig.tight_layout()
        fig.savefig(plots_dir / f"label_{i:02d}_{slugify(names[i])}.png", dpi=args.dpi)
        plt.close(fig)

    # overview (siatka max 9 lub 12 wykresów)
    max_grid = min(len(selected_idx), 12)
    if max_grid > 0:
        cols = 3 if max_grid <= 9 else 4
        rows = int(np.ceil(max_grid / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * w, rows * h))
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = np.array([axes])
        axes = axes.reshape(rows, cols)

        for j in range(rows * cols):
            r, c = divmod(j, cols)
            ax = axes[r, c]
            if j >= max_grid:
                ax.axis("off")
                continue
            i = selected_idx[j]
            pl = summary_obj["per_label"][i]
            centers = np.asarray(pl["curve"]["bin_centers"], dtype=np.float64)
            mean_pred = np.asarray(pl["curve"]["mean_pred"], dtype=np.float64)
            mean_true = np.asarray(pl["curve"]["mean_true"], dtype=np.float64)
            counts = np.asarray(pl["curve"]["counts"], dtype=np.int64)
            title = f"[{i}] {names[i]} | ECE={pl['ece']:.2f}"
            plot_reliability_one(ax, centers, mean_pred, mean_true, counts, title=title)

        fig.suptitle(
            f"Calibration overview | macro ECE={summary_obj['ece_macro']:.3f}, RMCE={summary_obj['rmce_macro']:.3f}",
            fontsize=12
        )
        fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        fig.savefig(plots_dir / "overview.png", dpi=args.dpi)
        plt.close(fig)

    # krótkie podsumowanie w konsoli
    print(f"[OK] Saved calibration summary to: {Path(outdir) / 'calibration_summary.json'}")
    print(f"[OK] Saved per-label plots to: {plots_dir}")
    print(f"[OK] Saved table to: {Path(outdir) / 'calibration_table.csv'}")
    print(f"[MACRO] ECE={summary_obj['ece_macro']:.4f} | RMCE={summary_obj['rmce_macro']:.4f}")
    

if __name__ == "__main__":
    main()
