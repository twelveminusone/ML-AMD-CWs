#!/usr/bin/env python
# -*- coding: utf-8 -*-

# --- add sibling 'utils/' (from imputation_order_sensitivity) to PYTHONPATH ---
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
SIBLING = HERE.parent / "imputation_order_sensitivity"
sys.path.insert(0, str(SIBLING))  # not strictly needed here, but keeps imports consistent
# -----------------------------------------------------------------------------

"""
Input Perturbation — Heatmap (Delta vs Baseline)
------------------------------------------------
Read the results produced by input_perturb_sensitivity.py and plot a heatmap
of metric deltas (perturbed - baseline), with rows = Feature±scale and
columns = Targets.

Usage:
  python make_input_perturb_heatmap.py \
    --input ../../outputs/input_perturbation_sensitivity/results.xlsx \
    --outdir ../../outputs/input_perturbation_sensitivity/figures \
    --metric R2 --fmt png --dpi 200
"""

from __future__ import annotations
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Heatmap of input-perturbation deltas vs baseline.")
    p.add_argument("--input", type=Path,
                   default=Path("../../outputs/input_perturbation_sensitivity/results.xlsx"),
                   help="Excel file produced by input_perturb_sensitivity.py (sheet=results).")
    p.add_argument("--sheet", type=str, default="results", help="Sheet name, default: results")
    p.add_argument("--outdir", type=Path,
                   default=Path("../../outputs/input_perturbation_sensitivity/figures"),
                   help="Directory to save figures.")
    p.add_argument("--metric", type=str, default="R2",
                   choices=["R2", "RMSE", "MAE"], help="Metric to visualize.")
    p.add_argument("--fmt", type=str, default="png", help="Figure format: png|tif|pdf ...")
    p.add_argument("--dpi", type=int, default=200, help="Figure DPI.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(args.input, sheet_name=args.sheet)
    needed = {"Target", "FeaturePerturbed", "Scale", "Label", "R2", "RMSE", "MAE"}
    missing = needed.difference(df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns in {args.input}: {missing}")

    # Split baseline vs perturbed
    base = df[df["FeaturePerturbed"].str.lower() == "baseline"].copy()
    pert = df[df["FeaturePerturbed"].str.lower() != "baseline"].copy()

    if base.empty or pert.empty:
        raise RuntimeError("No baseline or no perturbed rows found. Did you run input_perturb_sensitivity.py?")

    # For delta, subtract baseline per target (same split/model setting)
    metric = args.metric
    base_m = base[["Target", metric]].rename(columns={metric: "baseline"})
    pert = pert.merge(base_m, on="Target", how="left")
    pert["Delta"] = pert[metric] - pert["baseline"]

    # Friendly row labels: e.g., "i_pH (+10%)"
    def _lab(r):
        if r.get("Label", "") == "error":
            return f"{r['FeaturePerturbed']} (err)"
        try:
            pct = (float(r["Scale"]) - 1.0) * 100.0
            sign = "+" if pct >= 0 else ""
            return f"{r['FeaturePerturbed']} ({sign}{pct:.0f}%)"
        except Exception:
            return f"{r['FeaturePerturbed']} ({r.get('Label','?')})"

    pert["RowKey"] = pert.apply(_lab, axis=1)

    # Pivot: rows = Feature±scale, cols = Target
    pivot = pert.pivot_table(index="RowKey", columns="Target", values="Delta", aggfunc="mean")
    pivot = pivot.sort_index()

    # Color limits symmetric around zero
    vmax = np.nanmax(np.abs(pivot.values)) if pivot.size else 1.0
    vmax = 1.0 if not np.isfinite(vmax) or vmax == 0 else float(vmax)
    vlim = vmax

    fig, ax = plt.subplots(figsize=(max(6, 0.35 * pivot.shape[1] + 3), max(6, 0.28 * pivot.shape[0] + 2)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu_r", vmin=-vlim, vmax=vlim)
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(pivot.index.tolist())
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns.tolist(), rotation=45, ha="right")
    ax.set_title(f"Δ{metric} (Perturbed − Baseline)")
    ax.grid(False)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f"Delta {metric}")

    fig.tight_layout()
    outpath = args.outdir / f"heatmap_delta_{metric}.{args.fmt}"
    fig.savefig(outpath, dpi=args.dpi)
    plt.close(fig)

    # Optionally save pivot as CSV
    pivot_path = args.outdir / f"heatmap_delta_{metric}_pivot.csv"
    pivot.to_csv(pivot_path, index=True)
    print(f"[OK] Saved figure to: {outpath}")
    print(f"[OK] Saved pivot CSV to: {pivot_path}")


if __name__ == "__main__":
    main()
