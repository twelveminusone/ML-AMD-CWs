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
Input Perturbation — Line Plots
-------------------------------
For each Target, draw a line chart over the perturbation "scales" for every
feature, either on absolute metric values or delta vs baseline.

Usage:
  python make_input_perturb_lineplots.py \
    --input ../../outputs/input_perturbation_sensitivity/results.xlsx \
    --outdir ../../outputs/input_perturbation_sensitivity/figures \
    --metric R2 --mode delta --fmt png --dpi 200
"""

from __future__ import annotations
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Line plots per Target across feature perturbation scales.")
    p.add_argument("--input", type=Path,
                   default=Path("../../outputs/input_perturbation_sensitivity/results.xlsx"),
                   help="Excel file produced by input_perturb_sensitivity.py (sheet=results).")
    p.add_argument("--sheet", type=str, default="results", help="Sheet name, default: results")
    p.add_argument("--outdir", type=Path,
                   default=Path("../../outputs/input_perturbation_sensitivity/figures"),
                   help="Directory to save figures.")
    p.add_argument("--metric", type=str, default="R2",
                   choices=["R2", "RMSE", "MAE"], help="Metric to plot.")
    p.add_argument("--mode", type=str, default="delta",
                   choices=["absolute", "delta"], help="Plot absolute metric or delta vs baseline.")
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

    metric = args.metric
    # get unique, sorted scales for consistent x-axis
    pert = df[df["FeaturePerturbed"].str.lower() != "baseline"].copy()
    if pert.empty:
        raise RuntimeError("No perturbed rows found. Did you run input_perturb_sensitivity.py?")
    scales_sorted = sorted({float(s) for s in pert["Scale"].dropna().values})
    scale_labels = [f"{'+' if (s-1.0)>=0 else ''}{(s-1.0)*100:.0f}%" for s in scales_sorted]

    # Merge baseline to compute delta if needed
    if args.mode == "delta":
        base = df[df["FeaturePerturbed"].str.lower() == "baseline"][["Target", metric]].rename(columns={metric: "baseline"})
        pert = pert.merge(base, on="Target", how="left")
        pert["Value"] = pert[metric] - pert["baseline"]
        ylab = f"Δ{metric} (Perturbed − Baseline)"
        ttl_suffix = " (delta)"
    else:
        pert["Value"] = pert[metric]
        ylab = f"{metric}"
        ttl_suffix = " (absolute)"

    # For each Target, line per Feature across scales
    for tgt, sub in pert.groupby("Target"):
        # Build pivot: rows=Scale, cols=Feature, values=Value
        try:
            sub["Scale"] = sub["Scale"].astype(float)
        except Exception:
            pass
        pv = sub.pivot_table(index="Scale", columns="FeaturePerturbed", values="Value", aggfunc="mean")
        pv = pv.reindex(scales_sorted)

        fig, ax = plt.subplots(figsize=(7, 4.5))
        for feat in pv.columns:
            y = pv[feat].values
            ax.plot(scales_sorted, y, marker="o", linewidth=2, label=feat)

        ax.set_xticks(scales_sorted)
        ax.set_xticklabels(scale_labels, rotation=0)
        ax.set_xlabel("Perturbation")
        ax.set_ylabel(ylab)
        ax.set_title(f"Input Perturbation — {tgt}{ttl_suffix}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9, ncol=2)

        fig.tight_layout()
        outpath = args.outdir / f"line_{tgt.replace(' ', '_').replace('%','pct')}_{args.mode}_{metric}.{args.fmt}"
        fig.savefig(outpath, dpi=args.dpi)
        plt.close(fig)
        print(f"[OK] Saved: {outpath}")


if __name__ == "__main__":
    main()
