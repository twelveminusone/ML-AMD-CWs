#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plot Multi-Horizon Results (ML vs Naive)
---------------------------------------
Read the multi-horizon results.xlsx and produce:
  1) Per-(Imputation, FeatureSet, Target) line chart of Delta_R2_mean vs Horizon.
  2) A faceted summary: boxplots of Delta_R2_mean across metals for each FeatureSet at each Horizon.

Usage:
  python make_ts_baseline_multihorizon_figs.py \
    --input ../../outputs/imputation_order_sensitivity/timeseries_cv_multihorizon/results.xlsx \
    --outdir ../../outputs/imputation_order_sensitivity/figures

Notes:
- Matplotlib only (no seaborn) to keep dependencies light.
- English labels; relative paths; will create outdir if missing.
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Make multi-horizon comparison figures.")
    p.add_argument("--input", type=Path,
                   default=Path("../../outputs/imputation_order_sensitivity/timeseries_cv_multihorizon/results.xlsx"),
                   help="Path to multi-horizon results Excel file (results sheet).")
    p.add_argument("--sheet", type=str, default="results", help="Sheet name to read. Default: results")
    p.add_argument("--outdir", type=Path,
                   default=Path("../../outputs/imputation_order_sensitivity/figures"),
                   help="Directory to save figures.")
    return p.parse_args()


def _safe_str(x: str) -> str:
    return str(x).replace("/", "_").replace("\\", "_").replace(" ", "_").replace("%", "pct")


def per_group_lineplots(df: pd.DataFrame, outdir: Path) -> None:
    """For each (Imputation, FeatureSet, Target), plot Delta_R2_mean vs Horizon."""
    groups = df.groupby(["Imputation", "FeatureSet", "Target"], dropna=False)
    for (imp, fs, tgt), sub in groups:
        sub = sub.sort_values("Horizon")
        if sub["Delta_R2_mean"].isna().all():
            continue

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(sub["Horizon"], sub["Delta_R2_mean"], marker="o", linewidth=2)
        ax.axhline(0.0, linestyle="--", linewidth=1)
        ax.set_title(f"ΔR² vs Horizon\nImputation={imp}, FeatureSet={fs}, Target={tgt}")
        ax.set_xlabel("Horizon")
        ax.set_ylabel("Delta R² (BestML - BestNaive)")
        ax.grid(True, alpha=0.3)

        fname = outdir / f"deltaR2_{_safe_str(imp)}_{_safe_str(fs)}_{_safe_str(tgt)}.png"
        fig.tight_layout()
        fig.savefig(fname, dpi=200)
        plt.close(fig)


def faceted_boxplot(df: pd.DataFrame, outdir: Path) -> None:
    """A simple faceted summary: boxplots of Delta_R2_mean across targets per Horizon per FeatureSet."""
    fsets = sorted(df["FeatureSet"].dropna().unique().tolist())
    horizons = sorted(df["Horizon"].dropna().unique().tolist())
    if not fsets or not horizons:
        return

    # For each feature set, draw a boxplot across metals for every Horizon (x-axis)
    for fs in fsets:
        sub = df[df["FeatureSet"] == fs].copy()
        # build data: for each horizon, the list of Delta_R2_mean across targets and imputations
        data = []
        labels = []
        for h in horizons:
            vals = sub[sub["Horizon"] == h]["Delta_R2_mean"].dropna().values
            if len(vals) == 0:
                vals = np.array([np.nan])
            data.append(vals)
            labels.append(str(h))

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.boxplot(data, labels=labels, showmeans=True)
        ax.axhline(0.0, linestyle="--", linewidth=1)
        ax.set_title(f"ΔR² across targets by Horizon (FeatureSet={fs})")
        ax.set_xlabel("Horizon")
        ax.set_ylabel("Delta R² (BestML - BestNaive)")
        ax.grid(True, axis="y", alpha=0.3)

        fname = outdir / f"deltaR2_boxplot_{_safe_str(fs)}.png"
        fig.tight_layout()
        fig.savefig(fname, dpi=200)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input not found: {args.input}")

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(args.input, sheet_name=args.sheet)
    # Basic checks
    needed = {"Imputation", "FeatureSet", "Target", "Horizon", "Delta_R2_mean"}
    missing = needed.difference(df.columns)
    if missing:
        raise RuntimeError(f"Missing columns in results: {missing}")

    per_group_lineplots(df, outdir)
    faceted_boxplot(df, outdir)
    print(f"[OK] Saved figures to: {outdir}")


if __name__ == "__main__":
    main()
