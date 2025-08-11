#!/usr/bin/env python
# -*- coding: utf-8 -*-

# --- add sibling 'utils/' (from imputation_order_sensitivity) to PYTHONPATH ---
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
SIBLING = HERE / "imputation_order_sensitivity"   # <- script lives in Additional analysis/
sys.path.insert(0, str(SIBLING))  # so: from utils.io import ... works
# -----------------------------------------------------------------------------

"""
Comparative Analysis of Feature Importance (Open-source version)
----------------------------------------------------------------
Three methods on a fixed train/test split per target:
  (1) Permutation Importance (sklearn)
  (2) LIME (mean absolute weights across n samples) [optional dependency]
  (3) Occlusion/Ablation at inference time: set ONE feature to zero on X_test,
      compute ΔR² = R²_base - R²_drop (matches the local script behavior)

Open-source principles:
- Relative paths; English-only comments; unified time column 'day' (legacy 'day_z' auto-renamed).
- Input uses the fully-imputed dataset: full_dataset.xlsx
- Outputs: Excel with 'results' + 'meta'. Optional figures saved under outdir/figures.

Example (run from this folder: Additional analysis/):
  python feature_importance_comparison.py \
    --input ../data/full_dataset.xlsx \
    --output ../outputs/feature_importance_comparison/results.xlsx \
    --features i_COD,i_pH,i_acidity,i_EC,day,height \
    --targets "TFe (%),Zn (%),Al (%),Mn (%),Ni (%),Co (%),Cr (%)" \
    --test_size 0.2 --seed 42 --perm_repeats 10 --lime_n 5 --grid --fig
"""

from __future__ import annotations
import argparse
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor

# optional XGBoost
try:
    from xgboost import XGBRegressor  # type: ignore
except Exception:
    XGBRegressor = None

# optional LIME
try:
    from lime.lime_tabular import LimeTabularExplainer  # type: ignore
except Exception:
    LimeTabularExplainer = None

# project utils (reused from sibling package)
from utils.io import load_numeric_df
from utils.meta import write_results_with_meta
from utils.common import set_seed


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Comparative feature-importance analysis (PI, LIME, Occlusion).")
    p.add_argument("--input", type=Path, default=Path("../data/full_dataset.xlsx"),
                   help="Path to fully-imputed dataset (full_dataset.xlsx).")
    p.add_argument("--sheet", type=str, default=None, help="Excel sheet; None = first sheet.")
    p.add_argument("--output", type=Path, default=Path("../outputs/feature_importance_comparison/results.xlsx"),
                   help="Path to Excel output (results + meta).")
    p.add_argument("--time_col", type=str, default="day", help="Time column; legacy 'day_z' auto-renamed.")
    p.add_argument("--features", type=str, default="i_COD,i_pH,i_acidity,i_EC,day,height",
                   help="Comma-separated features (default = Feature Set C).")
    p.add_argument("--targets", type=str,
                   default="TFe (%),Zn (%),Al (%),Mn (%),Ni (%),Co (%),Cr (%)",
                   help="Comma-separated target columns (metal % removal).")
    p.add_argument("--test_size", type=float, default=0.2, help="Test fraction. Default: 0.2")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--grid", action="store_true",
                   help="Enable small GridSearchCV (cv=3, scoring='r2') for XGB when available.")
    p.add_argument("--perm_repeats", type=int, default=10, help="n_repeats for permutation importance.")
    p.add_argument("--lime_n", type=int, default=5,
                   help="Number of test samples to explain for LIME (mean abs weights).")
    p.add_argument("--methods", type=str, default="PI,LIME,Occlusion",
                   help="Subset of methods to run, from: PI,LIME,Occlusion.")
    p.add_argument("--fig", action="store_true", help="Also save global bar plots for each method.")
    return p.parse_args()


# -----------------------------
# Models
# -----------------------------
def make_estimator(seed: int, use_grid: bool):
    """Prefer XGB; fallback to RF. Return (estimator, model_name, used_grid_flag)."""
    if XGBRegressor is not None:
        name = "XGBRegressor"
        base = XGBRegressor(random_state=seed, n_estimators=200)
        if use_grid:
            grid = {"n_estimators": [100, 200], "max_depth": [3, 5, 7], "learning_rate": [0.05, 0.1]}
            est = GridSearchCV(base, grid, cv=3, scoring="r2", n_jobs=-1)
            return est, name, True
        return base, name, False
    name = "RandomForestRegressor"
    return RandomForestRegressor(random_state=seed, n_estimators=400), name, False


# -----------------------------
# Core
# -----------------------------
def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    # load + sanitize
    if not args.input.exists():
        raise FileNotFoundError(f"Input not found: {args.input}")
    df = load_numeric_df(args.input, sheet=args.sheet, time_col=args.time_col)

    feats: List[str] = [f.strip() for f in args.features.split(",") if f.strip()]
    tcols: List[str] = [t.strip() for t in args.targets.split(",") if t.strip()]
    methods = [m.strip().upper() for m in args.methods.split(",") if m.strip()]
    valid_methods = {"PI", "LIME", "OCCLUSION"}
    for m in methods:
        if m not in valid_methods:
            raise ValueError(f"Unknown method '{m}'. Choose from: PI,LIME,Occlusion")

    # Check columns
    miss_f = [c for c in feats if c not in df.columns]
    miss_t = [c for c in tcols if c not in df.columns]
    if miss_f or miss_t:
        raise RuntimeError(f"Missing columns — features: {miss_f}, targets: {miss_t}")

    # Keep only used columns and drop residual NaNs just in case
    used = list(dict.fromkeys(feats + tcols))  # preserve order
    dff = df[used].replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any").reset_index(drop=True)
    print(f"[INFO] Data ready: shape={dff.shape}")

    rows: List[dict] = []
    baseline_r2: Dict[str, float] = {}
    model_name_used, grid_used = None, False

    for tgt in tcols:
        X = dff[feats].copy()
        y = dff[tgt].copy()

        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=args.test_size,
                                              random_state=args.seed, shuffle=True)

        est, model_name, used_grid = make_estimator(args.seed, args.grid)
        model_name_used = model_name
        grid_used = grid_used or used_grid

        # fit
        est.fit(Xtr, ytr)
        est_used = est.best_estimator_ if isinstance(est, GridSearchCV) else est

        # baseline R2
        yhat_base = est_used.predict(Xte)
        r2_base = r2_score(yte, yhat_base)
        baseline_r2[tgt] = float(r2_base)

        # (1) Permutation Importance
        if "PI" in methods:
            pi = permutation_importance(est_used, Xte, yte, n_repeats=args.perm_repeats,
                                        random_state=args.seed, n_jobs=-1)
            for f, imp in zip(feats, pi.importances_mean):
                rows.append({"Method": "PI", "Target": tgt, "Feature": f, "Importance": float(imp)})

        # (2) LIME
        if "LIME" in methods:
            if LimeTabularExplainer is None:
                print("[WARN] LIME not installed, skipping.")
            else:
                explainer = LimeTabularExplainer(
                    Xtr.values, feature_names=feats, class_names=[tgt],
                    mode="regression", discretize_continuous=True, random_state=args.seed
                )
                weights = np.zeros(len(feats), dtype=float)
                n_take = int(min(args.lime_n, len(Xte)))
                for i in range(n_take):
                    exp = explainer.explain_instance(Xte.values[i], est_used.predict, num_features=len(feats))
                    for feat_label, val in exp.as_list():
                        for jf, f in enumerate(feats):
                            if f in feat_label:
                                weights[jf] += abs(val)
                                break
                if n_take > 0:
                    weights = weights / n_take
                for f, w in zip(feats, weights):
                    rows.append({"Method": "LIME", "Target": tgt, "Feature": f, "Importance": float(w)})

        # (3) Occlusion (set feature to 0 on test, like the original script)
        if "OCCLUSION" in methods:
            for f in feats:
                X_occ = Xte.copy()
                X_occ[f] = 0.0  # keep same behavior as the provided local script
                yhat_occ = est_used.predict(X_occ)
                r2_occ = r2_score(yte, yhat_occ)
                delta = r2_base - r2_occ  # ΔR² >=0 implies importance
                rows.append({"Method": "Occlusion", "Target": tgt, "Feature": f, "Importance": float(delta)})

    # Build results DataFrame
    results_df = pd.DataFrame(rows)

    # Add global mean rows (across targets) for quick comparison
    if not results_df.empty:
        g = results_df.groupby(["Method", "Feature"], as_index=False)["Importance"].mean()
        g["Target"] = "ALL (mean)"
        results_df = pd.concat([results_df, g[["Method", "Target", "Feature", "Importance"]]], ignore_index=True)

    # Save results + meta
    write_results_with_meta(
        results_df=results_df,
        output_path=args.output,
        input_path=args.input,
        args={
            **vars(args),
            "model_used": model_name_used or "N/A",
            "grid_used": bool(grid_used),
            "baseline_R2_per_target": baseline_r2,
            "lime_available": bool(LimeTabularExplainer is not None),
        },
        df_shape=dff.shape,
    )
    print(f"[OK] Saved results to: {args.output}")

    # Figures: global bars per method
    if args.fig and not results_df.empty:
        figdir = args.output.parent / "figures"
        figdir.mkdir(parents=True, exist_ok=True)

        for method in sorted(results_df["Method"].dropna().unique()):
            sub = results_df[(results_df["Method"] == method) & (results_df["Target"] == "ALL (mean)")]
            if sub.empty:
                continue
            sub = sub.sort_values("Importance", ascending=False)
            plt.figure(figsize=(7, 4))
            plt.bar(sub["Feature"], sub["Importance"])
            plt.xticks(rotation=30, ha="right")
            plt.ylabel("Global importance (mean across targets)")
            plt.title(f"Global Feature Importance — {method}")
            plt.tight_layout()
            outp = figdir / f"global_{method.lower().replace(' ', '_')}_importance.png"
            plt.savefig(outp, dpi=300)
            plt.close()
            print(f"[OK] Saved figure: {outp}")


if __name__ == "__main__":
    run(parse_args())
