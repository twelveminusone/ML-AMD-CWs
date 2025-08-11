#!/usr/bin/env python
# -*- coding: utf-8 -*-

# --- add sibling 'utils/' (from imputation_order_sensitivity) to PYTHONPATH ---
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent               # this file lives in: Additional analysis/
SIBLING = HERE / "imputation_order_sensitivity"      # reuse ../Additional analysis/imputation_order_sensitivity/utils/
sys.path.insert(0, str(SIBLING))
# -----------------------------------------------------------------------------

"""
Leave-One-Group-Out (LOGO) Cross-Validation for CWs
---------------------------------------------------
Open-source version of the LOGO analysis. Evaluates generalization across wetlands
(groups) by training on all-but-one wetland and testing on the left-out wetland.

Principles:
- Relative paths only; English-only comments.
- Uses the fully-imputed dataset with group IDs: full_dataset_for_LOGO.xlsx
- Time column is unified to 'day' by utils.io (legacy 'day_z' auto-handled).
- Saves an Excel with two sheets: 'results' (long format) and 'meta'.

Example (run from 'Additional analysis/'):
  python LOGO_cv_analysis.py \
    --input ../data/full_dataset_for_LOGO.xlsx \
    --output ../outputs/logo_cv/results.xlsx \
    --group_col wetland_ID --seed 42 --grid --save_wide

Outputs:
- results.xlsx: long table with per-(Group, Target) R2/MAE and fold sizes
- (optional) wide CSVs (R2 and MAE matrices) when --save_wide is set
"""

from __future__ import annotations
import argparse
from typing import List, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

# Optional XGBoost
try:
    from xgboost import XGBRegressor  # type: ignore
except Exception:
    XGBRegressor = None

# project utils (reused)
from utils.io import load_numeric_df
from utils.meta import write_results_with_meta
from utils.common import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LOGO cross-validation across wetlands.")
    p.add_argument("--input", type=Path, default=Path("../data/full_dataset_for_LOGO.xlsx"),
                   help="Path to fully-imputed dataset including group IDs (full_dataset_for_LOGO.xlsx).")
    p.add_argument("--sheet", type=str, default=None, help="Excel sheet name; None = first sheet.")
    p.add_argument("--output", type=Path, default=Path("../outputs/logo_cv/results.xlsx"),
                   help="Path to Excel output (results + meta).")
    p.add_argument("--group_col", type=str, default="wetland_ID",
                   help="Group column used by LOGO (e.g., wetland_ID).")
    p.add_argument("--targets", type=str,
                   default="Zn (%),Cr (%),TFe (%),Al (%),Mn (%),Ni (%),Co (%)",
                   help="Comma-separated target columns (metal removal percentages).")
    p.add_argument("--exclude_cols", type=str, default="",
                   help="Extra columns to exclude from features (comma-separated).")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--grid", action="store_true",
                   help="Enable a small GridSearchCV (cv=3, scoring='r2') when XGB is available.")
    p.add_argument("--save_wide", action="store_true",
                   help="Also save wide R2/MAE matrices as CSV next to the Excel output.")
    return p.parse_args()


def make_model(seed: int, use_grid: bool):
    """Prefer XGB; fallback to RandomForest. Return (estimator, model_name, grid_used_flag, param_grid_if_any)."""
    if XGBRegressor is not None:
        name = "XGBRegressor"
        base = XGBRegressor(random_state=seed, n_estimators=200, n_jobs=-1, verbosity=0)
        if use_grid:
            grid = {
                "n_estimators": [100, 200, 300],
                "max_depth": [4, 6, 8],
                "learning_rate": [0.05, 0.1, 0.3],
            }
            est = GridSearchCV(base, grid, cv=3, scoring="r2", n_jobs=-1)
            return est, name, True, grid
        return base, name, False, None
    # fallback
    name = "RandomForestRegressor"
    base = RandomForestRegressor(random_state=seed, n_estimators=400, n_jobs=-1)
    return base, name, False, None


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    if not args.input.exists():
        raise FileNotFoundError(f"Input not found: {args.input}")

    # Load numeric-only DataFrame, standardize 'day' naming, drop all-NaN columns
    df = load_numeric_df(args.input, sheet=args.sheet, time_col="day")

    # If group_col was non-numeric and got dropped by load_numeric_df, reattach it from raw Excel.
    if args.group_col not in df.columns:
        raw = pd.read_excel(args.input, sheet_name=args.sheet)
        if args.group_col not in raw.columns:
            raise RuntimeError(f"Group column '{args.group_col}' not found in dataset.")
        # align by row order
        df[args.group_col] = raw[args.group_col].values

    print(f"[INFO] Loaded dataset: {args.input} | shape={df.shape}")

    # Parse targets, exclusions
    targets: List[str] = [t.strip() for t in args.targets.split(",") if t.strip()]
    extra_exclude: List[str] = [c.strip() for c in args.exclude_cols.split(",") if c.strip()]

    # Check critical columns
    missing_targets = [c for c in targets if c not in df.columns]
    if missing_targets:
        raise RuntimeError(f"Missing target columns: {missing_targets}")
    if args.group_col not in df.columns:
        raise RuntimeError(f"Group column '{args.group_col}' not found after loading.")

    # Features = all columns except group + targets + extra excludes
    exclude = set([args.group_col] + targets + extra_exclude)
    feature_cols = [c for c in df.columns if c not in exclude]
    if len(feature_cols) == 0:
        raise RuntimeError("No feature columns left after exclusions.")
    print(f"[INFO] Features: {len(feature_cols)} cols")

    # Build arrays; drop rows with any NaNs in features
    X_all = df[feature_cols].replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    common_index = X_all.index
    y_frame = df.loc[common_index, targets]
    groups = df.loc[common_index, args.group_col]

    # LOGO splitter
    logo = LeaveOneGroupOut()
    unique_groups = groups.unique().tolist()
    print(f"[INFO] Unique groups (n={len(unique_groups)}): {unique_groups}")

    rows = []
    est, model_name, grid_used, param_grid = make_model(seed=args.seed, use_grid=args.grid)

    # Iterate targets
    for tgt in targets:
        y_all = y_frame[tgt].values

        for train_idx, test_idx in logo.split(X_all.values, y_all, groups.values):
            grp_label = str(groups.iloc[test_idx].iloc[0])  # left-out group label

            # Scale inside each fold
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_all.values[train_idx])
            X_te = scaler.transform(X_all.values[test_idx])
            y_tr, y_te = y_all[train_idx], y_all[test_idx]

            # Fit model
            try:
                est.fit(X_tr, y_tr)
                used_est = est.best_estimator_ if isinstance(est, GridSearchCV) else est
            except Exception as e:
                rows.append({
                    "Group": grp_label, "Target": tgt, "R2": np.nan, "MAE": np.nan,
                    "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
                    "Model": model_name, "note": f"Fit failed: {type(e).__name__}: {e}",
                })
                continue

            # Predict & metrics
            y_pred = used_est.predict(X_te)
            r2 = r2_score(y_te, y_pred)
            mae = mean_absolute_error(y_te, y_pred)

            rows.append({
                "Group": grp_label, "Target": tgt, "R2": float(r2), "MAE": float(mae),
                "n_train": int(len(y_tr)), "n_test": int(len(y_te)), "Model": model_name,
            })

    # Long-format results
    results_df = pd.DataFrame(rows)

    # Save results + meta
    meta_args = {
        **vars(args),
        "model_used": model_name,
        "grid_used": bool(grid_used),
        "param_grid": param_grid,
        "n_groups": int(len(unique_groups)),
        "groups": [str(g) for g in unique_groups],
        "n_features": int(len(feature_cols)),
        "feature_cols": feature_cols,
    }
    write_results_with_meta(
        results_df=results_df,
        output_path=args.output,
        input_path=args.input,
        args=meta_args,
        df_shape=df.shape,
    )
    print(f"[OK] Saved results to: {args.output}")

    # Optional wide matrices
    if args.save_wide and not results_df.empty:
        out_base = args.output.with_suffix("")
        r2_wide = results_df.pivot_table(index="Group", columns="Target", values="R2", aggfunc="mean")
        mae_wide = results_df.pivot_table(index="Group", columns="Target", values="MAE", aggfunc="mean")
        r2_path = out_base.with_name(args.output.stem + "_R2_wide.csv")
        mae_path = out_base.with_name(args.output.stem + "_MAE_wide.csv")
        r2_wide.to_csv(r2_path)
        mae_wide.to_csv(mae_path)
        print(f"[OK] Saved wide R2 to:  {r2_path}")
        print(f"[OK] Saved wide MAE to: {mae_path}")


if __name__ == "__main__":
    run(parse_args())
