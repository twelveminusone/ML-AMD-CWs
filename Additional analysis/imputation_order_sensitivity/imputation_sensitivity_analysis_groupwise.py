#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Imputation-Order Sensitivity — Groupwise (Single Split)
-------------------------------------------------------
Evaluate the setting where we FIRST split into train/test, THEN fit the
imputer on TRAIN ONLY, and ONLY transform TEST. On TEST, we simulate missing
for the target column to measure imputation accuracy against ground truth.

Notes:
- Use the INITIAL dataset that contains missing values (initial_dataset.xlsx).
- English-only prints/comments, relative paths, unified 'day' time column.
- Save 'results' + 'meta' sheets for transparency and reproducibility.

Example:
  python imputation_sensitivity_analysis_groupwise.py \
    --input ../../data/initial_dataset.xlsx \
    --output ../../outputs/imputation_order_sensitivity/groupwise/results.xlsx \
    --test_size 0.2 --seed 42 --methods all --save_details
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# project utils
from utils.io import load_numeric_df
from utils.imputers import get_imputers, make_iterative_imputer
from utils.meta import write_results_with_meta
from utils.common import set_seed


# ------------------------
# CLI
# ------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Groupwise single-split evaluation for imputation-order sensitivity."
    )
    p.add_argument(
        "--input",
        type=Path,
        default=Path("../../data/initial_dataset.xlsx"),
        help="Path to Excel dataset with missing values (initial_dataset.xlsx).",
    )
    p.add_argument(
        "--sheet",
        type=str,
        default=None,
        help="Excel sheet name; if None, use the first sheet.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("../../outputs/imputation_order_sensitivity/groupwise/results.xlsx"),
        help="Path to output Excel file (results + meta).",
    )
    p.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Test fraction for the single split. Default: 0.20",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. Default: 42",
    )
    p.add_argument(
        "--time_col",
        type=str,
        default="day",
        help="Time column name to unify/sort by (we standardize to 'day').",
    )
    p.add_argument(
        "--methods",
        type=str,
        default="all",
        help="Comma-separated methods to evaluate (e.g., 'RF,KNN,MICE'). Use 'all' for every method including MICE.",
    )
    p.add_argument(
        "--save_details",
        action="store_true",
        help="If set, also save per-split metrics as a CSV next to the Excel output.",
    )
    return p.parse_args()


# ------------------------
# Core evaluation (GROUPWISE single split)
# ------------------------
def eval_feature_groupwise_single_split(
    df: pd.DataFrame,
    target_col: str,
    estimators: Dict[str, object],
    test_size: float,
    seed: int,
) -> Tuple[List[dict], List[dict]]:
    """
    For one target column (that has missing values):
      - Keep rows where target is non-missing (to have ground truth).
      - Single train/test split.
      - Fit imputer on TRAIN ONLY (X+target with their original NaNs).
      - On TEST, set target to NaN (simulate-missing), transform, and compare
        imputed target vs ground truth.
    """
    mask = df[target_col].notna()
    X = df.loc[mask].drop(columns=[target_col]).to_numpy()
    y = df.loc[mask, target_col].to_numpy()

    n = len(y)
    summary_rows: List[dict] = []
    detail_rows: List[dict] = []

    # Basic safety checks
    if n < 3 or X.shape[1] == 0:
        summary_rows.append(
            {
                "Feature": target_col,
                "Method": "N/A",
                "R2": np.nan,
                "RMSE": np.nan,
                "MAE": np.nan,
                "n_train": 0,
                "n_test": 0,
                "note": "Insufficient data for single split.",
            }
        )
        return summary_rows, detail_rows

    # Ensure at least 1 sample per side
    test_size_eff = min(max(test_size, 1.0 / n), 1 - (1.0 / n))

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size_eff, random_state=seed, shuffle=True
    )

    n_tr, n_te = len(y_tr), len(y_te)

    # Methods: estimators dict + MICE baseline (estimator=None)
    full_methods = {**estimators, "MICE": None}

    for method_name, base_est in full_methods.items():
        try:
            # Build TRAIN array with original NaNs, TEST with target masked to NaN
            arr_tr = np.hstack([X_tr, y_tr.reshape(-1, 1)])
            arr_te = np.hstack([X_te, np.full((n_te, 1), np.nan, dtype=float)])

            imputer = make_iterative_imputer(base_est, seed=seed)
            imputer.fit(arr_tr)                # fit on TRAIN only
            arr_te_imp = imputer.transform(arr_te)
            y_pred = arr_te_imp[:, -1]

            r2 = float(r2_score(y_te, y_pred))
            rmse = float(mean_squared_error(y_te, y_pred, squared=False))
            mae = float(mean_absolute_error(y_te, y_pred))

            summary_rows.append(
                {
                    "Feature": target_col,
                    "Method": method_name,
                    "R2": r2,
                    "RMSE": rmse,
                    "MAE": mae,
                    "n_train": int(n_tr),
                    "n_test": int(n_te),
                    "note": "",
                }
            )
            detail_rows.append(
                {
                    "Feature": target_col,
                    "Method": method_name,
                    "Split": 1,
                    "R2": r2,
                    "RMSE": rmse,
                    "MAE": mae,
                    "n_train": int(n_tr),
                    "n_test": int(n_te),
                }
            )

        except Exception as e:
            summary_rows.append(
                {
                    "Feature": target_col,
                    "Method": method_name,
                    "R2": np.nan,
                    "RMSE": np.nan,
                    "MAE": np.nan,
                    "n_train": int(n_tr),
                    "n_test": int(n_te),
                    "note": f"Failed: {type(e).__name__}: {e}",
                }
            )
            detail_rows.append(
                {
                    "Feature": target_col,
                    "Method": method_name,
                    "Split": 1,
                    "R2": np.nan,
                    "RMSE": np.nan,
                    "MAE": np.nan,
                    "n_train": int(n_tr),
                    "n_test": int(n_te),
                    "note": f"Failed: {type(e).__name__}: {e}",
                }
            )

    return summary_rows, detail_rows


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    if not args.input.exists():
        raise FileNotFoundError(
            f"Input file not found: {args.input}. "
            "Place your dataset under ../../data or pass --input accordingly."
        )

    # Load and standardize (numeric-only, unify day/day_z to 'day')
    df = load_numeric_df(args.input, sheet=args.sheet, time_col=args.time_col)
    print(f"[INFO] Loaded dataset: {args.input} | shape={df.shape}")

    # Methods
    all_estimators = get_imputers(seed=args.seed)
    method_arg = args.methods.strip().lower()
    if method_arg == "all":
        selected_methods = list(all_estimators.keys())  # 'MICE' handled inside eval
    else:
        request = [m.strip() for m in args.methods.split(",") if m.strip()]
        unknown = [m for m in request if m not in all_estimators and m != "MICE"]
        if unknown:
            raise ValueError(f"Unknown method(s): {unknown}. Valid: {list(all_estimators.keys()) + ['MICE']}")
        selected_methods = [m for m in request if m != "MICE"]
    estimators_sub = {k: all_estimators[k] for k in selected_methods}

    # Columns with missing values
    missing_cols = [c for c in df.columns if df[c].isnull().sum() > 0]
    if not missing_cols:
        print("[WARN] No columns with missing values found in the INITIAL dataset.")
        # write empty results but keep meta for reproducibility
        empty_cols = ["Feature","Method","R2","RMSE","MAE","n_train","n_test","note"]
        write_results_with_meta(
            results_df=pd.DataFrame(columns=empty_cols),
            output_path=args.output,
            input_path=args.input,
            args=vars(args),
            df_shape=df.shape,
        )
        print(f"[OK] Saved empty results to: {args.output}")
        return

    print(f"[INFO] Features with missing values: {len(missing_cols)} columns")
    print(f"[INFO] Methods: {selected_methods + ['MICE']} | test_size={args.test_size} | seed={args.seed}")

    all_summary_rows: List[dict] = []
    all_detail_rows: List[dict] = []

    for col in missing_cols:
        summary_rows, detail_rows = eval_feature_groupwise_single_split(
            df=df,
            target_col=col,
            estimators=estimators_sub,
            test_size=args.test_size,
            seed=args.seed,
        )
        all_summary_rows.extend(summary_rows)
        all_detail_rows.extend(detail_rows)

    # Save
    results_df = pd.DataFrame(all_summary_rows)
    write_results_with_meta(
        results_df=results_df,
        output_path=args.output,
        input_path=args.input,
        args=vars(args),
        df_shape=df.shape,
    )
    print(f"[OK] Saved summary results to: {args.output}")

    if args.save_details:
        csv_path = args.output.with_suffix("").with_name(args.output.stem + "_split_metrics.csv")
        pd.DataFrame(all_detail_rows).to_csv(csv_path, index=False)
        print(f"[OK] Saved split-level metrics to: {csv_path}")


if __name__ == "__main__":
    run(parse_args())
