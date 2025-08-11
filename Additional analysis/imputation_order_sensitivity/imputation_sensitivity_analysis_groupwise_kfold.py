#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Imputation-Order Sensitivity (Groupwise + KFold)
------------------------------------------------
Evaluate how the order "split -> fit imputer on train -> transform test"
affects imputation accuracy, by K-fold simulate-missing on columns that
contain missing values.

Notes:
- Use the INITIAL dataset that still contains missing values (initial_dataset.xlsx).
- No absolute paths. English-only comments/prints.
- Unify time column as 'day' (auto-rename legacy 'day_z' if found).
- Save results + meta info for transparency.

Example (run from this folder):
  python imputation_sensitivity_analysis_groupwise_kfold.py \
    --input ../../data/initial_dataset.xlsx \
    --output ../../outputs/imputation_order_sensitivity/groupwise_kfold/results.xlsx \
    --n_splits 5 --seed 42 --methods all --save_fold_details
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# project utils
from utils.io import load_numeric_df
from utils.imputers import get_imputers, make_iterative_imputer
from utils.meta import write_results_with_meta
from utils.common import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Groupwise KFold evaluation for imputation-order sensitivity."
    )
    p.add_argument(
        "--input",
        type=Path,
        # ✅ default to initial dataset with missing values
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
        default=Path("../../outputs/imputation_order_sensitivity/groupwise_kfold/results.xlsx"),
        help="Path to output Excel file (results + meta).",
    )
    p.add_argument(
        "--n_splits",
        type=int,
        default=5,
        help="Number of KFold splits. Default: 5",
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
        "--save_fold_details",
        action="store_true",
        help="If set, also save per-fold metrics as a CSV next to the Excel output.",
    )
    return p.parse_args()


def eval_feature_groupwise_kfold(
    df: pd.DataFrame,
    target_col: str,
    estimators: Dict[str, object],
    kf: KFold,
    seed: int,
) -> Tuple[List[dict], List[dict]]:
    """
    For one target column (that has missing values), run KFold. For each fold:
      - split rows where target is non-missing
      - fit imputer on TRAIN rows only (X+target with their original NaNs)
      - set target in TEST rows to NaN (simulate-missing)
      - transform TEST rows and compare the imputed target against ground truth
    """
    mask = df[target_col].notna()
    X_full = df.loc[mask].drop(columns=[target_col])
    y_full = df.loc[mask, target_col].to_numpy()

    summary_rows: List[dict] = []
    fold_rows: List[dict] = []

    if X_full.shape[0] < kf.get_n_splits() or X_full.shape[1] == 0:
        summary_rows.append(
            {
                "Feature": target_col,
                "Method": "N/A",
                "R2_mean": np.nan,
                "R2_std": np.nan,
                "RMSE_mean": np.nan,
                "RMSE_std": np.nan,
                "MAE_mean": np.nan,
                "MAE_std": np.nan,
                "n_samples": int(X_full.shape[0]),
                "n_features": int(X_full.shape[1]),
                "note": "Insufficient data for KFold.",
            }
        )
        return summary_rows, fold_rows

    def _aggregate(method_name: str, r2, rmse, mae) -> dict:
        ddof = 1 if len(r2) > 1 else 0
        return {
            "Feature": target_col,
            "Method": method_name,
            "R2_mean": float(np.mean(r2)),
            "R2_std": float(np.std(r2, ddof=ddof)),
            "RMSE_mean": float(np.mean(rmse)),
            "RMSE_std": float(np.std(rmse, ddof=ddof)),
            "MAE_mean": float(np.mean(mae)),
            "MAE_std": float(np.std(mae, ddof=ddof)),
            "n_samples": int(X_full.shape[0]),
            "n_features": int(X_full.shape[1]),
            "note": "",
        }

    full_methods = {**estimators, "MICE": None}

    for method_name, base_est in full_methods.items():
        r2_list, rmse_list, mae_list = [], [], []

        for fold_id, (tr_idx, te_idx) in enumerate(kf.split(X_full), start=1):
            X_tr = X_full.iloc[tr_idx].to_numpy()
            y_tr = y_full[tr_idx]
            X_te = X_full.iloc[te_idx].to_numpy()
            y_te = y_full[te_idx]

            arr_tr = np.hstack([X_tr, y_tr.reshape(-1, 1)])
            arr_te = np.hstack([X_te, np.full((len(te_idx), 1), np.nan, dtype=float)])

            try:
                imputer = make_iterative_imputer(base_est, seed=seed)
                imputer.fit(arr_tr)               # fit on TRAIN only
                arr_te_imp = imputer.transform(arr_te)  # transform TEST
                y_pred = arr_te_imp[:, -1]
                y_true = y_te

                r2_list.append(r2_score(y_true, y_pred))
                rmse_list.append(mean_squared_error(y_true, y_pred, squared=False))
                mae_list.append(mean_absolute_error(y_true, y_pred))

                fold_rows.append(
                    {
                        "Feature": target_col,
                        "Method": method_name,
                        "Fold": fold_id,
                        "R2": float(r2_list[-1]),
                        "RMSE": float(rmse_list[-1]),
                        "MAE": float(mae_list[-1]),
                        "n_train": int(len(tr_idx)),
                        "n_test": int(len(te_idx)),
                    }
                )
            except Exception as e:
                r2_list.append(np.nan)
                rmse_list.append(np.nan)
                mae_list.append(np.nan)
                fold_rows.append(
                    {
                        "Feature": target_col,
                        "Method": method_name,
                        "Fold": fold_id,
                        "R2": np.nan,
                        "RMSE": np.nan,
                        "MAE": np.nan,
                        "n_train": int(len(tr_idx)),
                        "n_test": int(len(te_idx)),
                        "note": f"Failed: {type(e).__name__}: {e}",
                    }
                )

        summary_rows.append(_aggregate(method_name, r2_list, rmse_list, mae_list))

    return summary_rows, fold_rows


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    if not args.input.exists():
        raise FileNotFoundError(
            f"Input file not found: {args.input}. "
            "Place your dataset under ../../data or pass --input accordingly."
        )

    df = load_numeric_df(args.input, sheet=args.sheet, time_col=args.time_col)
    print(f"[INFO] Loaded dataset: {args.input} | shape={df.shape}")

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

    missing_cols = [c for c in df.columns if df[c].isnull().sum() > 0]
    if not missing_cols:
        print("[WARN] No columns with missing values found in the INITIAL dataset.")
        write_results_with_meta(
            results_df=pd.DataFrame(columns=[
                "Feature","Method","R2_mean","R2_std","RMSE_mean","RMSE_std","MAE_mean","MAE_std","n_samples","n_features","note"
            ]),
            output_path=args.output,
            input_path=args.input,
            args=vars(args),
            df_shape=df.shape,
        )
        print(f"[OK] Saved empty results to: {args.output}")
        return

    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    all_summary_rows: List[dict] = []
    all_fold_rows: List[dict] = []

    print(f"[INFO] Features with missing values: {len(missing_cols)} columns")
    print(f"[INFO] Methods: {selected_methods + ['MICE']} | n_splits={args.n_splits} | seed={args.seed}")

    estimators_sub = {k: all_estimators[k] for k in selected_methods}

    for col in missing_cols:
        summary_rows, fold_rows = eval_feature_groupwise_kfold(
            df=df,
            target_col=col,
            estimators=estimators_sub,
            kf=kf,
            seed=args.seed,
        )
        all_summary_rows.extend(summary_rows)
        all_fold_rows.extend(fold_rows)

    results_df = pd.DataFrame(all_summary_rows)
    write_results_with_meta(
        results_df=results_df,
        output_path=args.output,
        input_path=args.input,
        args=vars(args),
        df_shape=df.shape,
    )
    print(f"[OK] Saved summary results to: {args.output}")

    if args.save_fold_details:
        csv_path = args.output.with_suffix("").with_name(args.output.stem + "_fold_metrics.csv")
        pd.DataFrame(all_fold_rows).to_csv(csv_path, index=False)
        print(f"[OK] Saved fold-level metrics to: {csv_path}")


if __name__ == "__main__":
    run(parse_args())
