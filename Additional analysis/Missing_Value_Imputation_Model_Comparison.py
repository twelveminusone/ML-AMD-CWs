#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Missing-Value Imputation Model Comparison
-----------------------------------------
This script compares multiple estimators used inside IterativeImputer
(Sklearn) for imputing missing values, via a "simulate-missing K-fold"
evaluation on columns that already contain missing values.

Principles for open-source:
- No absolute local paths. Use relative defaults and CLI args.
- All code and comments are in English.
- Reproducibility via fixed random seeds and metadata logging.

Usage:
  python missing_value_filling.py \
      --input data/initial_dataset.xlsx \
      --output outputs/missing_value_filling/imputation_model_comparison.xlsx \
      --splits 5 --seed 42

Outputs:
- Excel file with two sheets:
  * 'results': mean/std of R2, RMSE, MAE per (feature, method)
  * 'meta': dataset hash, shapes, sklearn/pandas/python versions, args
"""

from __future__ import annotations
import argparse
from pathlib import Path
import hashlib
import sys
import platform
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Candidate estimators
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    BaggingRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare imputers (IterativeImputer + different estimators) via simulate-missing K-fold."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/initial_dataset.xlsx"),
        help="Path to the input dataset (.xlsx). Default: data/initial_dataset.xlsx",
    )
    parser.add_argument(
        "--sheet",
        type=str,
        default=None,
        help="Excel sheet name. If None, use the first sheet.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/missing_value_filling/imputation_model_comparison.xlsx"),
        help="Path to the output Excel file.",
    )
    parser.add_argument(
        "--splits",
        type=int,
        default=5,
        help="K-fold splits for evaluation. Default: 5",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. Default: 42",
    )
    return parser.parse_args()


def md5sum(path: Path) -> str:
    """Compute MD5 hash of a file (for reproducibility reporting)."""
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_numeric_dataframe(path: Path, sheet: str | None) -> pd.DataFrame:
    """Load Excel and coerce all columns to numeric where possible."""
    df = pd.read_excel(path, sheet_name=sheet)
    # Coerce to numeric; non-numeric becomes NaN (safe for imputation)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Drop columns that are fully NaN after coercion (no usable info)
    df = df.dropna(axis=1, how="all")
    return df


def get_estimators(seed: int) -> Dict[str, object]:
    """Return a dict of candidate base estimators for IterativeImputer."""
    return {
        "RF": RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1),
        "KNN": KNeighborsRegressor(n_neighbors=5),
        "GradientBoosting": GradientBoostingRegressor(random_state=seed),
        "SVM": SVR(),  # can be slower on large data; kept for completeness
        "MLP": MLPRegressor(random_state=seed, max_iter=500),
        "DecisionTree": DecisionTreeRegressor(random_state=seed),
        "AdaBoost": AdaBoostRegressor(random_state=seed),
        "Bagging": BaggingRegressor(random_state=seed),
        "HGBR": HistGradientBoostingRegressor(random_state=seed),
        # Historical alias retained for parity with earlier reports
        "RegressionTree": DecisionTreeRegressor(random_state=seed),
        # "MICE" handled separately via IterativeImputer(estimator=None)
    }


def simulate_missing_eval(
    X: pd.DataFrame,
    y: np.ndarray,
    estimator,
    kfold: KFold,
    seed: int,
) -> Tuple[List[float], List[float], List[float]]:
    """
    Simulate-missing evaluation for one target column using IterativeImputer
    with a given base estimator:
      - For each fold, set y[test_idx] to NaN, impute, and compare y_pred vs y_true.
    Returns lists of R2, RMSE, MAE for all folds.
    """
    r2_list, rmse_list, mae_list = [], [], []

    # Combine X and y into a single numeric array for IterativeImputer
    arr_all = np.hstack([X.to_numpy(), y.reshape(-1, 1)])

    for train_idx, test_idx in kfold.split(X):
        arr = arr_all.copy()
        # Simulate missingness only on the target column for test rows
        arr[test_idx, -1] = np.nan

        imputer = IterativeImputer(
            estimator=estimator,
            max_iter=10,
            random_state=seed,
            sample_posterior=False,
        )
        arr_imputed = imputer.fit_transform(arr)

        y_pred = arr_imputed[test_idx, -1]
        y_true = y[test_idx]

        r2_list.append(r2_score(y_true, y_pred))
        rmse_list.append(mean_squared_error(y_true, y_pred, squared=False))
        mae_list.append(mean_absolute_error(y_true, y_pred))

    return r2_list, rmse_list, mae_list


def evaluate_all_features(
    df: pd.DataFrame, splits: int, seed: int
) -> pd.DataFrame:
    """
    Evaluate all columns that contain missing values.
    For each such feature, run K-fold simulate-missing with each estimator and MICE baseline.
    """
    rng = np.random.RandomState(seed)
    kf = KFold(n_splits=splits, shuffle=True, random_state=seed)

    # Columns with at least one missing value
    missing_features = [c for c in df.columns if df[c].isnull().sum() > 0]

    est_dict = get_estimators(seed)
    results = []

    for feat in missing_features:
        # Use rows where the target feature is non-missing (we have a "ground truth")
        mask = df[feat].notna()
        X_full = df.loc[mask].drop(columns=[feat])
        y_full = df.loc[mask, feat].to_numpy()

        # If there are no predictors after dropping, skip
        if X_full.shape[1] == 0 or X_full.shape[0] < splits:
            results.append(
                {
                    "Feature": feat,
                    "Method": "N/A",
                    "R2_mean": np.nan,
                    "R2_std": np.nan,
                    "RMSE_mean": np.nan,
                    "RMSE_std": np.nan,
                    "MAE_mean": np.nan,
                    "MAE_std": np.nan,
                    "n_samples": int(X_full.shape[0]),
                    "n_features": int(X_full.shape[1]),
                    "note": "Insufficient data after filtering.",
                }
            )
            continue

        # Evaluate each estimator wrapped by IterativeImputer
        for method, base_est in est_dict.items():
            try:
                r2_list, rmse_list, mae_list = simulate_missing_eval(
                    X_full, y_full, base_est, kf, seed
                )
                results.append(
                    {
                        "Feature": feat,
                        "Method": method,
                        "R2_mean": float(np.mean(r2_list)),
                        "R2_std": float(np.std(r2_list, ddof=1)),
                        "RMSE_mean": float(np.mean(rmse_list)),
                        "RMSE_std": float(np.std(rmse_list, ddof=1)),
                        "MAE_mean": float(np.mean(mae_list)),
                        "MAE_std": float(np.std(mae_list, ddof=1)),
                        "n_samples": int(X_full.shape[0]),
                        "n_features": int(X_full.shape[1]),
                        "note": "",
                    }
                )
            except Exception as e:
                results.append(
                    {
                        "Feature": feat,
                        "Method": method,
                        "R2_mean": np.nan,
                        "R2_std": np.nan,
                        "RMSE_mean": np.nan,
                        "RMSE_std": np.nan,
                        "MAE_mean": np.nan,
                        "MAE_std": np.nan,
                        "n_samples": int(X_full.shape[0]),
                        "n_features": int(X_full.shape[1]),
                        "note": f"Failed: {type(e).__name__}: {e}",
                    }
                )

        # Separate MICE baseline (default estimator inside IterativeImputer)
        try:
            r2_list, rmse_list, mae_list = simulate_missing_eval(
                X_full, y_full, estimator=None, kf=kf, seed=seed
            )
            results.append(
                {
                    "Feature": feat,
                    "Method": "MICE",
                    "R2_mean": float(np.mean(r2_list)),
                    "R2_std": float(np.std(r2_list, ddof=1)),
                    "RMSE_mean": float(np.mean(rmse_list)),
                    "RMSE_std": float(np.std(rmse_list, ddof=1)),
                    "MAE_mean": float(np.mean(mae_list)),
                    "MAE_std": float(np.std(mae_list, ddof=1)),
                    "n_samples": int(X_full.shape[0]),
                    "n_features": int(X_full.shape[1]),
                    "note": "",
                }
            )
        except Exception as e:
            results.append(
                {
                    "Feature": feat,
                    "Method": "MICE",
                    "R2_mean": np.nan,
                    "R2_std": np.nan,
                    "RMSE_mean": np.nan,
                    "RMSE_std": np.nan,
                    "MAE_mean": np.nan,
                    "MAE_std": np.nan,
                    "n_samples": int(X_full.shape[0]),
                    "n_features": int(X_full.shape[1]),
                    "note": f"Failed: {type(e).__name__}: {e}",
                }
            )

    return pd.DataFrame(results)


def save_with_metadata(
    df_results: pd.DataFrame,
    output_path: Path,
    input_path: Path,
    args: argparse.Namespace,
    df_shape: Tuple[int, int],
):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build metadata table
    try:
        dataset_hash = md5sum(input_path)
    except Exception:
        dataset_hash = "unavailable"

    meta = {
        "input_path": str(input_path),
        "dataset_md5": dataset_hash,
        "n_rows": df_shape[0],
        "n_cols": df_shape[1],
        "splits": args.splits,
        "seed": args.seed,
        "python": sys.version.replace("\n", " "),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "sklearn": __import__("sklearn").__version__,
        "platform": platform.platform(),
    }
    df_meta = pd.DataFrame([meta])

    # Write two sheets
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df_results.to_excel(writer, index=False, sheet_name="results")
        df_meta.to_excel(writer, index=False, sheet_name="meta")


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Input file not found: {args.input}. "
            f"Place your dataset under ./data and pass --input if needed."
        )

    df = load_numeric_dataframe(args.input, args.sheet)
    print(f"[INFO] Loaded dataset: {args.input} | shape={df.shape}")

    df_results = evaluate_all_features(df, splits=args.splits, seed=args.seed)
    save_with_metadata(
        df_results=df_results,
        output_path=args.output,
        input_path=args.input,
        args=args,
        df_shape=df.shape,
    )
    print(f"[OK] Saved comparison to: {args.output}")


if __name__ == "__main__":
    main()
