#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Imputation-Order Sensitivity — Groupwise TimeSeries CV (Forward-Chaining)
------------------------------------------------------------------------
This script evaluates how imputers affect downstream model performance
under *temporal* splits. For each fold, we:
  1) sort by 'day' (legacy 'day_z' will be auto-renamed);
  2) fit the imputer on the TRAIN window only;
  3) transform the TEST window;
  4) compute removal efficiency y = (1 - o_metal / i_metal) * 100 on each window;
  5) train/evaluate ML models per metal target.

Open-source principles:
- Use the *initial* dataset with missing values (initial_dataset.xlsx).
- Relative paths only; English-only comments/prints.
- Unified time column: 'day'.
- Save transparent outputs: 'results' + 'meta' sheets; optional fold CSV.

Example:
  python imputation_sensitivity_analysis_groupwise_timeseries_cv.py \
    --input ../../data/initial_dataset.xlsx \
    --output ../../outputs/imputation_order_sensitivity/timeseries_cv/results.xlsx \
    --n_splits 3 --seed 42 --methods all --feature_sets A,B,C \
    --models XGB,RF,KNN,SVR,ANN --grid --save_fold_details
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Optional: xgboost & missforest may not be installed in all environments
try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None

from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer

try:
    from missforest import MissForest
except Exception:
    MissForest = None

# --- project utils (same folder's utils/)
from utils.io import load_numeric_df
from utils.meta import write_results_with_meta
from utils.common import set_seed


# ------------------------------------------------
# CLI
# ------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Groupwise TimeSeries CV evaluation for imputation-order sensitivity."
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
        default=Path("../../outputs/imputation_order_sensitivity/timeseries_cv/results.xlsx"),
        help="Path to output Excel file (results + meta).",
    )
    p.add_argument(
        "--n_splits",
        type=int,
        default=3,
        help="Number of forward-chaining (TimeSeriesSplit) folds. Default: 3",
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
        help="Time column to sort by. 'day_z' will be auto-renamed to this.",
    )
    p.add_argument(
        "--methods",
        type=str,
        default="all",
        help="Imputation methods: comma-separated among RF,KNN,MICE,MissForest; or 'all'.",
    )
    p.add_argument(
        "--feature_sets",
        type=str,
        default="A,B,C",
        help="Which feature sets to evaluate: subset of 'A,B,C'. Default: A,B,C.",
    )
    p.add_argument(
        "--models",
        type=str,
        default="XGB,RF,KNN,SVR,ANN",
        help="Which models to evaluate. Default: XGB,RF,KNN,SVR,ANN.",
    )
    p.add_argument(
        "--grid",
        action="store_true",
        help="If set, run a small GridSearchCV (cv=3) and use the best estimator."
    )
    p.add_argument(
        "--save_fold_details",
        action="store_true",
        help="If set, also save per-fold metrics as a CSV next to the Excel output.",
    )
    return p.parse_args()


# ------------------------------------------------
# Helpers
# ------------------------------------------------
def _normalize_columns_for_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove units in parentheses, drop obviously irrelevant columns like TDS/record,
    and trim spaces so feature names like 'i_COD (mg/L)' become 'i_COD'.
    """
    df = df.copy()
    df.columns = df.columns.str.replace(r"\(.*?\)", "", regex=True).str.strip()
    for col in ("record",):
        if col in df.columns:
            df = df.drop(columns=[col])
    for col in ("i_TDS", "o_TDS"):
        if col in df.columns:
            df = df.drop(columns=[col])
    return df


def _get_feature_sets(df_cols: List[str]) -> Dict[str, List[str]]:
    """
    Construct A/B/C feature sets following the local script convention:
      A: all non-target columns (i.e., drop targets like 'TFe (%)', etc.)
      B: a focused set from inflow/outflow + EC/SO4
      C: the 6-variable practical set
    """
    metals = ["TFe", "Zn", "Al", "Mn", "Ni", "Co", "Cr"]
    target_names = [f"{m} (%)" for m in metals]
    # A = all columns except explicit percentage targets
    A = [c for c in df_cols if c not in target_names]
    # B and C as per prior local convention (units already stripped)
    B = [c for c in ["i_COD", "i_pH", "o_EC", "o_Mn", "o_TFe", "o_SO42-"] if c in df_cols]
    C = [c for c in ["i_pH", "i_COD", "day", "i_EC", "height", "i_acidity"] if c in df_cols]
    return {"A": A, "B": B, "C": C}


def _build_imputer(method_key: str, seed: int):
    """
    Map method key to an actual imputer.
      - 'RF'   -> IterativeImputer with RandomForestRegressor (max_iter=1)
      - 'KNN'  -> KNNImputer
      - 'MICE' -> IterativeImputer (estimator=None)
      - 'MissForest' -> MissForest (if available)
    """
    mk = method_key.lower()
    if mk == "rf":
        est = RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1)
        return IterativeImputer(estimator=est, initial_strategy="mean", max_iter=1, random_state=seed)
    if mk == "knn":
        return KNNImputer(n_neighbors=5)
    if mk == "mice":
        return IterativeImputer(initial_strategy="mean", max_iter=10, random_state=seed)
    if mk == "missforest":
        if MissForest is None:
            raise RuntimeError("MissForest is not installed; please `pip install missforest` or remove it from --methods.")
        return MissForest(random_state=seed)
    raise ValueError(f"Unknown imputation method: {method_key}")


def _get_models(seed: int):
    """
    Return {name: (ctor, param_grid)}; when --grid is enabled, use param_grid via GridSearchCV(cv=3, scoring='r2').
    """
    models = {}
    if XGBRegressor is not None:
        models["XGB"] = (
            lambda: XGBRegressor(random_state=seed, n_estimators=200),
            {"n_estimators": [200, 400], "max_depth": [4, 6], "learning_rate": [0.05, 0.1]},
        )
    # Always available sklearn models
    models["RF"] = (
        lambda: RandomForestRegressor(random_state=seed),
        {"n_estimators": [200, 400], "max_depth": [None, 6, 10]},
    )
    models["KNN"] = (
        lambda: KNeighborsRegressor(),
        {"n_neighbors": [3, 5, 7]},
    )
    models["SVR"] = (
        lambda: SVR(),
        {"C": [1, 10], "gamma": ["scale"], "epsilon": [0.01, 0.1]},
    )
    models["ANN"] = (
        lambda: MLPRegressor(random_state=seed, max_iter=500),
        {"hidden_layer_sizes": [(50,), (100,)], "alpha": [1e-4, 1e-3], "activation": ["relu"], "solver": ["adam"]},
    )
    return models


# ------------------------------------------------
# Core evaluation
# ------------------------------------------------
def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    if not args.input.exists():
        raise FileNotFoundError(
            f"Input file not found: {args.input}. "
            "Place your dataset under ../../data or pass --input accordingly."
        )

    # Load & standardize numeric; unify day_z -> day; sort by day
    df_raw = load_numeric_df(args.input, sheet=args.sheet, time_col=args.time_col)
    if args.time_col not in df_raw.columns:
        raise RuntimeError(f"Time column '{args.time_col}' not found after loading. "
                           f"If your data had 'day_z', it should be auto-renamed to '{args.time_col}'.")
    df = _normalize_columns_for_features(df_raw)
    df = df.sort_values(args.time_col).reset_index(drop=True)

    # Metals and target names for removal efficiency
    metals = ["TFe", "Zn", "Al", "Mn", "Ni", "Co", "Cr"]
    target_names = [f"{m} (%)" for m in metals]

    # Build feature sets
    fsets_all = _get_feature_sets(df.columns.tolist())
    requested_sets = [s.strip().upper() for s in args.feature_sets.split(",") if s.strip()]
    for s in requested_sets:
        if s not in fsets_all:
            raise ValueError(f"Unknown feature set '{s}'. Valid options: {list(fsets_all.keys())}")
    feature_sets = {k: fsets_all[k] for k in requested_sets}

    # Methods
    all_methods = ["RF", "KNN", "MICE"] + (["MissForest"] if MissForest is not None else [])
    if args.methods.strip().lower() == "all":
        selected_methods = all_methods
    else:
        selected_methods = [m.strip() for m in args.methods.split(",") if m.strip()]
        for m in selected_methods:
            if m not in all_methods:
                raise ValueError(f"Unknown method '{m}'. Valid: {all_methods}")

    # Models
    all_models = _get_models(seed=args.seed)
    if args.models.strip().lower() == "all":
        selected_models = list(all_models.keys())
    else:
        selected_models = [m.strip() for m in args.models.split(",") if m.strip()]
        unknown = [m for m in selected_models if m not in all_models]
        if unknown:
            raise ValueError(f"Unknown model(s): {unknown}. Valid: {list(all_models.keys())}")

    print(f"[INFO] Loaded dataset: {args.input} | shape={df.shape}")
    print(f"[INFO] TimeSeriesSplit n_splits={args.n_splits} | seed={args.seed}")
    print(f"[INFO] Imputation methods: {selected_methods}")
    print(f"[INFO] Feature sets: {list(feature_sets.keys())}")
    print(f"[INFO] Models: {selected_models} | grid={'ON' if args.grid else 'OFF'}")

    tscv = TimeSeriesSplit(n_splits=args.n_splits)

    # Collect results
    rows_summary: List[dict] = []
    rows_fold: List[dict] = []

    for method in selected_methods:
        # For each feature set (A/B/C)
        for fs_name, fs_cols in feature_sets.items():
            # Extract X (features) in sorted order
            X_all = df[fs_cols]
            # Iterate each metal target
            for target in target_names:
                metal = target.split()[0]  # "TFe (%)" -> "TFe"
                ci_col = f"i_{metal}"
                co_col = f"o_{metal}"
                if ci_col not in df.columns or co_col not in df.columns:
                    # skip if the inflow/outflow columns are not present
                    continue

                # Per-fold metrics
                r2_list, rmse_list, mae_list = [], [], []

                fold_id = 0
                for train_idx, test_idx in tscv.split(X_all):
                    fold_id += 1
                    X_train_raw = X_all.iloc[train_idx].reset_index(drop=True)
                    X_test_raw = X_all.iloc[test_idx].reset_index(drop=True)

                    # Fit imputer on TRAIN only; transform TEST
                    imputer = _build_imputer(method, seed=args.seed)
                    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train_raw), columns=X_train_raw.columns)
                    X_test_imp = pd.DataFrame(imputer.transform(X_test_raw), columns=X_test_raw.columns)

                    # Compute y on the corresponding windows
                    df_train_full = df.iloc[train_idx].reset_index(drop=True)
                    df_test_full = df.iloc[test_idx].reset_index(drop=True)

                    ci_train = df_train_full[ci_col]
                    co_train = df_train_full[co_col]
                    ci_test = df_test_full[ci_col]
                    co_test = df_test_full[co_col]

                    y_train = (1.0 - (co_train / ci_train)) * 100.0
                    y_test = (1.0 - (co_test / ci_test)) * 100.0
                    # avoid division issues
                    y_train[ci_train == 0] = np.nan
                    y_test[ci_test == 0] = np.nan

                    # Drop NaNs in y
                    mask_tr = ~y_train.isna()
                    mask_te = ~y_test.isna()
                    X_train = X_train_imp.loc[mask_tr]
                    y_train = y_train.loc[mask_tr]
                    X_test = X_test_imp.loc[mask_te]
                    y_test = y_test.loc[mask_te]

                    # Require minimum sizes (as in your local script)
                    if len(y_train) < 10 or len(y_test) < 5:
                        # record but skip modeling for this fold
                        rows_fold.append({
                            "Imputation": method,
                            "FeatureSet": fs_name,
                            "Model": "N/A",
                            "Target": target,
                            "Fold": fold_id,
                            "R2": np.nan, "RMSE": np.nan, "MAE": np.nan,
                            "n_train": int(len(y_train)), "n_test": int(len(y_test)),
                            "note": "Insufficient samples after filtering."
                        })
                        continue

                    # Scale X and y
                    scaler_X = StandardScaler().fit(X_train)
                    scaler_y = StandardScaler().fit(y_train.values.reshape(-1, 1))
                    X_tr_s = scaler_X.transform(X_train)
                    X_te_s = scaler_X.transform(X_test)
                    y_tr_s = scaler_y.transform(y_train.values.reshape(-1, 1)).ravel()

                    # Evaluate each model; pick the *best R2* on this fold
                    best_fold_r2 = -np.inf
                    best_fold_rmse = np.inf
                    best_fold_mae = np.inf
                    best_model_name = None

                    for mdl_name in selected_models:
                        ctor, grid = all_models[mdl_name]
                        try:
                            if args.grid:
                                model = GridSearchCV(ctor(), grid, cv=3, scoring="r2", n_jobs=-1)
                                model.fit(X_tr_s, y_tr_s)
                                est = model.best_estimator_
                            else:
                                est = ctor()
                                est.fit(X_tr_s, y_tr_s)

                            y_pred_s = est.predict(X_te_s)
                            y_pred = scaler_y.inverse_transform(np.asarray(y_pred_s).reshape(-1, 1)).ravel()

                            r2 = r2_score(y_test.values, y_pred)
                            rmse = mean_squared_error(y_test.values, y_pred, squared=False)
                            mae = mean_absolute_error(y_test.values, y_pred)

                            # keep the best model on this fold
                            if r2 > best_fold_r2:
                                best_fold_r2 = r2
                                best_fold_rmse = rmse
                                best_fold_mae = mae
                                best_model_name = mdl_name

                        except Exception as e:
                            # record failure of this model but continue others
                            rows_fold.append({
                                "Imputation": method, "FeatureSet": fs_name,
                                "Model": mdl_name, "Target": target, "Fold": fold_id,
                                "R2": np.nan, "RMSE": np.nan, "MAE": np.nan,
                                "n_train": int(len(y_train)), "n_test": int(len(y_test)),
                                "note": f"Failed: {type(e).__name__}: {e}",
                            })

                    if best_model_name is not None:
                        r2_list.append(best_fold_r2)
                        rmse_list.append(best_fold_rmse)
                        mae_list.append(best_fold_mae)
                        rows_fold.append({
                            "Imputation": method, "FeatureSet": fs_name,
                            "Model": best_model_name, "Target": target, "Fold": fold_id,
                            "R2": best_fold_r2, "RMSE": best_fold_rmse, "MAE": best_fold_mae,
                            "n_train": int(len(y_train)), "n_test": int(len(y_test)),
                        })
                    else:
                        rows_fold.append({
                            "Imputation": method, "FeatureSet": fs_name,
                            "Model": "N/A", "Target": target, "Fold": fold_id,
                            "R2": np.nan, "RMSE": np.nan, "MAE": np.nan,
                            "n_train": int(len(y_train)), "n_test": int(len(y_test)),
                            "note": "No model succeeded on this fold.",
                        })

                # Aggregate across folds for this (method, fs, target)
                if len(r2_list) > 0:
                    ddof = 1 if len(r2_list) > 1 else 0
                    rows_summary.append({
                        "Imputation": method,
                        "FeatureSet": fs_name,
                        "Model": "BestPerFold",  # per fold best model
                        "Target": target,
                        "R2_mean": float(np.mean(r2_list)),
                        "R2_std": float(np.std(r2_list, ddof=ddof)),
                        "RMSE_mean": float(np.mean(rmse_list)),
                        "RMSE_std": float(np.std(rmse_list, ddof=ddof)),
                        "MAE_mean": float(np.mean(mae_list)),
                        "MAE_std": float(np.std(mae_list, ddof=ddof)),
                        "n_folds": int(len(r2_list)),
                    })
                else:
                    rows_summary.append({
                        "Imputation": method,
                        "FeatureSet": fs_name,
                        "Model": "BestPerFold",
                        "Target": target,
                        "R2_mean": np.nan, "R2_std": np.nan,
                        "RMSE_mean": np.nan, "RMSE_std": np.nan,
                        "MAE_mean": np.nan, "MAE_std": np.nan,
                        "n_folds": 0,
                        "note": "No valid folds.",
                    })

    # Save results + meta
    results_df = pd.DataFrame(rows_summary)
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
        pd.DataFrame(rows_fold).to_csv(csv_path, index=False)
        print(f"[OK] Saved fold-level metrics to: {csv_path}")


if __name__ == "__main__":
    run(parse_args())
