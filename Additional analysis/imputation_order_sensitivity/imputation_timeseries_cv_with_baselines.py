#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TimeSeries CV with Naive Baselines (Single Horizon)
---------------------------------------------------
Forward-chaining evaluation of imputers + ML models against naive baselines.

Pipeline per fold (sorted by 'day'):
  1) Fit imputer on TRAIN window only; transform TEST window.
  2) Build target y = (1 - o_metal / i_metal) * 100 aligned to a given horizon H:
     y_aligned[t] = y_raw[t + H].
  3) Train models on (X_t, y_{t+H}) in TRAIN; evaluate on TEST.
  4) Compare to naive baselines (Last, MA3, MA5), all shifted by 1 to avoid leakage.

Open-source principles:
- Use INITIAL dataset with missing values (initial_dataset.xlsx).
- Relative paths; English-only comments; unified time column 'day'.
- Save 'results' + 'meta'; optional per-fold CSV.

Example:
  python imputation_timeseries_cv_with_baselines.py \
    --input ../../data/initial_dataset.xlsx \
    --output ../../outputs/imputation_order_sensitivity/timeseries_cv_baseline/results.xlsx \
    --n_splits 3 --horizon 1 --seed 42 --methods all \
    --feature_sets A,B,C --models XGB,RF,KNN,SVR,ANN --grid --save_fold_details
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

# Optional: xgboost & missforest may not be installed everywhere
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

# project utils
from utils.io import load_numeric_df
from utils.meta import write_results_with_meta
from utils.common import set_seed


# ---------------------------
# CLI
# ---------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Forward-chaining TimeSeries CV with naive baselines (single horizon)."
    )
    p.add_argument("--input", type=Path, default=Path("../../data/initial_dataset.xlsx"),
                   help="Path to INITIAL dataset with missing values.")
    p.add_argument("--sheet", type=str, default=None, help="Excel sheet name; if None, first sheet.")
    p.add_argument("--output", type=Path,
                   default=Path("../../outputs/imputation_order_sensitivity/timeseries_cv_baseline/results.xlsx"),
                   help="Path to Excel output (results + meta).")
    p.add_argument("--n_splits", type=int, default=3, help="TimeSeriesSplit folds. Default: 3")
    p.add_argument("--horizon", type=int, default=1, help="Prediction horizon H (>=1). Default: 1")
    p.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42")
    p.add_argument("--time_col", type=str, default="day", help="Time column (legacy 'day_z' auto-renamed).")
    p.add_argument("--methods", type=str, default="all",
                   help="Imputers among RF,KNN,MICE,MissForest or 'all'.")
    p.add_argument("--feature_sets", type=str, default="A,B,C",
                   help="Feature sets to evaluate: subset of 'A,B,C'.")
    p.add_argument("--models", type=str, default="XGB,RF,KNN,SVR,ANN",
                   help="ML models to evaluate; best per fold is recorded.")
    p.add_argument("--grid", action="store_true",
                   help="Run small GridSearchCV (cv=3, scoring='r2') for each model.")
    p.add_argument("--baselines", type=str, default="last,ma3,ma5",
                   help="Naive baselines: comma list from last,ma3,ma5.")
    p.add_argument("--save_fold_details", action="store_true",
                   help="Also save fold-level metrics CSV next to the Excel output.")
    return p.parse_args()


# ---------------------------
# Helpers
# ---------------------------
def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip units '(...)', drop obvious non-features (record, TDS), trim spaces."""
    df = df.copy()
    df.columns = df.columns.str.replace(r"\(.*?\)", "", regex=True).str.strip()
    for col in ("record",):
        if col in df.columns:
            df = df.drop(columns=[col])
    for col in ("i_TDS", "o_TDS"):
        if col in df.columns:
            df = df.drop(columns=[col])
    return df


def _get_feature_sets(cols: List[str]) -> Dict[str, List[str]]:
    """Construct A/B/C feature sets (same convention as other scripts)."""
    metals = ["TFe", "Zn", "Al", "Mn", "Ni", "Co", "Cr"]
    targets = [f"{m} (%)" for m in metals]
    A = [c for c in cols if c not in targets]
    B = [c for c in ["i_COD", "i_pH", "o_EC", "o_Mn", "o_TFe", "o_SO42-"] if c in cols]
    C = [c for c in ["i_pH", "i_COD", "day", "i_EC", "height", "i_acidity"] if c in cols]
    return {"A": A, "B": B, "C": C}


def _build_imputer(method_key: str, seed: int):
    """Map method key to an imputer instance."""
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
            raise RuntimeError("MissForest not installed. `pip install missforest` or remove from --methods.")
        return MissForest(random_state=seed)
    raise ValueError(f"Unknown imputation method: {method_key}")


def _get_models(seed: int):
    """Return {name: (ctor, grid)}; use grid when --grid is set."""
    models = {}
    if XGBRegressor is not None:
        models["XGB"] = (
            lambda: XGBRegressor(random_state=seed, n_estimators=200),
            {"n_estimators": [200, 400], "max_depth": [4, 6], "learning_rate": [0.05, 0.1]},
        )
    models["RF"] = (
        lambda: RandomForestRegressor(random_state=seed),
        {"n_estimators": [200, 400], "max_depth": [None, 6, 10]},
    )
    models["KNN"] = (lambda: KNeighborsRegressor(), {"n_neighbors": [3, 5, 7]})
    models["SVR"] = (lambda: SVR(), {"C": [1, 10], "gamma": ["scale"], "epsilon": [0.01, 0.1]})
    models["ANN"] = (
        lambda: MLPRegressor(random_state=seed, max_iter=500),
        {"hidden_layer_sizes": [(50,), (100,)], "alpha": [1e-4, 1e-3], "activation": ["relu"], "solver": ["adam"]},
    )
    return models


def _build_y_aligned(df: pd.DataFrame, metal: str, horizon: int) -> pd.Series:
    """
    Build y_aligned for a metal at horizon H:
      y_raw[t] = (1 - o/i) * 100
      y_aligned[t] = y_raw[t + H]
    Implemented via shift(-H) on i/o before computing y.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    ci = df.get(f"i_{metal}", pd.Series([np.nan] * len(df)))
    co = df.get(f"o_{metal}", pd.Series([np.nan] * len(df)))
    ci_f = ci.shift(-horizon)
    co_f = co.shift(-horizon)
    y = (1.0 - (co_f / ci_f)) * 100.0
    y[ci_f == 0] = np.nan
    return y


def _naive_baselines(y_aligned: pd.Series, kind_list: List[str]) -> Dict[str, pd.Series]:
    """
    Build naive baseline predictors aligned with y_aligned:
      - 'last': y_pred[t] = y_aligned[t-1]
      - 'ma3' : mean(y_aligned[t-3 : t-1])
      - 'ma5' : mean(y_aligned[t-5 : t-1])
    All baselines use shift(1) to avoid using current/future information.
    """
    preds = {}
    if "last" in kind_list:
        preds["Naive-Last"] = y_aligned.shift(1)
    if "ma3" in kind_list:
        preds["Naive-MA3"] = y_aligned.rolling(window=3, min_periods=1).mean().shift(1)
    if "ma5" in kind_list:
        preds["Naive-MA5"] = y_aligned.rolling(window=5, min_periods=1).mean().shift(1)
    return preds


# ---------------------------
# Core
# ---------------------------
def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    if not args.input.exists():
        raise FileNotFoundError(f"Input not found: {args.input}. Place under ../../data or pass --input.")

    # Load + unify time column + numeric + sort
    df_raw = load_numeric_df(args.input, sheet=args.sheet, time_col=args.time_col)
    if args.time_col not in df_raw.columns:
        raise RuntimeError(f"Time column '{args.time_col}' not found after loading.")
    df = _normalize_columns(df_raw)
    df = df.sort_values(args.time_col).reset_index(drop=True)

    # Metals & targets
    metals = ["TFe", "Zn", "Al", "Mn", "Ni", "Co", "Cr"]
    target_names = [f"{m} (%)" for m in metals]

    # Feature sets
    fsets_all = _get_feature_sets(df.columns.tolist())
    req_sets = [s.strip().upper() for s in args.feature_sets.split(",") if s.strip()]
    for s in req_sets:
        if s not in fsets_all:
            raise ValueError(f"Unknown feature set '{s}'. Valid: {list(fsets_all.keys())}")
    feature_sets = {k: fsets_all[k] for k in req_sets}

    # Imputers
    all_methods = ["RF", "KNN", "MICE"] + (["MissForest"] if MissForest is not None else [])
    if args.methods.strip().lower() == "all":
        methods = all_methods
    else:
        methods = [m.strip() for m in args.methods.split(",") if m.strip()]
        for m in methods:
            if m not in all_methods:
                raise ValueError(f"Unknown method '{m}'. Valid: {all_methods}")

    # Models
    model_bank = _get_models(seed=args.seed)
    if args.models.strip().lower() == "all":
        models = list(model_bank.keys())
    else:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
        unknown = [m for m in models if m not in model_bank]
        if unknown:
            raise ValueError(f"Unknown model(s): {unknown}. Valid: {list(model_bank.keys())}")

    # Baselines
    baseline_kinds = [b.strip().lower() for b in args.baselines.split(",") if b.strip()]
    for b in baseline_kinds:
        if b not in {"last", "ma3", "ma5"}:
            raise ValueError("Baselines must be chosen from: last, ma3, ma5")

    print(f"[INFO] Data: {args.input} shape={df.shape}")
    print(f"[INFO] TSCV n_splits={args.n_splits} | horizon={args.horizon} | seed={args.seed}")
    print(f"[INFO] Imputers: {methods}")
    print(f"[INFO] Feature sets: {list(feature_sets.keys())}")
    print(f"[INFO] Models: {models} | grid={'ON' if args.grid else 'OFF'}")
    print(f"[INFO] Baselines: {baseline_kinds}")

    tscv = TimeSeriesSplit(n_splits=args.n_splits)

    rows_summary: List[dict] = []
    rows_fold: List[dict] = []

    for method in methods:
        for fs_name, fs_cols in feature_sets.items():
            X_all_raw = df[fs_cols]

            for target in target_names:
                metal = target.split()[0]  # "TFe (%)" -> "TFe"

                # Build horizon-aligned y
                y_aligned = _build_y_aligned(df, metal=metal, horizon=args.horizon)

                # Naive predictions aligned with y
                naive_preds = _naive_baselines(y_aligned, baseline_kinds)

                # Per-fold metrics for best ML and best Naive
                ml_r2, ml_rmse, ml_mae = [], [], []
                nv_r2, nv_rmse, nv_mae = [], [], []

                fold_id = 0
                for train_idx, test_idx in tscv.split(X_all_raw):
                    fold_id += 1

                    # Prepare X windows
                    X_tr_raw = X_all_raw.iloc[train_idx].reset_index(drop=True)
                    X_te_raw = X_all_raw.iloc[test_idx].reset_index(drop=True)

                    # Impute on TRAIN only
                    imputer = _build_imputer(method, seed=args.seed)
                    X_tr_imp = pd.DataFrame(imputer.fit_transform(X_tr_raw), columns=X_tr_raw.columns)
                    X_te_imp = pd.DataFrame(imputer.transform(X_te_raw), columns=X_te_raw.columns)

                    # y windows (already aligned to horizon)
                    y_tr = y_aligned.iloc[train_idx].reset_index(drop=True)
                    y_te = y_aligned.iloc[test_idx].reset_index(drop=True)

                    # Drop NaNs in y
                    mtr = ~y_tr.isna()
                    mte = ~y_te.isna()
                    X_tr = X_tr_imp.loc[mtr]
                    X_te = X_te_imp.loc[mte]
                    y_tr = y_tr.loc[mtr]
                    y_te = y_te.loc[mte]

                    # Need minimal sizes
                    if len(y_tr) < 10 or len(y_te) < 5:
                        rows_fold.append({
                            "Imputation": method, "FeatureSet": fs_name,
                            "Track": "ML", "Model": "N/A", "Target": target, "Fold": fold_id,
                            "R2": np.nan, "RMSE": np.nan, "MAE": np.nan,
                            "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
                            "note": "Insufficient samples after filtering."
                        })
                        rows_fold.append({
                            "Imputation": method, "FeatureSet": fs_name,
                            "Track": "Naive", "Model": "N/A", "Target": target, "Fold": fold_id,
                            "R2": np.nan, "RMSE": np.nan, "MAE": np.nan,
                            "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
                            "note": "Insufficient samples after filtering."
                        })
                        continue

                    # ===== Naive baselines on TEST =====
                    best_nv_name = None
                    best_nv_r2 = -np.inf
                    best_nv_rmse = np.inf
                    best_nv_mae = np.inf

                    for name, series_pred in naive_preds.items():
                        y_pred_nv = series_pred.iloc[test_idx].reset_index(drop=True)
                        y_pred_nv = y_pred_nv.loc[mte]  # align with filtered test mask

                        if len(y_pred_nv) != len(y_te) or len(y_te) == 0:
                            r2 = rmse = mae = np.nan
                        else:
                            r2 = r2_score(y_te.values, y_pred_nv.values)
                            rmse = mean_squared_error(y_te.values, y_pred_nv.values, squared=False)
                            mae = mean_absolute_error(y_te.values, y_pred_nv.values)

                        rows_fold.append({
                            "Imputation": method, "FeatureSet": fs_name,
                            "Track": "Naive", "Model": name, "Target": target, "Fold": fold_id,
                            "R2": r2, "RMSE": rmse, "MAE": mae,
                            "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
                        })

                        if np.isfinite(r2) and r2 > best_nv_r2:
                            best_nv_r2, best_nv_rmse, best_nv_mae = r2, rmse, mae
                            best_nv_name = name

                    nv_r2.append(best_nv_r2)
                    nv_rmse.append(best_nv_rmse)
                    nv_mae.append(best_nv_mae)

                    # ===== ML models =====
                    scaler_X = StandardScaler().fit(X_tr)
                    scaler_y = StandardScaler().fit(y_tr.values.reshape(-1, 1))
                    X_tr_s = scaler_X.transform(X_tr)
                    X_te_s = scaler_X.transform(X_te)
                    y_tr_s = scaler_y.transform(y_tr.values.reshape(-1, 1)).ravel()

                    best_ml_name = None
                    best_ml_r2 = -np.inf
                    best_ml_rmse = np.inf
                    best_ml_mae = np.inf

                    for mdl_name in models:
                        ctor, grid = model_bank[mdl_name]
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

                            r2 = r2_score(y_te.values, y_pred)
                            rmse = mean_squared_error(y_te.values, y_pred, squared=False)
                            mae = mean_absolute_error(y_te.values, y_pred)

                            rows_fold.append({
                                "Imputation": method, "FeatureSet": fs_name,
                                "Track": "ML", "Model": mdl_name, "Target": target, "Fold": fold_id,
                                "R2": r2, "RMSE": rmse, "MAE": mae,
                                "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
                            })

                            if r2 > best_ml_r2:
                                best_ml_r2, best_ml_rmse, best_ml_mae = r2, rmse, mae
                                best_ml_name = mdl_name

                        except Exception as e:
                            rows_fold.append({
                                "Imputation": method, "FeatureSet": fs_name,
                                "Track": "ML", "Model": mdl_name, "Target": target, "Fold": fold_id,
                                "R2": np.nan, "RMSE": np.nan, "MAE": np.nan,
                                "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
                                "note": f"Failed: {type(e).__name__}: {e}",
                            })

                    ml_r2.append(best_ml_r2)
                    ml_rmse.append(best_ml_rmse)
                    ml_mae.append(best_ml_mae)

                # Aggregate across folds
                def _agg(vals):
                    vals = [v for v in vals if np.isfinite(v)]
                    if len(vals) == 0:
                        return np.nan, np.nan, 0
                    ddof = 1 if len(vals) > 1 else 0
                    return float(np.mean(vals)), float(np.std(vals, ddof=ddof)), len(vals)

                ml_r2_m, ml_r2_s, n_ml = _agg(ml_r2)
                ml_rmse_m, ml_rmse_s, _ = _agg(ml_rmse)
                ml_mae_m, ml_mae_s, _ = _agg(ml_mae)

                nv_r2_m, nv_r2_s, n_nv = _agg(nv_r2)
                nv_rmse_m, nv_rmse_s, _ = _agg(nv_rmse)
                nv_mae_m, nv_mae_s, _ = _agg(nv_mae)

                rows_summary.append({
                    "Imputation": method,
                    "FeatureSet": fs_name,
                    "Target": target,
                    "Horizon": int(args.horizon),
                    "BestML_R2_mean": ml_r2_m, "BestML_R2_std": ml_r2_s,
                    "BestML_RMSE_mean": ml_rmse_m, "BestML_RMSE_std": ml_rmse_s,
                    "BestML_MAE_mean": ml_mae_m, "BestML_MAE_std": ml_mae_s,
                    "BestNaive_R2_mean": nv_r2_m, "BestNaive_R2_std": nv_r2_s,
                    "BestNaive_RMSE_mean": nv_rmse_m, "BestNaive_RMSE_std": nv_rmse_s,
                    "BestNaive_MAE_mean": nv_mae_m, "BestNaive_MAE_std": nv_mae_s,
                    "n_folds_effective": int(min(n_ml, n_nv)),
                    "note": "",
                })

    # Save
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
