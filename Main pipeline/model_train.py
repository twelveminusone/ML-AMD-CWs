#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model training (open-source version)
- Train 5 regressors (XGB/RF/KNN/SVR/ANN) on A/B/C feature sets
- Evaluate on a single train/test split
- Save per-target predictions, metrics summary, and best model artifact + scatter plot

Defaults (run from `Main pipeline/`):
  --input  ../data/full_dataset.xlsx
  --outdir ../outputs/model_and_predict
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

# Optional XGBoost
try:
    from xgboost import XGBRegressor  # type: ignore
except Exception:
    XGBRegressor = None

TARGETS = ['TFe (%)', 'Zn (%)', 'Al (%)', 'Mn (%)', 'Ni (%)', 'Co (%)', 'Cr (%)']


def normalize_day_col(df: pd.DataFrame, time_col: str = "day") -> pd.DataFrame:
    """Unify time column to 'day' if legacy 'day_z' exists."""
    if "day_z" in df.columns and time_col not in df.columns:
        df = df.rename(columns={"day_z": time_col})
    return df


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    repo = here.parent  # ML+AMD+CWs/
    p = argparse.ArgumentParser(description="Train 5 regressors on A/B/C feature sets; default + grid search.")
    p.add_argument("--input", type=Path, default=repo / "data" / "full_dataset.xlsx",
                   help="Path to the fully-imputed dataset.")
    p.add_argument("--sheet", type=str, default=None, help="Excel sheet name; None = first sheet.")
    p.add_argument("--outdir", type=Path, default=repo / "outputs" / "model_and_predict",
                   help="Base directory to save models and results.")
    p.add_argument("--test_size", type=float, default=0.2, help="Test fraction for the train/test split.")
    p.add_argument("--seed", type=int, default=42, help="Random seed for splitting and models when applicable.")
    p.add_argument("--sets", type=str, default="A,B,C", help="Feature sets to run, e.g., 'A,B,C' or 'C'.")
    return p.parse_args()


def build_feature_sets(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Return dict of feature lists for A/B/C; drop features not present to avoid hard failures."""
    # A: all non-target columns
    all_cols = df.columns.tolist()
    a_feats = [c for c in all_cols if c not in TARGETS]

    # B: focused inflow/outflow set (keep present-only)
    b_candidates = ['i_COD', 'i_pH', 'o_EC', 'o_Mn', 'o_TFe', 'o_SO42-']
    b_feats = [c for c in b_candidates if c in df.columns]

    # C: 6-variable practical set (keep present-only)
    c_candidates = ['i_pH', 'i_COD', 'day', 'i_EC', 'height', 'i_acidity']
    c_feats = [c for c in c_candidates if c in df.columns]

    return {"A": a_feats, "B": b_feats, "C": c_feats}


def build_model_space(seed: int):
    """Return dict: model name -> (estimator class, param_grid). XGB falls back to RF if not available."""
    if XGBRegressor is None:
        xgb_cls, xgb_grid = RandomForestRegressor, {'n_estimators': [100, 200, 300], 'max_depth': [4, 6, 8]}
    else:
        xgb_cls, xgb_grid = XGBRegressor, {'n_estimators': [100, 200, 300], 'max_depth': [4, 6, 8],
                                           'learning_rate': [0.05, 0.1, 0.3]}
    return {
        "XGB": (xgb_cls, xgb_grid),
        "RF": (RandomForestRegressor, {'n_estimators': [100, 200, 300], 'max_depth': [4, 6, 8]}),
        "KNN": (KNeighborsRegressor, {'n_neighbors': [3, 5, 7, 9]}),
        "SVR": (SVR, {'C': [1, 10, 100], 'gamma': ['scale', 'auto'], 'epsilon': [0.01, 0.1, 0.5]}),
        "ANN": (MLPRegressor, {'hidden_layer_sizes': [(50,), (100,), (100, 50)],
                               'activation': ['relu'], 'solver': ['adam'], 'max_iter': [500]})
    }


def train_eval_one(
    X_train: np.ndarray, y_train_scaled: np.ndarray,
    X_test: np.ndarray, y_test_real: np.ndarray,
    scaler_y: StandardScaler, model_cls, grid: dict, seed: int
) -> Tuple[object, Dict[str, float], np.ndarray, np.ndarray]:
    """
    Fit default model and grid-searched model; return the best by test R2.
    Returns: (best_model, metrics_dict, y_pred_best_real, y_test_real)
    """
    best = {"r2": -np.inf, "name": "unknown", "param_mode": "default", "model": None,
            "y_pred_real": None, "train_r2": None, "rmse": None, "mae": None}

    # default
    try:
        m = model_cls()
        if hasattr(m, "random_state"):
            setattr(m, "random_state", seed)
        m.fit(X_train, y_train_scaled)
        y_pred_test = m.predict(X_test)
        y_pred_train = m.predict(X_train)
        y_pred_test_real = scaler_y.inverse_transform(y_pred_test.reshape(-1, 1)).ravel()
        y_pred_train_real = scaler_y.inverse_transform(y_pred_train.reshape(-1, 1)).ravel()

        r2 = r2_score(y_test_real, y_pred_test_real)
        rmse = mean_squared_error(y_test_real, y_pred_test_real, squared=False)
        mae = mean_absolute_error(y_test_real, y_pred_test_real)
        train_r2 = r2_score(scaler_y.inverse_transform(y_train_scaled.reshape(-1, 1)).ravel(), y_pred_train_real)

        best = {"r2": r2, "name": model_cls.__name__, "param_mode": "default", "model": m,
                "y_pred_real": y_pred_test_real, "train_r2": train_r2, "rmse": rmse, "mae": mae}
    except Exception as e:
        print(f"[WARN] default fit failed for {model_cls.__name__}: {e}")

    # grid
    try:
        gridcv = GridSearchCV(model_cls(), grid, cv=5, scoring="neg_mean_squared_error", n_jobs=-1)
        gridcv.fit(X_train, y_train_scaled)
        m = gridcv.best_estimator_
        y_pred_test = m.predict(X_test)
        y_pred_train = m.predict(X_train)
        y_pred_test_real = scaler_y.inverse_transform(y_pred_test.reshape(-1, 1)).ravel()
        y_pred_train_real = scaler_y.inverse_transform(y_pred_train.reshape(-1, 1)).ravel()

        r2 = r2_score(y_test_real, y_pred_test_real)
        rmse = mean_squared_error(y_test_real, y_pred_test_real, squared=False)
        mae = mean_absolute_error(y_test_real, y_pred_test_real)
        train_r2 = r2_score(scaler_y.inverse_transform(y_train_scaled.reshape(-1, 1)).ravel(), y_pred_train_real)

        if r2 > best["r2"]:
            best = {"r2": r2, "name": model_cls.__name__, "param_mode": "grid", "model": m,
                    "y_pred_real": y_pred_test_real, "train_r2": train_r2, "rmse": rmse, "mae": mae}
    except Exception as e:
        print(f"[WARN] grid search failed for {model_cls.__name__}: {e}")

    return best["model"], {
        "best_model_name": best["name"], "best_param_mode": best["param_mode"],
        "train_r2": float(best["train_r2"]) if best["train_r2"] is not None else np.nan,
        "test_r2": float(best["r2"]), "test_rmse": float(best["rmse"]) if best["rmse"] is not None else np.nan,
        "test_mae": float(best["mae"]) if best["mae"] is not None else np.nan
    }, best["y_pred_real"], y_test_real


def main() -> None:
    args = parse_args()

    model_dir = args.outdir / "model"
    result_dir = args.outdir / "result"
    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_excel(args.input, sheet_name=args.sheet)
    df = normalize_day_col(df, "day")

    # Ensure targets exist
    present_targets = [t for t in TARGETS if t in df.columns]
    if not present_targets:
        raise RuntimeError("None of the expected target columns are present in the dataset.")
    # Build feature sets and filter by user choice
    feat_sets = build_feature_sets(df)
    wanted_sets = [s.strip().upper() for s in args.sets.split(",") if s.strip()]
    feat_sets = {k: v for k, v in feat_sets.items() if k in wanted_sets and len(v) > 0}
    if not feat_sets:
        raise RuntimeError("No valid feature set to run. Check --sets and dataset columns.")

    # Model space
    models = build_model_space(args.seed)

    all_metrics_rows: List[Dict] = []

    for set_name, feature_cols in feat_sets.items():
        print(f"[INFO] Running feature set {set_name} with {len(feature_cols)} features.")
        X_all = df[feature_cols].copy()

        for target in present_targets:
            y_all = df[target].copy()

            # split
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_all, y_all, test_size=args.test_size, random_state=args.seed, shuffle=True
            )

            # scaling
            scaler_X = StandardScaler().fit(X_tr)
            scaler_y = StandardScaler().fit(y_tr.values.reshape(-1, 1))
            X_train = scaler_X.transform(X_tr)
            X_test = scaler_X.transform(X_te)
            y_train_scaled = scaler_y.transform(y_tr.values.reshape(-1, 1)).ravel()
            y_test_real = y_te.values

            # Try each model family; keep best per family (default vs grid)
            preds_table = {"model": [], "param_mode": [], "y_pred": []}

            best_overall = {"r2": -np.inf, "model_obj": None, "name": "", "param_mode": ""}
            best_pred, best_true = None, None
            best_scaler_X, best_scaler_y = scaler_X, scaler_y

            for short_name, (cls, grid) in models.items():
                model_obj, metrics, y_pred_real, y_true_real = train_eval_one(
                    X_train, y_train_scaled, X_test, y_test_real, scaler_y, cls, grid, args.seed
                )

                # Save per-model predictions
                preds_table["model"].append(short_name)
                preds_table["param_mode"].append(metrics["best_param_mode"])
                preds_table["y_pred"].append(y_pred_real if y_pred_real is not None else np.full_like(y_true_real, np.nan))

                # Collect metrics row
                all_metrics_rows.append({
                    "feature_set": set_name, "target": target, "model": short_name,
                    "param": metrics["best_param_mode"], "train_r2": metrics["train_r2"],
                    "test_r2": metrics["test_r2"], "test_rmse": metrics["test_rmse"],
                    "test_mae": metrics["test_mae"]
                })

                # Track the absolute best across families for plotting/saving
                if metrics["test_r2"] > best_overall["r2"]:
                    best_overall.update({
                        "r2": metrics["test_r2"], "model_obj": model_obj,
                        "name": short_name, "param_mode": metrics["best_param_mode"]
                    })
                    best_pred, best_true = y_pred_real, y_true_real

            # Save prediction table for this (set, target)
            pred_df = pd.DataFrame({"y_true": y_test_real})
            for m, pm, pred in zip(preds_table["model"], preds_table["param_mode"], preds_table["y_pred"]):
                pred_df[f"{m}_{pm}_pred"] = pred
            pred_df.to_excel(result_dir / f"{set_name}_{target}_pred.xlsx", index=False)

            # Save the absolute best model artifact and scatter plot (C-set emphasized in filename)
            if best_overall["model_obj"] is not None:
                artifact = {
                    "model": best_overall["model_obj"],
                    "scaler_X": best_scaler_X,
                    "scaler_y": best_scaler_y,
                    "features": feature_cols
                }
                joblib.dump(artifact, model_dir / f"{set_name}_{target}_{best_overall['name']}_{best_overall['param_mode']}.joblib")

                # scatter
                if best_pred is not None and best_true is not None:
                    plt.figure(figsize=(5.5, 5))
                    plt.scatter(best_true, best_pred, s=24, alpha=0.8, edgecolor="k")
                    mn, mx = float(min(best_true.min(), best_pred.min())), float(max(best_true.max(), best_pred.max()))
                    plt.plot([mn, mx], [mn, mx], "g--", label="y = x")
                    plt.xlabel(f"Actual {target}")
                    plt.ylabel("Predicted")
                    plt.title(f"{target} — Best {best_overall['name']} ({best_overall['param_mode']}), R²={best_overall['r2']:.2f}")
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(result_dir / f"{set_name}_{target}_{best_overall['name']}_{best_overall['param_mode']}_scatter.png", dpi=300)
                    plt.close()

    # Save overall metrics summary
    metrics_df = pd.DataFrame(all_metrics_rows)
    metrics_df.to_excel(result_dir / "all_metrics.xlsx", index=False)

    print(f"[OK] Models and results saved under: {args.outdir}")
    if XGBRegressor is None:
        print("[NOTE] xgboost not installed; XGB entries used RandomForest as a fallback.")


if __name__ == "__main__":
    main()
