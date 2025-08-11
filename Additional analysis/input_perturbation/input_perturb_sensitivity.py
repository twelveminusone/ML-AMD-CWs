#!/usr/bin/env python
# -*- coding: utf-8 -*-

# --- add sibling 'utils/' (from imputation_order_sensitivity) to PYTHONPATH ---
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
SIBLING = HERE.parent / "imputation_order_sensitivity"
sys.path.insert(0, str(SIBLING))  # so 'from utils.io import ...' works
# -----------------------------------------------------------------------------

"""
Input Perturbation Sensitivity (Inference-time)
-----------------------------------------------
Evaluate how multiplicative perturbations on input features affect model
performance at inference time (no retraining). We:
  1) load the fully-imputed dataset (full_dataset.xlsx);
  2) use Feature Set C by default: [i_COD, i_pH, i_acidity, i_EC, day, height];
  3) per target metal percentage, train a baseline model on a single
     train/test split (80/20);
  4) on the SAME X_test, multiply ONE feature at a time by given scales
     (default: 0.9 and 1.1, i.e., ±10%) and re-predict;
  5) record metrics (R2/RMSE/MAE) for baseline and each perturbation.

Open-source principles:
- Relative paths only; English-only comments/prints.
- Time column standardized to 'day' (legacy 'day_z' auto-renamed by utils/io.py).
- Save transparent outputs: results + meta sheets.

Example:
  python input_perturb_sensitivity.py \
    --input ../../data/full_dataset.xlsx \
    --output ../../outputs/input_perturbation_sensitivity/results.xlsx \
    --test_size 0.2 --seed 42 --grid \
    --scales 0.9,1.1
"""

from __future__ import annotations
import argparse
from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

# Optional XGBoost
try:
    from xgboost import XGBRegressor  # type: ignore
except Exception:
    XGBRegressor = None  # will fallback to RF

# project utils (sourced from ../imputation_order_sensitivity/utils/)
from utils.io import load_numeric_df
from utils.meta import write_results_with_meta
from utils.common import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Input perturbation sensitivity (inference-time).")
    p.add_argument("--input", type=Path, default=Path("../../data/full_dataset.xlsx"),
                   help="Path to the fully-imputed dataset (full_dataset.xlsx).")
    p.add_argument("--sheet", type=str, default=None, help="Excel sheet name; None = first sheet.")
    p.add_argument("--output", type=Path,
                   default=Path("../../outputs/input_perturbation_sensitivity/results.xlsx"),
                   help="Path to Excel output (results + meta).")
    p.add_argument("--time_col", type=str, default="day",
                   help="Time column name (legacy 'day_z' is auto-renamed).")
    p.add_argument("--test_size", type=float, default=0.2,
                   help="Test fraction for single split. Default: 0.2")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--targets", type=str,
                   default="TFe (%),Zn (%),Al (%),Mn (%),Ni (%),Co (%),Cr (%)",
                   help="Comma-separated target columns (metal removal percentages).")
    p.add_argument("--features", type=str,
                   default="i_COD,i_pH,i_acidity,i_EC,day,height",
                   help="Comma-separated features to use (default Feature Set C).")
    p.add_argument("--scales", type=str, default="0.9,1.1",
                   help="Comma-separated multiplicative scales for perturbation, e.g., '0.9,1.1'.")
    p.add_argument("--grid", action="store_true",
                   help="Enable a small GridSearchCV for XGB if available.")
    return p.parse_args()


def _make_model(seed: int, use_grid: bool):
    """
    Prefer XGBRegressor; if unavailable, fallback to RandomForestRegressor.
    When --grid is on and XGB is available, run a small grid search (cv=3, scoring='r2').
    Returns a tuple (estimator, model_name, grid_used_flag).
    """
    if XGBRegressor is not None:
        model_name = "XGBRegressor"
        base = XGBRegressor(random_state=seed, n_estimators=200)
        if use_grid:
            grid = {"n_estimators": [100, 200], "max_depth": [3, 5, 7], "learning_rate": [0.05, 0.1]}
            est = GridSearchCV(base, grid, cv=3, scoring="r2", n_jobs=-1)
            return est, model_name, True
        return base, model_name, False
    # Fallback
    model_name = "RandomForestRegressor"
    base = RandomForestRegressor(random_state=seed, n_estimators=400)
    return base, model_name, False


def _metric_row(target: str, feature: str, scale: float, kind: str,
                r2: float, rmse: float, mae: float,
                model_name: str, n_train: int, n_test: int, note: str = "") -> dict:
    return {
        "Target": target,
        "FeaturePerturbed": feature,         # 'baseline' for the unperturbed test
        "Scale": float(scale),               # 1.0 for baseline
        "Label": kind,                       # 'baseline' or '-10%' / '+10%' / 'x0.90'
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
        "Model": model_name,
        "n_train": int(n_train),
        "n_test": int(n_test),
        "note": note,
    }


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    if not args.input.exists():
        raise FileNotFoundError(
            f"Input file not found: {args.input}. "
            "Place your fully-imputed dataset under ../../data or pass --input accordingly."
        )

    # Load numeric, unify day_z -> day, sort by day (if present)
    df = load_numeric_df(args.input, sheet=args.sheet, time_col=args.time_col)
    print(f"[INFO] Loaded dataset: {args.input} | shape={df.shape}")

    # Parse targets & features
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    features = [f.strip() for f in args.features.split(",") if f.strip()]
    scales = [float(s.strip()) for s in args.scales.split(",") if s.strip()]

    # Check columns presence
    missing_features = [c for c in features if c not in df.columns]
    if missing_features:
        raise RuntimeError(f"Missing features in dataset: {missing_features}")

    missing_targets = [c for c in targets if c not in df.columns]
    if missing_targets:
        raise RuntimeError(f"Missing targets in dataset: {missing_targets}")

    # Drop rows with non-finite values in used columns (should be none after full imputation, but be safe)
    used_cols = list(set(features + targets))
    dff = df[used_cols].replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any").reset_index(drop=True)
    print(f"[INFO] After sanitizing used columns: shape={dff.shape}")

    # Prepare outputs
    rows: List[dict] = []
    model_used = None
    grid_used = False

    for tgt in targets:
        X = dff[features].copy()
        y = dff[tgt].copy()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.seed, shuffle=True
        )

        est, model_name, used_grid = _make_model(seed=args.seed, use_grid=args.grid)
        model_used = model_name  # record last used
        grid_used = grid_used or used_grid

        # Fit
        try:
            est.fit(X_train, y_train)
            est_used = est.best_estimator_ if isinstance(est, GridSearchCV) else est
        except Exception as e:
            note = f"Training failed for target '{tgt}': {type(e).__name__}: {e}"
            print("[WARN]", note)
            # record NaNs and continue
            rows.append(_metric_row(
                target=tgt, feature="baseline", scale=1.0, kind="baseline",
                r2=np.nan, rmse=np.nan, mae=np.nan,
                model_name=model_name, n_train=len(y_train), n_test=len(y_test),
                note=note
            ))
            continue

        # Baseline performance on original X_test
        y_pred = est_used.predict(X_test)
        r2_base = r2_score(y_test, y_pred)
        rmse_base = mean_squared_error(y_test, y_pred, squared=False)
        mae_base = mean_absolute_error(y_test, y_pred)

        rows.append(_metric_row(
            target=tgt, feature="baseline", scale=1.0, kind="baseline",
            r2=float(r2_base), rmse=float(rmse_base), mae=float(mae_base),
            model_name=model_name, n_train=len(y_train), n_test=len(y_test),
        ))

        # Perturb ONE feature at a time on X_test
        for feat in features:
            for s in scales:
                Xp = X_test.copy()
                try:
                    Xp[feat] = Xp[feat] * float(s)
                    yp = est_used.predict(Xp)
                    r2 = r2_score(y_test, yp)
                    rmse = mean_squared_error(y_test, yp, squared=False)
                    mae = mean_absolute_error(y_test, yp)

                    # friendly label like '-10%' for 0.9
                    pct = (s - 1.0) * 100.0
                    sign = "+" if pct >= 0 else ""
                    label = f"{sign}{pct:.0f}%"

                    rows.append(_metric_row(
                        target=tgt, feature=feat, scale=float(s), kind=label,
                        r2=float(r2), rmse=float(rmse), mae=float(mae),
                        model_name=model_name, n_train=len(y_train), n_test=len(y_test),
                    ))
                except Exception as e:
                    rows.append(_metric_row(
                        target=tgt, feature=feat, scale=float(s), kind="error",
                        r2=np.nan, rmse=np.nan, mae=np.nan,
                        model_name=model_name, n_train=len(y_train), n_test=len(y_test),
                        note=f"Predict failed: {type(e).__name__}: {e}",
                    ))

    # Save results + meta
    results_df = pd.DataFrame(rows)
    write_results_with_meta(
        results_df=results_df,
        output_path=args.output,
        input_path=args.input,
        args={
            **vars(args),
            "model_used": model_used or "N/A",
            "grid_used": bool(grid_used),
        },
        df_shape=df.shape,
    )
    print(f"[OK] Saved results to: {args.output}")


if __name__ == "__main__":
    run(parse_args())
