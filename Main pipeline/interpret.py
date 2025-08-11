#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Interpretability suite (open-source version)
- For each target: fit XGBoost (with a small grid search),
  export per-target XGB importances, SHAP summary plots, PDP (1D/2D),
  and save the trained model.
- Also export an overall XGB importance (mean across targets) and a combined SHAP summary.

Defaults (run from `Main pipeline/`):
  --input  ../data/full_dataset.xlsx
  --outdir ../outputs/interpretability
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Tuple

# Non-interactive backend for figure saving
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.inspection import PartialDependenceDisplay

# Required dependencies (clear errors if missing)
try:
    import xgboost as xgb  # type: ignore
except Exception as e:
    raise ImportError("This script requires xgboost. Please `pip install xgboost`.") from e

try:
    import shap  # type: ignore
except Exception as e:
    raise ImportError("This script requires shap. Please `pip install shap`.") from e


TARGETS = ['TFe (%)', 'Zn (%)', 'Al (%)', 'Mn (%)', 'Ni (%)', 'Co (%)', 'Cr (%)']
C_FEATS = ['i_pH', 'i_COD', 'day', 'i_EC', 'height', 'i_acidity']


def normalize_day_col(df: pd.DataFrame, time_col: str = "day") -> pd.DataFrame:
    """Unify time column name to 'day' if a legacy 'day_z' exists."""
    if "day_z" in df.columns and time_col not in df.columns:
        df = df.rename(columns={"day_z": time_col})
    return df


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    repo = here.parent  # ML+AMD+CWs/
    p = argparse.ArgumentParser(description="XGB + SHAP + PDP interpretability suite.")
    p.add_argument("--input", type=Path, default=repo / "data" / "full_dataset.xlsx",
                   help="Path to the fully-imputed dataset.")
    p.add_argument("--sheet", type=str, default=None, help="Excel sheet name; None = first sheet.")
    p.add_argument("--outdir", type=Path, default=repo / "outputs" / "interpretability",
                   help="Base directory to save models and figures.")
    p.add_argument("--test_size", type=float, default=0.2, help="Test fraction for the train/test split.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--features", type=str, default=",".join(C_FEATS),
                   help="Comma-separated feature list (defaults to the 6-variable practical set).")
    return p.parse_args()


def ensure_dirs(base: Path) -> Tuple[Path, Path, Path, Path]:
    imp_model_dir = base / "imp_model"
    imp_dir = base / "importances"
    shap_dir = base / "shap"
    pdp_dir = base / "pdp"
    for d in [imp_model_dir, imp_dir, shap_dir, pdp_dir]:
        d.mkdir(parents=True, exist_ok=True)
    return imp_model_dir, imp_dir, shap_dir, pdp_dir


def main() -> None:
    args = parse_args()
    imp_model_dir, imp_dir, shap_dir, pdp_dir = ensure_dirs(args.outdir)

    # Load and sanitize
    df = pd.read_excel(args.input, sheet_name=args.sheet)
    df = normalize_day_col(df, "day")

    # Features & targets
    feats = [f.strip() for f in args.features.split(",") if f.strip()]
    feats = [f for f in feats if f in df.columns]  # drop missing silently
    if len(feats) == 0:
        raise RuntimeError("No valid features found in the dataset for the requested feature list.")
    present_targets = [t for t in TARGETS if t in df.columns]
    if not present_targets:
        raise RuntimeError("None of the expected target columns are present in the dataset.")

    X_full = df[feats]
    xgb_param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [4, 6],
        "learning_rate": [0.1, 0.3],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }

    per_target_importances: List[np.ndarray] = []
    shap_values_all: List[np.ndarray] = []
    X_for_shap_all: List[pd.DataFrame] = []

    for target in present_targets:
        y_full = df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X_full, y_full, test_size=args.test_size, random_state=args.seed
        )

        # Fit XGB with small grid
        model = xgb.XGBRegressor(objective="reg:squarederror", random_state=args.seed, n_jobs=-1)
        grid = GridSearchCV(model, xgb_param_grid, cv=5, scoring="neg_mean_squared_error", n_jobs=-1)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_

        # Save model
        model_path = imp_model_dir / f"C_{target}_XGB_grid.json"
        best_model.save_model(model_path)

        # Save per-target XGB importances (gain)
        booster = best_model.get_booster()
        gain = booster.get_score(importance_type="gain")
        importances = np.array([gain.get(f"f{j}", 0.0) for j in range(X_full.shape[1])], dtype=float)
        per_target_importances.append(importances)

        plt.figure(figsize=(7, 4))
        plt.barh(feats, importances)
        plt.xlabel("Importance (gain)")
        plt.title(f"Feature Importances (XGBoost) — {target}")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(imp_dir / f"C_{target}_XGB_importance_gain.png", dpi=300)
        plt.close()

        # SHAP (summary plot per target)
        explainer = shap.Explainer(best_model, X_train)
        shap_values = explainer(X_test)  # explain on test to keep runtime moderate
        plt.figure(figsize=(7, 4))
        shap.summary_plot(shap_values.values, X_test, plot_type="dot", show=False)
        plt.title(f"SHAP Summary — {target}")
        plt.tight_layout()
        plt.savefig(shap_dir / f"C_{target}_SHAP_summary.png", dpi=300)
        plt.close()

        # Keep for global summary
        shap_values_all.append(shap_values.values)
        X_for_shap_all.append(X_test.copy())

        # PDP (1D)
        try:
            PartialDependenceDisplay.from_estimator(best_model, X_train, features=feats, kind="average")
            plt.tight_layout()
            plt.savefig(pdp_dir / f"C_{target}_PDP_1D.png", dpi=250)
            plt.close()
        except Exception as e:
            print(f"[WARN] 1D PDP failed for {target}: {type(e).__name__}: {e}")

        # PDP (2D pairs)
        try:
            for i in range(len(feats)):
                for j in range(i + 1, len(feats)):
                    pair = (feats[i], feats[j])
                    PartialDependenceDisplay.from_estimator(best_model, X_train, features=[pair], kind="average")
                    ax = plt.gca()
                    if hasattr(ax, "collections") and ax.collections:
                        plt.colorbar(ax.collections[0], ax=ax)
                    plt.title(f"2D PDP: {pair[0]} & {pair[1]} → {target}")
                    plt.xlabel(pair[0]); plt.ylabel(pair[1])
                    plt.tight_layout()
                    plt.savefig(pdp_dir / f"C_{target}_PDP_2D_{pair[0]}_{pair[1]}.png", dpi=250)
                    plt.close()
        except Exception as e:
            print(f"[WARN] 2D PDP failed for {target}: {type(e).__name__}: {e}")

    # Overall XGB importance (mean across targets)
    if per_target_importances:
        mean_imp = np.mean(np.vstack(per_target_importances), axis=0)
        plt.figure(figsize=(7, 5))
        plt.barh(feats, mean_imp)
        plt.xlabel("Importance (gain)")
        plt.title("XGBoost Feature Importances — All Targets (mean)")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(imp_dir / "C_XGB_importance_overall_gain.png", dpi=300)
        plt.close()
        pd.Series(mean_imp, index=feats).to_excel(imp_dir / "C_XGB_importance_overall_gain.xlsx")

    # Combined SHAP summary (stack per-target values)
    try:
        if shap_values_all:
            vals = np.vstack(shap_values_all)
            X_rep = pd.concat(X_for_shap_all, axis=0).reset_index(drop=True)
            plt.figure(figsize=(7, 4))
            shap.summary_plot(vals, X_rep, plot_type="dot", show=False)
            plt.title("SHAP Summary — All Targets Combined")
            plt.tight_layout()
            plt.savefig(shap_dir / "C_SHAP_summary_all_targets.png", dpi=300)
            plt.close()
    except Exception as e:
        print(f"[WARN] Global SHAP summary failed: {type(e).__name__}: {e}")

    print(f"[OK] All interpretability outputs saved under: {args.outdir}")


if __name__ == "__main__":
    main()
