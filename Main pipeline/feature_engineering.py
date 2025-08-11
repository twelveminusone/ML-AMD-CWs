#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Feature engineering (open-source version)
- Correlation matrix (optionally clustered if SciPy is available)
- (Optional) hierarchical clustering dendrogram
- (Optional) model-based feature importance (XGBoost preferred; fallback to RandomForest)
- Export A/B/C feature sets as Excel files

Defaults (run from `Main pipeline/`):
  --input  ../data/full_dataset.xlsx
  --outdir ../outputs/feature_engineering
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional dependencies
try:
    from xgboost import XGBRegressor  # type: ignore
except Exception:
    XGBRegressor = None

try:
    from scipy.spatial.distance import squareform  # type: ignore
    from scipy.cluster.hierarchy import linkage, dendrogram, fcluster  # type: ignore
    _SCIPY_OK = True
except Exception:
    _SCIPY_OK = False


TARGETS = ['TFe (%)', 'Zn (%)', 'Al (%)', 'Mn (%)', 'Ni (%)', 'Co (%)', 'Cr (%)']
C_FEATURES = ['i_pH', 'i_COD', 'day', 'i_EC', 'height', 'i_acidity']


def normalize_day_col(df: pd.DataFrame, time_col: str = "day") -> pd.DataFrame:
    """Unify time column to 'day' if legacy 'day_z' exists."""
    if "day_z" in df.columns and time_col not in df.columns:
        df = df.rename(columns={"day_z": time_col})
    return df


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    repo = here.parent  # ML+AMD+CWs/
    p = argparse.ArgumentParser(description="Feature engineering: correlation, clustering, importance, A/B/C sets.")
    p.add_argument("--input", type=Path, default=repo / "data" / "full_dataset.xlsx",
                   help="Path to fully-imputed dataset.")
    p.add_argument("--sheet", type=str, default=None, help="Excel sheet name; None = first sheet.")
    p.add_argument("--outdir", type=Path, default=repo / "outputs" / "feature_engineering",
                   help="Directory to save figures and tables.")
    p.add_argument("--n_repeat", type=int, default=5,
                   help="Repetitions for model-based importance averaging.")
    p.add_argument("--cluster_threshold", type=float, default=0.4,
                   help="Distance threshold for clustering (if SciPy available).")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    return p.parse_args()


def _safe_corr(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Compute correlation on numeric subset, dropping all-NaN columns."""
    dff = df[cols].select_dtypes(include=[np.number]).copy()
    dff = dff.dropna(axis=1, how="all")
    return dff.corr()


def _cluster_order_by_abs_corr(corr: pd.DataFrame) -> List[str]:
    """Return feature order by hierarchical clustering on (1 - |corr|). Requires SciPy."""
    if not _SCIPY_OK:
        return list(corr.columns)
    dist = 1.0 - np.abs(corr.values)
    # squareform expects condensed distance vector; ensure symmetry + zero diagonal
    dist = (dist + dist.T) / 2.0
    np.fill_diagonal(dist, 0.0)
    Z = linkage(squareform(dist, checks=False), method="average")
    leaves = dendrogram(Z, labels=corr.columns.tolist(), no_plot=True)["leaves"]
    return [corr.columns[int(i)] for i in leaves]


def _plot_corr_heatmap(corr: pd.DataFrame, order: Optional[List[str]], outpath: Path) -> None:
    """Plain matplotlib heatmap to keep dependencies minimal."""
    if order is None:
        order = list(corr.columns)
    corr_sorted = corr.loc[order, order]
    fig, ax = plt.subplots(figsize=(0.45 * len(order) + 6, 0.45 * len(order) + 4))
    im = ax.imshow(corr_sorted.values, vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(order, rotation=90)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(order)
    ax.set_title("Correlation Matrix", fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def _plot_input_dendrogram(df_inputs: pd.DataFrame, outpath: Path, threshold: float) -> None:
    """Plot dendrogram for input features only if SciPy is available."""
    if not _SCIPY_OK:
        print("[WARN] SciPy not available; skip dendrogram.")
        return
    corr_in = df_inputs.corr()
    dist_in = 1.0 - np.abs(corr_in.values)
    dist_in = (dist_in + dist_in.T) / 2.0
    np.fill_diagonal(dist_in, 0.0)
    Z = linkage(squareform(dist_in, checks=False), method="average")
    fig, ax = plt.subplots(figsize=(0.32 * df_inputs.shape[1] + 6, 5))
    dendrogram(Z, labels=df_inputs.columns.tolist(), color_threshold=threshold, leaf_rotation=90, ax=ax)
    ax.set_title("Hierarchical Clustering Dendrogram (Input Features)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def _compute_model_importance(df: pd.DataFrame, features: List[str], targets: List[str],
                              n_repeat: int, seed: int) -> pd.Series:
    """
    Compute mean feature importance across targets and repeats.
    Prefer XGB (gain); fallback to RandomForest feature_importances_.
    Return a pandas Series indexed by feature name.
    """
    from sklearn.ensemble import RandomForestRegressor

    X = df[features].values
    rng = np.random.RandomState(seed)

    importances = np.zeros(len(features), dtype=float)
    counts = 0

    for rep in range(n_repeat):
        for tgt in targets:
            y = df[tgt].values
            if XGBRegressor is not None:
                model = XGBRegressor(
                    random_state=int(rng.randint(0, 1_000_000)),
                    n_estimators=200, learning_rate=0.1, max_depth=6
                )
                model.fit(X, y)
                booster = model.get_booster()
                gain = booster.get_score(importance_type="gain")
                vec = np.array([gain.get(f"f{j}", 0.0) for j in range(len(features))], dtype=float)
            else:
                model = RandomForestRegressor(
                    random_state=int(rng.randint(0, 1_000_000)),
                    n_estimators=400
                )
                model.fit(X, y)
                vec = np.asarray(getattr(model, "feature_importances_", np.zeros(len(features))), dtype=float)
            importances += vec
            counts += 1

    if counts > 0:
        importances /= counts
    return pd.Series(importances, index=features)


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    # Load and sanitize
    df = pd.read_excel(args.input, sheet_name=args.sheet)
    df = normalize_day_col(df, "day")

    # Separate inputs and targets
    present_targets = [t for t in TARGETS if t in df.columns]
    if not present_targets:
        raise RuntimeError("None of the expected target columns are present.")
    input_cols = [c for c in df.columns if c not in present_targets]

    # Basic correlation (all variables)
    corr_all = _safe_corr(df, input_cols + present_targets)
    order_all = _cluster_order_by_abs_corr(corr_all) if _SCIPY_OK else list(corr_all.columns)
    _plot_corr_heatmap(corr_all, order_all, args.outdir / "correlation_heatmap.png")
    corr_all.to_csv(args.outdir / "correlation_matrix.csv", index=True)

    # Input-only dendrogram (optional)
    df_inputs = df[input_cols].select_dtypes(include=[np.number]).dropna(axis=1, how="all")
    _plot_input_dendrogram(df_inputs, args.outdir / "feature_dendrogram_input.png", args.cluster_threshold)

    # Model-based importance (average over targets & repeats)
    fi_series = _compute_model_importance(df, list(df_inputs.columns), present_targets,
                                          n_repeat=args.n_repeat, seed=args.seed)
    fi_series.sort_values(ascending=False).to_excel(args.outdir / "feature_importance_global.xlsx")

    # ----- Build A / B / C sets -----
    # A-set: all input features (numeric-only columns used above)
    A_features = list(df_inputs.columns)
    pd.DataFrame({"feature": A_features}).to_excel(args.outdir / "A_features_full.xlsx", index=False)

    # B-set: one representative per cluster (if SciPy); else top-k by global importance
    if _SCIPY_OK and len(A_features) >= 2:
        corr_in = df_inputs.corr()
        dist_in = 1.0 - np.abs(corr_in.values)
        dist_in = (dist_in + dist_in.T) / 2.0
        np.fill_diagonal(dist_in, 0.0)
        Z = linkage(squareform(dist_in, checks=False), method="average")
        clusters = fcluster(Z, t=args.cluster_threshold, criterion="distance")
        cluster_df = pd.DataFrame({"feature": A_features, "cluster": clusters})
        B_features = []
        for cid in sorted(cluster_df["cluster"].unique()):
            sub_feats = cluster_df.loc[cluster_df.cluster == cid, "feature"].tolist()
            best = max(sub_feats, key=lambda f: float(fi_series.get(f, 0.0)))
            B_features.append(best)
    else:
        k = min(12, max(6, len(A_features) // 5))  # simple heuristic
        B_features = list(fi_series.sort_values(ascending=False).head(k).index)

    pd.DataFrame({"feature": B_features}).to_excel(args.outdir / "B_features_selected.xlsx", index=False)

    # C-set: fixed 6-feature practical set (ensure existence; drop missing silently)
    C_features = [f for f in C_FEATURES if f in df.columns]
    pd.DataFrame({"feature": C_features}).to_excel(args.outdir / "C_features_manual.xlsx", index=False)

    # Export convenience tables with targets appended
    df_A = df[[*A_features, *present_targets]]
    df_B = df[[*B_features, *present_targets]]
    df_C = df[[*C_features, *present_targets]]
    df_A.to_excel(args.outdir / "A_features_with_targets.xlsx", index=False)
    df_B.to_excel(args.outdir / "B_features_with_targets.xlsx", index=False)
    df_C.to_excel(args.outdir / "C_features_with_targets.xlsx", index=False)

    print(f"[OK] Saved feature-engineering outputs to: {args.outdir}")
    if XGBRegressor is None:
        print("[NOTE] xgboost not installed; global importance used RandomForest fallback.")
    if not _SCIPY_OK:
        print("[NOTE] SciPy not installed; skipped dendrogram and used a fallback for B-set.")

if __name__ == "__main__":
    main()
