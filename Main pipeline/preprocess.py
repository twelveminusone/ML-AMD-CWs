import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import ks_2samp
from scipy.spatial.distance import cdist

# ===== Threshold Parameters =====
R2_THRESHOLD = 0.9
RMSE_THRESHOLD = 30
HGBR_R2_THRESHOLD = 0.99       # Extremely high R2 required for o_acidity
HGBR_RMSE_THRESHOLD = 2        # Extremely low RMSE required for o_acidity
HOTDECK_KS_P_THRESHOLD = 0.05  # You may fine-tune this value
RANDOM_STATE = 42
CV_FOLDS = 5
SPECIAL_HGBR = {"o_acidity(mg/L)": True}
FINAL_REMOVE = ["i_TDS(mg/L)", "o_TDS(mg/L)"]

# Parameters for hot-deck multiple criteria
HOTDECK_TOP_CORR_N = 3         # Number of top correlated features for o_COD imputation
COD_KS_P = 0.01                # Criteria for o_COD
COD_MEAN_RATIO = 0.2
COD_STD_RATIO = 0.5

def safe_filename(s):
    """Replace special characters in filename."""
    return re.sub(r'[\\/*?:"<>|()\s]', '_', s)

def save_data(df, path):
    """Save DataFrame to Excel or CSV depending on file extension."""
    if path.endswith('.xlsx'):
        df.to_excel(path, index=False)
    else:
        df.to_csv(path, index=False)

def plot_violin_before_after(before, after, title, out_path):
    """Plot violin plots of feature distributions before and after imputation."""
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.violinplot(before.dropna())
    plt.title(f'{title} Before Imputation')
    plt.ylabel(title)
    plt.subplot(1,2,2)
    plt.violinplot(after.dropna(), showmeans=True)
    plt.title(f'{title} After Imputation')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def get_top_corr_features(df, target, n=3):
    """Return the top n most correlated features with the target."""
    corrs = df.corr()[target].abs().sort_values(ascending=False)
    corrs = corrs.drop(target)
    return corrs.index[:n].tolist()

def impute_hotdeck_strong_corr(df, target_col, n_corr=3, similarity='euclidean'):
    """Hot-deck imputation using top correlated features as similarity space."""
    df = df.copy()
    mask_na = df[target_col].isna()
    n_missing = mask_na.sum()
    notna_idx = df[~mask_na].index
    na_idx = df[mask_na].index
    top_corr_feats = get_top_corr_features(df, target_col, n=n_corr)
    for idx in na_idx:
        sample = df.loc[idx, top_corr_feats]
        candidates = df.loc[notna_idx, top_corr_feats]
        valid_cols = sample.dropna().index.intersection(candidates.columns)
        if len(valid_cols) == 0:
            continue
        X = sample[valid_cols].values.reshape(1, -1)
        Y = candidates[valid_cols].values
        dists = cdist(X, Y, metric=similarity)[0]
        nearest = notna_idx[np.argmin(dists)]
        df.loc[idx, target_col] = df.loc[nearest, target_col]
    return df

def judge_distribution_acceptable(before, after, ks_p_threshold=0.01, mean_ratio=0.2, std_ratio=0.5):
    """Evaluate if the distribution after imputation matches the original."""
    mean_before, mean_after = before.mean(), after.mean()
    std_before, std_after = before.std(), after.std()
    mean_change = abs(mean_after - mean_before) / (abs(mean_before) + 1e-8)
    std_change = abs(std_after - std_before) / (std_before + 1e-8)
    ks_stat, ks_p = ks_2samp(before.dropna(), after.dropna())
    return (ks_p > ks_p_threshold) and (mean_change < mean_ratio) and (std_change < std_ratio), ks_p, mean_change, std_change

def impute_rf(df, out_prefix='rf'):
    """Random Forest imputation for numerical columns with missing values."""
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    impute_cols = [col for col in numeric_cols if df[col].isnull().sum() > 0]
    eval_report = []
    na_indices_dict = {}
    for col in impute_cols:
        notna_idx = df[df[col].notna()].index
        na_idx = df[df[col].isna()].index
        na_indices_dict[col] = na_idx.copy()
        X = df.loc[notna_idx, df.columns.difference([col])]
        y = df.loc[notna_idx, col]
        X_pred = df.loc[na_idx, df.columns.difference([col])]
        if len(X_pred) == 0 or len(X) < 10:
            continue
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }
        grid = GridSearchCV(
            RandomForestRegressor(random_state=RANDOM_STATE), param_grid,
            scoring='neg_root_mean_squared_error', cv=CV_FOLDS, n_jobs=-1)
        grid.fit(X, y)
        model = grid.best_estimator_
        y_pred = model.predict(X)
        rmse = mean_squared_error(y, y_pred, squared=False)
        r2 = r2_score(y, y_pred)
        eval_report.append({"feature": col, "n_missing": len(na_idx), "rmse_train": rmse, "r2_train": r2})
        imputed = model.predict(X_pred)
        df.loc[na_idx, col] = imputed
    report = pd.DataFrame(eval_report)
    report_path = f"{out_prefix}_impute_report.csv"
    report.to_csv(report_path, index=False)
    print(f"RF imputation finished. Report saved to: {report_path}")
    return df, report, na_indices_dict

def impute_hgbr(df, target_cols, out_prefix='hgbr'):
    """Imputation using HistGradientBoostingRegressor for specified columns."""
    df = df.copy()
    eval_report = []
    na_indices_dict = {}
    for col in target_cols:
        notna_idx = df[df[col].notna()].index
        na_idx = df[df[col].isna()].index
        na_indices_dict[col] = na_idx.copy()
        X = df.loc[notna_idx, df.columns.difference([col])]
        y = df.loc[notna_idx, col]
        X_pred = df.loc[na_idx, df.columns.difference([col])]
        if len(X_pred) == 0 or len(X) < 10:
            continue
        if col in SPECIAL_HGBR:
            param_grid = {
                'max_iter': [100, 200, 500, 1000],
                'max_leaf_nodes': [15, 31, 63, 127, 255],
                'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
            }
            this_r2_thresh = HGBR_R2_THRESHOLD
            this_rmse_thresh = HGBR_RMSE_THRESHOLD
        else:
            param_grid = {
                'max_iter': [100, 200, 300],
                'max_leaf_nodes': [31, 63, 127],
                'learning_rate': [0.01, 0.05, 0.1]
            }
            this_r2_thresh = R2_THRESHOLD
            this_rmse_thresh = RMSE_THRESHOLD
        grid = GridSearchCV(
            HistGradientBoostingRegressor(random_state=RANDOM_STATE), param_grid,
            scoring='neg_root_mean_squared_error', cv=CV_FOLDS, n_jobs=-1)
        grid.fit(X, y)
        model = grid.best_estimator_
        y_pred = model.predict(X)
        rmse = mean_squared_error(y, y_pred, squared=False)
        r2 = r2_score(y, y_pred)
        eval_report.append({"feature": col, "n_missing": len(na_idx), "rmse_train": rmse, "r2_train": r2})
        imputed = model.predict(X_pred)
        df.loc[na_idx, col] = imputed
    report = pd.DataFrame(eval_report)
    report_path = f"{out_prefix}_impute_report.csv"
    report.to_csv(report_path, index=False)
    print(f"HGBR imputation finished. Report saved to: {report_path}")
    return df, report, na_indices_dict

def impute_hotdeck(df, target_cols, similarity='euclidean', out_prefix='hotdeck', output_dir='.'):
    """Hot-deck imputation for given columns."""
    df = df.copy()
    eval_report = []
    os.makedirs(output_dir, exist_ok=True)
    for col in target_cols:
        mask_na = df[col].isna()
        n_missing = mask_na.sum()
        notna_idx = df[~mask_na].index
        na_idx = df[mask_na].index
        other_cols = df.columns.difference([col])
        before_impute = df[col].copy()
        for idx in na_idx:
            sample = df.loc[idx, other_cols]
            candidates = df.loc[notna_idx, other_cols]
            valid_cols = sample.dropna().index.intersection(candidates.columns)
            if len(valid_cols) == 0:
                continue
            X = sample[valid_cols].values.reshape(1, -1)
            Y = candidates[valid_cols].values
            dists = cdist(X, Y, metric=similarity)[0]
            nearest = notna_idx[np.argmin(dists)]
            df.loc[idx, col] = df.loc[nearest, col]
        after_impute = df[col].copy()
        x = df.loc[notna_idx, col]
        y = df.loc[na_idx, col]
        if len(x) > 0 and len(y) > 0:
            ks_stat, ks_p = ks_2samp(x, y)
        else:
            ks_stat, ks_p = np.nan, np.nan
        before_mean = before_impute.mean()
        before_std = before_impute.std()
        after_mean = after_impute.mean()
        after_std = after_impute.std()
        violin_path = os.path.join(output_dir, f"violin_{safe_filename(col)}_before_after.png")
        plot_violin_before_after(before_impute, after_impute, col, violin_path)
        eval_report.append({
            "feature": col, "n_missing": n_missing,
            "ks_stat": ks_stat, "ks_p": ks_p,
            "before_mean": before_mean, "before_std": before_std,
            "after_mean": after_mean, "after_std": after_std,
            "violin_plot": violin_path
        })
    report = pd.DataFrame(eval_report)
    report_path = f"{out_prefix}_impute_report.csv"
    report.to_csv(report_path, index=False)
    print(f"Hot-deck imputation finished. Report saved to: {report_path}")
    return df, report

def get_poor_impute_features(report, r2_thresh, rmse_thresh, override_good=None):
    """Return features with unsatisfactory imputation based on thresholds."""
    poor_features = []
    for _, row in report.iterrows():
        if override_good and row['feature'] in override_good:
            continue
        if (row['r2_train'] < r2_thresh) or (row['rmse_train'] > rmse_thresh):
            poor_features.append(row['feature'])
    return poor_features

def auto_remove_features(df, features):
    """Remove specified features and their paired input/output columns."""
    to_drop = set(features)
    for f in features:
        if f.startswith('i_'):
            maybe_out = f.replace('i_', 'o_', 1)
            if maybe_out in df.columns:
                to_drop.add(maybe_out)
        if f.startswith('o_'):
            maybe_in = f.replace('o_', 'i_', 1)
            if maybe_in in df.columns:
                to_drop.add(maybe_in)
    df2 = df.drop(columns=list(to_drop), errors='ignore')
    print(f"Features removed in the final dataset: {list(to_drop)}")
    return df2

def run_ideal_preprocess(
    input_path: str,
    output_dir: str
):
    """Main workflow: RF imputation -> HGBR imputation -> Hot-deck -> Remove problematic features."""
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_excel(input_path) if input_path.endswith('.xlsx') else pd.read_csv(input_path)
    # Step 1. RF imputation
    rf_prefix = os.path.join(output_dir, 'rf')
    df_rf, rf_report, rf_na_idx = impute_rf(df, out_prefix=rf_prefix)
    rf_outfile = os.path.join(output_dir, "after_RF_fill.xlsx")
    save_data(df_rf, rf_outfile)
    print(f"RF-imputed data saved: {rf_outfile}")
    poor_rf_features = get_poor_impute_features(rf_report, R2_THRESHOLD, RMSE_THRESHOLD)
    print("Features with poor RF imputation performance:", poor_rf_features)
    for f in poor_rf_features:
        idx = rf_na_idx.get(f, [])
        df_rf.loc[idx, f] = np.nan
    rf_restore_outfile = os.path.join(output_dir, "after_RF_restore_NA.xlsx")
    save_data(df_rf, rf_restore_outfile)
    print(f"Features with poor imputation reset to NaN: {rf_restore_outfile}")
    # Step 2. HGBR imputation
    if poor_rf_features:
        hgbr_prefix = os.path.join(output_dir, 'hgbr')
        df_hgbr, hgbr_report, hgbr_na_idx = impute_hgbr(df_rf, poor_rf_features, out_prefix=hgbr_prefix)
        hgbr_outfile = os.path.join(output_dir, "after_HGBR_fill.xlsx")
        save_data(df_hgbr, hgbr_outfile)
        poor_hgbr_features = get_poor_impute_features(
            hgbr_report,
            HGBR_R2_THRESHOLD, HGBR_RMSE_THRESHOLD,
            override_good=["o_acidity(mg/L)"]
        )
        print("Features with poor HGBR imputation performance:", poor_hgbr_features)
        for f in poor_hgbr_features:
            idx = hgbr_na_idx.get(f, [])
            df_hgbr.loc[idx, f] = np.nan
        hgbr_restore_outfile = os.path.join(output_dir, "after_HGBR_restore_NA.xlsx")
        save_data(df_hgbr, hgbr_restore_outfile)
    else:
        df_hgbr = df_rf
        poor_hgbr_features = []
    # Step 3. Hot-deck imputation + o_COD enhancement
    if poor_hgbr_features:
        hotdeck_prefix = os.path.join(output_dir, 'hotdeck')
        df_hotdeck, hotdeck_report = impute_hotdeck(df_hgbr, poor_hgbr_features, out_prefix=hotdeck_prefix, output_dir=output_dir)
        hotdeck_outfile = os.path.join(output_dir, "after_HOTDECK_fill.xlsx")
        save_data(df_hotdeck, hotdeck_outfile)
        # For o_COD(mg/L), use hot-deck with strong correlation + multiple criteria
        if "o_COD(mg/L)" in poor_hgbr_features:
            print("Applying hot-deck imputation for o_COD(mg/L) using top correlated features and multi-criteria assessment...")
            df_hotdeck_cod = impute_hotdeck_strong_corr(df_hgbr, "o_COD(mg/L)", n_corr=HOTDECK_TOP_CORR_N)
            before = df_hgbr["o_COD(mg/L)"].dropna()
            after = df_hotdeck_cod["o_COD(mg/L)"].dropna()
            accept, ks_p, mean_diff, std_diff = judge_distribution_acceptable(
                before, after, ks_p_threshold=COD_KS_P, mean_ratio=COD_MEAN_RATIO, std_ratio=COD_STD_RATIO)
            print(f"o_COD distribution after imputation acceptable? {accept} (ks_p={ks_p:.3g}, mean change={mean_diff:.2%}, std change={std_diff:.2%})")
            if accept:
                df_hotdeck["o_COD(mg/L)"] = df_hotdeck_cod["o_COD(mg/L)"]
            else:
                print("o_COD distribution still problematic. Will continue to remove.")
        # Automatically check for features to be removed
        still_nan_features = []
        for _, row in hotdeck_report.iterrows():
            if row['feature'] == "o_COD(mg/L)":
                if not accept:
                    still_nan_features.append("o_COD(mg/L)")
            else:
                if row['ks_p'] < HOTDECK_KS_P_THRESHOLD:
                    still_nan_features.append(row['feature'])
    else:
        df_hotdeck = df_hgbr
        still_nan_features = []
    # Step 4. Force remove all TDS features at the end
    all_remove_features = list(set(still_nan_features + FINAL_REMOVE))
    if all_remove_features:
        df_final = auto_remove_features(df_hotdeck, all_remove_features)
    else:
        df_final = df_hotdeck
    final_outfile = os.path.join(output_dir, "full_dataset.xlsx")
    save_data(df_final, final_outfile)
    print(f"All imputation and feature removal completed! Final dataset saved: {final_outfile}")
    return df_final

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Preprocess the raw dataset and perform the full imputation pipeline.")
    HERE = Path(__file__).resolve().parent
    REPO = HERE.parent  # repository root (ML+AMD+CWs/)

    parser.add_argument("--input", type=Path, default=REPO / "data" / "initial_dataset.xlsx",
                        help="Path to the initial dataset (with missing values).")
    parser.add_argument("--outdir", type=Path, default=REPO / "data",
                        help="Directory to save the processed dataset.")
    parser.add_argument("--outfile", type=str, default="full_dataset.xlsx",
                        help="Filename for the processed dataset. Default: full_dataset.xlsx")

    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    df_final = run_ideal_preprocess(input_path=str(args.input), output_dir=str(args.outdir))

    # Save to a standardized filename in addition to whatever the function created
    final_path = args.outdir / args.outfile
    try:
        df_final.to_excel(final_path, index=False)
        print(f"[OK] Saved standardized output to: {final_path}")
    except Exception as e:
        print(f"[WARN] Failed to save standardized output: {type(e).__name__}: {e}")
