# Main pipeline

This folder contains the **primary end‑to‑end workflow** used in the paper: data preprocessing → feature engineering → model training → interpretability. All scripts follow the open‑source principles used across the repo: **relative paths**, **English‑only comments**, and a unified time column **`day`** (legacy `day_z` is auto‑renamed inside the scripts).

> Install dependencies at the repository root: `pip install -r requirements.txt`.

---

## Data I/O

- **Input (raw with missing values):** `../data/initial_dataset.xlsx`
- **Output (fully imputed/processed):** `../data/full_dataset.xlsx` (created by `preprocess.py`)
- All run artifacts are written under `../outputs/` (ignored by git via the top‑level `.gitignore`).

---

## Scripts

### 1) `preprocess.py`
Preprocess the initial dataset and write a standardized fully‑imputed file.

**Default run (from `Main pipeline/`):**
```bash
python preprocess.py   --input ../data/initial_dataset.xlsx   --outdir ../data   --outfile full_dataset.xlsx
```
**Notes**
- Only the CLI and paths were changed; internal logic is preserved.
- The final file is named `full_dataset.xlsx` to keep a stable contract for downstream steps.

---

### 2) `feature_engineering.py`
Correlation matrix (optionally clustered), optional dendrogram, optional model‑based global importance, and export of A/B/C feature sets.

**Default run:**
```bash
python feature_engineering.py   --input ../data/full_dataset.xlsx   --outdir ../outputs/feature_engineering
```
**Outputs (examples)**
- `correlation_heatmap.png`, `correlation_matrix.csv`
- `feature_dendrogram_input.png` (if SciPy available)
- `feature_importance_global.xlsx` (XGB gain or RF fallback)
- `A_features_full.xlsx`, `B_features_selected.xlsx`, `C_features_manual.xlsx`
- Convenience tables: `*_features_with_targets.xlsx`

---

### 3) `model_train.py`
Train **XGB / RF / KNN / SVR / ANN** on **A/B/C** feature sets using a single split; save per‑target predictions, a metrics summary, and the best model artifact + scatter plot.

**Default run:**
```bash
python model_train.py   --input ../data/full_dataset.xlsx   --outdir ../outputs/model_and_predict   --test_size 0.2 --seed 42
```
**Outputs (examples)**
- `result/ALL_metrics.xlsx` (per feature set × model × target)
- `result/{SET}_{Target}_{Model}_{param}_scatter.png`
- `result/{SET}_{Target}_pred.xlsx` (y_true + per‑model predictions)
- `model/{SET}_{Target}_{Model}_{param}.joblib`

---

### 4) `interpret.py`
For each target, fit XGBoost with a small grid, export **XGB importances**, **SHAP** summaries, and **PDP (1D/2D)**. Uses the practical **C feature set** by default: `i_pH,i_COD,day,i_EC,height,i_acidity`.

**Default run:**
```bash
python interpret.py   --input ../data/full_dataset.xlsx   --outdir ../outputs/interpretability   --test_size 0.2 --seed 42
```
**Dependencies**
- Requires `xgboost` and `shap`. If not installed: `pip install xgboost shap`.

**Outputs (examples)**
- `imp_model/C_{Target}_XGB_grid.json`
- `importances/C_{Target}_XGB_importance_gain.png` and `C_XGB_importance_overall_gain.*`
- `shap/C_{Target}_SHAP_summary.png` and `C_SHAP_summary_all_targets.png`
- `pdp/C_{Target}_PDP_1D.png`, `pdp/C_{Target}_PDP_2D_{feat1}_{feat2}.png`

---

## End‑to‑end quick start

```bash
# 1) Preprocess (creates ../data/full_dataset.xlsx)
python preprocess.py --input ../data/initial_dataset.xlsx --outdir ../data --outfile full_dataset.xlsx

# 2) Feature engineering
python feature_engineering.py --input ../data/full_dataset.xlsx --outdir ../outputs/feature_engineering

# 3) Model training
python model_train.py --input ../data/full_dataset.xlsx --outdir ../outputs/model_and_predict --test_size 0.2 --seed 42

# 4) Interpretability (XGB + SHAP + PDP)
python interpret.py --input ../data/full_dataset.xlsx --outdir ../outputs/interpretability --test_size 0.2 --seed 42
```

---

## Reproducibility & notes

- All scripts accept CLI arguments and default to **relative repo paths**.  
- Figures/tables are written under `../outputs/` and are **ignored by git** to keep the repo lightweight.  
- Feature set C is kept fixed for interpretability to mirror the paper’s main configuration.  
- The scripts are self‑contained; no `utils/` is required inside this folder.
