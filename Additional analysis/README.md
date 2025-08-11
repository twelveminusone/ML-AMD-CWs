# Additional analyses

This folder contains all **supplemental analyses** added beyond the main pipeline to improve transparency and address reviewer requests. Every script uses **relative paths**, **English-only comments**, and a unified time column **`day`** (legacy `day_z` is auto-renamed by shared utilities). Each run writes an Excel file with two sheets: **`results`** and **`meta`** (dataset hash, library versions, CLI args).

> Install dependencies at the repository root: `pip install -r requirements.txt`.

---

## Layout

```
Additional analysis/
├─ imputation_order_sensitivity/      # 8 scripts + utils/ (shared helpers)
├─ input_perturbation/                # input perturbation sensitivity (3 scripts)
├─ feature_importance_comparison.py   # PI / LIME / Occlusion
├─ LOGO_cv_analysis.py                # Leave-One-Group-Out CV across wetlands
└─ Missing_Value_Imputation_Model_Comparison.py
```

The subfolder `imputation_order_sensitivity/utils/` is a small shared toolbox (`io.py`, `meta.py`, `common.py`, `imputers.py`). Other analyses reuse it via a tiny `sys.path` shim at the top of their scripts (no absolute paths). Ensure `utils/__init__.py` exists.

---

## 1) Imputation-order sensitivity (`imputation_order_sensitivity/`)

**Goal:** Evaluate how the **order** of *imputation vs. data splitting* affects results, including KFold and forward-chaining (time series) setups, with naive baselines and multi-horizon analyses.

- **Data:** `../data/initial_dataset.xlsx` (contains missing values).
- **Outputs:** `../outputs/imputation_order_sensitivity/...`
- **Scripts:** global leakage baseline, groupwise split, KFold, TimeSeries CV, baselines (single & multi-horizon), and figure generation.
- **Note:** Other analyses reuse this folder’s `utils/`.

**Example:**

```bash
# from this subfolder
python imputation_sensitivity_analysis_groupwise_kfold.py   --input ../../data/initial_dataset.xlsx   --output ../../outputs/imputation_order_sensitivity/groupwise_kfold/results.xlsx
```

---

## 2) Input perturbation sensitivity (`input_perturbation/`)

**Goal:** Test **inference-time robustness** by multiplying one input feature at a time (e.g., ±10%) and observing metric changes without retraining.

- **Data:** `../data/full_dataset.xlsx` (fully imputed).
- **Outputs:** `../outputs/input_perturbation_sensitivity/`
- **Scripts:** 
  - `input_perturb_sensitivity.py` (core experiment), 
  - `make_input_perturb_heatmap.py` (Δ heatmap), 
  - `make_input_perturb_lineplots.py` (per-target line plots).
- **Model:** XGB preferred; falls back to RF when xgboost is unavailable.

**Example:**

```bash
# from this subfolder
python input_perturb_sensitivity.py   --input ../../data/full_dataset.xlsx   --output ../../outputs/input_perturbation_sensitivity/results.xlsx   --scales 0.9,1.1 --grid
```

---

## 3) Comparative feature importance (`feature_importance_comparison.py`)

**Goal:** Compare three model-agnostic importance measures on a fixed split per target.

- **Methods:** Permutation Importance (sklearn), **LIME** (optional dependency), **Occlusion** (set one feature to zero at inference; report ΔR²).
- **Data:** `../data/full_dataset.xlsx` (fully imputed).
- **Outputs:** `../outputs/feature_importance_comparison/` (+ optional figures with `--fig`).
- **Model:** XGB preferred; falls back to RF.

**Example:**

```bash
# from this folder (Additional analysis/)
python feature_importance_comparison.py   --input ../data/full_dataset.xlsx   --output ../outputs/feature_importance_comparison/results.xlsx   --features i_COD,i_pH,i_acidity,i_EC,day,height --grid --fig
```

---

## 4) Leave-One-Group-Out CV (`LOGO_cv_analysis.py`)

**Goal:** Measure generalization across wetlands by training on all-but-one **group** and testing on the left-out group.

- **Grouping column:** `wetland_ID`.
- **Data:** `../data/full_dataset_for_LOGO.xlsx` (same variables as `full_dataset.xlsx` + first column `wetland_ID`).
- **Outputs:** `../outputs/logo_cv/` (+ optional wide matrices with `--save_wide`).

**Example:**

```bash
# from this folder (Additional analysis/)
python LOGO_cv_analysis.py   --input ../data/full_dataset_for_LOGO.xlsx   --output ../outputs/logo_cv/results.xlsx   --group_col wetland_ID --grid --save_wide
```

---

## 5) Missing-value imputation model comparison (`Missing_Value_Imputation_Model_Comparison.py`)

**Goal:** Compare multiple **imputation models** and select the best strategy for filling missing entries before downstream ML.

- **Data:** `../data/initial_dataset.xlsx` (with missing values).
- **Outputs:** `../outputs/missing_value_imputation/` (results + meta).

**Example:**

```bash
# from this folder (Additional analysis/)
python Missing_Value_Imputation_Model_Comparison.py   --input ../data/initial_dataset.xlsx   --output ../outputs/missing_value_imputation/results.xlsx
```

---

### Notes
- All scripts accept CLI arguments; defaults target the relative paths above.
- Optional dependencies (e.g., `xgboost`, `lime`) are detected at runtime. If not installed, the script either falls back (RF) or skips the method with a clear warning, and the `meta` sheet records this.
- Put generated files under `../outputs/` (ignored by git via top-level `.gitignore`).

