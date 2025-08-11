# Imputation-Order Sensitivity

Supplemental analyses for **imputation vs. split order** and **temporal evaluation**.
This folder contains 8 runnable scripts plus a tiny `utils/` toolbox. Everything uses
**relative paths**, **English-only comments**, and the unified time column **`day`**
(legacy `day_z` is auto-renamed by `utils/io.py`).

> **Input dataset**: `../../data/initial_dataset.xlsx` (the original file **with missing values**).  
> **Outputs**: Excel files under `../../outputs/imputation_order_sensitivity/...` with two sheets:  
> - `results` (metrics) and `meta` (dataset MD5, shapes, library versions, CLI args).  
> Optional per-fold CSVs can also be saved.

---

## Contents

```
imputation_order_sensitivity/
├─ utils/                       # shared helpers: io.py, imputers.py, meta.py, common.py
├─ imputation_sensitivity_analysis.py                   # Global once (leakage baseline)
├─ imputation_sensitivity_analysis_groupwise.py         # Groupwise: single split
├─ imputation_sensitivity_analysis_groupwise_kfold.py   # Groupwise: KFold (default 5)
├─ imputation_sensitivity_analysis_groupwise_kfold3.py  # Groupwise: KFold (3 folds)
├─ imputation_sensitivity_analysis_groupwise_timeseries_cv.py  # Groupwise: TimeSeries CV (forward-chaining)
├─ imputation_timeseries_cv_with_baselines.py                   # TimeSeries CV + naive baselines (single horizon)
├─ imputation_timeseries_cv_with_baselines_multihorizon.py      # TimeSeries CV + naive baselines (multi-horizon)
└─ make_ts_baseline_multihorizon_figs.py                        # Figures for multi-horizon results
```

### Utilities (`utils/`)
- `io.py` — load Excel → numeric, drop all-NaN cols, unify `day_z`→`day`, sort by `day`.
- `imputers.py` — factory for IterativeImputer + base estimators.
- `meta.py` — save results + meta (dataset hash, shapes, versions, args).
- `common.py` — set random seeds.

---

## Environment

From the repository root:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Quick start (run from *this* folder)

> Replace output paths as you like; all are **relative**.

1) **Global (leakage baseline)** — imputer fits on the **entire** table, then KFold simulate-missing.
```bash
python imputation_sensitivity_analysis.py   --input ../../data/initial_dataset.xlsx   --output ../../outputs/imputation_order_sensitivity/global/results.xlsx   --n_splits 5 --seed 42 --methods all --save_fold_details
```

2) **Groupwise (single split)** — fit imputer on **train only**, transform **test**.
```bash
python imputation_sensitivity_analysis_groupwise.py   --input ../../data/initial_dataset.xlsx   --output ../../outputs/imputation_order_sensitivity/groupwise/results.xlsx   --test_size 0.2 --seed 42 --methods all --save_details
```

3) **Groupwise + KFold (default 5)**  
```bash
python imputation_sensitivity_analysis_groupwise_kfold.py   --input ../../data/initial_dataset.xlsx   --output ../../outputs/imputation_order_sensitivity/groupwise_kfold/results.xlsx   --n_splits 5 --seed 42 --methods all --save_fold_details
```

4) **Groupwise + KFold = 3**  
```bash
python imputation_sensitivity_analysis_groupwise_kfold3.py   --input ../../data/initial_dataset.xlsx   --output ../../outputs/imputation_order_sensitivity/groupwise_kfold3/results.xlsx   --n_splits 3 --seed 42 --methods all --save_fold_details
```

5) **Groupwise + TimeSeries CV (forward-chaining)** — per fold: fit imputer on train window, transform test window; compute metal removal efficiency and evaluate ML models.
```bash
python imputation_sensitivity_analysis_groupwise_timeseries_cv.py   --input ../../data/initial_dataset.xlsx   --output ../../outputs/imputation_order_sensitivity/timeseries_cv/results.xlsx   --n_splits 3 --seed 42 --methods all   --feature_sets A,B,C --models XGB,RF,KNN,SVR,ANN --grid --save_fold_details
```

6) **TimeSeries CV + naive baselines (single horizon)** — compare Best-ML vs naive (Last/MA3/MA5) at horizon `H`.
```bash
python imputation_timeseries_cv_with_baselines.py   --input ../../data/initial_dataset.xlsx   --output ../../outputs/imputation_order_sensitivity/timeseries_cv_baseline/results.xlsx   --n_splits 3 --horizon 1 --seed 42 --methods all   --feature_sets A,B,C --models XGB,RF,KNN,SVR,ANN --grid   --baselines last,ma3,ma5 --save_fold_details
```

7) **TimeSeries CV + naive baselines (multi-horizon)** — horizons like `1,2,3,5,7,10`.
```bash
python imputation_timeseries_cv_with_baselines_multihorizon.py   --input ../../data/initial_dataset.xlsx   --output ../../outputs/imputation_order_sensitivity/timeseries_cv_multihorizon/results.xlsx   --n_splits 3 --horizons 1,2,3,5,7,10 --seed 42 --methods all   --feature_sets A,B,C --models XGB,RF,KNN,SVR,ANN --grid   --baselines last,ma3,ma5 --save_fold_details
```

8) **Figures for multi-horizon results** — ΔR² (BestML − BestNaive) vs horizon.
```bash
python make_ts_baseline_multihorizon_figs.py   --input ../../outputs/imputation_order_sensitivity/timeseries_cv_multihorizon/results.xlsx   --outdir ../../outputs/imputation_order_sensitivity/figures
```

---

## Common CLI options

- `--input` / `--output` — relative paths; default input is `../../data/initial_dataset.xlsx`.
- `--time_col` — defaults to `day` (legacy `day_z` is auto-handled).
- `--methods` — imputation methods: `RF`, `KNN`, `MICE`, optional `MissForest`; use `all` for every available one.
- `--feature_sets` — any subset of `A,B,C`:
  - **A**: all non-target columns.
  - **B**: a focused inflow/outflow set (`i_COD,i_pH,o_EC,o_Mn,o_TFe,o_SO42-`) when present.
  - **C**: a 6-variable practical set (`i_pH,i_COD,day,i_EC,height,i_acidity`) when present.
- `--models` — `XGB`, `RF`, `KNN`, `SVR`, `ANN`. Use `--grid` to enable a small GridSearchCV (cv=3, scoring=`r2`).
- `--save_fold_details` / `--save_details` — additionally write per-fold CSVs next to the Excel output.

> Optional dependencies: `xgboost` (for XGB) and `missforest` (for MissForest). If not installed, the scripts will skip them or raise a clear message.

---

## Reproducibility & privacy

- **No absolute paths**. All scripts accept CLI arguments and default to relative paths.
- Each run saves **`meta`**: dataset MD5, shapes, Python/pandas/numpy/sklearn versions, and the exact CLI arguments.
- Time column is unified to **`day`** across scripts to avoid naming drift.

---

## License & citation

See the repository-level `LICENSE` and `README.md`. Please cite our paper when using this code or the datasets.
