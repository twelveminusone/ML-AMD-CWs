# ML-AMD-CWs: Machine Learning for Constructed Wetlands Treating Acid Mine Drainage

This repository contains the complete workflow code, evaluation scripts, and result archiving system for the research article:

> **Critical operational parameters for metal removal efficiency in acid mine drainage treated by constructed wetlands: An explainable machine learning approach**  
> *Jingkang Zhang, Xingjie Wang, Liyuan Ma, et al.*

---

## Overview

This project predicts the removal efficiency of multiple heavy metals (Fe, Zn, Al, Mn, Ni, Co, Cr) in constructed wetlands (CWs) treating acid mine drainage (AMD) using a suite of machine learning models.  
The pipeline includes data preprocessing, feature engineering, model training, evaluation, interpretability analysis (XGBoost/SHAP/PDP), and fully automated result archiving and documentation.

- **Data Preprocessing:** Advanced imputation (RF, HGBR, hot-deck), abnormal feature removal, dataset cleaning.
- **Feature Engineering:** Correlation analysis, hierarchical clustering, multi-target feature importance selection (XGBoost).
- **Model Training:** Five algorithms (XGBoost, Random Forest, KNN, SVR, ANN) with auto hyperparameter tuning and cross-validation.
- **Model Evaluation:** Metrics export for each model/target, best model auto-detection, visual prediction assessment.
- **Interpretability:** SHAP value analysis, feature importance ranking, partial dependence plots (1D/2D).
- **Reporting:** Automated archiving, summary tables, workflow logs, and reproducibility README.

---

## Project Structure

```plaintext
ML-AMD-CWs/
│
├── preprocess.py              # Data imputation and preprocessing
├── feature_engineering.py     # Correlation, clustering, feature selection
├── model_train.py             # Model training, tuning, and evaluation
├── interpret.py               # Interpretability analysis (SHAP, XGB, PDP)
├── reporting.py               # Automated result collection, summary, README/ZIP
│
├── requirements.txt           # Python dependencies
├── README.md                  # This documentation file
├── LICENSE                    # Open-source license (MIT recommended)
│
├── data/                      # (Optional) Example data or data structure templates
├── outputs/                   # (Generated) Collected results, models, and figures
└── docs/                      # (Optional) Additional documentation or figures
```
---

## Quick Start

1. **Install dependencies**
   Python 3.8+ is recommended.

   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare your data**

   * Place your original or preprocessed dataset in the correct directory (edit script paths if needed).
   * A sample data structure is provided in `data/`.

3. **Run the workflow**
   Execute the scripts in order:

   ```bash
   python preprocess.py
   python feature_engineering.py
   python model_train.py
   python interpret.py
   python reporting.py
   ```

   All results and model outputs will be automatically archived in `outputs/` (or your specified output path).

4. **Review outputs**

   * Key outputs: model metrics, best model summary, feature importances, SHAP plots, PDP plots, prediction results, and a workflow log.
   * Summary and file index: see `outputs/README.txt`.

---

## Main Scripts and Functions

| Script                | Functionality                                              |
|-----------------------|-----------------------------------------------------------|
| preprocess.py         | Data imputation (RF/HGBR/hot-deck), cleaning, export      |
| feature_engineering.py| Feature clustering, correlation, XGBoost importance       |
| model_train.py        | Train/evaluate 5 ML models for all metals, auto tuning    |
| interpret.py          | SHAP/XGB importance, summary plots, 1D/2D PDP             |
| reporting.py          | Archive all results, generate summary/index/ZIP           |


## Data Availability

Due to privacy or size limits, full datasets are not included. See the original paper for supplementary data, or contact the corresponding author.


## Citation

If you use this code or workflow, please cite our paper:

Jingkang Zhang, Xingjie Wang, Liyuan Ma, et al.  
Critical operational parameters for metal removal efficiency in acid mine drainage treated by constructed wetlands: An explainable machine learning approach


## License

This repository is open-source under the MIT License. See [LICENSE](./LICENSE) for details.


## Contact

For questions or collaboration, please contact:
- Jingkang Zhang (倞康 张): [zjk1202321889@cug.edu.cn]
- Or open an issue on GitHub.

## Example: Adding Images or Diagrams

You can embed images (e.g., research flowcharts, model architecture, data examples) and share sample data in your README.

### Research Flowchart

![Research Workflow](GitHub/Flowchart.tif)

- The above figure shows the overall workflow of this research, from raw data processing to model interpretation.

### Example Datasets

#### Initial Dataset

[Download initial dataset](GitHub/initial%20dataset.xlsx)

#### Full Dataset

[Download full dataset](GitHub/full%20dataset.xlsx)

---

