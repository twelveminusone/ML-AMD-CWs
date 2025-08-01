import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['figure.max_open_warning'] = 100

import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
import shap
from sklearn.inspection import PartialDependenceDisplay

# =================== Path and Parameter Settings ===================
input_path = r'D:\data\ML+CWs\full dataset\jupyter notebook\full dataset of 5CWs without TDS gai.xlsx'
imp_dir = r'D:\data\ML+CWs\interpret\feature importance'
imp_model_dir = r'D:\data\ML+CWs\interpret\feature importance\model'
pdp_dir = r'D:\data\ML+CWs\interpret\PDP'
os.makedirs(imp_dir, exist_ok=True)
os.makedirs(imp_model_dir, exist_ok=True)
os.makedirs(pdp_dir, exist_ok=True)

input_feats = ['i_pH', 'i_COD', 'day', 'i_EC', 'height', 'i_acidity']
target_feats_raw = ['TFe (%)', 'Zn (%)', 'Al (%)', 'Mn (%)', 'Ni (%)', 'Co (%)', 'Cr (%)']
target_feats = [name.split(' ')[0] for name in target_feats_raw]

# =================== Load Data and Define C Dataset ===================
df = pd.read_excel(input_path)
df_targets = df[target_feats_raw].copy()
df_targets.columns = target_feats
X = df[input_feats]
Y = df_targets

# =================== XGBoost Model Parameter Settings ===================
xgb_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [4, 6],
    'learning_rate': [0.1, 0.3],
    'subsample': [0.8, 1.0],
}
random_state = 42

xgb_best_models = []
xgb_importances = []
shap_values_list = []

# =================== Train a Model for Each Target ===================
for idx, target in enumerate(target_feats):
    y = Y[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state)
    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=random_state)
    grid = GridSearchCV(model, xgb_param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    xgb_best_models.append(best_model)
    xgb_importances.append(best_model.feature_importances_)
    # Save the model for each metal
    model_json = os.path.join(imp_model_dir, f'C_{target}_XGB_grid.json')
    best_model.save_model(model_json)
    # SHAP values
    explainer = shap.Explainer(best_model, X)
    shap_values = explainer(X)
    shap_values_list.append(shap_values.values)
    # SHAP summary plot for each target
    plt.figure(figsize=(7, 4))
    shap.summary_plot(shap_values.values, X, plot_type="dot", show=False)
    plt.title(f"SHAP Summary Plot for {target}", fontname='Arial')
    plt.tight_layout()
    plt.savefig(os.path.join(imp_dir, f"C_{target}_SHAP_summary.png"), dpi=300)
    plt.close()
    # XGB feature importance bar plot for each target
    plt.figure(figsize=(7, 4))
    plt.barh(input_feats, best_model.feature_importances_, color='#82b8d4')
    plt.xlabel('Importance', fontname='Arial')
    plt.title(f'Feature Importances (XGBoost) for {target}', fontname='Arial')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(imp_dir, f"C_{target}_XGBimp_gain.png"), dpi=300)
    plt.close()

# =========== Overall XGB Importance Bar Plot (mean over all targets) ===========
xgb_importances_arr = np.array(xgb_importances)
xgb_mean_imp = np.mean(xgb_importances_arr, axis=0)
plt.figure(figsize=(7, 5))
plt.barh(input_feats, xgb_mean_imp, color='#589ad3')
plt.xlabel('Importance', fontname='Arial')
plt.title('Feature Importances (XGBoost, All Targets)', fontname='Arial')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(imp_dir, 'C_XGBimp_overall_gain.png'), dpi=300)
plt.close()

# =========== Overall SHAP Summary Swarm Plot (all targets combined) ===========
shap_values_all = np.vstack(shap_values_list)
X_all = pd.concat([X] * len(target_feats), ignore_index=True)
plt.figure(figsize=(8, 5))
shap.summary_plot(shap_values_all, X_all, plot_type="dot", show=False)
plt.title("Overall SHAP Summary Plot for All Targets", fontname='Arial')
plt.tight_layout()
plt.savefig(os.path.join(imp_dir, "C_SHAP_summary_overall.png"), dpi=300)
plt.close()

# =========== Compute Combined Importance (XGB+SHAP), select top 2 for 2D PDP ===========
shap_values_abs = np.abs(shap_values_all)
mean_shap_imp = shap_values_abs.mean(axis=0)
xgb_norm = xgb_mean_imp / (np.max(xgb_mean_imp) + 1e-8)
shap_norm = mean_shap_imp / (np.max(mean_shap_imp) + 1e-8)
combined_imp = 0.5 * xgb_norm + 0.5 * shap_norm
sorted_idx = np.argsort(-combined_imp)
feat1, feat2 = input_feats[sorted_idx[0]], input_feats[sorted_idx[1]]

print("X_train type:", type(X_train))
print("X_train columns:", X_train.columns.tolist())
print("feat1:", feat1)
print("feat2:", feat2)

# =========== PDP Analysis (1D & 2D, top 2 by combined XGB+SHAP) ===========
for idx, target in enumerate(target_feats):
    y = Y[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state)
    best_model = xgb_best_models[idx]
    # 1D PDP
    for feat in input_feats:
        plt.figure(figsize=(5, 3))
        try:
            PartialDependenceDisplay.from_estimator(best_model, X_train, [feat])
            plt.title(f'{feat} → {target} (PDP)', fontname='Arial')
            plt.xlabel(feat, fontname='Arial')
            plt.ylabel(target, fontname='Arial')
            plt.tight_layout()
            plt.savefig(os.path.join(pdp_dir, f'C_{target}_1D_PDP_{feat}.png'), dpi=200)
            plt.close()
        except Exception as e:
            print(f'PDP failed for {feat} of {target}: {e}')
    # 2D PDP
    try:
        PartialDependenceDisplay.from_estimator(
            best_model, X_train, [(feat1, feat2)], grid_resolution=30
        )
        ax = plt.gca()
        # Attempt to automatically add colorbar
        quadmesh = None
        for child in ax.get_children():
            if 'QuadMesh' in str(type(child)):
                quadmesh = child
                break
        if quadmesh is not None:
            plt.colorbar(quadmesh, ax=ax)
        elif hasattr(ax, 'collections') and ax.collections:
            plt.colorbar(ax.collections[0], ax=ax)
        plt.title(f'2D PDP: {feat1} & {feat2} → {target}', fontname='Arial')
        plt.xlabel(feat1, fontname='Arial')
        plt.ylabel(feat2, fontname='Arial')
        plt.tight_layout()
        plt.savefig(os.path.join(pdp_dir, f'C_{target}_2D_PDP_{feat1}_{feat2}.png'), dpi=200)
        plt.close()
    except Exception as e:
        print(f'2D PDP failed for {target}: {e}')

print('All interpretability analyses completed! XGBoost/SHAP feature importances, summary plots, and PDP (1D/2D) have all been saved.')
