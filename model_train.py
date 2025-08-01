import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import joblib

# ===== Parameter Section =====
data_path = r'D:\data\ML+CWs\full dataset\jupyter notebook\full dataset of 5CWs without TDS gai.xlsx'
model_dir = r'D:\data\ML+CWs\model and predict\model'
result_dir = r'D:\data\ML+CWs\model and predict\result'
os.makedirs(model_dir, exist_ok=True)
os.makedirs(result_dir, exist_ok=True)

target_names = ['TFe (%)', 'Zn (%)', 'Al (%)', 'Mn (%)', 'Ni (%)', 'Co (%)', 'Cr (%)']

# Define feature sets
A_X = [c for c in pd.read_excel(data_path, nrows=1).columns if c not in target_names]
B_X = ['i_COD', 'i_pH', 'o_EC', 'o_Mn', 'o_TFe', 'o_SO42-']
C_X = ['i_pH', 'i_COD', 'day', 'i_EC', 'height', 'i_acidity']
dataset_dict = {'A': A_X, 'B': B_X, 'C': C_X}

# Five models and their parameters
model_dict = {
    'XGB': (XGBRegressor, {
        'n_estimators': [100, 200, 300],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.05, 0.1, 0.3]
    }),
    'RF': (RandomForestRegressor, {
        'n_estimators': [100, 200, 300],
        'max_depth': [4, 6, 8]
    }),
    'KNN': (KNeighborsRegressor, {
        'n_neighbors': [3, 5, 7, 9]
    }),
    'SVR': (SVR, {
        'C': [1, 10, 100],
        'gamma': ['scale', 'auto'],
        'epsilon': [0.01, 0.1, 0.5]
    }),
    'ANN': (MLPRegressor, {
        'hidden_layer_sizes': [(50,), (100,), (100, 50)],
        'activation': ['relu'],
        'solver': ['adam'],
        'max_iter': [500]
    })
}

# ===== Main Workflow =====
data = pd.read_excel(data_path)
all_metrics = []

for ds_name, feats in dataset_dict.items():
    X_all = data[feats]
    for target in target_names:
        y_all = data[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=0.2, random_state=42)
        scaler_X = StandardScaler().fit(X_train)
        scaler_y = StandardScaler().fit(y_train.values.reshape(-1, 1))
        X_train_scaled = scaler_X.transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        y_train_scaled = scaler_y.transform(y_train.values.reshape(-1, 1)).ravel()
        y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).ravel()

        pred_save = {'model': [], 'param_mode': [], 'y_true': y_test.values, 'y_pred': [], 'train_r2': [], 'test_r2': [], 'test_rmse': [], 'test_mae': []}
        best_r2 = -np.inf
        best_model = None
        best_model_name = None
        best_param_mode = None
        best_test_pred = None
        best_test_true = None
        best_scaler_X = None
        best_scaler_y = None

        for model_name, (model_cls, param_grid) in model_dict.items():
            # -------- Default parameters --------
            try:
                model = model_cls()
                model.fit(X_train_scaled, y_train_scaled)
                y_pred_test = model.predict(X_test_scaled)
                y_pred_train = model.predict(X_train_scaled)
                y_pred_test_real = scaler_y.inverse_transform(y_pred_test.reshape(-1, 1)).ravel()
                y_pred_train_real = scaler_y.inverse_transform(y_pred_train.reshape(-1, 1)).ravel()
                test_r2 = r2_score(y_test.values, y_pred_test_real)
                test_rmse = mean_squared_error(y_test.values, y_pred_test_real, squared=False)
                test_mae = mean_absolute_error(y_test.values, y_pred_test_real)
                train_r2 = r2_score(y_train.values, y_pred_train_real)
                pred_save['model'].append(model_name)
                pred_save['param_mode'].append('default')
                pred_save['y_pred'].append(y_pred_test_real)
                pred_save['train_r2'].append(train_r2)
                pred_save['test_r2'].append(test_r2)
                pred_save['test_rmse'].append(test_rmse)
                pred_save['test_mae'].append(test_mae)
                if test_r2 > best_r2:
                    best_r2 = test_r2
                    best_model = model
                    best_model_name = model_name
                    best_param_mode = 'default'
                    best_test_pred = y_pred_test_real
                    best_test_true = y_test.values
                    best_scaler_X = scaler_X
                    best_scaler_y = scaler_y
            except Exception as e:
                print(f"{ds_name}-{target}-{model_name}-default error: {e}")

            # -------- Grid search (auto hyperparameter tuning) --------
            try:
                grid = GridSearchCV(model_cls(), param_grid, cv=5, n_jobs=-1, scoring='r2')
                grid.fit(X_train_scaled, y_train_scaled)
                model = grid.best_estimator_
                y_pred_test = model.predict(X_test_scaled)
                y_pred_train = model.predict(X_train_scaled)
                y_pred_test_real = scaler_y.inverse_transform(y_pred_test.reshape(-1, 1)).ravel()
                y_pred_train_real = scaler_y.inverse_transform(y_pred_train.reshape(-1, 1)).ravel()
                test_r2 = r2_score(y_test.values, y_pred_test_real)
                test_rmse = mean_squared_error(y_test.values, y_pred_test_real, squared=False)
                test_mae = mean_absolute_error(y_test.values, y_pred_test_real)
                train_r2 = r2_score(y_train.values, y_pred_train_real)
                pred_save['model'].append(model_name)
                pred_save['param_mode'].append('grid')
                pred_save['y_pred'].append(y_pred_test_real)
                pred_save['train_r2'].append(train_r2)
                pred_save['test_r2'].append(test_r2)
                pred_save['test_rmse'].append(test_rmse)
                pred_save['test_mae'].append(test_mae)
                if test_r2 > best_r2:
                    best_r2 = test_r2
                    best_model = model
                    best_model_name = model_name
                    best_param_mode = 'grid'
                    best_test_pred = y_pred_test_real
                    best_test_true = y_test.values
                    best_scaler_X = scaler_X
                    best_scaler_y = scaler_y
            except Exception as e:
                print(f"{ds_name}-{target}-{model_name}-grid error: {e}")

        # Save predictions for the current target
        result_df = pd.DataFrame({
            'y_true': y_test.values
        })
        for i, (m, pm, pred) in enumerate(zip(pred_save['model'], pred_save['param_mode'], pred_save['y_pred'])):
            result_df[f'{m}_{pm}_pred'] = pred
        result_df.to_excel(os.path.join(result_dir, f'{ds_name}_{target}_pred.xlsx'), index=False)

        # Save evaluation metrics
        for m, pm, tr, te, rmse, mae in zip(pred_save['model'], pred_save['param_mode'], pred_save['train_r2'], pred_save['test_r2'], pred_save['test_rmse'], pred_save['test_mae']):
            all_metrics.append({
                'Dataset': ds_name,
                'Target': target,
                'Model': m,
                'ParamMode': pm,
                'TrainR2': tr,
                'TestR2': te,
                'TestRMSE': rmse,
                'TestMAE': mae
            })

        # Save best model for C set
        if ds_name == 'C' and best_model is not None:
            joblib.dump({'model': best_model, 'scaler_X': best_scaler_X, 'scaler_y': best_scaler_y},
                        os.path.join(model_dir, f'C_{target}_{best_model_name}_{best_param_mode}.pkl'))

# Save overall metrics
pd.DataFrame(all_metrics).to_excel(os.path.join(result_dir, 'all_model_metrics.xlsx'), index=False)

# ===== Determine Best Model for C-set (main four metals) =====
df_metrics = pd.DataFrame(all_metrics)
C_metrics = df_metrics[df_metrics['Dataset'] == 'C']
main_targets = ['TFe (%)', 'Zn (%)', 'Al (%)', 'Mn (%)']

main_model_r2 = (
    C_metrics[C_metrics['Target'].isin(main_targets)]
    .groupby(['Model', 'ParamMode'])
    .agg({'TestR2':'mean'})
    .sort_values('TestR2', ascending=False)
    .reset_index()
)
best_row = main_model_r2.iloc[0]
best_model_name = best_row['Model']
best_param_mode = best_row['ParamMode']
print(f"After evaluation on four main metals, the best model for C-set is: {best_model_name} (params: {best_param_mode}), mean R²: {best_row['TestR2']:.2f}")

# ===== Plot scatter for all 7 metals (best model for C-set) =====
for target in target_names:
    pred_df = pd.read_excel(os.path.join(result_dir, f'C_{target}_pred.xlsx'))
    pred_col = [col for col in pred_df.columns if col.startswith(f'{best_model_name}_{best_param_mode}_')][0]
    y_true = pred_df['y_true']
    y_pred = pred_df[pred_col]
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, c='blue', label='Test', alpha=0.7)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'g--', label='y=x')
    r2 = r2_score(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    plt.xlabel(f'Actual {target}')
    plt.ylabel('Predicted')
    plt.title(f'Prediction for {target} (C-set)\nR²={r2:.2f}, RMSE={rmse:.2f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, f'C_{target}_{best_model_name}_{best_param_mode}_scatter.png'), dpi=300)
    plt.close()

print('All results, best models, and prediction plots have been saved!')
