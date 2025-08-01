import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from xgboost import XGBRegressor

# ===== Parameters =====
input_path = r'D:\data\ML+CWs\full dataset\jupyter notebook\full dataset of 5CWs without TDS gai.xlsx'
output_dir = r'D:\data\ML+CWs\feature engineering'
os.makedirs(output_dir, exist_ok=True)

target_names = ['TFe (%)', 'Zn (%)', 'Al (%)', 'Mn (%)', 'Ni (%)', 'Co (%)', 'Cr (%)']
df = pd.read_excel(input_path)
all_features = df.columns.tolist()
input_features = [col for col in all_features if col not in target_names]

# ===== Correlation Heatmap with Star Annotations (all features) =====
corr_matrix = df[all_features].corr()
dist_all = 1 - np.abs(corr_matrix)
linkage_all = linkage(squareform(dist_all, checks=False), method='average')
dendro_all = dendrogram(linkage_all, labels=all_features, no_plot=True)
sorted_features = [all_features[int(idx)] for idx in dendro_all['leaves']]

plt.figure(figsize=(18, 15))
ax = plt.gca()
sns.heatmap(
    corr_matrix.loc[sorted_features, sorted_features],
    annot=False, cmap='coolwarm', square=True, cbar=True, ax=ax
)
for i in range(len(sorted_features)):
    for j in range(len(sorted_features)):
        value = corr_matrix.loc[sorted_features[i], sorted_features[j]]
        star = ''
        if abs(value) > 0.9:
            star = '***'
        elif abs(value) > 0.8:
            star = '**'
        elif abs(value) > 0.7:
            star = '*'
        ax.text(j + 0.5, i + 0.5, star, ha='center', va='center', color='black', fontsize=11)
plt.title('Correlation Matrix (Clustered Order, with Stars)')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'correlation_heatmap_stars.png'), dpi=300)
plt.close()

# ===== Hierarchical Clustering Dendrogram (input features only) =====
dist_input = 1 - np.abs(df[input_features].corr())
linkage_input = linkage(squareform(dist_input, checks=False), method='average')
dendro_input = dendrogram(linkage_input, labels=input_features, no_plot=True)
sorted_input = [input_features[int(idx)] for idx in dendro_input['leaves']]

plt.figure(figsize=(14, 7))
dendrogram(linkage_input, labels=input_features, color_threshold=0.4, leaf_rotation=90)
plt.title('Hierarchical Clustering Dendrogram (Input Features Only)')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'feature_dendrogram_input.png'), dpi=300)
plt.close()

# ===== XGBoost Feature Importance (Gain, mean of all targets, average of 5 repetitions) =====
N_REPEAT = 5
params = dict(n_estimators=200, learning_rate=0.1, max_depth=6)
X = df[input_features].values

feature_importance_mat_repeat = []
for repeat in range(N_REPEAT):
    feature_importance_mat = []
    for target in target_names:
        y = df[target].values
        xgb = XGBRegressor(**params, random_state=42+repeat)
        xgb.fit(X, y)
        booster = xgb.get_booster()
        score_dict = booster.get_score(importance_type='gain')
        scores = [score_dict.get(f'f{idx}', 0.0) for idx in range(len(input_features))]
        feature_importance_mat.append(scores)
    feature_importance_mat_repeat.append(np.array(feature_importance_mat))
mat = np.stack(feature_importance_mat_repeat)  # (repeat, 7, n_features)
feature_importance = mat.mean(axis=(0,1))  # Average over all targets and repetitions
fi_series = pd.Series(feature_importance, index=input_features).loc[sorted_input].sort_values(ascending=False)

# Output feature importance table
fi_series.to_excel(os.path.join(output_dir, 'feature_importance_XGB_gain_mean.xlsx'))

# Plot feature importance bar chart
plt.figure(figsize=(7, 13))
fi_series.plot(kind='barh')
plt.title('Main Feature Importances (XGBoost Gain, All Targets, Mean, 5x Avg)')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'feature_importance_XGB_gain_mean.png'), dpi=300)
plt.close()

# ===== Automatic B-set selection (hierarchical clustering + max feature importance) =====
clusters = fcluster(linkage_input, t=0.4, criterion='distance')
cluster_df = pd.DataFrame({'feature': sorted_input, 'cluster': clusters})
B_features = []
for cluster_id in sorted(cluster_df['cluster'].unique()):
    cluster_feats = cluster_df[cluster_df['cluster'] == cluster_id]['feature'].tolist()
    valid_feats = [f for f in cluster_feats if f in fi_series.index]
    if len(valid_feats) > 0:
        best_feat = fi_series[valid_feats].idxmax()
        B_features.append(best_feat)
    else:
        B_features.append(cluster_feats[0])
df_B = df[B_features + target_names]
df_B.to_excel(os.path.join(output_dir, 'B_features_selected.xlsx'), index=False)

# ===== A-set =====
df_A = df[input_features + target_names]
df_A.to_excel(os.path.join(output_dir, 'A_features_full.xlsx'), index=False)

# ===== C-set (custom) =====
C_features = ['i_pH', 'i_COD', 'day', 'i_EC', 'height', 'i_acidity']  # Custom C-set for paper
df_C = df[C_features + target_names]
df_C.to_excel(os.path.join(output_dir, 'C_features_manual.xlsx'), index=False)

print(f"\nAll feature engineering outputs have been saved to {output_dir}")
print("Main workflow: XGB gain, average across all targets and 5 repetitions. Datasets A/B/C and all charts/tables have been exported.")
