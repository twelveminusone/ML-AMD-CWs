import os
import shutil
import pandas as pd
from datetime import datetime
from zipfile import ZipFile

def copy_outputs(src_dirs, dst_dir, file_types=None):
    """Recursively collect and copy all output files from multiple source directories to a unified archive directory."""
    os.makedirs(dst_dir, exist_ok=True)
    count = 0
    for src_dir in src_dirs:
        if not os.path.exists(src_dir):
            continue
        for root, dirs, files in os.walk(src_dir):
            for file in files:
                if (file_types is None) or any(file.endswith(ext) for ext in file_types):
                    full_path = os.path.join(root, file)
                    dst_path = os.path.join(dst_dir, file)
                    shutil.copy2(full_path, dst_path)
                    count += 1
    print(f"{count} files have been archived to: {dst_dir}")

def generate_output_index(output_dir, out_file="outputs_index.md"):
    """Generate a markdown file listing all output files in the archive directory."""
    entries = []
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            rel_path = os.path.relpath(os.path.join(root, file), output_dir)
            entries.append(rel_path)
    entries.sort()
    with open(os.path.join(output_dir, out_file), 'w', encoding='utf-8') as f:
        f.write(f"# Output File Index ({datetime.now():%Y-%m-%d %H:%M})\n\n")
        for file in entries:
            f.write(f"- {file}\n")
    print(f"Output file index generated: {out_file}")

def generate_summary_report(output_dir):
    """Summarize main model metrics and produce a final evaluation table (best model, performance for key metals, top-5 feature importances)."""
    result_dir = output_dir
    metrics_file = None
    for f in os.listdir(result_dir):
        if f.endswith('all_model_metrics.xlsx'):
            metrics_file = os.path.join(result_dir, f)
            break
    if not metrics_file:
        print("all_model_metrics.xlsx not found, unable to generate the summary table.")
        return
    df = pd.read_excel(metrics_file)
    main_targets = ['TFe (%)', 'Zn (%)', 'Al (%)', 'Mn (%)']
    df_c = df[df['Dataset'] == 'C']
    best_rows = []
    for tgt in main_targets:
        row = df_c[df_c['Target'] == tgt].sort_values('TestR2', ascending=False).iloc[0]
        best_rows.append(row)
    df_best = pd.DataFrame(best_rows)
    main_model_r2 = (
        df_c[df_c['Target'].isin(main_targets)]
        .groupby(['Model', 'ParamMode'])
        .agg({'TestR2':'mean'}).sort_values('TestR2', ascending=False)
        .reset_index()
    )
    fi_file = None
    for f in os.listdir(result_dir):
        if 'feature_importance' in f and f.endswith('.xlsx'):
            fi_file = os.path.join(result_dir, f)
            break
    if not fi_file:
        fe_dir = os.path.join(os.path.dirname(result_dir), 'feature engineering')
        if os.path.exists(fe_dir):
            for f in os.listdir(fe_dir):
                if 'feature_importance' in f and f.endswith('.xlsx'):
                    fi_file = os.path.join(fe_dir, f)
                    break
    if fi_file:
        fi = pd.read_excel(fi_file, index_col=0)
        top5 = fi.head(5)
    else:
        top5 = pd.DataFrame()
    with pd.ExcelWriter(os.path.join(output_dir, 'final_summary.xlsx')) as writer:
        df_best.to_excel(writer, sheet_name='Best Model for Each Metal', index=False)
        main_model_r2.to_excel(writer, sheet_name='Average R2 for Four Major Metals', index=False)
        if not top5.empty:
            top5.to_excel(writer, sheet_name='Top-5 Feature Importances')
    print("final_summary.xlsx generated.")

def save_experiment_log(steps, output_dir, out_file="experiment_log.md"):
    """Save a markdown log of the main analysis workflow steps."""
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, out_file), 'w', encoding='utf-8') as f:
        f.write(f"# Experimental/Analysis Workflow Log ({datetime.now():%Y-%m-%d %H:%M})\n\n")
        for idx, step in enumerate(steps, 1):
            f.write(f"{idx}. {step}\n")
    print(f"Experiment workflow log saved: {out_file}")

def auto_generate_readme(output_dir, out_file="README.txt"):
    """Auto-generate a README file with a short explanation for each output file type."""
    file_map = {
        '.xlsx': 'Data table / Evaluation metrics / Feature importance / Prediction results',
        '.csv': 'Data table / Imputation report / Analysis results',
        '.png': 'Plots (e.g., feature importance, PDP, SHAP, scatter, etc.)',
        '.pdf': 'Report or printable figures',
        '.pkl': 'Saved machine learning model file, loadable for further prediction',
        '.json': 'Model structure parameters (e.g., XGBoost model)',
        '.md': 'Automatically generated file index or workflow log',
    }
    entries = []
    for f in sorted(os.listdir(output_dir)):
        ext = os.path.splitext(f)[-1]
        desc = file_map.get(ext, 'Other')
        entries.append(f"- {f}: {desc}")
    with open(os.path.join(output_dir, out_file), 'w', encoding='utf-8') as f:
        f.write("Project output auto-archive directory\n\n")
        for line in entries:
            f.write(line + "\n")
        f.write("\nAll models and prediction files are traceable and reproducible. For details, see experiment_log.md.\n")
    print(f"README generated: {out_file}")

def archive_project_results(src_project_dir, archive_zip="project_outputs.zip"):
    """Zip all output files for easy archiving and sharing."""
    with ZipFile(archive_zip, 'w') as zipf:
        for root, dirs, files in os.walk(src_project_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, src_project_dir)
                zipf.write(file_path, arcname)
    print(f"Project results zipped as: {archive_zip}")

if __name__ == "__main__":
    src_dirs = [
        r"D:\data\ML+CWs\full dataset",
        r"D:\data\ML+CWs\feature engineering",
        r"D:\data\ML+CWs\model and predict\result",
        r"D:\data\ML+CWs\model and predict\model",
        r"D:\data\ML+CWs\interpret\feature importance",
        r"D:\data\ML+CWs\interpret\PDP"
    ]
    dst_dir = r"D:\data\ML+CWs\reporting"
    file_types = [".csv", ".xlsx", ".png", ".pdf", ".pkl", ".json"]

    steps = [
        "Data preprocessing: multiple rounds of imputation, abnormal feature removal, standardization",
        "Feature engineering: correlation analysis, hierarchical clustering and feature importance selection, output of A/B/C feature sets",
        "Model training and hyperparameter tuning: five models, grid search with cross-validation",
        "Model prediction and evaluation: output prediction results and evaluation metrics for all targets and model-parameter combinations",
        "Automatic selection and saving of best model, summary of performance for main metals",
        "Interpretability analysis: feature importance (XGB/SHAP), PDP, summary plots, etc.",
        "Automatic archiving of all outputs, models, figures, and reports"
    ]

    copy_outputs(src_dirs, dst_dir, file_types=file_types)
    generate_output_index(dst_dir, out_file="outputs_index.md")
    generate_summary_report(dst_dir)
    save_experiment_log(steps, dst_dir, out_file="experiment_log.md")
    auto_generate_readme(dst_dir, out_file="README.txt")
    archive_project_results(dst_dir, archive_zip=os.path.join(dst_dir, "project_outputs.zip"))
