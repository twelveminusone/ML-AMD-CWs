#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Project-level reporting aggregator (open-source version)
-------------------------------------------------------
Scan one or more source directories (recursively) for Excel outputs, read the
'results' and 'meta' sheets (when present), and build lightweight indexes and
summaries under a single reporting folder.

Example (from repo root):
    python reporting.py --sources ./outputs --out ./outputs/reporting --zip
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate analysis outputs into project-level indexes and summaries.")
    p.add_argument("--sources", type=str, nargs="+", default=["./outputs"],
                   help="One or more directories or glob patterns to scan recursively for Excel files.")
    p.add_argument("--pattern", type=str, default="**/*.xlsx",
                   help="Glob pattern (relative to each source) for files to parse. Default: **/*.xlsx")
    p.add_argument("--results_sheet", type=str, default="results", help="Sheet name for results.")
    p.add_argument("--meta_sheet", type=str, default="meta", help="Sheet name for meta info.")
    p.add_argument("--out", type=Path, default=Path("./outputs/reporting"), help="Output directory for summaries.")
    p.add_argument("--zip", action="store_true", help="Zip the reporting folder after creation.")
    return p.parse_args()


def _infer_analysis_label(rel_path: str) -> str:
    rel = rel_path.lower()
    mapping = [
        ("imputation_order_sensitivity", "imputation_order_sensitivity"),
        ("input_perturbation_sensitivity", "input_perturbation_sensitivity"),
        ("feature_engineering", "feature_engineering"),
        ("model_and_predict", "model_and_predict"),
        ("interpretability", "interpretability"),
        ("logo_cv", "logo_cv"),
        ("feature_importance_comparison", "feature_importance_comparison"),
        ("missing_value_imputation", "missing_value_imputation"),
    ]
    for key, label in mapping:
        if key in rel:
            return label
    return "unknown"


def _try_read_results(path: Path, sheet: str):
    info = {"has_results": False, "n_rows": 0, "n_cols": 0, "columns": []}
    try:
        df = pd.read_excel(path, sheet_name=sheet)
        if isinstance(df, dict):  # if sheet=None accidentally
            df = list(df.values())[0]
        info["has_results"] = True
        info["n_rows"] = int(df.shape[0])
        info["n_cols"] = int(df.shape[1])
        info["columns"] = [str(c) for c in df.columns]
        return df, info
    except Exception:
        return pd.DataFrame(), info


def _try_read_meta(path: Path, sheet: str):
    info = {"has_meta": False}
    meta_kv: Dict[str, str] = {}
    try:
        m = pd.read_excel(path, sheet_name=sheet)
        if isinstance(m, dict):
            m = list(m.values())[0]
        info["has_meta"] = True
        if m.shape[1] >= 2:
            kcol, vcol = m.columns[:2]
            for k, v in zip(m[kcol], m[vcol]):
                if pd.isna(k):
                    continue
                meta_kv[str(k)] = "" if pd.isna(v) else str(v)
        else:
            meta_kv["__raw__"] = m.to_json(orient="records")
    except Exception:
        pass
    return meta_kv, info


def _summarize_metrics(df: pd.DataFrame):
    """Compute simple means/medians for common metric columns if present."""
    summary = {}
    candidates = [
        "R2", "RMSE", "MAE",
        "train_r2", "test_r2", "test_rmse", "test_mae",
        "r2", "rmse", "mae",
    ]
    for col in candidates:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").values
            if vals.size:
                summary[f"mean_{col}"] = float(np.nanmean(vals))
                summary[f"median_{col}"] = float(np.nanmedian(vals))
    return summary


def gather_files(sources: List[str], pattern: str) -> List[Path]:
    files: List[Path] = []
    for src in sources:
        if any(ch in src for ch in ["*", "?", "["]):  # glob pattern
            files.extend([p for p in Path(".").glob(src)])
        else:
            base = Path(src)
            if base.is_file():
                files.append(base)
            elif base.exists():
                files.extend(base.glob(pattern))
    uniq = sorted({f.resolve() for f in files if f.suffix.lower() == ".xlsx"})
    return list(uniq)


def main() -> None:
    args = parse_args()
    outdir = args.out
    outdir.mkdir(parents=True, exist_ok=True)

    files = gather_files(args.sources, args.pattern)
    if not files:
        print("[WARN] No Excel files found. Check --sources and --pattern.")
        return

    index_rows = []
    metrics_rows = []
    meta_rows = []

    for fpath in files:
        try:
            rel = str(fpath.resolve().relative_to(Path('.').resolve()))
        except Exception:
            rel = str(fpath)

        label = _infer_analysis_label(rel)
        res_df, res_info = _try_read_results(fpath, args.results_sheet)
        meta_kv, meta_info = _try_read_meta(fpath, args.meta_sheet)

        index_rows.append({
            "file": str(fpath),
            "rel_path": rel,
            "analysis": label,
            **res_info,
            **meta_info,
        })

        if res_info["has_results"] and not res_df.empty:
            msum = _summarize_metrics(res_df)
            metrics_rows.append({"file": str(fpath), "rel_path": rel, "analysis": label, **msum})

        if meta_info["has_meta"] and meta_kv:
            row = {"file": str(fpath), "rel_path": rel, "analysis": label}
            for k, v in meta_kv.items():
                row[f"meta.{k}"] = v
            meta_rows.append(row)

    df_index = pd.DataFrame(index_rows).sort_values(["analysis", "rel_path"])
    df_metrics = pd.DataFrame(metrics_rows).sort_values(["analysis", "rel_path"])
    df_meta = pd.DataFrame(meta_rows).sort_values(["analysis", "rel_path"])

    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    index_path = outdir / f"report_index_{ts}.xlsx"
    metrics_path = outdir / f"report_metrics_summary_{ts}.xlsx"
    meta_path = outdir / f"report_meta_summary_{ts}.xlsx"

    with pd.ExcelWriter(index_path, engine="openpyxl") as w:
        df_index.to_excel(w, index=False, sheet_name="index")

    with pd.ExcelWriter(metrics_path, engine="openpyxl") as w:
        df_metrics.to_excel(w, index=False, sheet_name="metrics")

    with pd.ExcelWriter(meta_path, engine="openpyxl") as w:
        df_meta.to_excel(w, index=False, sheet_name="meta")

    # Write a small README for the reporting output
    readme = outdir / "README_reporting.txt"
    txt = (
        "This folder was generated by reporting.py. Files:\n"
        f"- {index_path.name}: one row per Excel file with presence of 'results'/'meta'.\n"
        f"- {metrics_path.name}: means/medians of common metric columns per file.\n"
        f"- {meta_path.name}: flattened key/value pairs from 'meta' sheets.\n"
        "Notes: paths are relative to the repository root where the script was executed.\n"
    )
    readme.write_text(txt, encoding="utf-8")

    if args.zip:
        zip_path = outdir.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for p in [index_path, metrics_path, meta_path, readme]:
                zf.write(p, arcname=p.name)
        print(f"[OK] Wrote zip: {zip_path}")

    print(f"[OK] Reporting written to: {outdir}")
    print(f"[OK] Files: {index_path.name}, {metrics_path.name}, {meta_path.name}")


if __name__ == "__main__":
    main()
