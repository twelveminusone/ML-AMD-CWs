from pathlib import Path
import hashlib, sys, platform
import pandas as pd
import numpy as np

def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def write_results_with_meta(results_df: pd.DataFrame, output_path: Path, input_path: Path, args: dict, df_shape: tuple[int,int]):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        dataset_hash = md5sum(input_path)
    except Exception:
        dataset_hash = "unavailable"

    meta = {
        "input_path": str(input_path),
        "dataset_md5": dataset_hash,
        "n_rows": df_shape[0],
        "n_cols": df_shape[1],
        **args,
        "python": sys.version.replace("\n", " "),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "sklearn": __import__("sklearn").__version__,
        "platform": platform.platform(),
    }
    with pd.ExcelWriter(output_path, engine="openpyxl") as w:
        results_df.to_excel(w, index=False, sheet_name="results")
        pd.DataFrame([meta]).to_excel(w, index=False, sheet_name="meta")
