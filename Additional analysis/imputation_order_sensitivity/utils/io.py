from pathlib import Path
import pandas as pd
import numpy as np

def load_numeric_df(path: Path, sheet: str | None = None, time_col: str = "day") -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet)
    # unify "day_z" -> "day"
    if time_col not in df.columns and "day_z" in df.columns:
        df = df.rename(columns={"day_z": time_col})
    # coerce numeric
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # drop all-NaN columns
    df = df.dropna(axis=1, how="all")
    # if day exists, make sure sortable
    if time_col in df.columns:
        df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
        df = df.sort_values(time_col).reset_index(drop=True)
    return df
