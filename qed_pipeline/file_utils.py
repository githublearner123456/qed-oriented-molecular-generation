import csv
import json
import os
import platform
from typing import Dict, Optional, Sequence

import pandas as pd


def save_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_csv(path: str, rows: Sequence[Dict], fieldnames: Optional[Sequence[str]] = None) -> None:
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_table(file_path: str) -> pd.DataFrame:
    ext = os.path.splitext(file_path)[1].lower()
    # Tolerate common typo ".xlxs" and read it as an Excel workbook.
    if ext in {".xlsx", ".xls", ".xlxs"}:
        return pd.read_excel(file_path, engine="openpyxl")
    if ext == ".csv":
        return pd.read_csv(file_path)
    if ext in {".tsv", ".txt"}:
        return pd.read_csv(file_path, sep="\t")
    raise ValueError("Only .xlsx, .xls, .csv, .tsv, and .txt files are supported.")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {c: str(c).strip().lower() for c in df.columns}
    return df.rename(columns=rename)


def base_environment_info() -> Dict:
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
