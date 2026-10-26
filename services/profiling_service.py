"""Dataset statistics, column profiling and table previews."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

MAX_CELL_CHARS = 80


def is_numeric_column(series: pd.Series) -> bool:
    """True for integer/float columns (booleans are excluded)."""
    return ptypes.is_numeric_dtype(series) and not ptypes.is_bool_dtype(series)


def is_text_column(series: pd.Series) -> bool:
    """True for object or pandas string columns."""
    return series.dtype == object or isinstance(series.dtype, pd.StringDtype)


def is_datetime_column(series: pd.Series) -> bool:
    """True for timezone-naive or timezone-aware datetime columns."""
    return ptypes.is_datetime64_any_dtype(series)


def friendly_dtype(series: pd.Series) -> str:
    """Describe a column's type in beginner-friendly words."""
    if ptypes.is_bool_dtype(series):
        return "boolean"
    if ptypes.is_integer_dtype(series):
        return "integer"
    if ptypes.is_float_dtype(series):
        return "decimal"
    if is_datetime_column(series):
        return "datetime"
    if is_text_column(series):
        return "text"
    return str(series.dtype)


def to_native(value: Any) -> Any:
    """Convert numpy/pandas scalars to JSON-friendly Python values."""
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(value))
    return value


def count_duplicate_rows(df: pd.DataFrame, subset: list[str] | None = None) -> int:
    """Number of rows that repeat an earlier row (optionally on ``subset`` columns)."""
    if df.empty:
        return 0
    return int(df.duplicated(subset=subset or None, keep="first").sum())


def detect_mixed_types(series: pd.Series) -> str | None:
    """Return a warning if a text column mixes numeric-looking and other values."""
    if not is_text_column(series):
        return None
    values = series.dropna()
    if values.empty:
        return None
    as_text = values.astype(str).str.strip()
    as_text = as_text[as_text != ""]
    if as_text.empty:
        return None
    numeric_like = int(pd.to_numeric(as_text, errors="coerce").notna().sum())
    other = len(as_text) - numeric_like
    if numeric_like and other:
        return (
            f"Column '{series.name}' mixes {numeric_like} numeric-looking value(s) "
            f"with {other} non-numeric value(s). Consider converting its type."
        )
    return None


def numeric_stats(series: pd.Series) -> dict[str, float | None] | None:
    """Minimum, maximum, mean and median for a numeric column."""
    if not is_numeric_column(series):
        return None
    values = series.dropna()
    if values.empty:
        return {"min": None, "max": None, "mean": None, "median": None}
    return {
        "min": to_native(values.min()),
        "max": to_native(values.max()),
        "mean": to_native(values.mean()),
        "median": to_native(values.median()),
    }


def profile_column(series: pd.Series, total_rows: int) -> dict[str, Any]:
    """Profile a single column."""
    missing = int(series.isna().sum())
    return {
        "name": str(series.name),
        "dtype": friendly_dtype(series),
        "raw_dtype": str(series.dtype),
        "missing": missing,
        "missing_pct": round(missing / total_rows * 100, 1) if total_rows else 0.0,
        "unique": int(series.nunique(dropna=True)),
        "is_numeric": is_numeric_column(series),
        "is_text": is_text_column(series),
        "stats": numeric_stats(series),
        "warning": detect_mixed_types(series),
    }


def profile_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Return summary statistics and per-column profiles for ``df``."""
    rows = int(df.shape[0])
    columns = [profile_column(df[name], rows) for name in df.columns]
    return {
        "rows": rows,
        "columns": int(df.shape[1]),
        "missing_total": int(df.isna().sum().sum()),
        "duplicate_rows": count_duplicate_rows(df),
        "column_profiles": columns,
        "warnings": [col["warning"] for col in columns if col["warning"]],
    }


def format_cell(value: Any) -> str | None:
    """Format a value for display; ``None`` means the value is missing."""
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (float, np.floating)):
        text = f"{float(value):.6f}".rstrip("0").rstrip(".")
        return text if text not in ("", "-0") else "0"
    if isinstance(value, pd.Timestamp):
        text = str(value)
        return text[:-9] if text.endswith(" 00:00:00") else text
    text = str(value)
    if len(text) > MAX_CELL_CHARS:
        text = text[: MAX_CELL_CHARS - 1] + "…"
    return text


def build_preview(df: pd.DataFrame, max_rows: int = 15) -> dict[str, Any]:
    """Build a display-ready preview of the first ``max_rows`` rows.

    Row numbers are 1-based positions in the original uploaded data.
    """
    head = df.head(max_rows)
    rows = []
    for label, record in zip(head.index, head.itertuples(index=False, name=None)):
        row_number = int(label) + 1 if isinstance(label, (int, np.integer)) else str(label)
        rows.append({"row_number": row_number, "cells": [format_cell(v) for v in record]})
    return {
        "headers": [str(c) for c in df.columns],
        "rows": rows,
        "shown": len(rows),
        "total": int(df.shape[0]),
    }
