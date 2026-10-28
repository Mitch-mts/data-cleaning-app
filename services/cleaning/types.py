"""Convert columns to integer, decimal, text, or datetime."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from services.cleaning.checks import require_columns, row_number
from services.cleaning.constants import CONVERSION_ERROR_ACTIONS, MAX_EXAMPLES, TARGET_TYPES
from services.cleaning.records import CleaningError, StepRecord
from services.profiling_service import is_datetime_column, is_text_column


def strip_text_cells(series: pd.Series) -> pd.Series:
    """Remove leading and trailing spaces from text cells. Other cells stay as they are."""
    return series.map(lambda value: value.strip() if isinstance(value, str) else value)


def convert_to_number(series: pd.Series, target: str) -> tuple[pd.Series, pd.Series]:
    """Turn a column into integers or decimals.

    Text is stripped first. True and False become 1.0 and 0.0. Dates cannot
    become numbers, so every date is treated as invalid. For integers, 3.5 is
    invalid because it is not a whole number.

    Returns ``(converted, invalid_mask)``. Invalid cells become missing.
    """
    had_a_value = series.notna()
    if is_text_column(series):
        source = strip_text_cells(series)
    else:
        source = series

    if ptypes.is_bool_dtype(source):
        source = source.astype("float64")

    numbers = pd.to_numeric(source, errors="coerce")
    if is_datetime_column(series):
        numbers = pd.Series(np.nan, index=series.index)

    invalid = had_a_value & numbers.isna()
    if target == "float":
        return numbers.astype("float64"), invalid

    has_fraction = numbers.notna() & (numbers % 1 != 0)
    invalid = invalid | has_fraction
    whole_numbers = numbers.mask(has_fraction)
    try:
        return whole_numbers.astype("Int64"), invalid
    except (TypeError, ValueError, OverflowError) as error:
        raise CleaningError("values are too large to store as integers") from error


def convert_to_text(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Turn every value into text. Nothing is invalid; missing stays missing."""
    invalid = pd.Series(False, index=series.index)
    return series.astype("string"), invalid


def convert_to_datetime(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Parse dates. Values that are not real dates become missing.

    A column that is already dates is returned as it is. Mixed time zones are
    stored in UTC so the column has one type.
    """
    had_a_value = series.notna()
    if is_datetime_column(series):
        return series, pd.Series(False, index=series.index)

    if is_text_column(series):
        source = strip_text_cells(series)
    else:
        source = series

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        converted = pd.to_datetime(source, errors="coerce", format="mixed")
        if not is_datetime_column(converted):
            converted = pd.to_datetime(source, errors="coerce", format="mixed", utc=True)

    invalid = had_a_value & converted.isna()
    return converted, invalid


def convert_series(series: pd.Series, target: str) -> tuple[pd.Series, pd.Series]:
    """Convert ``series`` to ``target`` type.

    Returns ``(converted, invalid_mask)`` where ``invalid_mask`` marks values that
    were present before but could not be converted (they become missing).
    """
    if target in ("integer", "float"):
        return convert_to_number(series, target)
    if target == "string":
        return convert_to_text(series)
    if target == "datetime":
        return convert_to_datetime(series)
    raise CleaningError(f"Unsupported target type: {target}")


def _failure_examples(column: pd.Series, invalid: pd.Series) -> list[dict[str, Any]]:
    """The first few values that could not be converted, with their row numbers."""
    examples = column[invalid].head(MAX_EXAMPLES)
    return [
        {"row": row_number(label), "value": str(value)}
        for label, value in examples.items()
    ]


def convert_column_types(
    df: pd.DataFrame, conversions: dict[str, str], on_invalid: str = "coerce"
) -> tuple[pd.DataFrame, StepRecord]:
    """Convert columns to new types and report every value that failed.

    ``on_invalid="coerce"`` turns invalid values into missing values.
    ``on_invalid="skip"`` leaves a column unchanged when any value is invalid.
    """
    if on_invalid not in CONVERSION_ERROR_ACTIONS:
        raise CleaningError(f"Unknown invalid-value action: {on_invalid}")
    require_columns(df, list(conversions))

    out = df.copy()
    converted_columns: dict[str, str] = {}
    failures: dict[str, Any] = {}
    issues = []
    for column, target in conversions.items():
        if target not in TARGET_TYPES or target == "keep":
            continue

        try:
            converted, invalid = convert_series(out[column], target)
        except CleaningError as error:
            issues.append(f"'{column}' could not be converted to {target}: {error}.")
            continue

        invalid_count = int(invalid.sum())
        skipped = bool(invalid_count) and on_invalid == "skip"
        if invalid_count:
            if skipped:
                action = "column left unchanged"
            else:
                action = "invalid values set to missing"
            failures[column] = {
                "target_type": target,
                "invalid_count": invalid_count,
                "action": action,
                "examples": _failure_examples(out[column], invalid),
            }

        if not skipped:
            out[column] = converted
            converted_columns[column] = target

    total_invalid = sum(failure["invalid_count"] for failure in failures.values())
    record = StepRecord(
        operation="Convert data types",
        description=(
            f"Converted {len(converted_columns)} column(s); "
            f"{total_invalid} value(s) could not be converted."
        ),
        columns=list(converted_columns),
        details={"converted": converted_columns, "conversion_failures": failures},
        issues=issues,
    )
    return out, record
