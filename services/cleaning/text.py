"""Trim text, change capitalisation, and turn blanks into missing values."""

from __future__ import annotations

import numpy as np
import pandas as pd

from services.cleaning.checks import require_columns
from services.cleaning.constants import TEXT_CASES
from services.cleaning.records import CleaningError, StepRecord
from services.profiling_service import is_text_column


def change_capitalisation(text: pd.Series, case: str) -> pd.Series:
    """Return ``text`` in the requested case. ``none`` leaves it unchanged."""
    if case == "lower":
        return text.str.lower()
    if case == "upper":
        return text.str.upper()
    if case == "title":
        return text.str.title()
    return text


def clean_one_text_column(
    column: pd.Series,
    trim: bool,
    case: str,
    empty_to_missing: bool,
) -> tuple[pd.Series, int, int]:
    """Clean the text cells in one column.

    Numbers, dates, and missing cells in a text column are left alone.
    Returns the updated column, how many cells changed, and how many blanks
    became missing.
    """
    is_real_text = column.map(lambda value: isinstance(value, str))
    original_text = column[is_real_text]
    cleaned_text = original_text

    if trim:
        cleaned_text = cleaned_text.str.strip()
    cleaned_text = change_capitalisation(cleaned_text, case)

    blank_count = 0
    if empty_to_missing:
        is_blank = cleaned_text.str.strip() == ""
        blank_count = int(is_blank.sum())
        cleaned_text = cleaned_text.mask(is_blank, np.nan)

    updated = column.copy()
    updated[is_real_text] = cleaned_text
    cells_changed = int(original_text.ne(cleaned_text).sum())
    return updated, cells_changed, blank_count


def clean_text_columns(
    df: pd.DataFrame,
    columns: list[str],
    trim: bool = True,
    case: str = "none",
    empty_to_missing: bool = True,
) -> tuple[pd.DataFrame, StepRecord]:
    """Trim whitespace, change capitalisation and blank-to-missing in text columns."""
    if case not in TEXT_CASES:
        raise CleaningError(f"Unknown capitalisation option: {case}")
    require_columns(df, columns)

    out = df.copy()
    changed: dict[str, int] = {}
    blanks: dict[str, int] = {}
    issues = []
    for column in columns:
        if not is_text_column(out[column]):
            issues.append(f"'{column}' is not a text column and was skipped.")
            continue
        updated, cells_changed, blank_count = clean_one_text_column(
            out[column], trim, case, empty_to_missing
        )
        out[column] = updated
        changed[column] = cells_changed
        blanks[column] = blank_count

    actions = [
        name
        for selected, name in (
            (trim, "trimmed whitespace"),
            (case != "none", f"converted to {case} case"),
            (empty_to_missing, "turned blank values into missing values"),
        )
        if selected
    ]
    summary = ", ".join(actions) or "no text changes selected"
    record = StepRecord(
        operation="Clean text",
        description=summary.capitalize() + f"; {sum(changed.values())} cell(s) changed.",
        columns=list(changed),
        details={"cells_changed": changed, "blanks_to_missing": blanks},
        issues=issues,
    )
    return out, record
