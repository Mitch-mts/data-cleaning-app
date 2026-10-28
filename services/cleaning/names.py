"""Make column names consistent: trim, lower case, and underscores."""

from __future__ import annotations

import re

import pandas as pd

from services.cleaning.records import StepRecord


def make_unique_names(names: list[str]) -> list[str]:
    """Append ``_2``, ``_3``… to repeated names so every name is unique."""
    used: set[str] = set()
    result = []
    for name in names:
        candidate = name
        counter = 2
        while candidate in used:
            candidate = f"{name}_{counter}"
            counter += 1
        used.add(candidate)
        result.append(candidate)
    return result


def _proposed_name(
    name: object,
    position: int,
    strip: bool,
    lowercase: bool,
    underscores: bool,
) -> str:
    """Turn one column name into the form the user asked for.

    A blank name becomes ``column_1``, ``column_2``, and so on, using the
    column's position so it still has a name.
    """
    new = str(name)
    if strip:
        new = new.strip()
    if lowercase:
        new = new.lower()
    if underscores:
        # When trimming is on, spaces were already removed from the ends.
        # When it is off, leading and trailing spaces become underscores too.
        source = new.strip() if strip else new
        new = re.sub(r"\s+", "_", source)
    return new or f"column_{position}"


def standardize_column_names(
    df: pd.DataFrame,
    strip: bool = True,
    lowercase: bool = True,
    underscores: bool = True,
) -> tuple[pd.DataFrame, StepRecord]:
    """Trim, lower-case and underscore column names, resolving duplicates safely."""
    proposed = [
        _proposed_name(name, position, strip, lowercase, underscores)
        for position, name in enumerate(df.columns, start=1)
    ]
    unique = make_unique_names(proposed)
    renamed = {
        str(old): new
        for old, new in zip(df.columns, unique)
        if str(old) != new
    }
    duplicates_resolved = [new for new, original in zip(unique, proposed) if new != original]

    out = df.copy()
    out.columns = unique
    record = StepRecord(
        operation="Standardize column names",
        description=f"Renamed {len(renamed)} column(s).",
        columns=list(renamed),
        details={"renamed": renamed, "duplicates_resolved": duplicates_resolved},
    )
    return out, record
