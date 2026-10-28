"""Small checks shared by the cleaning operations."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from services.cleaning.records import CleaningError


def row_number(label: Any) -> Any:
    """1-based row number in the original data for an index label."""
    return int(label) + 1 if isinstance(label, (int, np.integer)) else str(label)


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Raise ``CleaningError`` if any requested column is no longer in the table.

    An earlier step may have removed it, for example a column that was mostly empty.
    """
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise CleaningError(
            "Column(s) not found (they may have been removed by an earlier step): "
            + ", ".join(missing)
        )
