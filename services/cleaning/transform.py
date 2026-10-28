"""Create a new column from one numeric column and a fixed number."""

from __future__ import annotations

import pandas as pd

from services.cleaning.checks import require_columns
from services.cleaning.constants import TRANSFORM_OPERATIONS
from services.cleaning.records import CleaningError, StepRecord
from services.profiling_service import is_numeric_column

OPERATION_SYMBOLS = {
    "add": "+",
    "subtract": "-",
    "multiply": "×",
    "divide": "÷",
}


def calculate(numbers: pd.Series, operation: str, constant: float) -> pd.Series:
    """Apply one of the four allowed operations. Nothing else is evaluated."""
    if operation == "add":
        return numbers + constant
    if operation == "subtract":
        return numbers - constant
    if operation == "multiply":
        return numbers * constant
    if operation == "divide":
        return numbers / constant
    raise CleaningError(f"Unsupported operation: {operation}")


def add_computed_column(
    df: pd.DataFrame, source: str, operation: str, constant: float, new_name: str
) -> tuple[pd.DataFrame, StepRecord]:
    """Create ``new_name`` = ``source`` (add, subtract, multiply, or divide) ``constant``.

    Only the four operations in ``TRANSFORM_OPERATIONS`` are supported. The
    user's text is never evaluated as code.
    """
    require_columns(df, [source])
    if operation not in TRANSFORM_OPERATIONS:
        raise CleaningError(f"Unsupported operation: {operation}")
    if new_name in df.columns:
        raise CleaningError(f"A column named '{new_name}' already exists.")
    if operation == "divide" and constant == 0:
        raise CleaningError("Cannot divide by zero.")

    series = df[source]
    if not is_numeric_column(series):
        raise CleaningError(f"'{source}' is not numeric; convert it to a number type first.")

    out = df.copy()
    out[new_name] = calculate(series.astype("float64"), operation, constant)
    symbol = OPERATION_SYMBOLS[operation]
    record = StepRecord(
        operation="Create computed column",
        description=f"Created '{new_name}' = '{source}' {symbol} {constant:g}.",
        columns=[new_name],
        details={"source": source, "operation": operation, "constant": constant},
    )
    return out, record
