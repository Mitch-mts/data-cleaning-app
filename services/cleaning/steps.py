"""Run one cleaning operation without stopping the whole pipeline."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

from services.cleaning.records import CleaningError, StepRecord


def run_step(
    data: pd.DataFrame,
    steps: list[StepRecord],
    errors: list[str],
    step_name: str,
    operation: Callable[..., tuple[pd.DataFrame, StepRecord]],
    *args: Any,
    **kwargs: Any,
) -> pd.DataFrame:
    """Run one cleaning operation.

    On success, save its report and return the new table.
    On failure, save the error and return the old table so the next step can
    still run.
    """
    try:
        new_data, record = operation(data, *args, **kwargs)
    except CleaningError as error:
        errors.append(f"{step_name}: {error}")
        return data

    steps.append(record)
    return new_data
