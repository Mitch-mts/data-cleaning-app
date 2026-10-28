"""The report objects every cleaning operation returns."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd


class CleaningError(ValueError):
    """Raised when a cleaning operation cannot be performed."""


@dataclass
class StepRecord:
    """What one cleaning step did. The results page is built from these."""

    operation: str
    description: str
    columns: list[str] = field(default_factory=list)
    rows_removed: int = 0
    columns_removed: list[str] = field(default_factory=list)
    missing_filled: int = 0
    details: dict[str, Any] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dictionary."""
        return asdict(self)


@dataclass
class PipelineResult:
    """The table after cleaning, plus what each step did and what failed."""

    df: pd.DataFrame
    steps: list[StepRecord]
    errors: list[str]
    warnings: list[str]
