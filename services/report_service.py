"""Cleaning report generation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from services.cleaning_service import PipelineResult
from services.profiling_service import count_duplicate_rows, to_native


def summarize(df: pd.DataFrame) -> dict[str, int]:
    """Headline numbers used in the before/after comparison."""
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": count_duplicate_rows(df),
    }


def compare(before: dict[str, int], after: dict[str, int]) -> list[dict[str, Any]]:
    """Rows for a before/after table: metric, before, after and change."""
    labels = {
        "rows": "Rows",
        "columns": "Columns",
        "missing_values": "Missing values",
        "duplicate_rows": "Duplicate rows",
    }
    return [
        {
            "metric": label,
            "before": before[key],
            "after": after[key],
            "change": after[key] - before[key],
        }
        for key, label in labels.items()
    ]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return to_native(value)


def build_report(
    source_name: str, original: pd.DataFrame, result: PipelineResult
) -> dict[str, Any]:
    """Build the cleaning report for a pipeline run."""
    before = summarize(original)
    after = summarize(result.df)
    steps = [step.to_dict() for step in result.steps]

    conversion_failures: dict[str, Any] = {}
    outliers: dict[str, Any] = {}
    for step in result.steps:
        conversion_failures.update(step.details.get("conversion_failures", {}))
        outliers.update(step.details.get("outliers", {}))

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_file": source_name,
        "summary": {
            "before": before,
            "after": after,
            "rows_removed": before["rows"] - after["rows"],
            "columns_removed": [c for s in result.steps for c in s.columns_removed],
            "missing_values_handled": sum(s.missing_filled for s in result.steps),
        },
        "operations": steps,
        "affected_columns": sorted({c for s in result.steps for c in s.columns}),
        "conversion_failures": conversion_failures,
        "outliers": outliers,
        "errors": list(result.errors),
        "warnings": list(result.warnings),
    }
    return _json_safe(report)


def report_to_json(report: dict[str, Any]) -> str:
    """Serialise a report as pretty-printed JSON."""
    return json.dumps(report, indent=2, ensure_ascii=False)
