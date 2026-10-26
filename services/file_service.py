"""File validation, reading, exporting and per-session workspace storage."""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import re
import secrets
import shutil
import time
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS: frozenset[str] = frozenset({".csv", ".xlsx"})
EXCEL_MAX_ROWS = 1_048_575  # Excel's sheet limit minus the header row
ZIP_SIGNATURE = b"PK\x03\x04"
FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")
CSV_DELIMITERS = ",;\t|"

WORKSPACE_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
ORIGINAL_BASENAME = "original"


class FileValidationError(ValueError):
    """Raised when an uploaded file is missing, unsupported or unreadable."""


class ExportError(ValueError):
    """Raised when a dataset cannot be exported in the requested format."""


# --------------------------------------------------------------------------- #
# Validation and reading
# --------------------------------------------------------------------------- #
def get_extension(filename: str) -> str:
    """Return the lower-case extension of ``filename`` (including the dot)."""
    return Path(filename or "").suffix.lower()


def display_filename(filename: str, max_length: int = 100) -> str:
    """Return a short, path-free version of the user's filename for display only.

    The value is never used to build filesystem paths.
    """
    name = Path((filename or "").replace("\\", "/")).name
    name = "".join(ch for ch in name if ch.isprintable())
    return name[:max_length] or "uploaded file"


def validate_upload(filename: str, data: bytes, max_bytes: int) -> str:
    """Validate an uploaded file and return its normalised extension.

    Checks the extension, emptiness, size and a basic content signature so a
    renamed binary file is rejected before pandas tries to parse it.
    """
    if not filename:
        raise FileValidationError("Please choose a file to upload.")

    extension = get_extension(filename)
    if extension not in ALLOWED_EXTENSIONS:
        raise FileValidationError(
            "Unsupported file type. Please upload a .csv or .xlsx file."
        )
    if not data:
        raise FileValidationError("The uploaded file is empty.")
    if len(data) > max_bytes:
        limit_mb = max_bytes / (1024 * 1024)
        raise FileValidationError(f"The file is too large. The limit is {limit_mb:g} MB.")

    if extension == ".xlsx" and not data.startswith(ZIP_SIGNATURE):
        raise FileValidationError(
            "This file does not look like a valid Excel (.xlsx) workbook."
        )
    if extension == ".csv" and b"\x00" in data[:8192]:
        raise FileValidationError(
            "This file does not look like a text CSV file. "
            "If it was exported as UTF-16, please re-save it as UTF-8."
        )
    return extension


def _decode_text(data: bytes) -> str:
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")


def _detect_delimiter(text: str) -> str:
    sample = text[:16384]
    try:
        return csv.Sniffer().sniff(sample, delimiters=CSV_DELIMITERS).delimiter
    except csv.Error:
        return ","


def read_csv_bytes(data: bytes) -> pd.DataFrame:
    """Parse CSV bytes into a DataFrame, raising ``FileValidationError`` on failure."""
    text = _decode_text(data)
    if not text.strip():
        raise FileValidationError("The uploaded file is empty.")
    try:
        return pd.read_csv(
            io.StringIO(text), sep=_detect_delimiter(text), low_memory=False
        )
    except pd.errors.EmptyDataError as exc:
        raise FileValidationError("The CSV file has no columns to read.") from exc
    except pd.errors.ParserError as exc:
        raise FileValidationError(
            "The CSV file is malformed and could not be parsed. "
            "Check that every row has the same number of fields."
        ) from exc


def read_excel_bytes(data: bytes) -> pd.DataFrame:
    """Parse the first sheet of an .xlsx workbook.

    pandas opens workbooks with ``data_only=True``, so only cached cell values
    are read and formulas are never evaluated.
    """
    try:
        return pd.read_excel(io.BytesIO(data), sheet_name=0, engine="openpyxl")
    except Exception as exc:  # openpyxl raises many different error types
        logger.info("Excel parsing failed: %s", type(exc).__name__)
        raise FileValidationError(
            "The Excel file could not be read. It may be corrupted or password protected."
        ) from exc


def load_dataset(data: bytes, extension: str) -> pd.DataFrame:
    """Read a CSV or Excel file and ensure it contains usable data."""
    if extension == ".csv":
        df = read_csv_bytes(data)
    elif extension == ".xlsx":
        df = read_excel_bytes(data)
    else:
        raise FileValidationError("Unsupported file type. Please upload a .csv or .xlsx file.")

    if df.shape[1] == 0:
        raise FileValidationError("The file does not contain any columns.")
    if df.shape[0] == 0:
        raise FileValidationError("The file has a header row but no data rows.")

    df.columns = [str(column) for column in df.columns]
    return df


# --------------------------------------------------------------------------- #
# Exporting
# --------------------------------------------------------------------------- #
def _neutralize_cell(value: Any) -> Any:
    if isinstance(value, str) and value.startswith(FORMULA_PREFIXES):
        try:
            float(value)
            return value
        except ValueError:
            return "'" + value
    return value


def neutralize_formulas(df: pd.DataFrame) -> pd.DataFrame:
    """Prefix text cells that spreadsheet apps would treat as formulas.

    Protects anyone opening the exported file from CSV/Excel formula injection.
    Plain negative numbers stored as text (e.g. ``"-5"``) are left untouched.
    """
    out = df.copy()
    for column in out.columns:
        series = out[column]
        if series.dtype == object or isinstance(series.dtype, pd.StringDtype):
            out[column] = series.map(_neutralize_cell, na_action="ignore")
    return out


def _excel_compatible(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in out.columns:
        if isinstance(out[column].dtype, pd.DatetimeTZDtype):
            out[column] = out[column].dt.tz_localize(None)
    return out


def export_csv(df: pd.DataFrame) -> bytes:
    """Return the DataFrame as UTF-8 CSV bytes (with BOM so Excel detects UTF-8)."""
    return neutralize_formulas(df).to_csv(index=False).encode("utf-8-sig")


def export_excel(df: pd.DataFrame) -> bytes:
    """Return the DataFrame as .xlsx bytes."""
    if len(df) > EXCEL_MAX_ROWS:
        raise ExportError(
            f"The dataset has {len(df):,} rows, which exceeds Excel's limit. "
            "Please download it as CSV instead."
        )
    buffer = io.BytesIO()
    _excel_compatible(neutralize_formulas(df)).to_excel(
        buffer, index=False, engine="openpyxl", sheet_name="cleaned_data"
    )
    return buffer.getvalue()


# --------------------------------------------------------------------------- #
# Workspace storage (one private temporary directory per browser session)
# --------------------------------------------------------------------------- #
def create_workspace(root: Path) -> tuple[str, Path]:
    """Create a new private workspace directory and return ``(id, path)``."""
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    workspace_id = secrets.token_hex(16)
    path = root / workspace_id
    path.mkdir(mode=0o700)
    return workspace_id, path


def get_workspace(root: Path, workspace_id: str | None) -> Path | None:
    """Return the workspace path for a valid id, or ``None``.

    The id must be 32 hex characters, which rules out path traversal.
    """
    if not workspace_id or not WORKSPACE_ID_PATTERN.fullmatch(workspace_id):
        return None
    path = root / workspace_id
    if not path.is_dir():
        return None
    os.utime(path)  # mark as recently used so cleanup keeps it
    return path


def delete_workspace(root: Path, workspace_id: str | None) -> None:
    """Delete a workspace directory if it exists."""
    path = get_workspace(root, workspace_id)
    if path is not None:
        shutil.rmtree(path, ignore_errors=True)


def cleanup_expired_workspaces(root: Path, max_age_seconds: int) -> int:
    """Remove workspaces not used for ``max_age_seconds``; return how many were removed."""
    if not root.is_dir():
        return 0
    cutoff = time.time() - max_age_seconds
    removed = 0
    for entry in root.iterdir():
        if entry.is_dir() and WORKSPACE_ID_PATTERN.fullmatch(entry.name):
            if entry.stat().st_mtime < cutoff:
                shutil.rmtree(entry, ignore_errors=True)
                removed += 1
    return removed


def save_original(workspace: Path, data: bytes, extension: str) -> Path:
    """Store the uploaded bytes under a fixed internal name."""
    path = workspace / f"{ORIGINAL_BASENAME}{extension}"
    path.write_bytes(data)
    return path


def load_original(workspace: Path, extension: str) -> pd.DataFrame:
    """Re-read the untouched original dataset from the workspace."""
    if extension not in ALLOWED_EXTENSIONS:
        raise FileValidationError("Unsupported file type.")
    path = workspace / f"{ORIGINAL_BASENAME}{extension}"
    return load_dataset(path.read_bytes(), extension)


def save_json(workspace: Path, name: str, payload: Any) -> None:
    """Write JSON data to ``workspace/name``."""
    (workspace / name).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_json(workspace: Path, name: str) -> Any | None:
    """Read JSON data from ``workspace/name``; return ``None`` if it is missing."""
    path = workspace / name
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def delete_file(workspace: Path, name: str) -> None:
    """Delete ``workspace/name`` if it exists."""
    (workspace / name).unlink(missing_ok=True)
