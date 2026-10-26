"""Shared pytest fixtures."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app import create_app  # noqa: E402

SAMPLE_CSV = PROJECT_ROOT / "sample_data" / "messy_customers.csv"


@pytest.fixture
def sample_bytes() -> bytes:
    return SAMPLE_CSV.read_bytes()


@pytest.fixture
def messy_df() -> pd.DataFrame:
    """Small hand-made frame with the usual problems."""
    return pd.DataFrame(
        {
            " First Name ": [" Alice ", "bob", "bob", "   ", None, "Eve"],
            "Age": [30, 25, 25, None, 40, 1000],
            "City": ["Paris", "Rome", "Rome", "", "Oslo", None],
            "Score": ["10", "x", "x", "7.5", None, "3"],
        }
    )


@pytest.fixture
def xlsx_bytes() -> bytes:
    buffer = io.BytesIO()
    pd.DataFrame({"Name": ["Ann", "Ben"], "Amount": [1.5, 2.5]}).to_excel(
        buffer, index=False, engine="openpyxl"
    )
    return buffer.getvalue()


@pytest.fixture
def app(tmp_path):
    app = create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "test-secret",
            "WORKSPACE_ROOT": tmp_path / "workspaces",
        }
    )
    return app


@pytest.fixture
def client(app):
    return app.test_client()


def csrf_token(client) -> str:
    """Load a page so the session gets a CSRF token, then return it."""
    client.get("/")
    with client.session_transaction() as session:
        return session["csrf_token"]


def upload(client, data: bytes, filename: str = "data.csv", follow_redirects: bool = False):
    token = csrf_token(client)
    return client.post(
        "/upload",
        data={"csrf_token": token, "file": (io.BytesIO(data), filename)},
        content_type="multipart/form-data",
        follow_redirects=follow_redirects,
    )
