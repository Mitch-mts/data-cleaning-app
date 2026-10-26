"""End-to-end tests of the Flask routes."""

from __future__ import annotations

import io
import json

import pandas as pd

from services import file_service
from tests.conftest import csrf_token, upload


def clean(client, pairs):
    token = csrf_token(client)
    data = [("csrf_token", token)] + list(pairs)
    from werkzeug.datastructures import MultiDict

    return client.post("/clean", data=MultiDict(data))


def test_home_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Upload and analyse" in response.data
    assert response.headers["X-Content-Type-Options"] == "nosniff"


def test_upload_and_overview(client, sample_bytes):
    response = upload(client, sample_bytes, "messy.csv")
    assert response.status_code == 302
    assert response.headers["Location"].endswith("/overview")

    page = client.get("/overview")
    assert page.status_code == 200
    assert b"Dataset overview" in page.data
    assert b"mixed types" in page.data
    assert b"Customer ID" in page.data


def test_upload_xlsx(client, xlsx_bytes):
    response = upload(client, xlsx_bytes, "book.xlsx", follow_redirects=True)
    assert b"Uploaded successfully: 2 rows" in response.data


def test_upload_rejects_bad_extension(client):
    response = upload(client, b"a,b\n1,2", "evil.exe", follow_redirects=True)
    assert b"Unsupported file type" in response.data
    with client.session_transaction() as session:
        assert "workspace_id" not in session


def test_upload_rejects_too_large(app, client):
    app.config["MAX_UPLOAD_BYTES"] = 100
    app.config["MAX_CONTENT_LENGTH"] = 100_000
    response = upload(client, b"a\n" + b"1\n" * 200, "big.csv", follow_redirects=True)
    assert b"too large" in response.data


def test_upload_requires_csrf(client, sample_bytes):
    response = client.post(
        "/upload",
        data={"file": (io.BytesIO(sample_bytes), "a.csv")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400


def test_user_html_is_escaped(client):
    upload(client, b"<script>alert(1)</script>\n<b>x</b>\n", "xss.csv")
    page = client.get("/overview")
    assert b"<script>alert(1)</script>" not in page.data
    assert b"&lt;script&gt;" in page.data


def test_pages_require_upload(client):
    for path in ["/overview", "/clean", "/results", "/download/data/csv", "/download/report"]:
        response = client.get(path)
        assert response.status_code == 302, path


def test_tampered_session_is_ignored(client):
    with client.session_transaction() as session:
        session["workspace_id"] = "../../etc"
    assert client.get("/overview").status_code == 302


def test_full_cleaning_workflow(client, sample_bytes):
    upload(client, sample_bytes, "messy.csv")

    clean_page = client.get("/clean")
    assert clean_page.status_code == 200
    assert b"Clean Data" in clean_page.data

    response = clean(
        client,
        [
            ("text_columns", "Full Name"),
            ("text_columns", "City"),
            ("text_trim", "on"),
            ("text_case", "title"),
            ("text_empty_to_missing", "on"),
            ("duplicates_remove", "on"),
            ("rename_strip", "on"),
            ("rename_lowercase", "on"),
            ("rename_underscores", "on"),
        ],
    )
    assert response.status_code == 302
    results = client.get("/results")
    assert results.status_code == 200
    assert b"Before and after" in results.data
    assert b"Remove duplicate rows" in results.data

    csv_response = client.get("/download/data/csv")
    assert csv_response.status_code == 200
    assert "attachment" in csv_response.headers["Content-Disposition"]
    assert "cleaned_messy.csv" in csv_response.headers["Content-Disposition"]
    df = file_service.load_dataset(csv_response.data, ".csv")
    assert len(df) == 13
    assert list(df.columns)[:2] == ["customer_id", "full_name"]
    assert "city_2" in df.columns
    assert df["full_name"].iloc[0] == "Alice Smith"
    assert df["notes"].dropna().iloc[-1].startswith("'=")  # formula neutralized

    xlsx_response = client.get("/download/data/xlsx")
    assert xlsx_response.status_code == 200
    assert len(pd.read_excel(io.BytesIO(xlsx_response.data))) == 13

    report_response = client.get("/download/report")
    report = json.loads(report_response.data)
    assert report["summary"]["rows_removed"] == 2
    assert report["source_file"] == "messy.csv"


def test_advanced_cleaning_workflow(client, sample_bytes):
    upload(client, sample_bytes, "messy.csv")
    response = clean(
        client,
        [
            ("convert_type_3", "integer"),   # Age
            ("convert_type_4", "datetime"),  # Signup Date
            ("convert_type_5", "float"),     # Purchase Amount
            ("conversion_errors", "coerce"),
            ("duplicates_remove", "on"),
            ("outlier_columns", "Purchase Amount"),
            ("outlier_multiplier", "1.5"),
            ("filter_numeric_column", "Age"),
            ("filter_numeric_min", "0"),
            ("transform_source", "Purchase Amount"),
            ("transform_operation", "multiply"),
            ("transform_value", "1.2"),
            ("transform_new_name", "amount_with_tax"),
            ("sort_primary", "Age"),
            ("sort_direction", "desc"),
        ],
    )
    assert response.status_code == 302

    results = client.get("/results")
    assert results.status_code == 200
    assert b"Values that could not be converted" in results.data
    assert b"Outliers found (IQR)" in results.data
    assert b"thirty" in results.data

    report = json.loads(client.get("/download/report").data)
    failures = report["conversion_failures"]
    assert failures["Age"]["invalid_count"] == 3  # "thirty" x2 and 36.5
    assert failures["Signup Date"]["invalid_count"] == 2
    assert failures["Purchase Amount"]["invalid_count"] == 1
    assert report["outliers"]["Purchase Amount"]["count"] == 1
    assert report["errors"] == []

    df = file_service.load_dataset(client.get("/download/data/csv").data, ".csv")
    assert df["Age"].tolist() == sorted(df["Age"].tolist(), reverse=True)
    assert (df["Age"] >= 0).all()
    assert "amount_with_tax" in df.columns
    outlier_step = next(op for op in report["operations"] if op["operation"].startswith("Detect outliers"))
    assert outlier_step["rows_removed"] == 0  # outliers are only reported unless removal is ticked


def test_default_text_columns_skip_mixed_type_columns(client, sample_bytes):
    upload(client, sample_bytes)
    html = client.get("/clean").get_data(as_text=True)
    text_select = html.split('name="text_columns"', 1)[1].split("</select>", 1)[0]
    assert '<option value="Full Name" selected>' in text_select
    assert '<option value="Age" >' in text_select


def test_clean_form_remembers_previous_options(client, sample_bytes):
    upload(client, sample_bytes)
    clean(client, [("outlier_columns", "Age"), ("outlier_multiplier", "2.5")])
    page = client.get("/clean")
    assert b'value="2.5"' in page.data


def test_invalid_clean_options_show_errors(client, sample_bytes):
    upload(client, sample_bytes)
    response = clean(client, [("missing_threshold_enabled", "on"), ("missing_threshold", "500")])
    assert response.status_code == 400
    assert b"Please fix the following" in response.data


def test_select_columns_limits_scope(client, sample_bytes):
    upload(client, sample_bytes)
    token = csrf_token(client)
    response = client.post("/overview", data={"csrf_token": token, "columns": ["Age", "City"]})
    assert response.status_code == 302
    page = client.get("/clean")
    assert b"Columns selected for cleaning (2 of 9)" in page.data


def test_unknown_download_format(client, sample_bytes):
    upload(client, sample_bytes)
    assert client.get("/download/data/exe").status_code == 404


def test_reset_deletes_workspace(app, client, sample_bytes):
    upload(client, sample_bytes)
    with client.session_transaction() as session:
        workspace_id = session["workspace_id"]
    root = app.config["WORKSPACE_ROOT"]
    assert (root / workspace_id).is_dir()
    client.post("/reset", data={"csrf_token": csrf_token(client)})
    assert not (root / workspace_id).exists()


def test_new_upload_replaces_previous(app, client, sample_bytes, xlsx_bytes):
    upload(client, sample_bytes)
    with client.session_transaction() as session:
        first = session["workspace_id"]
    upload(client, xlsx_bytes, "b.xlsx")
    root = app.config["WORKSPACE_ROOT"]
    assert not (root / first).exists()
    assert len(list(root.iterdir())) == 1
