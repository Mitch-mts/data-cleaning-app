"""Flask application: routes only. Business logic lives in ``services/``."""

from __future__ import annotations

import hmac
import io
import logging
import os
import re
import secrets
import tempfile
import time
from pathlib import Path
from typing import Any

from flask import (
    Blueprint,
    Flask,
    abort,
    current_app,
    flash,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)
from werkzeug.datastructures import MultiDict
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

import pandas as pd

from services import file_service
from services.cleaning_service import PipelineResult, run_cleaning_pipeline
from services.file_service import ExportError, FileValidationError
from services.options_service import OptionsError, parse_cleaning_options
from services.profiling_service import build_preview, format_cell, profile_dataset
from services.report_service import build_report, compare, report_to_json

logger = logging.getLogger(__name__)

META_FILE = "meta.json"
FORM_FILE = "form.json"
REPORT_FILE = "report.json"

bp = Blueprint("main", __name__)


# --------------------------------------------------------------------------- #
# Workspace helpers
# --------------------------------------------------------------------------- #
def _root() -> Path:
    return Path(current_app.config["WORKSPACE_ROOT"])


def _workspace() -> Path | None:
    return file_service.get_workspace(_root(), session.get("workspace_id"))


def _load_dataset() -> tuple[Path, dict[str, Any], pd.DataFrame] | None:
    """Return ``(workspace, meta, original_df)`` for this session, or ``None``."""
    workspace = _workspace()
    if workspace is None:
        return None
    meta = file_service.load_json(workspace, META_FILE)
    if not meta:
        return None
    try:
        df = file_service.load_original(workspace, meta["extension"])
    except (FileValidationError, OSError, KeyError):
        return None
    return workspace, meta, df


def _no_dataset_redirect():
    flash("Please upload a dataset first. Uploaded files expire after a period of inactivity.", "warning")
    return redirect(url_for("main.index"))


def _form_to_dict(form: MultiDict) -> dict[str, list[str]]:
    return {key: form.getlist(key) for key in form.keys() if key != "csrf_token"}


def _dict_to_form(data: dict[str, list[str]]) -> MultiDict:
    return MultiDict([(key, value) for key, values in data.items() for value in values])


def _default_clean_form(profiles: dict[str, dict[str, Any]], scope: list[str]) -> MultiDict:
    """Pre-select text columns, skipping numbers stored as text (mixed-type columns)."""
    text_columns = [c for c in scope if profiles[c]["is_text"] and not profiles[c]["warning"]]
    return MultiDict([("text_columns", c) for c in text_columns])


def _run_saved_cleaning(
    workspace: Path, meta: dict[str, Any], df: pd.DataFrame
) -> PipelineResult | None:
    saved = file_service.load_json(workspace, FORM_FILE)
    if saved is None:
        return None
    try:
        options = parse_cleaning_options(_dict_to_form(saved), list(df.columns), meta["scope"])
    except OptionsError:
        return None
    return run_cleaning_pipeline(df, options)


def _download_stem(meta: dict[str, Any]) -> str:
    stem = Path(meta.get("original_filename", "data")).stem
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_")[:50]
    return stem or "data"


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@bp.get("/")
def index():
    file_service.cleanup_expired_workspaces(_root(), current_app.config["WORKSPACE_MAX_AGE_SECONDS"])
    workspace = _workspace()
    meta = file_service.load_json(workspace, META_FILE) if workspace else None
    return render_template("index.html", meta=meta, active_step="upload")


@bp.post("/upload")
def upload():
    max_bytes = current_app.config["MAX_UPLOAD_BYTES"]
    uploaded = request.files.get("file")
    if uploaded is None or not uploaded.filename:
        flash("Please choose a file to upload.", "danger")
        return redirect(url_for("main.index"))

    data = uploaded.read(max_bytes + 1)
    try:
        extension = file_service.validate_upload(uploaded.filename, data, max_bytes)
        df = file_service.load_dataset(data, extension)
    except FileValidationError as exc:
        flash(str(exc), "danger")
        return redirect(url_for("main.index"))

    root = _root()
    file_service.cleanup_expired_workspaces(root, current_app.config["WORKSPACE_MAX_AGE_SECONDS"])
    file_service.delete_workspace(root, session.get("workspace_id"))
    workspace_id, workspace = file_service.create_workspace(root)
    file_service.save_original(workspace, data, extension)
    file_service.save_json(
        workspace,
        META_FILE,
        {
            "original_filename": file_service.display_filename(uploaded.filename),
            "extension": extension,
            "uploaded_at": int(time.time()),
            "scope": list(df.columns),
        },
    )
    session["workspace_id"] = workspace_id
    logger.info("Dataset uploaded: %s, %d bytes, %d rows x %d columns",
                extension, len(data), df.shape[0], df.shape[1])
    flash(f"Uploaded successfully: {df.shape[0]:,} rows and {df.shape[1]} columns.", "success")
    return redirect(url_for("main.overview"))


@bp.get("/overview")
def overview():
    loaded = _load_dataset()
    if loaded is None:
        return _no_dataset_redirect()
    _, meta, df = loaded
    return render_template(
        "overview.html",
        meta=meta,
        profile=profile_dataset(df),
        preview=build_preview(df, current_app.config["PREVIEW_ROWS"]),
        scope=meta["scope"],
        active_step="overview",
    )


@bp.post("/overview")
def select_columns():
    loaded = _load_dataset()
    if loaded is None:
        return _no_dataset_redirect()
    workspace, meta, df = loaded
    selected = [c for c in request.form.getlist("columns") if c in df.columns]
    if not selected:
        flash("Select at least one column to clean.", "danger")
        return redirect(url_for("main.overview"))
    if selected != meta["scope"]:
        meta["scope"] = selected
        file_service.save_json(workspace, META_FILE, meta)
        file_service.delete_file(workspace, FORM_FILE)
        file_service.delete_file(workspace, REPORT_FILE)
    return redirect(url_for("main.clean"))


@bp.route("/clean", methods=["GET", "POST"])
def clean():
    loaded = _load_dataset()
    if loaded is None:
        return _no_dataset_redirect()
    workspace, meta, df = loaded
    columns = list(df.columns)
    scope = [c for c in meta["scope"] if c in columns] or columns
    errors: list[str] = []
    profiles = {c["name"]: c for c in profile_dataset(df)["column_profiles"]}

    if request.method == "POST":
        form = request.form
        try:
            options = parse_cleaning_options(form, columns, scope)
        except OptionsError as exc:
            errors = exc.errors
        else:
            result = run_cleaning_pipeline(df, options)
            report = build_report(meta["original_filename"], df, result)
            file_service.save_json(workspace, FORM_FILE, _form_to_dict(form))
            file_service.save_json(workspace, REPORT_FILE, report)
            if result.errors:
                flash("Cleaning finished, but some operations could not be performed. See details below.", "warning")
            else:
                flash("Data cleaned successfully.", "success")
            return redirect(url_for("main.results"))
    else:
        saved = file_service.load_json(workspace, FORM_FILE)
        form = _dict_to_form(saved) if saved is not None else _default_clean_form(profiles, scope)

    return render_template(
        "clean.html",
        meta=meta,
        form=form,
        errors=errors,
        scope=scope,
        columns=columns,
        column_profiles=profiles,
        active_step="clean",
    ), (400 if errors else 200)


@bp.get("/results")
def results():
    loaded = _load_dataset()
    if loaded is None:
        return _no_dataset_redirect()
    workspace, meta, df = loaded
    report = file_service.load_json(workspace, REPORT_FILE)
    result = _run_saved_cleaning(workspace, meta, df)
    if report is None or result is None:
        flash("Choose your cleaning options and click Clean Data first.", "info")
        return redirect(url_for("main.clean"))
    return render_template(
        "results.html",
        meta=meta,
        report=report,
        comparison=compare(report["summary"]["before"], report["summary"]["after"]),
        preview=build_preview(result.df, current_app.config["PREVIEW_ROWS"]),
        active_step="results",
    )


@bp.get("/download/data/<fmt>")
def download_data(fmt: str):
    if fmt not in ("csv", "xlsx"):
        abort(404)
    loaded = _load_dataset()
    if loaded is None:
        return _no_dataset_redirect()
    workspace, meta, df = loaded
    result = _run_saved_cleaning(workspace, meta, df)
    if result is None:
        flash("Clean the data before downloading it.", "info")
        return redirect(url_for("main.clean"))
    try:
        if fmt == "csv":
            payload, mimetype = file_service.export_csv(result.df), "text/csv"
        else:
            payload = file_service.export_excel(result.df)
            mimetype = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    except ExportError as exc:
        flash(str(exc), "danger")
        return redirect(url_for("main.results"))
    return send_file(
        io.BytesIO(payload),
        mimetype=mimetype,
        as_attachment=True,
        download_name=f"cleaned_{_download_stem(meta)}.{fmt}",
    )


@bp.get("/download/report")
def download_report():
    workspace = _workspace()
    meta = file_service.load_json(workspace, META_FILE) if workspace else None
    report = file_service.load_json(workspace, REPORT_FILE) if workspace else None
    if not meta or report is None:
        flash("Clean the data before downloading the report.", "info")
        return redirect(url_for("main.index"))
    return send_file(
        io.BytesIO(report_to_json(report).encode("utf-8")),
        mimetype="application/json",
        as_attachment=True,
        download_name=f"cleaning_report_{_download_stem(meta)}.json",
    )


@bp.post("/reset")
def reset():
    file_service.delete_workspace(_root(), session.pop("workspace_id", None))
    flash("Your data was removed from the server.", "info")
    return redirect(url_for("main.index"))


# --------------------------------------------------------------------------- #
# Application factory
# --------------------------------------------------------------------------- #
def _csrf_token() -> str:
    if "csrf_token" not in session:
        session["csrf_token"] = secrets.token_urlsafe(32)
    return session["csrf_token"]


def create_app(test_config: dict[str, Any] | None = None) -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__)
    app.config.update(
        SECRET_KEY=os.environ.get("SECRET_KEY") or secrets.token_hex(32),
        MAX_UPLOAD_BYTES=10 * 1024 * 1024,
        WORKSPACE_ROOT=Path(tempfile.gettempdir()) / "data_cleaning_app",
        WORKSPACE_MAX_AGE_SECONDS=60 * 60,
        PREVIEW_ROWS=15,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
    )
    if test_config:
        app.config.update(test_config)
    # Allow a little room for multipart form overhead on top of the file limit.
    app.config["MAX_CONTENT_LENGTH"] = app.config["MAX_UPLOAD_BYTES"] + 64 * 1024

    app.register_blueprint(bp)
    app.jinja_env.globals["csrf_token"] = _csrf_token
    app.jinja_env.filters["cell"] = format_cell

    @app.before_request
    def protect_against_csrf():
        if request.method == "POST":
            expected = session.get("csrf_token")
            sent = request.form.get("csrf_token", "")
            if not expected or not hmac.compare_digest(expected, sent):
                abort(400, description="Your session expired or the form was invalid. Please try again.")

    @app.after_request
    def set_security_headers(response):
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "same-origin")
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; "
            "script-src 'self' https://cdn.jsdelivr.net; "
            "style-src 'self' https://cdn.jsdelivr.net; "
            "img-src 'self' data:; font-src 'self' https://cdn.jsdelivr.net",
        )
        return response

    @app.errorhandler(RequestEntityTooLarge)
    def file_too_large(_error):
        limit_mb = app.config["MAX_UPLOAD_BYTES"] / (1024 * 1024)
        flash(f"The file is too large. The limit is {limit_mb:g} MB.", "danger")
        return redirect(url_for("main.index"))

    @app.errorhandler(HTTPException)
    def http_error(error: HTTPException):
        return render_template("error.html", error=error), error.code

    @app.errorhandler(Exception)
    def unexpected_error(error: Exception):
        # Exception messages can quote cell values, so tracebacks are debug-only.
        logger.error("Unhandled error: %s", type(error).__name__, exc_info=app.debug)
        return render_template("error.html", error=None), 500

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1")
