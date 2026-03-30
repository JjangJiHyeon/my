"""
Parser registry and common orchestration.

parse_document() is the single entry point used by app.py.
It dispatches to the correct format-specific parser, measures time,
and guarantees a consistent JSON structure regardless of outcome.
"""

from __future__ import annotations

import hashlib
import os
import time
from typing import Any, Callable

from .pdf_parser import parse_pdf
from .doc_parser import parse_doc
from .hwp_parser import parse_hwp
from .xlsx_parser import parse_excel

ParserFn = Callable[[str], dict[str, Any]]

PARSER_MAP: dict[str, ParserFn] = {
    ".pdf": parse_pdf,
    ".doc": parse_doc,
    ".docx": parse_doc,
    ".hwp": parse_hwp,
    ".xlsx": parse_excel,
    ".xls": parse_excel,
}

SUPPORTED_EXTENSIONS = set(PARSER_MAP.keys())


def _make_base(filepath: str) -> dict[str, Any]:
    filename = os.path.basename(filepath)
    ext = os.path.splitext(filename)[1].lower()
    return {
        "id": hashlib.md5(filepath.encode("utf-8")).hexdigest(),
        "filename": filename,
        "filepath": filepath,
        "file_type": ext.lstrip("."),
        "file_size": os.path.getsize(filepath),
    }


def _ensure_metadata(result: dict[str, Any]) -> dict[str, Any]:
    """Guarantee metadata is always a dict with common keys."""
    meta = result.get("metadata")
    if not isinstance(meta, dict):
        meta = {}
    result["metadata"] = meta
    return meta


def parse_document(filepath: str) -> dict[str, Any]:
    base = _make_base(filepath)
    ext = os.path.splitext(filepath)[1].lower()
    parser = PARSER_MAP.get(ext)

    if parser is None:
        return {
            **base,
            "status": "error",
            "error": f"Unsupported format: {ext}",
            "pages": [],
            "metadata": {},
        }

    start = time.time()
    try:
        result = parser(filepath)
        elapsed = round(time.time() - start, 2)

        meta = _ensure_metadata(result)
        pages = result.get("pages", [])

        meta["total_chars"] = sum(len(p.get("text", "")) for p in pages)
        meta["parse_time_sec"] = elapsed
        meta["page_count"] = meta.get("page_count", len(pages))

        ocr_applied_any = any(p.get("ocr_applied") for p in pages)
        meta["ocr_applied"] = ocr_applied_any

        status = result.get("status", "success")
        if status == "success" and meta["total_chars"] == 0:
            status = "partial"

        return {**base, **result, "status": status}

    except Exception as exc:
        elapsed = round(time.time() - start, 2)
        return {
            **base,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "pages": [],
            "metadata": {"parse_time_sec": elapsed, "ocr_applied": False},
        }
