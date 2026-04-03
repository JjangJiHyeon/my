"""
FastAPI backend for the multi-format document parser.

Configuration via environment variables:
  DOC_DIR     – directory containing source documents  (default: ./documents)
  RESULT_DIR  – directory for cached parse results      (default: ./parsed_results)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

from parsers import parse_document, SUPPORTED_EXTENSIONS, PARSER_VERSION

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
logger = logging.getLogger(__name__)

DOC_DIR: str = os.getenv(
    "DOC_DIR",
    os.path.join(os.path.dirname(__file__), "documents"),
)
RESULT_DIR: str = os.getenv(
    "RESULT_DIR",
    os.path.join(os.path.dirname(__file__), "parsed_results"),
)
DEV_CLEAR_RESULTS_ON_START: bool = os.getenv("DEV_CLEAR_RESULTS_ON_START", "0") == "1"



os.makedirs(DOC_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


@asynccontextmanager
async def lifespan(application: FastAPI):
    # Tell the PDF parser where to store preview images
    try:
        from parsers.pdf_parser import set_result_dir
        set_result_dir(RESULT_DIR)
    except Exception as exc:
        logger.warning("Could not set RESULT_DIR on pdf_parser: %s", exc)

    if DEV_CLEAR_RESULTS_ON_START:
        _clear_cache_dir()

    _warm_cache()
    logger.info("DOC_DIR    = %s", DOC_DIR)
    logger.info("RESULT_DIR = %s", RESULT_DIR)
    yield


app = FastAPI(title="Document Parser", lifespan=lifespan)

parsed_cache: dict[str, dict[str, Any]] = {}


# ── helpers ──────────────────────────────────────────────────────────

def _clear_cache_dir() -> None:
    """Clear JSON files and preview directories for DEV mode."""
    import shutil
    json_removed = 0
    previews_removed = False
    logger.info("DEV mode: clearing parsed results on startup")
    try:
        for name in os.listdir(RESULT_DIR):
            if name.endswith(".json"):
                os.remove(os.path.join(RESULT_DIR, name))
                json_removed += 1
        
        preview_dir = os.path.join(RESULT_DIR, "previews")
        if os.path.exists(preview_dir):
            shutil.rmtree(preview_dir)
            previews_removed = True
            
        logger.info("Removed %d cached json files", json_removed)
        if previews_removed:
            logger.info("Removed preview directory %s", preview_dir)
            
        parsed_cache.clear()
        
    except Exception as exc:
        logger.warning("Failed to clear cache directory: %s", exc)


def _doc_id(filepath: str) -> str:
    return hashlib.md5(filepath.encode("utf-8")).hexdigest()


def _result_path(doc_id: str) -> Path:
    return Path(RESULT_DIR) / f"{doc_id}.json"


def _save_result(doc_id: str, data: dict[str, Any]) -> None:
    data["parser_version"] = PARSER_VERSION
    try:
        with open(_result_path(doc_id), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.warning("Failed to persist result %s: %s", doc_id, exc)


def _is_legacy_result(data: dict[str, Any]) -> bool:
    """Check if the cached JSON is from the older version of the parser."""
    if data.get("parser_version") != PARSER_VERSION:
        return True
        
    pages = data.get("pages", [])
    if not pages:
        return False  # Empty documents might not be legacy if they just failed, but let's re-parse to be safe or keep it. Actually, if there are no pages, it lacks new fields.
        # A safer check is to see if any new field exists in the first page (if there are pages).
    
    first_page = pages[0]
    required_fields = ("blocks", "preview_image", "parser_debug", "page_width")
    # It must have all required fields to be considered non-legacy
    return not all(field in first_page for field in required_fields)


def _load_result(doc_id: str) -> dict[str, Any] | None:
    p = _result_path(doc_id)
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
            if _is_legacy_result(data):
                logger.info("Ignoring legacy cached result for %s", doc_id)
                return None
            return data
    except Exception as exc:
        logger.warning("Failed to read cached result %s: %s", doc_id, exc)
        return None


def _warm_cache() -> None:
    """Load every .json file in RESULT_DIR into the memory cache."""
    for name in os.listdir(RESULT_DIR):
        if not name.endswith(".json"):
            continue
        did = name[:-5]
        if did in parsed_cache:
            continue
        data = _load_result(did)
        if data:
            parsed_cache[did] = data
    logger.info("Loaded %d cached results from disk", len(parsed_cache))


def _scan_documents() -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    if not os.path.isdir(DOC_DIR):
        return docs
    for fname in sorted(os.listdir(DOC_DIR)):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            continue
        fpath = os.path.join(DOC_DIR, fname)
        if not os.path.isfile(fpath):
            continue
        did = _doc_id(fpath)
        cached = parsed_cache.get(did)
        docs.append({
            "id": did,
            "filename": fname,
            "file_type": ext.lstrip("."),
            "file_size": os.path.getsize(fpath),
            "parsed": cached is not None,
            "status": cached["status"] if cached else "pending",
            "quality_score": cached.get("metadata", {}).get("quality_score") if cached else None,
            "quality_grade": cached.get("metadata", {}).get("quality_grade") if cached else None,
        })
    return docs


# ── routes ───────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse(
        os.path.join(os.path.dirname(__file__), "static", "index.html"),
        media_type="text/html",
    )


@app.get("/api/documents")
async def list_documents():
    return _scan_documents()


@app.post("/api/documents/{doc_id}/parse")
async def parse_single(doc_id: str):
    for fname in os.listdir(DOC_DIR):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            continue
        fpath = os.path.join(DOC_DIR, fname)
        if _doc_id(fpath) == doc_id:
            result = parse_document(fpath)
            parsed_cache[doc_id] = result
            _save_result(doc_id, result)
            return result
    raise HTTPException(404, "Document not found")


@app.get("/api/documents/{doc_id}")
async def get_parsed(doc_id: str):
    cached = parsed_cache.get(doc_id)
    if cached is None:
        cached = _load_result(doc_id)
        if cached:
            parsed_cache[doc_id] = cached
    if cached is None:
        raise HTTPException(404, "Document not parsed yet")
    return cached


@app.post("/api/parse-all")
async def parse_all():
    results: list[dict[str, Any]] = []
    for fname in sorted(os.listdir(DOC_DIR)):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            continue
        fpath = os.path.join(DOC_DIR, fname)
        if not os.path.isfile(fpath):
            continue
        did = _doc_id(fpath)
        try:
            result = parse_document(fpath)
        except Exception as exc:
            result = {
                "id": did,
                "filename": fname,
                "status": "error",
                "error": str(exc),
                "pages": [],
                "metadata": {},
            }
        parsed_cache[did] = result
        _save_result(did, result)
        results.append(result)
    return results


@app.get("/prototype-summary", response_class=HTMLResponse)
async def prototype_summary_page():
    return FileResponse(
        os.path.join(os.path.dirname(__file__), "static", "summary.html"),
        media_type="text/html",
    )


@app.get("/api/prototype-stats")
async def prototype_stats():
    harness_path = os.path.join(os.path.dirname(__file__), "harness_history.json")
    if not os.path.exists(harness_path):
        raise HTTPException(404, "Harness history not found")
    try:
        with open(harness_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        raise HTTPException(500, f"Error reading harness history: {exc}")


@app.get("/api/quality-report")
async def quality_report():
    report_path = os.path.join(os.path.dirname(__file__), "quality_report.md")
    if not os.path.exists(report_path):
        raise HTTPException(404, "Quality report not found")
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            return {"content": f.read()}
    except Exception as exc:
        raise HTTPException(500, f"Error reading quality report: {exc}")


# ── preview image serving ────────────────────────────────────────────

@app.get("/api/documents/{doc_id}/pages/{page_num}/preview")
async def get_page_preview(doc_id: str, page_num: int):
    """Serve a rendered page preview image (PNG)."""
    preview_path = os.path.join(
        RESULT_DIR, "previews", doc_id, f"page_{page_num}.png"
    )
    if not os.path.isfile(preview_path):
        raise HTTPException(404, "Preview image not found. Parse the document first.")
    return FileResponse(preview_path, media_type="image/png")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
