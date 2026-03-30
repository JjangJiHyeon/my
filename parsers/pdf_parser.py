"""
PDF parser – block-level extraction pipeline.

Extraction order
────────────────
1. PyMuPDF  ``get_text("dict")``  → text / title / footer blocks with bbox
2. pdfplumber table regions       → table blocks with bbox + cell data
3. Camelot fallback               → table blocks (when pdfplumber finds none)
4. Image / chart-like heuristics  → image & chart_like blocks via xref
5. OCR boxes (only when needed)   → text blocks from OCR for text-poor regions

Preview images are rendered via PyMuPDF pixmap and saved as PNG under
``<RESULT_DIR>/previews/<doc_id>/``.

Every failure is caught and recorded in ``parser_debug.parse_warnings``
so that partial results are always returned.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import uuid
from typing import Any

import fitz  # PyMuPDF
import numpy as np

logger = logging.getLogger(__name__)

# ── tunables ─────────────────────────────────────────────────────────

OCR_TEXT_THRESHOLD = 60       # chars per page below which OCR is considered
OCR_IMAGE_MIN = 1             # page must have ≥1 image to trigger OCR
PREVIEW_ZOOM = 1.5            # zoom factor for preview PNGs
PREVIEW_FORMAT = "png"
BLOCK_ZOOM = 2.0              # zoom for OCR rendering (higher → better quality)

CHART_MIN_ASPECT = 0.3        # aspect-ratio lower bound for chart heuristic
CHART_MAX_ASPECT = 3.0        # aspect-ratio upper bound
CHART_MIN_AREA_RATIO = 0.02   # image area / page area lower bound

# ── Camelot filters ──────────────────────────────────────────────────
CAM_TABLE_MIN_ROWS = 2
CAM_TABLE_MIN_COLS = 2
CAM_TABLE_MIN_FILLED_RATIO = 0.35
CAM_TABLE_MAX_PAGE_AREA_RATIO = 0.70
CAM_TABLE_MIN_ACCURACY = 60.0

# ── result dir (will be set from app.py via module-level helper) ─────

_result_dir: str = ""


def set_result_dir(path: str) -> None:
    global _result_dir
    _result_dir = path


def _preview_dir(doc_id: str) -> str:
    d = os.path.join(_result_dir, "previews", doc_id)
    os.makedirs(d, exist_ok=True)
    return d


# ── public entry point ──────────────────────────────────────────────

def parse_pdf(filepath: str) -> dict[str, Any]:
    import hashlib
    doc_id = hashlib.md5(filepath.encode("utf-8")).hexdigest()

    doc = fitz.open(filepath)
    metadata = _extract_metadata(doc)
    pages: list[dict[str, Any]] = []
    ocr_page_nums: list[int] = []

    # Pre-extract pdfplumber table regions (bbox + cell data)
    plumber_tables = _extract_tables_via_pdfplumber(filepath, doc.page_count)

    for idx in range(doc.page_count):
        try:
            page_result = _process_page(doc, idx, doc_id, plumber_tables.get(idx, []))
            if page_result.get("ocr_applied"):
                ocr_page_nums.append(idx + 1)
            pages.append(page_result)
        except Exception as exc:
            logger.warning("Page %d failed: %s", idx + 1, exc)
            pages.append({
                "page_num": idx + 1,
                "text": "",
                "tables": [],
                "blocks": [],
                "dimensions": "",
                "image_count": 0,
                "text_source": "error",
                "ocr_applied": False,
                "ocr_confidence": 0.0,
                "preview_image": None,
                "parser_debug": {
                    "fallback_reason": None,
                    "ocr_engine_used": None,
                    "merge_strategy": "error",
                    "parse_warnings": [f"Page {idx+1} error: {exc}"],
                    "extraction_order": [],
                },
                "error": str(exc),
            })

    doc.close()

    empty_pages = sum(1 for p in pages if len(p.get("text", "")) == 0)
    quality = _assess_quality(pages, ocr_page_nums, empty_pages)

    metadata.update({
        "parser_used": "PyMuPDF + pdfplumber + Camelot + EasyOCR/pytesseract",
        "ocr_pages": ocr_page_nums,
        "text_quality": quality,
        "empty_pages": empty_pages,
    })

    return {"pages": pages, "metadata": metadata, "status": "success"}


# ── per-page processing ─────────────────────────────────────────────

def _process_page(
    doc: fitz.Document,
    idx: int,
    doc_id: str,
    plumber_table_data: list[dict[str, Any]],
) -> dict[str, Any]:
    page: fitz.Page = doc[idx]
    rect = page.rect
    pw, ph = rect.width, rect.height
    dims = f"{round(pw, 1)} x {round(ph, 1)}"

    warnings: list[str] = []
    extraction_order: list[str] = []
    blocks: list[dict[str, Any]] = []
    block_counter = 0

    # ── 0. Preview image ──────────────────────────────────────────
    preview_path = _render_preview(page, doc_id, idx, warnings)

    # ── 1. PyMuPDF text blocks (dict mode) ────────────────────────
    native_text = ""
    try:
        text_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
        for bk in text_dict.get("blocks", []):
            if bk.get("type") == 0:  # text block
                bbox = [round(bk["bbox"][i], 2) for i in range(4)]
                block_text = ""
                for line in bk.get("lines", []):
                    for span in line.get("spans", []):
                        block_text += span.get("text", "")
                    block_text += "\n"
                block_text = block_text.strip()
                if not block_text:
                    continue

                btype = _classify_text_block(bk, bbox, ph)
                blocks.append({
                    "id": f"p{idx+1}_b{block_counter}",
                    "type": btype,
                    "bbox": bbox,
                    "text": block_text,
                    "confidence": 1.0,
                    "source": "pymupdf",
                    "extra": {},
                })
                block_counter += 1
                native_text += block_text + "\n"
        extraction_order.append("pymupdf_blocks")
    except Exception as exc:
        warnings.append(f"PyMuPDF dict extraction failed: {exc}")
        # fallback: plain text
        try:
            native_text = page.get_text("text").strip()
            extraction_order.append("pymupdf_plain_fallback")
        except Exception as exc2:
            warnings.append(f"PyMuPDF plain text also failed: {exc2}")

    native_text = native_text.strip()

    # ── 2. pdfplumber table blocks ────────────────────────────────
    table_bboxes: list[list[float]] = []
    tables_for_page: list[list[list[str]]] = []
    if plumber_table_data:
        for ti, tinfo in enumerate(plumber_table_data):
            tbl_bbox = tinfo.get("bbox", [0, 0, pw, ph])
            rows = tinfo.get("rows", [])
            table_bboxes.append(tbl_bbox)
            tables_for_page.append(rows)
            blocks.append({
                "id": f"p{idx+1}_b{block_counter}",
                "type": "table",
                "bbox": [round(v, 2) for v in tbl_bbox],
                "text": _table_to_text(rows),
                "confidence": 1.0,
                "source": "pdfplumber",
                "extra": {"rows": rows},
            })
            block_counter += 1
        extraction_order.append("pdfplumber_tables")

    # ── 3. Camelot fallback (only when pdfplumber found no tables) ─
    if not plumber_table_data:
        camelot_tables = _extract_tables_via_camelot(doc.name, idx, pw, ph, warnings)
        accepted_camelot_tables = []
        for ct in camelot_tables:
            is_accepted, reason = _accept_camelot_table(
                rows=ct["rows"],
                bbox=ct["bbox"],
                page_width=pw,
                page_height=ph,
                accuracy=ct.get("accuracy", 0.0)
            )
            if is_accepted:
                accepted_camelot_tables.append(ct)
            else:
                warnings.append(f"Camelot rejected on page {idx+1}: {reason}")

        if accepted_camelot_tables:
            for ct in accepted_camelot_tables:
                table_bboxes.append(ct["bbox"])
                tables_for_page.append(ct["rows"])
                blocks.append({
                    "id": f"p{idx+1}_b{block_counter}",
                    "type": "table",
                    "bbox": ct["bbox"],
                    "text": _table_to_text(ct["rows"]),
                    "confidence": ct.get("accuracy", 0.0) / 100.0,
                    "source": "camelot",
                    "extra": {"rows": ct["rows"], "accuracy": ct.get("accuracy")},
                })
                block_counter += 1
            extraction_order.append("camelot_tables")

    # ── 4. Image / chart-like heuristics ──────────────────────────
    image_list = page.get_images(full=True)
    image_count = len(image_list)
    try:
        for img_info in image_list:
            xref = img_info[0]
            img_rect = _find_image_rect(page, xref)
            if img_rect is None:
                continue
            ibbox = [round(img_rect.x0, 2), round(img_rect.y0, 2),
                     round(img_rect.x1, 2), round(img_rect.y1, 2)]
            iw = ibbox[2] - ibbox[0]
            ih = ibbox[3] - ibbox[1]
            if iw <= 0 or ih <= 0:
                continue
            aspect = iw / ih
            area_ratio = (iw * ih) / (pw * ph) if pw * ph else 0

            if (CHART_MIN_ASPECT <= aspect <= CHART_MAX_ASPECT
                    and area_ratio >= CHART_MIN_AREA_RATIO):
                btype = "chart_like"
            else:
                btype = "image"

            blocks.append({
                "id": f"p{idx+1}_b{block_counter}",
                "type": btype,
                "bbox": ibbox,
                "text": "",
                "confidence": 1.0,
                "source": "heuristic",
                "extra": {"xref": xref, "aspect_ratio": round(aspect, 3),
                          "area_ratio": round(area_ratio, 4)},
            })
            block_counter += 1
        extraction_order.append("image_heuristics")
    except Exception as exc:
        warnings.append(f"Image heuristic failed: {exc}")

    # ── 5. Remove text blocks that overlap table regions ──────────
    # Only use pdfplumber table bboxes for text suppression to avoid Camelot false positives
    # suppressing valid native text blocks.
    if plumber_table_data:
        plumber_bboxes = [tb.get("bbox", [0, 0, pw, ph]) for tb in plumber_table_data]
        blocks = _filter_overlapping_text_blocks(blocks, plumber_bboxes)

    # ── 6. OCR (only when needed) ─────────────────────────────────
    need_ocr = _should_ocr(native_text, image_count)
    ocr_applied = False
    ocr_confidence = 0.0
    ocr_engine: str | None = None
    fallback_reason: str | None = None
    merge_strategy = "native_only"
    final_text = native_text
    text_source = "native"

    if need_ocr:
        try:
            from .ocr_utils import run_ocr_on_image

            mat = fitz.Matrix(BLOCK_ZOOM, BLOCK_ZOOM)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img_arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)

            ocr_result = run_ocr_on_image(
                img_arr, page_width=pw, page_height=ph, zoom=BLOCK_ZOOM
            )
            ocr_applied = ocr_result["success"]
            ocr_confidence = ocr_result.get("confidence", 0.0)
            ocr_engine = ocr_result.get("engine")
            fallback_reason = ocr_result.get("fallback_reason")

            if ocr_applied:
                # Add OCR boxes as blocks
                for ob in ocr_result.get("boxes", []):
                    blocks.append({
                        "id": f"p{idx+1}_b{block_counter}",
                        "type": "text",
                        "bbox": ob["bbox"],
                        "text": ob["text"],
                        "confidence": ob["confidence"],
                        "source": f"ocr_{ocr_engine}",
                        "extra": {},
                    })
                    block_counter += 1

                ocr_text = ocr_result.get("text", "")
                if len(native_text) < OCR_TEXT_THRESHOLD:
                    # Native text too sparse → replace
                    final_text = ocr_text
                    text_source = "ocr"
                    merge_strategy = "ocr_replace"
                else:
                    # Native text exists → append OCR supplement
                    final_text = native_text + "\n\n[OCR]\n" + ocr_text
                    text_source = "hybrid"
                    merge_strategy = "hybrid_merge"

                extraction_order.append("ocr_boxes")
            else:
                if ocr_result.get("error"):
                    warnings.append(f"OCR failed: {ocr_result['error']}")
        except Exception as exc:
            warnings.append(f"OCR pipeline error: {exc}")

    parser_debug = {
        "fallback_reason": fallback_reason,
        "ocr_engine_used": ocr_engine,
        "merge_strategy": merge_strategy,
        "parse_warnings": warnings,
        "extraction_order": extraction_order,
    }

    return {
        "page_num": idx + 1,
        "text": final_text,
        "tables": tables_for_page,
        "blocks": blocks,
        "dimensions": dims,
        "image_count": image_count,
        "text_source": text_source,
        "ocr_applied": ocr_applied,
        "ocr_confidence": ocr_confidence,
        "preview_image": preview_path,
        "parser_debug": parser_debug,
    }


# ── block classification ────────────────────────────────────────────

def _classify_text_block(
    block: dict, bbox: list[float], page_height: float
) -> str:
    """Heuristic classification of a PyMuPDF text block."""
    y0, y1 = bbox[1], bbox[3]
    height = y1 - y0

    # Title heuristic: near top, large font
    lines = block.get("lines", [])
    if lines:
        max_size = max(
            (span.get("size", 0) for line in lines for span in line.get("spans", [])),
            default=0,
        )
        if max_size >= 14 and y0 < page_height * 0.15:
            return "title"

    # Footer heuristic: near bottom, small
    if y1 > page_height * 0.92 and height < 20:
        return "footer"

    return "text"


# ── preview rendering ───────────────────────────────────────────────

def _render_preview(
    page: fitz.Page, doc_id: str, idx: int, warnings: list[str]
) -> str | None:
    """Render page to PNG and return the API path. Never raises."""
    try:
        if not _result_dir:
            warnings.append("RESULT_DIR not set — preview skipped")
            return None
        out_dir = _preview_dir(doc_id)
        fname = f"page_{idx + 1}.{PREVIEW_FORMAT}"
        fpath = os.path.join(out_dir, fname)

        mat = fitz.Matrix(PREVIEW_ZOOM, PREVIEW_ZOOM)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        pix.save(fpath)

        return f"/api/documents/{doc_id}/pages/{idx + 1}/preview"
    except Exception as exc:
        warnings.append(f"Preview generation failed: {exc}")
        return None


# ── image rect lookup ────────────────────────────────────────────────

def _find_image_rect(page: fitz.Page, xref: int) -> fitz.Rect | None:
    """Find the bounding rect of an image on the page by its xref."""
    try:
        for img in page.get_image_info(xrefs=True):
            if img.get("xref") == xref:
                bbox = img.get("bbox")
                if bbox:
                    return fitz.Rect(bbox)
        return None
    except Exception:
        return None


# ── table-text overlap filter ────────────────────────────────────────

def _rects_overlap(a: list[float], b: list[float], threshold: float = 0.5) -> bool:
    """Check if box `a` overlaps `b` by ≥ threshold of a's area."""
    ix0 = max(a[0], b[0])
    iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2])
    iy1 = min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return False
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    if area_a <= 0:
        return False
    return (inter / area_a) >= threshold


def _filter_overlapping_text_blocks(
    blocks: list[dict[str, Any]], table_bboxes: list[list[float]]
) -> list[dict[str, Any]]:
    """Remove text blocks whose bbox overlaps ≥50 % with any table bbox."""
    filtered = []
    for blk in blocks:
        if blk["type"] not in ("text", "title", "footer"):
            filtered.append(blk)
            continue
        overlaps = any(_rects_overlap(blk["bbox"], tb) for tb in table_bboxes)
        if not overlaps:
            filtered.append(blk)
    return filtered


# ── OCR trigger logic ────────────────────────────────────────────────

def _should_ocr(native_text: str, image_count: int) -> bool:
    if len(native_text) >= OCR_TEXT_THRESHOLD:
        return False
    if image_count >= OCR_IMAGE_MIN:
        return True
    if len(native_text) == 0:
        return True
    return False


# ── table extraction: pdfplumber ─────────────────────────────────────

def _extract_tables_via_pdfplumber(
    filepath: str, page_count: int
) -> dict[int, list[dict[str, Any]]]:
    """
    Returns ``{page_index: [{"bbox": [...], "rows": [[...], ...]}, ...]}``.
    """
    result: dict[int, list[dict[str, Any]]] = {}
    try:
        import pdfplumber

        with pdfplumber.open(filepath) as pdf:
            for idx, pg in enumerate(pdf.pages):
                try:
                    tables = pg.find_tables()
                    if not tables:
                        continue
                    page_tables = []
                    for tbl in tables:
                        bbox_raw = tbl.bbox  # (x0, top, x1, bottom)
                        rows_raw = tbl.extract()
                        if rows_raw is None:
                            continue
                        cleaned_rows = [
                            [(c.strip() if c else "") for c in row]
                            for row in rows_raw
                        ]
                        page_tables.append({
                            "bbox": [
                                round(bbox_raw[0], 2),
                                round(bbox_raw[1], 2),
                                round(bbox_raw[2], 2),
                                round(bbox_raw[3], 2),
                            ],
                            "rows": cleaned_rows,
                        })
                    if page_tables:
                        result[idx] = page_tables
                except Exception:
                    pass
    except Exception as exc:
        logger.warning("pdfplumber table extraction failed: %s", exc)
    return result


# ── table extraction: Camelot fallback ───────────────────────────────

def _extract_tables_via_camelot(
    filepath: str,
    page_idx: int,
    page_width: float,
    page_height: float,
    warnings: list[str],
) -> list[dict[str, Any]]:
    """Camelot fallback — tries lattice then stream. Returns list of table dicts."""
    tables_out: list[dict[str, Any]] = []
    try:
        import camelot

        page_str = str(page_idx + 1)

        for flavour in ("lattice", "stream"):
            try:
                cam_tables = camelot.read_pdf(
                    filepath, pages=page_str, flavor=flavour,
                    suppress_stdout=True,
                )
                for ct in cam_tables:
                    df = ct.df
                    rows_list = [list(row) for _, row in df.iterrows()]
                    
                    cx0, cy0, cx1, cy1 = ct._bbox  # (x0, y0_bottom, x1, y1_bottom)
                    bbox = [
                        round(cx0, 2),
                        round(page_height - cy1, 2),
                        round(cx1, 2),
                        round(page_height - cy0, 2),
                    ]
                    
                    acc = round(ct.accuracy, 2) if hasattr(ct, "accuracy") else 0.0

                    tables_out.append({
                        "bbox": bbox,
                        "rows": rows_list,
                        "accuracy": acc,
                    })
                if tables_out:
                    break  # got results, no need for second flavour
            except Exception as exc:
                warnings.append(f"Camelot {flavour} failed on page {page_idx+1}: {exc}")

    except ImportError:
        warnings.append("camelot-py not installed — skipping Camelot fallback")
    except Exception as exc:
        warnings.append(f"Camelot error: {exc}")

    return tables_out


def _accept_camelot_table(
    rows: list[list[str]],
    bbox: list[float],
    page_width: float,
    page_height: float,
    accuracy: float
) -> tuple[bool, str]:
    """Validate Camelot table results before acceptance."""
    num_rows = len(rows)
    num_cols = len(rows[0]) if num_rows > 0 else 0

    if num_rows < CAM_TABLE_MIN_ROWS:
        return False, f"too_few_rows:{num_rows}"
    if num_cols < CAM_TABLE_MIN_COLS:
        return False, f"too_few_cols:{num_cols}"

    flat_cells = [c.strip() for r in rows for c in r]
    filled_cells = sum(1 for c in flat_cells if c)
    filled_ratio = filled_cells / (num_rows * num_cols) if (num_rows * num_cols) > 0 else 0
    if filled_ratio < CAM_TABLE_MIN_FILLED_RATIO:
        return False, f"low_filled_ratio:{filled_ratio:.2f}"

    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    page_area = page_width * page_height
    area_ratio = area / page_area if page_area > 0 else 0
    if area_ratio > CAM_TABLE_MAX_PAGE_AREA_RATIO:
        return False, f"bbox_too_large:{area_ratio:.2f}"

    if accuracy < CAM_TABLE_MIN_ACCURACY:
        return False, f"low_accuracy:{accuracy}"

    return True, "accepted"


# ── helper: table rows → plain text ─────────────────────────────────

def _table_to_text(rows: list[list[str]]) -> str:
    return "\n".join(" | ".join(cell for cell in row) for row in rows)


# ── metadata ─────────────────────────────────────────────────────────

def _extract_metadata(doc: fitz.Document) -> dict[str, Any]:
    raw = doc.metadata or {}
    return {
        "page_count": doc.page_count,
        "author": raw.get("author", "") or "",
        "title": raw.get("title", "") or "",
        "creator": raw.get("creator", "") or "",
        "producer": raw.get("producer", "") or "",
    }


# ── quality assessment ───────────────────────────────────────────────

def _assess_quality(
    pages: list[dict],
    ocr_page_nums: list[int],
    empty_pages: int,
) -> str:
    if not pages:
        return "empty"

    total_text = sum(len(p.get("text", "")) for p in pages)
    n = len(pages)
    avg = total_text / n

    ocr_ratio = len(ocr_page_nums) / n if n else 0
    empty_ratio = empty_pages / n if n else 0

    if avg > 200 and empty_ratio < 0.2:
        return "good"
    if avg > 80 or (ocr_ratio > 0 and avg > 40):
        return "partial"
    return "poor"
