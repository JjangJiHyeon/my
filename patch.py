import os

base_dir = r"c:\Users\jihyeon\Desktop\my"
pdf_parser_path = os.path.join(base_dir, "parsers", "pdf_parser.py")
app_path = os.path.join(base_dir, "app.py")

# 1. Update pdf_parser.py
with open(pdf_parser_path, "r", encoding="utf-8") as f:
    pdf_code = f.read()

# Replace _render_preview signature and body
render_preview_old = """def _render_preview(
    page: fitz.Page, doc_id: str, idx: int, warnings: list[str]
) -> str | None:
    \"\"\"Render page to PNG and return the API path. Never raises.\"\"\"
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
        return None"""

render_preview_new = """def _render_preview(
    page: fitz.Page, pw: float, ph: float, doc_id: str, idx: int, warnings: list[str]
) -> tuple[str | None, float | None, float | None, float | None, float | None, str | None]:
    \"\"\"Render page to PNG and return (api_path, pw_px, ph_px, scale_x, scale_y, error_msg).\"\"\"
    try:
        if not _result_dir:
            msg = "RESULT_DIR not set — preview skipped"
            warnings.append(msg)
            return None, None, None, None, None, msg
        out_dir = _preview_dir(doc_id)
        fname = f"page_{idx + 1}.{PREVIEW_FORMAT}"
        fpath = os.path.join(out_dir, fname)

        mat = fitz.Matrix(PREVIEW_ZOOM, PREVIEW_ZOOM)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        pix.save(fpath)

        path = f"/api/documents/{doc_id}/pages/{idx + 1}/preview"
        return path, float(pix.width), float(pix.height), float(pix.width)/pw, float(pix.height)/ph, None
    except Exception as exc:
        msg = f"Preview generation failed: {exc}"
        warnings.append(msg)
        return None, None, None, None, None, msg"""
pdf_code = pdf_code.replace(render_preview_old, render_preview_new)

# Add _validate_and_format_block function
validate_block_code = """
# ── block validation/normalization ──────────────────────────────────

def _validate_and_format_block(
    blk: dict,
    page_width: float,
    page_height: float,
    page_num: int,
    dropped_blocks: list,
    bbox_warnings: list
) -> dict | None:
    \"\"\"Normalize block schema and validate/clamp bbox.\"\"\"
    blkid = blk.get("id", "unknown")
    source = blk.get("source", "unknown")
    raw_bbox = blk.get("bbox")
    
    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
        dropped_blocks.append({"id": blkid, "source": source, "reason": "invalid_bbox_length"})
        return None
    
    try:
        x0, y0, x1, y1 = float(raw_bbox[0]), float(raw_bbox[1]), float(raw_bbox[2]), float(raw_bbox[3])
    except (ValueError, TypeError):
        dropped_blocks.append({"id": blkid, "source": source, "reason": "invalid_bbox_type"})
        return None

    # fix reversed bbox
    if x0 > x1:
        x0, x1 = x1, x0
    if y0 > y1:
        y0, y1 = y1, y0
        
    original_coords = (x0, y0, x1, y1)

    # clamp to page bounds
    x0 = max(0.0, min(float(page_width), x0))
    x1 = max(0.0, min(float(page_width), x1))
    y0 = max(0.0, min(float(page_height), y0))
    y1 = max(0.0, min(float(page_height), y1))

    if (x0, y0, x1, y1) != original_coords:
        reason = "negative_bbox_clamped" if any(v < 0 for v in original_coords) else "page_bounds_clamped"
        bbox_warnings.append(f"Block {blkid} ({source}): {reason} from {original_coords} to {(x0, y0, x1, y1)}")

    width = x1 - x0
    height = y1 - y0
    
    # drop if too small (< 2x2 points)
    if width < 2 or height < 2:
        dropped_blocks.append({"id": blkid, "source": source, "reason": "too_small_dropped"})
        return None
        
    btype = blk.get("type", "unknown")
    if btype == "chart_like":
        btype = "chart"
    if btype not in ("text", "table", "image", "chart", "title", "footer", "unknown"):
        btype = "unknown"

    meta = blk.get("extra", {})

    return {
        "id": blkid,
        "type": btype,
        "bbox": [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)],
        "text": blk.get("text", ""),
        "page_num": page_num,
        "source": source,
        "score": blk.get("confidence", 0.0),
        "meta": meta
    }

# ── per-page processing ─────────────────────────────────────────────
"""
pdf_code = pdf_code.replace("# ── per-page processing ─────────────────────────────────────────────\n", validate_block_code)

debug_vars_old = """    warnings: list[str] = []
    extraction_order: list[str] = []
    blocks: list[dict[str, Any]] = []
    block_counter = 0

    # ── 0. Preview image ──────────────────────────────────────────
    preview_path = _render_preview(page, doc_id, idx, warnings)"""

debug_vars_new = """    warnings: list[str] = []
    extraction_order: list[str] = []
    blocks: list[dict[str, Any]] = []
    block_counter = 0
    
    dropped_blocks: list[dict[str, Any]] = []
    bbox_warnings: list[str] = []

    # ── 0. Preview image ──────────────────────────────────────────
    preview_path, preview_w, preview_h, scale_x, scale_y, preview_err = _render_preview(page, pw, ph, doc_id, idx, warnings)"""
pdf_code = pdf_code.replace(debug_vars_old, debug_vars_new)

return_block_old = """    parser_debug = {
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
    }"""
    
return_block_new = """    valid_blocks = []
    block_type_counts = {"text": 0, "table": 0, "image": 0, "chart": 0, "title": 0, "footer": 0, "unknown": 0}
    
    candidate_counts = {
        "pymupdf_text_blocks": sum(1 for b in blocks if b["source"] == "pymupdf"),
        "pdfplumber_tables": sum(1 for b in blocks if b["source"] == "pdfplumber"),
        "camelot_tables": sum(1 for b in blocks if b["source"] == "camelot"),
        "image_candidates": sum(1 for b in blocks if b["source"] == "heuristic"),
        "ocr_boxes": sum(1 for b in blocks if str(b["source"]).startswith("ocr_")),
    }
    
    for bk in blocks:
        validated = _validate_and_format_block(bk, pw, ph, idx + 1, dropped_blocks, bbox_warnings)
        if validated:
            valid_blocks.append(validated)
            block_type_counts[validated["type"]] = block_type_counts.get(validated["type"], 0) + 1

    candidate_counts["final_blocks"] = len(valid_blocks)

    parser_debug = {
        "preview_generated": preview_path is not None,
        "preview_error": preview_err,
        "native_text_chars": len(native_text),
        "ocr_used": ocr_applied,
        "ocr_trigger_reason": "native_text_below_threshold_or_images_present" if need_ocr else "none",
        "ocr_engine_used": ocr_engine,
        "merge_strategy": merge_strategy,
        "fallback_reason": fallback_reason,
        "candidate_counts": candidate_counts,
        "block_type_counts": block_type_counts,
        "dropped_blocks": dropped_blocks,
        "bbox_warnings": bbox_warnings,
        "parse_warnings": warnings,
        "extraction_order": extraction_order,
    }

    return {
        "page_num": idx + 1,
        "page_width": pw,
        "page_height": ph,
        "preview_width": preview_w,
        "preview_height": preview_h,
        "coord_space": "page_points",
        "preview_image": preview_path,
        "text": final_text,
        "tables": tables_for_page,
        "blocks": valid_blocks,
        "image_count": image_count,
        "text_source": text_source,
        "ocr_applied": ocr_applied,
        "ocr_confidence": ocr_confidence,
        "parser_debug": parser_debug,
    }"""

pdf_code = pdf_code.replace(return_block_old, return_block_new)

with open(pdf_parser_path, "w", encoding="utf-8") as f:
    f.write(pdf_code)

# 2. Update app.py legacy result logic
with open(app_path, "r", encoding="utf-8") as f:
    app_code = f.read()
    
app_legacy_old = 'required_fields = ("blocks", "preview_image", "parser_debug")'
app_legacy_new = 'required_fields = ("blocks", "preview_image", "parser_debug", "page_width")'
app_code = app_code.replace(app_legacy_old, app_legacy_new)

with open(app_path, "w", encoding="utf-8") as f:
    f.write(app_code)

print("Patch applied successfully.")
