import logging
import re
import os
import hashlib
import fitz
from typing import Any

logger = logging.getLogger(__name__)

# ── Metadata & Quality ──────────────────────────────────────────────

def extract_metadata(doc: fitz.Document) -> dict[str, Any]:
    meta = doc.metadata
    return {
        "title": meta.get("title", ""),
        "author": meta.get("author", ""),
        "subject": meta.get("subject", ""),
        "keywords": meta.get("keywords", ""),
        "creator": meta.get("creator", ""),
        "producer": meta.get("producer", ""),
        "creationDate": meta.get("creationDate", ""),
        "modDate": meta.get("modDate", ""),
        "page_count": doc.page_count,
    }

def assess_quality(pages: list[dict], ocr_pages: list[int], empty_pages: int) -> float:
    if not pages: return 0.0
    
    total_score = 0.0
    for p in pages:
        p_score = 100.0
        debug = p.get("parser_debug", {})
        
        # Penalize for errors or warnings
        if p.get("error"): p_score -= 50
        if debug.get("parse_warnings"): p_score -= len(debug["parse_warnings"]) * 5
        
        # Penalize for low confidence OCR
        if p.get("ocr_applied") and p.get("ocr_confidence", 100) < 60:
            p_score -= 20
            
        total_score += max(0.0, p_score)
        
    avg_score = total_score / len(pages)
    
    # Global penalties
    if empty_pages > len(pages) * 0.3: avg_score *= 0.7
    
    return round(avg_score, 2)

# ── Block Validation ────────────────────────────────────────────────

def validate_and_format_block(
    blk: dict,
    page_width: float,
    page_height: float,
    page_num: int,
    dropped_blocks: list,
    bbox_warnings: list
) -> dict | None:
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

    if x0 > x1: x0, x1 = x1, x0
    if y0 > y1: y0, y1 = y1, y0
        
    x0 = max(0.0, min(float(page_width), x0))
    x1 = max(0.0, min(float(page_width), x1))
    y0 = max(0.0, min(float(page_height), y0))
    y1 = max(0.0, min(float(page_height), y1))

    width = x1 - x0
    height = y1 - y0
    
    if width < 2 or height < 2:
        dropped_blocks.append({"id": blkid, "source": source, "reason": "too_small_dropped"})
        return None
        
    btype = blk.get("type", "unknown")
    if btype == "chart_like": btype = "chart"
    if btype not in ("text", "table", "image", "chart", "title", "footer", "unknown"):
        btype = "unknown"

    # Support both 'meta' (new modular pipelines) and 'extra' (legacy internal)
    meta = blk.get("meta") or blk.get("extra") or {}

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

# ── Common Heuristics ──────────────────────────────────────────────

def get_text_similarity(a: str, b: str) -> float:
    an = "".join(a.split()).lower()
    bn = "".join(b.split()).lower()
    if not an or not bn: return 0.0
    set_a, set_b = set(an), set(bn)
    if not set_a or not set_b: return 0.0
    return len(set_a & set_b) / max(len(set_a), len(set_b))

def get_source_priority(src: str) -> int:
    s = str(src).lower()
    if "pymupdf" in s: return 3
    if "pdfplumber" in s: return 3
    if "camelot" in s: return 3
    if "ocr" in s: return 1
    return 0

def is_visual_noise_text(text: str) -> bool:
    if not text: return True
    clean = text.strip()
    if not clean: return True
    
    noise_patterns = ["2025년", "qoq", "%", "contents", "본 자", "1분기", "2024", "사업별 주"]
    lower_clean = clean.lower()
    
    if re.fullmatch(r'^(20[2-3]\d)(년|말|분기|\.|2)?$', lower_clean): return True
    if lower_clean in noise_patterns: return True
    
    if len(clean) <= 8 and not re.search(r'[가-힣a-zA-Z]{4,}', clean): return True
        
    meaningful = re.findall(r'[a-zA-Z가-힣]', clean)
    if not meaningful or len(meaningful) < 2: return True
    # ── Table Extraction ──────────────────────────────────────────────

def extract_tables_via_pdfplumber(filepath: str, page_count: int) -> dict[int, list]:
    import pdfplumber
    table_map = {}
    try:
        with pdfplumber.open(filepath) as pdf:
            for i in range(min(len(pdf.pages), page_count)):
                page = pdf.pages[i]
                tables = page.find_tables()
                if tables:
                    page_tables = []
                    for t in tables:
                        page_tables.append({
                            "bbox": t.bbox,
                            "cells": t.extract()
                        })
                    table_map[i] = page_tables
    except Exception as e:
        logger.warning(f"pdfplumber failed: {e}")
    return table_map

# ── OCR Helpers ───────────────────────────────────────────────────

def trigger_ocr_analysis(page, pw, ph):
    # Placeholder for OCR trigger logic
    # In real implementation, this would call EasyOCR or Tesseract
    return {"ocr_applied": False, "blocks": []}
