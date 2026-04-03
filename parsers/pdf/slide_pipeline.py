import logging
import re
from typing import Any
from .utils import get_source_priority, get_text_similarity, is_visual_noise_text, validate_and_format_block

logger = logging.getLogger(__name__)

def process_slide_ir(doc, page_idx, plumber_tables):
    """Main processing flow for slide/IR documents."""
    page = doc[page_idx]
    pw, ph = page.rect.width, page.rect.height
    
    # 1. Native Page Data
    raw_blocks = page.get_text("dict")["blocks"]
    blocks = []
    
    for i, b in enumerate(raw_blocks):
        if "lines" not in b: continue
        for j, line in enumerate(b["lines"]):
            text = "".join(span["text"] for span in line["spans"])
            if not text.strip(): continue
            
            # Slide-specific early noise filter (decorative icons/logos)
            if is_visual_noise_text(text) and len(text) < 10:
                continue
                
            blocks.append({
                "id": f"p{page_idx+1}_b{len(blocks)}",
                "type": "text",
                "bbox": list(line["bbox"]),
                "text": text,
                "source": "pymupdf_native",
                "confidence": 1.0,
                "extra": {"span_count": len(line["spans"])}
            })
            
    # 2. Table Integration (pdfplumber)
    for i, tbl in enumerate(plumber_tables):
        blocks.append({
            "id": f"p{page_idx+1}_t{i}",
            "type": "table",
            "bbox": list(tbl.get("bbox", [0,0,pw,ph])),
            "text": "[Table Data]",
            "source": "pdfplumber",
            "confidence": 1.0,
            "extra": {"normalized_table": tbl.get("cells"), "summary_priority": "high"}
        })
            
    # 3. Page Role Estimation
    role = _estimate_page_role(blocks, page_idx, doc.page_count)
    
    # ── Salvage Logic ──────────────────────────
    salvage_applied = False
    salvage_source = "native"
    near_empty_reason = None
    
    text_content = "".join(b["text"] for b in blocks if b["type"] == "text").strip()
    if not blocks or len(text_content) < 50:
        if not blocks:
            near_empty_reason = "no_blocks_found"
        else:
            near_empty_reason = "sparse_text_content"
            
        salvaged_blocks, source = _salvage_page(page, pw, ph)
        if salvaged_blocks:
            blocks = salvaged_blocks
            salvage_applied = True
            salvage_source = source

    # 4. KPI/Bullet Protection & Block Classification
    # Note: sorting is now handled centrally in pdf_parser.py
    slide_signals = []
    for b in blocks:
        _classify_slide_block(b, slide_signals)
        
    return {
        "page_num": page_idx + 1,
        "page_width": pw,
        "page_height": ph,
        "blocks": blocks,
        "parser_debug": {
            "pipeline_used": "slide_ir_pipeline",
            "page_role": role,
            "block_count": len(blocks),
            "slide_like_signals": list(set(slide_signals)),
            "visual_preservation_reason": "kpi_and_bullet_protection",
            "empty_page_flag": not bool(blocks),
            "near_empty_reason": near_empty_reason,
            "salvage_applied": salvage_applied,
            "salvage_source": salvage_source
        }
    }

# ... (Previous helper functions kept as they are specialized) ...

def _classify_slide_block(b, signals):
    text = b["text"].strip()
    meta = b.setdefault("meta", {})
    
    # 1. KPI detection (large numbers with units)
    if re.search(r'[\d,]+(\.\d+)?\s*(%|원|USD|pt|조|억|만|배|배성|천)', text):
        if len(text) < 40:
            meta["slide_role"] = "kpi_metric"
            meta["summary_priority"] = "high"
            signals.append("kpi_detected")
            
    # 2. Bullet detection
    bullet_chars = ("▪", "•", "-", "□", "◦", "*", "+")
    if text.startswith(bullet_chars):
        meta["slide_role"] = "bullet_item"
        signals.append("bullet_list_detected")
        
    # 3. Title preservation (Sets type="title" for central sorting)
    if b["bbox"][1] < 100 and len(text) < 60:
        b["type"] = "title"  # Crucial for central _sort_reading_order
        meta["summary_role"] = "title"
        meta["summary_priority"] = "high"

def _salvage_page(page, pw, ph) -> tuple[list[dict], str]:
    """Attempts to recover content when standard extraction fails."""
    salvaged = []
    
    # 1. Try "blocks" mode (standard fallback)
    try:
        raw_blocks = page.get_text("blocks")
        for b in raw_blocks:
            text = b[4].strip()
            if not text: continue
            salvaged.append({
                "id": f"salvage_b{len(salvaged)}",
                "type": "text",
                "bbox": list(b[:4]),
                "text": text,
                "source": "pymupdf_salvage_blocks",
                "confidence": 0.8
            })
    except:
        pass
        
    # 2. Aggressive Small Span Collection (for fragments)
    try:
        # Use a very permissive flag set
        full_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_PRESERVE_IMAGES)
        for b in full_dict["blocks"]:
            if b["type"] != 0: continue
            for line in b["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    if not text: continue
                    
                    # Already captured?
                    if any(get_text_similarity(text, s["text"]) > 0.9 for s in salvaged):
                        continue
                        
                    salvaged.append({
                        "id": f"salvage_s{len(salvaged)}",
                        "type": "text",
                        "bbox": list(span["bbox"]),
                        "text": text,
                        "source": "pymupdf_salvage_spans",
                        "confidence": 0.6
                    })
    except:
        pass
        
    # Sort salvaged blocks
    salvaged.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))
    
    # Tiered source identification
    source = "native_salvage"
    if any("spans" in b["source"] for b in salvaged):
        source = "native_spans_fallback"
    if any("blocks" in b["source"] for b in salvaged):
        source = "native_blocks_fallback"
        
    return salvaged, source

def _estimate_page_role(blocks, page_num, total_pages):
    all_text = " ".join(b["text"] for b in blocks).lower().replace(" ", "")
    
    if page_num == 0: return "cover"
    if page_num >= total_pages - 1: return "disclaimer"
    
    # TOC keywords
    if any(k in all_text for k in ["목차", "contents", "index"]):
        return "toc"
        
    # Divider heuristics (Centered text, few blocks)
    if len(blocks) < 5:
        return "section_divider"
        
    return "body"

def _sort_slide_blocks(blocks):
    """Sort: Titles first, then by Y coordinate."""
    def sort_key(b):
        text = b.get("text", "").strip()
        # Title heuristic for slides: top of the page, relatively large or alone
        is_title = b["bbox"][1] < 120 and len(text) < 100
        return (0 if is_title else 1, b["bbox"][1], b["bbox"][0])
        
    return sorted(blocks, key=sort_key)

def _classify_slide_block(b, signals):
    text = b["text"].strip()
    meta = b.setdefault("meta", {})
    
    # 1. KPI detection (large numbers with units)
    if re.search(r'[\d,]+(\.\d+)?\s*(%|원|USD|pt|조|억|만|배|배성|천)', text):
        if len(text) < 40:
            meta["slide_role"] = "kpi_metric"
            meta["summary_priority"] = "high"
            signals.append("kpi_detected")
            
    # 2. Bullet detection
    bullet_chars = ("▪", "•", "-", "□", "◦", "*", "+")
    if text.startswith(bullet_chars):
        meta["slide_role"] = "bullet_item"
        signals.append("bullet_list_detected")
        
    # 3. Title preservation
    if b["bbox"][1] < 100 and len(text) < 60:
        meta["summary_role"] = "title"
        meta["summary_priority"] = "high"
