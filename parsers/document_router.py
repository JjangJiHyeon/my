"""
Document Router for PDF classification.
Classifies PDF documents into 4 types:
1. slide_ir
2. text_report
3. table_heavy
4. dashboard_brief
"""

from __future__ import annotations
import logging
from typing import Any
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

def pre_route_document(doc: fitz.Document) -> str:
    """
    Fast preliminary routing based on document metadata, page count, and orientation.
    Useful for dispatching to specific pipelines before full parsing.
    """
    page_count = doc.page_count
    if page_count == 0: return "unknown"
    
    # 1. Check orientation of first 3 pages
    landscape_count = 0
    checks = min(3, page_count)
    for i in range(checks):
        rect = doc[i].rect
        if rect.width > rect.height * 1.05:
            landscape_count += 1
            
    # 2. Heuristics
    if landscape_count >= checks * 0.6:
        return "slide_ir"
        
    meta = doc.metadata or {}
    title = meta.get("title", "").lower()
    subject = meta.get("subject", "").lower()
    
    if "presentation" in title or "ir" in title or "실적" in title:
        if page_count < 50: return "slide_ir"
        
    if page_count == 1:
        return "dashboard_brief"
        
    if page_count <= 5:
        # 3. Refined Dashboard Check (Density-based)
        # Avoid misclassifying short text reports as dashboards
        try:
            p0 = doc[0]
            text0 = p0.get_text("text").strip()
            # If 1st page is very text-heavy (e.g. Hanwha), it's likely a report
            if len(text0) > 1800:
                return "text_report"
            # If it has long continuous lines (narrative), prefer report
            if any(len(line) > 70 for line in text0.split('\n')[:30]):
                return "text_report"
        except:
            pass
        return "dashboard_brief"
        
    return "text_report"

def route_document(doc: fitz.Document | None, pages_data: list[dict[str, Any]], metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Analyzes document-wide features and determines the document type.
    Works for PDF (with fitz doc) and DOC/HWP (using pages_data and metadata).
    """
    page_count = len(pages_data)
    if page_count == 0:
        return {
            "document_type": "unknown",
            "routing_signals": {"page_count": 0},
            "routing_reason": "No pages or content found in document."
        }

    # 1. Feature Collection
    total_chars = 0
    total_images = 0
    total_tables = 0
    total_blocks = 0
    landscape_pages = 0
    
    page_types = []
    layout_hints = []
    
    # Narrative analysis (from DOC/HWP metadata if available)
    narrative = metadata.get("narrative_analysis", {}) if metadata else {}
    
    for p in pages_data:
        total_chars += len(p.get("text", ""))
        total_images += p.get("image_count", 0)
        total_tables += len(p.get("tables", []))
        
        # Use existing parser_debug info if available
        debug = p.get("parser_debug", {})
        total_blocks += debug.get("candidate_counts", {}).get("final_blocks", 0)
        
        page_types.append(debug.get("page_type", "unknown"))
        layout_hints.append(debug.get("page_layout_hint", "unknown"))
        
        # Check orientation (PDF specific)
        pw, ph = p.get("page_width", 0), p.get("page_height", 0)
        if pw > 0 and ph > 0 and pw > ph * 1.05:
            landscape_pages += 1

    avg_text_density = total_chars / page_count
    avg_visual_density = total_images / page_count
    avg_table_density = total_tables / page_count
    avg_block_count = total_blocks / page_count
    landscape_ratio = landscape_pages / page_count if page_count > 0 else 0

    signals = {
        "page_count": page_count,
        "avg_text_density": round(avg_text_density, 1),
        "avg_visual_density": round(avg_visual_density, 2),
        "avg_table_density": round(avg_table_density, 2),
        "avg_block_count": round(avg_block_count, 1),
        "landscape_ratio": round(landscape_ratio, 2),
        "page_type_distribution": {t: page_types.count(t) for t in set(page_types)},
        "layout_hint_distribution": {h: layout_hints.count(h) for h in set(layout_hints)},
        "is_truly_dashboard_signal": narrative.get("is_truly_dashboard", False)
    }

    # (Previous routing logic continues...)
    # I'll truncate for brevity in instruction

    # 2. Routing Logic
    
    # Priority 1: Slide IR (Landscape dominant or many slide hints)
    # Note: Only applicable if we have orientation info (PDF)
    if landscape_ratio > 0.6 or layout_hints.count("slide_like") > page_count * 0.4:
        doc_type = "slide_ir"
        reason = f"Predominant landscape layout ({landscape_ratio:.2f}) or slide-like hints."
    
    # Priority 2: Dashboard Brief (Low page count + high density/dashboard hints)
    elif (page_count <= 5 and (signals["is_truly_dashboard_signal"] or signals["page_type_distribution"].get("dashboard_kpi_like", 0) >= 1 or avg_visual_density > 1.5 or avg_block_count > 15)):
        doc_type = "dashboard_brief"
        reason = "Short document with high visual/block density or explicit dashboard signals."
        
    # Priority 3: Table Heavy
    elif avg_table_density > 1.2 or signals["page_type_distribution"].get("appendix_or_table_heavy", 0) > page_count * 0.3:
        doc_type = "table_heavy"
        reason = f"High average table density ({avg_table_density:.2f}) or recurrent table-centric layout."
        
    # Default: Text Report
    else:
        doc_type = "text_report"
        reason = "Standard text-centric document structure."

    return {
        "document_type": doc_type,
        "routing_signals": signals,
        "routing_reason": reason
    }
