import logging
import re
from typing import Any
from .utils import get_source_priority, get_text_similarity, is_visual_noise_text, validate_and_format_block

logger = logging.getLogger(__name__)

def process_text_report(doc, page_idx, plumber_tables):
    """Main processing flow for text-heavy reports."""
    page = doc[page_idx]
    pw, ph = page.rect.width, page.rect.height
    
    # 1. Native Text Extraction
    raw_blocks = page.get_text("dict")["blocks"]
    blocks = []
    
    for i, b in enumerate(raw_blocks):
        if "lines" not in b: continue
        for j, line in enumerate(b["lines"]):
            text = "".join(span["text"] for span in line["spans"])
            if not text.strip(): continue
            
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
            "extra": {"normalized_table": tbl.get("cells"), "summary_priority": "medium"}
        })
        
    # Note: Paragraph merging and Reading order sorting are now handled 
    # centrally in pdf_parser.py -> _finalize_page_results()
    
    return {
        "page_num": page_idx + 1,
        "page_width": pw,
        "page_height": ph,
        "blocks": blocks,
        "parser_debug": {
            "pipeline_used": "text_report_pipeline",
            "block_count": len(blocks)
        }
    }

# Removed redundant _merge_paragraphs, _infer_layout, _sort_blocks, _filter_headers_footers
# as they are now handled by the unified normalization phase in pdf_parser.py

def _merge_paragraphs(blocks, pw, ph, hint):
    """Report-specific merging: multi-line paragraphs."""
    # To keep it simple but effective, we merge blocks with small Y gaps and similar X positions
    text_blocks = [b for b in blocks if b["type"] == "text"]
    other_blocks = [b for b in blocks if b["type"] != "text"]
    
    if not text_blocks: return blocks
    
    # Sort by Y then X
    sorted_b = sorted(text_blocks, key=lambda b: (b["bbox"][1], b["bbox"][0]))
    merged = []
    skip = set()
    
    for i in range(len(sorted_b)):
        if i in skip: continue
        curr = sorted_b[i].copy()
        
        for j in range(i+1, len(sorted_b)):
            if j in skip: continue
            next_b = sorted_b[j]
            
            # Y gap check
            y_gap = next_b["bbox"][1] - curr["bbox"][3]
            x_diff = abs(next_b["bbox"][0] - curr["bbox"][0])
            
            # Tight constraints for reports
            if 0 <= y_gap < 12 and x_diff < 5:
                curr["text"] = curr["text"].strip() + " " + next_b["text"].strip()
                curr["bbox"] = [
                    min(curr["bbox"][0], next_b["bbox"][0]),
                    min(curr["bbox"][1], next_b["bbox"][1]),
                    max(curr["bbox"][2], next_b["bbox"][2]),
                    max(curr["bbox"][3], next_b["bbox"][3])
                ]
                skip.add(j)
            else:
                if y_gap > 20: break
                
        merged.append(curr)
        
    return merged + other_blocks

def _infer_layout(blocks, pw):
    text_blocks = [b for b in blocks if b["type"] == "text"]
    if len(text_blocks) < 5: return "single_column"
    
    x0s = [b["bbox"][0] for b in text_blocks if (b["bbox"][2]-b["bbox"][0]) < pw * 0.45]
    left = sum(1 for x in x0s if x < pw * 0.3)
    right = sum(1 for x in x0s if x > pw * 0.5)
    
    if left > 2 and right > 2: return "multi_column"
    return "single_column"

def _sort_blocks(blocks, hint, pw):
    if hint == "multi_column":
        # Sort by Column (Left then Right) then Y
        return sorted(blocks, key=lambda b: (1 if b["bbox"][0] > pw/2 else 0, b["bbox"][1]))
    return sorted(blocks, key=lambda b: (b["bbox"][1], b["bbox"][0]))

def _filter_headers_footers(blocks, ph):
    for b in blocks:
        y0, y1 = b["bbox"][1], b["bbox"][3]
        # Common spots for headers/footers in portrait reports
        if y0 < ph * 0.08 or y1 > ph * 0.92:
            b.setdefault("meta", {})["summary_exclude"] = True
            b["meta"]["summary_exclude_reason"] = "report_header_footer"
            b["meta"]["summary_priority"] = "low"
