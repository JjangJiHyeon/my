import logging
import re
from typing import Any
from .utils import get_source_priority, get_text_similarity, is_visual_noise_text, validate_and_format_block

logger = logging.getLogger(__name__)

def process_dashboard_brief(doc, page_idx, plumber_tables):
    """
    Main processing flow for 1-page dashboards and briefs.
    Focuses on region-based segmentation (cards, KPI boxes).
    """
    page = doc[page_idx]
    pw, ph = page.rect.width, page.rect.height
    
    # 1. Native Data Extraction
    raw_blocks = page.get_text("dict")["blocks"]
    blocks = []
    
    # Early noise filter
    numeric_dropped = 0
    for i, b in enumerate(raw_blocks):
        if "lines" not in b: continue
        for j, line in enumerate(b["lines"]):
            text = "".join(span["text"] for span in line["spans"]).strip()
            if not text: continue
            
            # Filter extremely small numeric fragments that are likely noise
            if re.fullmatch(r'[\d\.\,%\+\-]{1,5}', text) and (line["bbox"][2]-line["bbox"][0]) < 10:
                numeric_dropped += 1
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
        
    # 3. Dashboard Segmentation (Adaptive Clustering)
    regions = _segment_dashboard_cluster(blocks, pw, ph)
    
    # 4. Merge fragments into regions (Cards)
    merged_blocks, absorbed_count = _merge_fragments_adaptive(blocks, regions)
    
    # Identify block types for debug
    block_types = [b.get("meta", {}).get("dashboard_role", "unknown") for b in merged_blocks]
    
    # Note: Global deduplication and Reading order sorting will be handled 
    # centrally in pdf_parser.py -> _finalize_page_results()
    
    return {
        "page_num": page_idx + 1,
        "page_width": pw,
        "page_height": ph,
        "blocks": merged_blocks,
        "parser_debug": {
            "pipeline_used": "dashboard_brief_pipeline",
            "dashboard_signals": ["adaptive_clustering_applied"],
            "segmented_regions": len(regions),
            "dashboard_block_types": list(set(block_types)),
            "fragment_absorbed_count": absorbed_count,
            "numeric_fragment_dropped_count": numeric_dropped
        }
    }

def _segment_dashboard_cluster(blocks: list[dict], pw: float, ph: float) -> list[list[float]]:
    """Adaptive clustering to find cards/boxes."""
    if not blocks: return []
    
    # Group blocks that are close to each other
    clusters = []
    visited = set()
    
    # Simple proximity-based clustering
    def is_close(b1, b2):
        # If boxes overlap or are within 30px
        box1 = b1["bbox"]
        box2 = b2["bbox"]
        
        # Horizontal closeness
        h_overlap = max(0, min(box1[2], box2[2]) - max(box1[0], box2[0]))
        v_dist = min(abs(box1[1]-box2[3]), abs(box1[3]-box2[1]))
        
        # Vertical closeness
        v_overlap = max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
        h_dist = min(abs(box1[0]-box2[2]), abs(box1[2]-box2[0]))
        
        if (h_overlap > 0 and v_dist < 40) or (v_overlap > 0 and h_dist < 60):
            return True
        return False

    for i, b in enumerate(blocks):
        if i in visited: continue
        cluster = [i]
        visited.add(i)
        
        # Grow cluster
        changed = True
        while changed:
            changed = False
            for j, other in enumerate(blocks):
                if j in visited: continue
                if any(is_close(blocks[c_idx], other) for c_idx in cluster):
                    cluster.append(j)
                    visited.add(j)
                    changed = True
        clusters.append(cluster)
        
    regions = []
    for cluster in clusters:
        bboxes = [blocks[idx]["bbox"] for idx in cluster]
        x0 = min(b[0] for b in bboxes)
        y0 = min(b[1] for b in bboxes)
        x1 = max(b[2] for b in bboxes)
        y1 = max(b[3] for b in bboxes)
        regions.append([x0, y0, x1, y1])
        
    return regions

def _merge_fragments_adaptive(blocks: list[dict], regions: list[list[float]]) -> tuple[list[dict], int]:
    """Merges blocks into cards based on clustered regions."""
    # Group blocks by which region they belong to
    region_map = {i: [] for i in range(len(regions))}
    absorbed_count = 0
    
    for b in blocks:
        bx0, by0, bx1, by1 = b["bbox"]
        center_x = (bx0 + bx1) / 2
        center_y = (by0 + by1) / 2
        
        assigned = False
        for i, (rx0, ry0, rx1, ry1) in enumerate(regions):
            if rx0 <= center_x <= rx1 and ry0 <= center_y <= ry1:
                region_map[i].append(b)
                assigned = True
                break
        # Fallback: find nearest region if tiny fragment outside
        if not assigned and regions:
            dists = []
            for rx0, ry0, rx1, ry1 in regions:
                dists.append(min(abs(center_x-rx0), abs(center_x-rx1)) + min(abs(center_y-ry0), abs(center_y-ry1)))
            nearest = dists.index(min(dists))
            region_map[nearest].append(b)
            absorbed_count += 1

    final_blocks = []
    for i, rb in region_map.items():
        if not rb: continue
        
        text_blocks = [b for b in rb if b["type"] == "text"]
        other_blocks = [b for b in rb if b["type"] != "text"]
        
        if text_blocks:
            # Sort naturally for the region
            text_blocks.sort(key=lambda b: (round(b["bbox"][1]/10), b["bbox"][0]))
            
            # Analyze content to assign role
            full_text = " ".join(b["text"] for b in text_blocks)
            role = "card"
            if len(text_blocks) >= 5 and any(re.search(r'\d', b["text"]) for b in text_blocks):
                role = "ranking" if "위" in full_text or "1." in full_text else "compact_table"
            elif "주요 내용" in full_text or "Issue" in full_text:
                role = "issue_box"
                
            combined_text = "\n".join(b["text"] for b in text_blocks)
            
            # Bounding box for the whole card
            x0 = min(b["bbox"][0] for b in text_blocks)
            y0 = min(b["bbox"][1] for b in text_blocks)
            x1 = max(b["bbox"][2] for b in text_blocks)
            y1 = max(b["bbox"][3] for b in text_blocks)
            
            final_blocks.append({
                "id": f"p_r{i}",
                "type": "text",
                "bbox": [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)],
                "text": combined_text,
                "source": "dashboard_segmentation",
                "meta": {
                    "dashboard_role": role,
                    "summary_priority": "high" if role != "card" else "medium"
                }
            })
            absorbed_count += len(text_blocks) - 1
            
        final_blocks.extend(other_blocks)
        
    return final_blocks, absorbed_count
