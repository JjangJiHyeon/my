"""
Table Pipeline V1 Utilities for Pinetree PDF Parser
Provides heuristics for scoring table candidates, segmenting dashboard pages, 
normalizing multi-engine table results, and choosing the best table format.
"""

import math
import re
from typing import Any, Dict, List

# ── 1. Table Candidate Selection ────────────────────────────────────────

def detect_table_candidates(page_width: float, page_height: float, blocks: List[Dict]) -> None:
    # Expanded candidate types to match project labels
    VALID_CANDIDATE_TYPES = ("table", "image", "chart", "unknown", "image_like", "chart_like", "figure_like", "visual_block")
    
    for b in blocks:
        meta = b.get("meta", {})
        b_type = b.get("type", "")
        b_source = str(b.get("source", ""))
        text = (b.get("text") or "").strip()
        
        # 1. Coverage Check
        if b_type not in VALID_CANDIDATE_TYPES and "ocr" not in b_source:
            continue

        signals = []
        rejects = []
        score = 1.0  # Start with a base neutral score

        # 2. Engine signals (High confidence)
        if b_source in ("camelot", "pdfplumber"):
            score += 3.0
            signals.append(f"engine={b_source}")

        # 3. Text content-based rejection (False Positives)
        text_upper = text.upper()
        
        # TOC / Appendix / Cover detection
        toc_keywords = ("CONTENTS", "목차", "INDEX", "APPENDIX", "부록")
        if any(kw in text_upper[:50] for kw in toc_keywords):
            score -= 5.0
            rejects.append("toc_or_appendix_keywords")
            
        # Disclaimer / Compliance / Branding
        disclaimer_keywords = ("DISCLAIMER", "COMPLIANCE", "면책", "준법", "COPYRIGHT", "ALL RIGHTS RESERVED")
        if any(kw in text_upper for kw in disclaimer_keywords):
            score -= 2.0
            rejects.append("disclaimer_compliance_zone")

        # Heading + Page Number pattern (TOC lines)
        if re.search(r'\.{5,}\s*\d+', text) or re.search(r'[\s_]{5,}\s*\d+', text):
            score -= 3.0
            rejects.append("toc_page_number_pattern")

        # 4. Layout & Density signals
        lines = text.split("\n")
        num_lines = len(lines)
        num_tokens = len(text.split())

        if num_lines >= 3:
            score += 1.0
            signals.append("min_rows_met")
            
            # Text length vs Row count
            avg_line_len = len(text) / num_lines
            if avg_line_len > 100:
                score -= 3.0
                rejects.append("paragraph_like_dump")
            elif avg_line_len < 40:
                score += 1.0
                signals.append("short_label_structure")

        # Numeric density
        num_numbers = len(re.findall(r'\d+', text))
        if num_numbers > 0:
            num_density = num_numbers / max(1, len(text.replace(" ", "")))
            if num_density > 0.15:
                score += 2.0
                signals.append("high_numeric_density")
            elif num_density < 0.02 and b_type != "image":
                score -= 2.0
                rejects.append("low_numeric_density")
        
        # 5. Visual/Geometric Rejection (KPI, Deco, Logo)
        bbox = b.get("bbox", [0, 0, page_width, page_height])
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        area_ratio = (w * h) / (page_width * page_height)
        
        # Very small area + few tokens = Logo/KPI/Badge
        if area_ratio < 0.005 or (area_ratio < 0.02 and num_tokens < 5):
            score -= 4.0
            rejects.append("small_kpi_or_logo_area")
        
        # Extreme aspect ratio (separators, vertical sidebars)
        if h > 0:
            aspect = w / h
            if aspect > 15 or aspect < 0.1:
                score -= 3.0
                rejects.append("decorative_separator_shape")

        # Apply final scores
        meta["table_candidate_score"] = float(score)
        meta["table_signals"] = signals
        meta["table_reject_signals"] = rejects
        
        # Threshold: if score is too low, it's definitely not a summary-ready table
        if score < 1.0:
            # Downgrade type to prevent summary generation
            if b_type == "table":
                b["type"] = "unknown"
                meta["classification_reason"] = f"table_pipeline_reject: {','.join(rejects)}"
            elif b_type in ("image", "chart_like", "figure_like", "visual_block"):
                # Mark as not-table-summary candidate
                meta["skip_table_normalization"] = True


# ── 2. Dashboard Subtable Segmentation ────────────────────────────────

def segment_dashboard_subtables(blocks: List[Dict]) -> None:
    """
    Groups numeric/label clusters into sub-tables using heading anchors.
    Assigns subtable_group_id, subtable_title, subtable_bbox to block metadata.
    """
    # Group text blocks that look like section headings
    headings = []
    candidates = []
    
    for b in blocks:
        meta = b.get("meta", {})
        if meta.get("summary_role") == "section_header" or b.get("type") == "title":
            headings.append(b)
        elif b.get("type") in ("table", "image", "chart_like") or meta.get("table_candidate_score", 0) > 1.0:
            candidates.append(b)
            
    if not headings or not candidates:
        return
        
    # Sort headings by Y
    headings.sort(key=lambda x: x["bbox"][1])

    # Assign candidates to the nearest heading above them
    for b in candidates:
        by1 = b["bbox"][1]
        best_heading = None
        min_dist = float('inf')
        
        for h in headings:
            hy2 = h["bbox"][3]
            # Heading must be above or slightly overlapping the block
            if hy2 <= by1 + 10:
                dist = by1 - hy2
                if dist < min_dist and dist < 200: # Threshold for 'nearest' (200 pts)
                    min_dist = dist
                    best_heading = h
                    
        if best_heading:
            meta = b.get("meta", {})
            meta["subtable_group_id"] = best_heading["id"]
            meta["subtable_title"] = best_heading.get("text", "").strip()
            
            # Subtable bbox could merge multiple blocks if needed, here we just mark it
            meta["subtable_bbox"] = b["bbox"]

# ── 2.5 Subtable Fragment Merge & Summary Checks ─────────────────────

def _estimate_expected_col_count(headers: List[str], rows: List[List[str]]) -> int:
    if headers and len(headers) > 1: return len(headers)
    return max((len(r) for r in rows), default=0)

def _has_row_label_structure(cells: List[str]) -> bool:
    if not cells or len(cells) < 2: return False
    first = cells[0]
    first_has_num = bool(re.search(r'\d', first))
    rest_nums = sum(1 for c in cells[1:] if re.search(r'\d', c))
    return not first_has_num and len(first) > 1 and rest_nums >= len(cells[1:]) * 0.5

def _rechunk_dense_numeric_cells(cells: List[str], expected_cols: int) -> List[str]:
    if not cells or expected_cols < 2: return cells
    if len(cells) <= expected_cols: return cells
    if _has_row_label_structure(cells):
        label = cells[0]
        numbers = cells[1:]
        chunk_size = math.ceil(len(numbers) / max(1, expected_cols - 1))
        new_cells = [label]
        for i in range(0, len(numbers), chunk_size):
            new_cells.append(" ".join(numbers[i:i+chunk_size]))
        return new_cells[:expected_cols]
    return cells

def _is_sparse_fragment_table(rows: List[List[str]], headers: List[str]) -> bool:
    if not rows: return True
    r_count = len(rows)
    c_count = max((len(r) for r in rows), default=0)
    total_cells = r_count * c_count
    filled_cells = sum(1 for r in rows for c in r if c.strip())
    if total_cells > 0 and filled_cells / total_cells < 0.2:
        return True
    return False

def _is_dashboard_mixed_table(norm_table: Dict[str, Any], raw_text: str) -> bool:
    rows = norm_table.get("rows", [])
    if not rows: return False
    
    narrative_score = 0
    if len(raw_text) > 100 and "다." in raw_text:
        sentences = [s.strip() for s in re.split(r'[.?!]\s+', raw_text) if len(s.strip()) > 15]
        if len(sentences) >= 2:
            narrative_score += 1
            
    r_lengths = [len([c for c in r if str(c).strip()]) for r in rows]
    has_long_row = any(l >= 3 for l in r_lengths)
    has_single_cell_long_row = any(l == 1 and len(str(rows[i][0])) > 50 for i, l in enumerate(r_lengths))
    
    if narrative_score > 0 and has_long_row and has_single_cell_long_row:
        return True
        
    return False

def is_dashboard_summary_ready_table(norm_table: Dict[str, Any]) -> tuple[bool, str]:
    if not norm_table: return False, "empty_table"
    
    rows = norm_table.get("rows", [])
    if len(rows) < 3:
        return False, "dashboard_compact_table_fail"
        
    reject_signals = norm_table.get("reject_signals", [])
    if "dashboard_mixed_table" in reject_signals or "giant_dump_table" in reject_signals or "mixed_narrative_table" in reject_signals:
        return False, "dashboard_rank_grid_mixed"
        
    # Check column/header consistency
    r_lengths = [len([c for c in r if str(c).strip()]) for r in rows]
    if not r_lengths: return False, "dashboard_compact_table_fail"
    mode_len = max(set(r_lengths), key=r_lengths.count)
    if mode_len < 2:
        return False, "dashboard_compact_table_fail"
        
    inconsistent_rows = sum(1 for l in r_lengths if l != mode_len and l > 0)
    if inconsistent_rows > len(rows) * 0.5:
        return False, "dashboard_multi_region_table"
        
    # Check narrative mixed
    for r in rows:
        for c in r:
            s_c = str(c).strip()
            if len(s_c) > 50 and s_c.endswith(("다.", "요.")):
                return False, "dashboard_rank_grid_mixed"
                
    # Check if multiple lists are merged
    header_counts = sum(1 for r in rows if sum(1 for c in r if isinstance(c, str) and not any(ch.isdigit() for ch in c)) > mode_len * 0.7)
    if header_counts > 3:
        return False, "dashboard_multi_region_table"
        
    return True, "dashboard_allowlist_kept"

def is_summary_ready_table(norm_table: Dict[str, Any]) -> tuple[bool, str]:
    if not norm_table: return False, "empty_table"
    reject_signals = norm_table.get("reject_signals", [])
    if "dashboard_mixed_table" in reject_signals: return False, "dashboard_mixed_table"
    if "giant_dump_table" in reject_signals: return False, "giant_dump_table"
    if "collapsed_row_table" in reject_signals: return False, "collapsed_row_table"
    if "micro_fragment_table" in reject_signals: return False, "micro_fragment_table"
    if "mixed_narrative_table" in reject_signals: return False, "mixed_narrative_table"
    if "non_rectangular_sparse_table" in reject_signals: return False, "non_rectangular_sparse_table"
    if "no_row_labels" in reject_signals: return False, "no_row_labels"
    
    rows = norm_table.get("rows", [])
    if not rows: return False, "no_rows"
    shape = norm_table.get("shape", {"rows": 0, "cols": 0})
    r_count, c_count = shape.get("rows", 0), shape.get("cols", 0)
    
    if r_count < 2: return False, "micro_fragment_table"
    headers = norm_table.get("headers", [])
    header_comps = sum(1 for h in headers if str(h).strip())
    if header_comps == 0: return False, "no_headers"
    
    first_col_nums = sum(1 for r in rows[1:] if r and re.search(r'\d', r[0]))
    data_rows = r_count - 1
    if data_rows > 0 and first_col_nums == data_rows and c_count <= 5:
        if data_rows < 3: return False, "no_row_labels"
        
    total_cells = r_count * c_count
    filled_cells = sum(1 for r in rows for c in r if str(c).strip())
    if total_cells > 0 and filled_cells / total_cells < 0.3:
        return False, "non_rectangular_sparse_table"
        
    return True, "summary_ready"

def merge_subtable_fragments(blocks: List[Dict], page_width: float, page_height: float) -> List[Dict]:
    groups = {}
    for b in blocks:
        meta = b.get("meta", {})
        gid = meta.get("subtable_group_id") or meta.get("subtable_title")
        if gid and (b.get("type") == "table" or meta.get("table_candidate_score", 0) > 1.0):
            groups.setdefault(gid, []).append(b)
            
    merged_dropped_ids = set()
    for gid, group_blocks in groups.items():
        if len(group_blocks) < 2: continue
            
        group_blocks.sort(key=lambda x: x.get("bbox", [0,0,0,0])[1])
        current_cluster = [group_blocks[0]]
        
        for i in range(1, len(group_blocks)):
            prev = current_cluster[-1]
            curr = group_blocks[i]
            
            p_bbox = prev.get("bbox", [0,0,0,0])
            c_bbox = curr.get("bbox", [0,0,0,0])
            y_dist = c_bbox[1] - p_bbox[3]
            
            x_overlap = max(0, min(p_bbox[2], c_bbox[2]) - max(p_bbox[0], c_bbox[0]))
            min_w = min(p_bbox[2]-p_bbox[0], c_bbox[2]-c_bbox[0])
            
            ptxt = prev.get("text", "")
            ctxt = curr.get("text", "")
            p_num = len(re.findall(r'\d', ptxt)) / max(1, len(ptxt.replace(" ","")))
            c_num = len(re.findall(r'\d', ctxt)) / max(1, len(ctxt.replace(" ","")))
            
            if y_dist < 50 and (x_overlap > 0 or min_w < 50 or (p_num > 0.1 and c_num > 0.1)):
                current_cluster.append(curr)
            else:
                if len(current_cluster) > 1:
                    _perform_merge(current_cluster, merged_dropped_ids)
                current_cluster = [curr]
                
        if len(current_cluster) > 1:
            _perform_merge(current_cluster, merged_dropped_ids)
            
    return [b for b in blocks if b["id"] not in merged_dropped_ids]

def _perform_merge(cluster: List[Dict], dropped_ids: set):
    base = cluster[0]
    base_meta = base.setdefault("meta", {})
    
    merged_ids = []
    merged_texts = [base.get("text", "")]
    
    b_box = base.get("bbox", [0,0,0,0])
    min_x, min_y, max_x, max_y = b_box[0], b_box[1], b_box[2], b_box[3]
    
    for b in cluster[1:]:
        dropped_ids.add(b["id"])
        merged_ids.append(b["id"])
        merged_texts.append(b.get("text", ""))
        
        bb = b.get("bbox", [0,0,0,0])
        min_x = min(min_x, bb[0])
        min_y = min(min_y, bb[1])
        max_x = max(max_x, bb[2])
        max_y = max(max_y, bb[3])
        
    merged_text = "\n".join(merged_texts)
    
    # Check summary readiness
    temp_norm = normalize_table_candidate(merged_text, "native")
    score_table_quality(temp_norm)
    is_ready, reason = is_summary_ready_table(temp_norm)
    
    base_meta["post_merge_summary_ready"] = is_ready
    base_meta["post_merge_reject_reason"] = reason if not is_ready else None
    
    if not is_ready:
        req = base_meta.setdefault("table_reject_signals", [])
        if reason not in req: req.append(reason)
        
    base["bbox"] = [min_x, min_y, max_x, max_y]
    base["text"] = merged_text
    base_meta["merged_from_ids"] = merged_ids
    base_meta["merged_fragment_count"] = len(cluster)
    base_meta["merged_bbox"] = base["bbox"]
    base_meta["merged_by"] = "subtable_group_merge"

# ── 3. Normalized Table Object ───────────────────────────────────────

def normalize_table_candidate(
    candidate_data: Any, 
    source_type: str, 
    title: str | None = None
) -> Dict[str, Any]:
    """
    Unified table object schema for Camelot, PDFPlumber, OCR, Native.
    """
    # Base schema
    obj = {
        "title": title,
        "headers": [],
        "rows": [],
        "header_rows": [],
        "shape": {"rows": 0, "cols": 0},
        "source": source_type,
        "row_label_col_index": 0,
        "numeric_col_indices": [],
        "ocr_fill_cells": 0,
        "ocr_only_cells": 0,
        "ocr_dependency_score": 0.0,
        "quality": {}
    }
    
    if not candidate_data:
        return obj

    rows = []
    
    if source_type in ("camelot", "pdfplumber"):
        rows = candidate_data
    else:
        if isinstance(candidate_data, str):
            raw_rows = []
            for line in candidate_data.strip().split("\n"):
                cells = [c.strip() for c in re.split(r'\s{2,}|\t', line) if c.strip()]
                if not cells: continue
                
                expanded_cells = []
                for c in cells:
                    tokens = c.split()
                    nums = sum(1 for t in tokens if re.search(r'\d', t))
                    if len(tokens) >= 3 and nums >= len(tokens) - 1:
                        expanded_cells.extend(tokens)
                    else:
                        expanded_cells.append(c)
                raw_rows.append(expanded_cells)
                
            expected_cols = _estimate_expected_col_count(raw_rows[0] if raw_rows else [], raw_rows)
            for r in raw_rows:
                if len(r) > expected_cols + 2 and expected_cols > 1:
                    r = _rechunk_dense_numeric_cells(r, expected_cols)
                rows.append(r)

    if not rows:
        return obj

    shape_rows = len(rows)
    shape_cols = max(len(r) for r in rows) if rows else 0
    obj["shape"]["rows"] = shape_rows
    obj["shape"]["cols"] = shape_cols

    # Ensure rectangular
    padded_rows = []
    for r in rows:
        diff = shape_cols - len(r)
        padded_rows.append(r + [""] * diff)
        
    obj["rows"] = padded_rows
    
    if len(padded_rows) > 0:
        obj["headers"] = padded_rows[0]
        obj["header_rows"] = [0]
        
    # Detect numeric columns (scan first few rows)
    num_cols = set()
    scan_limit = min(5, len(padded_rows))
    for i in range(1, scan_limit):
        for col_idx, cell in enumerate(padded_rows[i]):
            if any(char.isdigit() for char in cell):
                num_cols.add(col_idx)
    
    obj["numeric_col_indices"] = list(num_cols)
    
    return obj

# ── 4. Engine Selection & Scoring ────────────────────────────────────

def score_table_quality(norm_table: Dict[str, Any]) -> float:
    """
    Populates 'quality' metrics in normalized table.
    Returns overall score.
    """
    rows = norm_table["rows"]
    if not rows:
        norm_table["quality"] = {
            "empty_cell_ratio": 1.0,
            "header_completeness": 0.0,
            "numeric_alignment_score": 0.0,
            "duplication_score": 0.0,
            "shape_plausibility_score": 0.0,
            "overall_table_quality": 0.0
        }
        return 0.0

    total_cells = len(rows) * norm_table["shape"]["cols"]
    empty_cells = sum(1 for r in rows for c in r if not c.strip())
    empty_ratio = empty_cells / max(1, total_cells)
    
    # Header completeness
    headers = norm_table["headers"]
    header_comps = sum(1 for h in headers if h.strip()) / max(1, len(headers))
    
    unique_cells = set(c.strip() for r in rows for c in r if c.strip())
    filled_cells = total_cells - empty_cells
    dup_score = 1.0 - (len(unique_cells) / max(1, filled_cells))
    
    # Shape plausibility
    r_count = norm_table["shape"]["rows"]
    c_count = norm_table["shape"]["cols"]

    # Check giant dump / collapsed / mixed narrative
    is_giant_dump = False
    is_mixed_narrative = False
    has_single_cell_rows = False
    mixed_narrative_count = 0
    
    raw_text = "\n".join(" ".join(str(c) for c in r) for r in rows)
    if _is_dashboard_mixed_table(norm_table, raw_text):
        reject_list.append("dashboard_mixed_table")
        
    is_collapsed_row = False
    
    for r in rows:
        num_count = sum(1 for c in r if re.search(r'\d', c))
        if len(r) > 15 and num_count > 10:
            is_collapsed_row = True
        if len(r) > 0:
            first = r[0]
            if len(first) > 80 and num_count > 2:
                is_mixed_narrative = True
                
    if r_count < 3 and len("".join(c for r in rows for c in r)) > 300:
        is_giant_dump = True

    shape_score = 1.0
    if c_count < 2 or r_count < 2:
        shape_score = 0.2
    if r_count > 50:
        shape_score = 0.5
        
    reject_list = []
    if is_giant_dump or is_collapsed_row or is_mixed_narrative:
        shape_score = 0.1
        if is_giant_dump: reject_list.append("giant_dump_table")
        if is_collapsed_row: reject_list.append("collapsed_row_table")
        if is_mixed_narrative: reject_list.append("mixed_narrative_table")
        
    if _is_sparse_fragment_table(rows, norm_table.get("headers", [])):
        reject_list.append("non_rectangular_sparse_table")
        
    norm_table["reject_signals"] = reject_list
        
    num_score = len(norm_table["numeric_col_indices"]) / max(1, c_count)

    # OCR dependency penalty
    ocr_dep = norm_table.get("ocr_dependency_score", 0.0)

    overall = (
        (1.0 - empty_ratio) * 2.0 +
        (header_comps) * 1.5 +
        (num_score) * 1.0 +
        (shape_score) * 2.0 -
        (dup_score) * 1.0 -
        (ocr_dep) * 1.5
    )

    norm_table["quality"] = {
        "empty_cell_ratio": empty_ratio,
        "header_completeness": header_comps,
        "numeric_alignment_score": num_score,
        "duplication_score": dup_score,
        "shape_plausibility_score": shape_score,
        "overall_table_quality": overall
    }
    
    return overall

def choose_best_table_candidate(candidates: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    """
    Given a list of normalized tables from different engines, pick the highest quality one.
    """
    if not candidates:
        return None
        
    for c in candidates:
        score_table_quality(c)
        
    candidates.sort(key=lambda x: x["quality"]["overall_table_quality"], reverse=True)
    return candidates[0]
