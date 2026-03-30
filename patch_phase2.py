import re

with open(r"c:\Users\jihyeon\Desktop\my\parsers\pdf_parser.py", "r", encoding="utf-8") as f:
    code = f.read()

# 1. We need to inject the massive new heuristic functions just before _process_page
heuristic_functions = '''
# ── Phase 2 Heuristics ──────────────────────────────────────────────

import difflib

def _text_similarity(a: str, b: str) -> float:
    an = "".join(a.split()).lower()
    bn = "".join(b.split()).lower()
    if not an and not bn: return 1.0
    if not an or not bn: return 0.0
    return difflib.SequenceMatcher(None, an, bn).ratio()

def _get_source_priority(src: str) -> int:
    # higher is better
    s = str(src).lower()
    if "pymupdf" in s: return 10
    if "pdfplumber" in s: return 9
    if "camelot" in s: return 8
    if "ocr" in s: return 5
    return 1

def _reclassify_images(blocks: list[dict], pw: float, ph: float, quality_notes: list, classification_overrides: list, dropped: list) -> list[dict]:
    out = []
    page_area = pw * ph if pw * ph > 0 else 1
    for b in blocks:
        if b["type"] in ("image", "chart", "chart_like"):
            x0, y0, x1, y1 = b["bbox"]
            area = (x1 - x0) * (y1 - y0)
            area_ratio = area / page_area
            # Drop massive background regions
            if area_ratio > 0.85:
                dropped.append({"id": b["id"], "reason": "background_dropped", "source": b["source"]})
                quality_notes.append(f"Dropped {b['id']} as it covers {area_ratio*100:.1f}% of page")
                continue
            
            # Additional heuristic: if bounding box contains lots of text blocks entirely inside it?
            # For now, just maintain as image/chart but log reason.
            b["meta"]["classification_reason"] = "area_ratio_check"
        out.append(b)
    return out

def _deduplicate_blocks(blocks: list[dict], merge_events: list, dedup_stats: dict) -> list[dict]:
    out = []
    dropped_count = 0
    # Process from highest priority to lowest, so we drop the lowers
    # We sort by priority so high priority gets inserted first
    sorted_blocks = sorted(blocks, key=lambda x: _get_source_priority(x["source"]), reverse=True)
    
    for b in sorted_blocks:
        x0, y0, x1, y1 = b["bbox"]
        b_area = (x1-x0)*(y1-y0)
        if b_area <= 0: continue
        
        is_duplicate = False
        for ex in out:
            # Type compatible? Usually text vs text
            if b["type"] not in ("text", "title", "footer", "unknown") or ex["type"] not in ("text", "title", "footer", "unknown"):
                continue
            
            ex0, ey0, ex1, ey1 = ex["bbox"]
            ix0, iy0 = max(x0, ex0), max(y0, ey0)
            ix1, iy1 = min(x1, ex1), min(y1, ey1)
            iarea = max(0, ix1 - ix0) * max(0, iy1 - iy0)
            
            if iarea / b_area > 0.7 or iarea / max(1, (ex1-ex0)*(ey1-ey0)) > 0.7:
                # BBox overlaps significantly. Check text similarity.
                sim = _text_similarity(b["text"], ex["text"])
                if sim > 0.6 or (len(b["text"]) < 5 and sim > 0.4):
                    is_duplicate = True
                    merge_events.append({
                        "kept": ex["id"], "dropped": b["id"], 
                        "reason": f"dedup_overlap_sim_{sim:.2f}_priority_{ex['source']}_over_{b['source']}"
                    })
                    dropped_count += 1
                    break
        
        if not is_duplicate:
            out.append(b)
            
    dedup_stats["dropped_duplicates"] = dropped_count
    return sorted(out, key=lambda x: int(x["id"].split("_b")[-1]) if "_b" in x["id"] else 0)

def _merge_adjacent_text_blocks(blocks: list[dict], merge_events: list, text_merge_stats: dict, pw: float) -> list[dict]:
    # Group text blocks into paragraph-like clumps
    out = []
    merged_count = 0
    
    # Sort blocks natively by vertical Y then X
    # Using center points helps
    sorted_b = sorted(blocks, key=lambda b: (b["bbox"][1] + (b["bbox"][3]-b["bbox"][1])/2, b["bbox"][0]))
    
    merged = []
    skip = set()
    
    for i in range(len(sorted_b)):
        if i in skip: continue
        b1 = sorted_b[i]
        
        # We only try to merge text-like blocks generated natively or OCR (not tables/images)
        if b1["type"] != "text":
            merged.append(b1)
            continue
            
        current = b1.copy()
        
        for j in range(i+1, len(sorted_b)):
            if j in skip: continue
            b2 = sorted_b[j]
            
            if b2["type"] != "text":
                continue
                
            y_gap = b2["bbox"][1] - current["bbox"][3]
            x_diff = abs(b2["bbox"][0] - current["bbox"][0])
            width1 = current["bbox"][2] - current["bbox"][0]
            width2 = b2["bbox"][2] - b2["bbox"][0]
            w_diff_ratio = abs(width1 - width2) / max(1, width1)
            
            # Rules: 
            # - Must be close vertically (-10 to 20 pts)
            # - Must be roughly same left margin (< 30 pts diff)
            # - Width difference shouldn't be huge (> 50% diff)
            # - Neither should be tiny footer-like or title isolated (we only checked 'text' type anyway, title/footer are separate)
            if -15 <= y_gap <= 25 and x_diff < 30 and w_diff_ratio < 0.6:
                # Merge current and b2
                cx0 = min(current["bbox"][0], b2["bbox"][0])
                cy0 = min(current["bbox"][1], b2["bbox"][1])
                cx1 = max(current["bbox"][2], b2["bbox"][2])
                cy1 = max(current["bbox"][3], b2["bbox"][3])
                
                current["bbox"] = [round(cx0, 2), round(cy0, 2), round(cx1, 2), round(cy1, 2)]
                current["text"] = current["text"].strip() + "\\n" + b2["text"].strip()
                current["score"] = min(current.get("score", 1.0), b2.get("score", 1.0))
                
                merge_events.append({"kept": current["id"], "dropped": b2["id"], "reason": "paragraph_merge"})
                merged_count += 1
                skip.add(j)
            else:
                # If it's way below, we break search as they are sorted by Y.
                if y_gap > 100:
                    break
                    
        merged.append(current)
        
    text_merge_stats["merged_pairs"] = merged_count
    return merged

def _infer_page_layout_hint(blocks: list[dict], pw: float, ph: float) -> str:
    # Multi-column vs single vs slide
    text_blocks = [b for b in blocks if b["type"] in ("text", "title", "footer")]
    img_blocks = [b for b in blocks if b["type"] in ("image", "chart", "table")]
    
    if pw > ph * 1.1: 
        return "slide_like"
        
    # Check for multi-column by looking at x0 coordinates
    x0s = [b["bbox"][0] for b in text_blocks if (b["bbox"][2]-b["bbox"][0]) < pw * 0.45]
    if len(x0s) > 3:
        left_col = sum(1 for x in x0s if x < pw * 0.3)
        right_col = sum(1 for x in x0s if x > pw * 0.5)
        if left_col > 1 and right_col > 1:
            return "multi_column"
            
    if len(img_blocks) > len(text_blocks) / 2 and len(img_blocks) > 0:
        return "mixed_visual"
        
    return "single_column"

def _sort_reading_order(blocks: list[dict], hint: str, pw: float, ph: float) -> list[dict]:
    # Sort differently based on hint
    if hint == "multi_column":
        # Sort by X column roughly, then Y
        # We split page into left/right
        mid_x = pw / 2
        left_blocks = []
        right_blocks = []
        other_blocks = [] # titles span across etc
        for b in blocks:
            x0, y0, x1, y1 = b["bbox"]
            cx = x0 + (x1-x0)/2
            if (x1-x0) > pw * 0.6: 
                other_blocks.append(b) # Spans
            elif cx < mid_x:
                left_blocks.append(b)
            else:
                right_blocks.append(b)
                
        # Sort internal
        left_blocks.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))
        right_blocks.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))
        other_blocks.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))
        
        # Merge back: top-spanning, then left, then right, then bottom-spanning
        # Simplification: just sort everything by column index then Y
        # We assign a column_idx: 0 for spanning_top, 1 for left, 2 for right, 3 for spanning_bot
        # Actual implementation: sort all by:
        # Title first
        # Main text
        blocks_sorted = sorted(blocks, key=lambda b: (
            0 if b["type"] == "title" else 2 if b["type"] == "footer" else 1,
            min(1, int((b["bbox"][0] + (b["bbox"][2]-b["bbox"][0])/2) / (pw/2 + 1))) if (b["bbox"][2]-b["bbox"][0]) < pw * 0.6 else 0, # column
            b["bbox"][1] # y
        ))
        return blocks_sorted
        
    elif hint == "slide_like":
        # Titles top, then cluster by proximity to center, etc.
        # Fallback to simple top->down, left->right
        blocks_sorted = sorted(blocks, key=lambda b: (
            0 if b["type"] == "title" else 2 if b["type"] == "footer" else 1,
            b["bbox"][1] // 30, # bucket y
            b["bbox"][0]
        ))
        return blocks_sorted
        
    # single_column / mixed
    blocks_sorted = sorted(blocks, key=lambda b: (
        0 if b["type"] == "title" else 2 if b["type"] == "footer" else 1,
        b["bbox"][1] // 20, 
        b["bbox"][0]
    ))
    return blocks_sorted

def _assign_block_scores(blocks: list[dict]):
    for b in blocks:
        if b["type"] in ("title", "footer", "text"):
            # length-based penalization for native
            slen = len(b["text"].strip())
            if slen < 3:
                sc = min(b.get("score", 1.0), 0.5)
                b["score"] = sc
                b["meta"]["score_reason"] = "too_short_text"
            else:
                b["meta"]["score_reason"] = "text_completeness_ok"
        else:
            b["meta"]["score_reason"] = f"type_heuristic_{b['type']}"
'''

# 2. Inject right above `def _process_page`
injection_point = "# ── per-page processing ─────────────────────────────────────────────"
code = code.replace(injection_point, heuristic_functions + "\n" + injection_point)

# 3. Update the end of `_process_page`
# From:
#     for bk in blocks:
#         validated = _validate_and_format_block(bk, pw, ph, idx + 1, dropped_blocks, bbox_warnings)
#         if validated:
#             valid_blocks.append(validated)
#             block_type_counts[validated["type"]] = block_type_counts.get(validated["type"], 0) + 1
#     candidate_counts["final_blocks"] = len(valid_blocks)

# To include the pipeline
search_str = """
    valid_blocks = []
    block_type_counts = {"text": 0, "table": 0, "image": 0, "chart": 0, "title": 0, "footer": 0, "unknown": 0}
    
    candidate_counts = {
"""
post_processing = """
    valid_blocks = []
    for bk in blocks:
        validated = _validate_and_format_block(bk, pw, ph, idx + 1, dropped_blocks, bbox_warnings)
        if validated: valid_blocks.append(validated)
        
    quality_notes = []
    merge_events = []
    classification_overrides = []
    dedup_stats = {}
    text_merge_stats = {}
    
    valid_blocks, more_dropped = _reclassify_images(valid_blocks, pw, ph, quality_notes, classification_overrides, dropped_blocks)
    valid_blocks = _deduplicate_blocks(valid_blocks, merge_events, dedup_stats)
    valid_blocks = _merge_adjacent_text_blocks(valid_blocks, merge_events, text_merge_stats, pw)
    
    hint = _infer_page_layout_hint(valid_blocks, pw, ph)
    valid_blocks = _sort_reading_order(valid_blocks, hint, pw, ph)
    
    _assign_block_scores(valid_blocks)

    block_type_counts = {"text": 0, "table": 0, "image": 0, "chart": 0, "title": 0, "footer": 0, "unknown": 0}
    for b in valid_blocks:
        t = b["type"]
        block_type_counts[t] = block_type_counts.get(t, 0) + 1
        
    candidate_counts = {
"""
if search_str in code:
    code = code.replace(search_str, post_processing.strip() + "\n    candidate_counts = {\n")
else:
    print("Could not find search_str!")

# 4. We also need to remove the old iteration:
#     for bk in blocks:
#         validated = _validate_and_format_block(bk, pw, ph, idx + 1, dropped_blocks, bbox_warnings)
#         if validated:
#             valid_blocks.append(validated)
#             block_type_counts[validated["type"]] = block_type_counts.get(validated["type"], 0) + 1
# This logic is inside code. Let's regex it out so we don't do it twice.
old_val_loop = re.compile(r'for bk in blocks:\s*validated = _validate_and_format_block.*?block_type_counts\[validated\["type"\]\] \+ 1', re.DOTALL)
# Wait, my replacement above completely overshadowed it if I just replace it correctly.
# Oh, the original code had:
#     for bk in blocks:
#         validated = _validate_and_format_block(bk, pw, ph, idx + 1, dropped_blocks, bbox_warnings) ...
# Let's cleanly replace the entire block of code from 
# valid_blocks = []
# ... down to
# candidate_counts["final_blocks"] = len(valid_blocks)
matcher = re.compile(r'valid_blocks = \[\].*?candidate_counts\["final_blocks"\] = len\(valid_blocks\)', re.DOTALL)

clean_replace = """
    # ── 7. Phase 2 Heuristics Post-Processing ────────────────────
    validated_pre = []
    for bk in blocks:
        validated = _validate_and_format_block(bk, pw, ph, idx + 1, dropped_blocks, bbox_warnings)
        if validated: validated_pre.append(validated)
        
    quality_notes = []
    merge_events = []
    classification_overrides = []
    dedup_stats = {}
    text_merge_stats = {}
    
    valid_blocks, more_dropped = _reclassify_images(validated_pre, pw, ph, quality_notes, classification_overrides, dropped_blocks)
    valid_blocks = _deduplicate_blocks(valid_blocks, merge_events, dedup_stats)
    valid_blocks = _merge_adjacent_text_blocks(valid_blocks, merge_events, text_merge_stats, pw)
    
    page_layout_hint = _infer_page_layout_hint(valid_blocks, pw, ph)
    valid_blocks = _sort_reading_order(valid_blocks, page_layout_hint, pw, ph)
    
    _assign_block_scores(valid_blocks)
    
    block_type_counts = {"text": 0, "table": 0, "image": 0, "chart": 0, "title": 0, "footer": 0, "unknown": 0}
    for b in valid_blocks:
        t = b["type"]
        block_type_counts[t] = block_type_counts.get(t, 0) + 1
        
    candidate_counts = {
        "pymupdf_text_blocks": sum(1 for b in blocks if b["source"] == "pymupdf"),
        "pdfplumber_tables": sum(1 for b in blocks if b["source"] == "pdfplumber"),
        "camelot_tables": sum(1 for b in blocks if b["source"] == "camelot"),
        "image_candidates": sum(1 for b in blocks if b["source"] == "heuristic"),
        "ocr_boxes": sum(1 for b in blocks if str(b["source"]).startswith("ocr_")),
        "final_blocks": len(valid_blocks)
    }
"""
code = matcher.sub(clean_replace.strip(), code)

# 5. Finally, update the parser_debug dict creation
pd_old = """
    parser_debug = {
        "preview_generated": preview_path is not None,
        "preview_error": preview_err,
        "native_text_chars": len(native_text),
        "ocr_used": ocr_applied,
        "ocr_trigger_reason": ocr_reason,
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
"""
pd_new = """
    parser_debug = {
        "preview_generated": preview_path is not None,
        "preview_error": preview_err,
        "native_text_chars": len(native_text),
        "ocr_used": ocr_applied,
        "ocr_trigger_reason": ocr_reason,
        "ocr_engine_used": ocr_engine,
        "merge_strategy": merge_strategy,
        "fallback_reason": fallback_reason,
        "candidate_counts": candidate_counts,
        "block_type_counts": block_type_counts,
        "dropped_blocks": dropped_blocks,
        "bbox_warnings": bbox_warnings,
        "parse_warnings": warnings,
        "extraction_order": extraction_order,
        "quality_notes": quality_notes,
        "merge_events": merge_events,
        "classification_overrides": classification_overrides,
        "dedup_stats": dedup_stats,
        "text_merge_stats": text_merge_stats,
        "page_layout_hint": page_layout_hint,
        "reading_order_strategy": "column_aware_top_down" if page_layout_hint == "multi_column" else "y_bucket_sorted",
    }
"""
code = code.replace(pd_old.strip(), pd_new.strip())

with open(r"c:\Users\jihyeon\Desktop\my\parsers\pdf_parser.py", "w", encoding="utf-8") as f:
    f.write(code)

print("pdf_parser.py patched successfully!")
