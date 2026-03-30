import re

with open(r"c:\Users\jihyeon\Desktop\my\parsers\pdf_parser.py", "r", encoding="utf-8") as f:
    code = f.read()

# 1. Update text_similarity and source_priority
sim_old = """def _text_similarity(a: str, b: str) -> float:
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
    return 1"""

sim_new = """def _text_similarity(a: str, b: str) -> float:
    an = "".join(a.split()).lower()
    bn = "".join(b.split()).lower()
    if not an or not bn: return 0.0
    set_a, set_b = set(an), set(bn)
    if not set_a or not set_b: return 0.0
    return len(set_a & set_b) / max(len(set_a), len(set_b))

def _get_source_priority(src: str) -> int:
    s = str(src).lower()
    if "pymupdf" in s: return 3
    if "pdfplumber" in s: return 3
    if "camelot" in s: return 3
    if "ocr" in s: return 1
    return 0"""
code = code.replace(sim_old, sim_new)

# 2. Update _reclassify_images
# We want to calculate text_density. Since individual image blocks might not have "text" populated natively,
# we also check overlapping text blocks to see if this "image" is actually just a text region.
reclassify_old = """def _reclassify_images(blocks: list[dict], pw: float, ph: float, quality_notes: list, classification_overrides: list, dropped: list) -> list[dict]:
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
    return out"""

reclassify_new = """def _reclassify_images(blocks: list[dict], pw: float, ph: float, quality_notes: list, classification_overrides: list, dropped: list) -> list[dict]:
    out = []
    page_area = pw * ph if pw * ph > 0 else 1
    reclassified = False
    
    # Calculate contained text length for each image block
    for b in blocks:
        if b["type"] in ("image", "chart", "chart_like"):
            x0, y0, x1, y1 = b["bbox"]
            area = (x1 - x0) * (y1 - y0)
            area_ratio = area / page_area
            
            if area_ratio > 0.85:
                dropped.append({"id": b["id"], "reason": "background_dropped", "source": b["source"]})
                continue
                
            contained_text_len = len(b.get("text", ""))
            
            # Fallback: check other text blocks inside it
            if contained_text_len == 0:
                for tb in blocks:
                    if tb["type"] in ("text", "title", "footer"):
                        tx0, ty0, tx1, ty1 = tb["bbox"]
                        # if tb is mostly inside b
                        if tx0 >= x0 - 10 and ty0 >= y0 - 10 and tx1 <= x1 + 10 and ty1 <= y1 + 10:
                            contained_text_len += len(tb.get("text", ""))
                            
            text_density = contained_text_len / (area + 1)
            
            if contained_text_len > 50 and text_density > 0.0005:
                b["type"] = "text"
                b["meta"]["classification_reason"] = "high_text_density"
                classification_overrides.append({
                    "block_id": b["id"],
                    "from": "image",
                    "to": "text",
                    "reason": "high_text_density"
                })
                reclassified = True
            else:
                b["meta"]["classification_reason"] = "area_ratio_check"
                
        out.append(b)
        
    if reclassified:
        quality_notes.append("image_reclassified_to_text")
        
    return out"""
code = code.replace(reclassify_old, reclassify_new)


# 3. Update _deduplicate_blocks
dedup_old = """def _deduplicate_blocks(blocks: list[dict], merge_events: list, dedup_stats: dict) -> list[dict]:
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
    return sorted(out, key=lambda x: int(x["id"].split("_b")[-1]) if "_b" in x["id"] else 0)"""

dedup_new = """def _deduplicate_blocks(blocks: list[dict], merge_events: list, dedup_stats: dict, quality_notes: list) -> list[dict]:
    out = []
    dropped_count = 0
    sorted_blocks = sorted(blocks, key=lambda x: _get_source_priority(x["source"]), reverse=True)
    
    for b in sorted_blocks:
        x0, y0, x1, y1 = b["bbox"]
        b_area = (x1-x0)*(y1-y0)
        if b_area <= 0: continue
        
        is_duplicate = False
        for ex in out:
            if b["type"] not in ("text", "title", "footer", "unknown") or ex["type"] not in ("text", "title", "footer", "unknown"):
                continue
            
            ex0, ey0, ex1, ey1 = ex["bbox"]
            ix0, iy0 = max(x0, ex0), max(y0, ey0)
            ix1, iy1 = min(x1, ex1), min(y1, ey1)
            iarea = max(0, ix1 - ix0) * max(0, iy1 - iy0)
            
            overlap_ratio = iarea / min(b_area, max(1, (ex1-ex0)*(ey1-ey0)))
            
            if overlap_ratio >= 0.5:
                sim = _text_similarity(b["text"], ex["text"])
                if sim >= 0.8 or (len(b["text"]) < 10 and overlap_ratio >= 0.8):
                    is_duplicate = True
                    merge_events.append({
                        "kept": ex["id"],
                        "dropped": b["id"],
                        "reason": "duplicate_text_native_preferred" if _get_source_priority(ex["source"]) > _get_source_priority(b["source"]) else "duplicate_text_merged"
                    })
                    dropped_count += 1
                    break
                    
        if not is_duplicate:
            out.append(b)
            
    dedup_stats["dropped_duplicates"] = dropped_count
    if dropped_count > 0:
        quality_notes.append("duplicate_text_removed")
        
    return sorted(out, key=lambda x: int(x["id"].split("_b")[-1]) if "_b" in x["id"] else 0)"""
code = code.replace(dedup_old, dedup_new)


# 4. _merge_adjacent_text_blocks -> add quality_notes check
merge_old = """    text_merge_stats["merged_pairs"] = merged_count
    return merged"""

merge_new = """    text_merge_stats["merged_pairs"] = merged_count
    if merged_count > 0:
        # Avoid circular imports or issues; we need to pass quality_notes but it's not passed. 
        # I'll modify the caller below instead to check text_merge_stats["merged_pairs"] > 0
        pass
    return merged"""
code = code.replace(merge_old, merge_new)


# 5. Caller Update
caller_old = """    valid_blocks = _deduplicate_blocks(valid_blocks, merge_events, dedup_stats)
    valid_blocks = _merge_adjacent_text_blocks(valid_blocks, merge_events, text_merge_stats, pw)"""

caller_new = """    valid_blocks = _deduplicate_blocks(valid_blocks, merge_events, dedup_stats, quality_notes)
    valid_blocks = _merge_adjacent_text_blocks(valid_blocks, merge_events, text_merge_stats, pw)
    if text_merge_stats.get("merged_pairs", 0) > 0:
        quality_notes.append(f"text_blocks_merged:{text_merge_stats['merged_pairs']}")"""
code = code.replace(caller_old, caller_new)


# 6. _sort_reading_order for slide_like
order_old = """    elif hint == "slide_like":
        # Titles top, then cluster by proximity to center, etc.
        # Fallback to simple top->down, left->right
        blocks_sorted = sorted(blocks, key=lambda b: (
            0 if b["type"] == "title" else 2 if b["type"] == "footer" else 1,
            b["bbox"][1] // 30, # bucket y
            b["bbox"][0]
        ))
        return blocks_sorted"""

order_new = """    elif hint == "slide_like":
        blocks_sorted = sorted(blocks, key=lambda b: (
            0 if b["type"] == "title" else 1,
            b["bbox"][1]
        ))
        return blocks_sorted"""
code = code.replace(order_old, order_new)

with open(r"c:\Users\jihyeon\Desktop\my\parsers\pdf_parser.py", "w", encoding="utf-8") as f:
    f.write(code)

print("pdf_parser phase 2 fine-tuning applied!")
