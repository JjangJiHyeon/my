import re

with open(r"c:\Users\jihyeon\Desktop\my\parsers\pdf_parser.py", "r", encoding="utf-8") as f:
    code = f.read()

# 1. reading_order_strategy bug fix
# In _process_page, after _sort_reading_order, page_layout_hint is passed, so my previous patch worked, but maybe it got overwritten or I missed a spot?
# Let's check _process_page at the end.
old_order_assign = """    # Reading Order
    valid_blocks = _sort_reading_order(valid_blocks, page_layout_hint, pw, ph)
    
    # Strategy assignment
    if page_layout_hint == "slide_like":
        strategy = "slide_title_priority"
    elif page_layout_hint == "multi_column":
        strategy = "column_aware_top_down"
    else:
        strategy = "y_bucket_sorted"

    if not quality_notes:
        quality_notes.append("no_major_adjustments")"""

new_order_assign = """    # Reading Order
    valid_blocks = _sort_reading_order(valid_blocks, page_layout_hint, pw, ph)
    
    # Strategy assignment
    if page_layout_hint == "slide_like":
        strategy = "slide_title_priority"
    elif page_layout_hint == "multi_column":
        strategy = "column_aware_top_down"
    else:
        strategy = "y_bucket_sorted"

    if not quality_notes:
        quality_notes.append("no_major_adjustments")"""

# Actually, the function signature is _sort_reading_order(blocks: list[dict], hint: str, pw: float, ph: float)
# In my previous edit I replaced:
# `valid_blocks = _sort_reading_order(valid_blocks, pw, page_layout_hint)` -> which is args mismatch.

fix_sig = r'valid_blocks = _sort_reading_order\(valid_blocks, pw, page_layout_hint\)'
code = re.sub(fix_sig, 'valid_blocks = _sort_reading_order(valid_blocks, page_layout_hint, pw, ph)', code)

# Let's ensure strategy is logged correctly
fix_dict = r'parser_debug\["reading_order_strategy"\] = strategy'
code = re.sub(fix_dict, 'parser_debug["reading_order_strategy"] = strategy', code)

# 2. Dedup Logic Fix
# The overlap ratio was: iarea / min(b_area, max(1, (ex1-ex0)*(ey1-ey0)))
# But pdfplumber y coords might be inverted compared to pymupdf, leading to 0 overlap. 
# Also, we should ensure quality_notes gets "duplicate_text_removed" appended properly.
dedup_old = """            if overlap_ratio >= 0.75:
                sim = _text_similarity(b["text"], ex["text"])
                if sim >= 0.75 or (len(b["text"]) < 10 and overlap_ratio >= 0.75):"""
dedup_new = """            if overlap_ratio >= 0.5:
                sim = _text_similarity(b["text"], ex["text"])
                if sim >= 0.75 or (len(b["text"]) < 10 and overlap_ratio >= 0.75):"""
code = code.replace(dedup_old, dedup_new)

with open(r"c:\Users\jihyeon\Desktop\my\parsers\pdf_parser.py", "w", encoding="utf-8") as f:
    f.write(code)

print("Patch logic holes applied.")
