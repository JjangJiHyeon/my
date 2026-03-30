import re

with open(r"c:\Users\jihyeon\Desktop\my\parsers\pdf_parser.py", "r", encoding="utf-8") as f:
    code = f.read()

# 1. Update Dedup similarity from 0.8 to 0.75
code = code.replace("if sim >= 0.8 or", "if sim >= 0.75 or")
code = code.replace("overlap_ratio >= 0.8", "overlap_ratio >= 0.75")

# 2. Update reading_order_strategy
order_old = """    strategy = "y_bucket_sorted"  # default
    if hint == "single_column":
        blocks_sorted = sorted(blocks, key=lambda b: (
            0 if b["type"] == "title" else 2 if b["type"] == "footer" else 1,
            b["bbox"][1]
        ))
    elif hint == "slide_like":
        blocks_sorted = sorted(blocks, key=lambda b: (
            0 if b["type"] == "title" else 1,
            b["bbox"][1]
        ))
        return blocks_sorted
    elif hint == "multi_column":
        # Group by x to approximate columns
        strategy = "column_aware_top_down"
        blocks_sorted = sorted(blocks, key=lambda b: (
            0 if b["type"] == "title" else 1,
            b["bbox"][0] // (pw / 3) if pw else 0,
            b["bbox"][1]
        ))
    else:"""

order_new = """    strategy = "y_bucket_sorted"
    if hint == "single_column":
        blocks_sorted = sorted(blocks, key=lambda b: (
            0 if b["type"] == "title" else 2 if b["type"] == "footer" else 1,
            b["bbox"][1]
        ))
    elif hint == "slide_like":
        strategy = "slide_title_priority"
        blocks_sorted = sorted(blocks, key=lambda b: (
            0 if b["type"] == "title" else 1,
            b["bbox"][1]
        ))
        # Important: must set reading_order_strategy back on the caller via returning it, but wait, _sort_reading_order doesn't return strategy!
        # Ah, the caller does `parser_debug["reading_order_strategy"] = "y_bucket_sorted"` manually right now.
        pass
    elif hint == "multi_column":
        strategy = "column_aware_top_down"
        blocks_sorted = sorted(blocks, key=lambda b: (
            0 if b["type"] == "title" else 1,
            b["bbox"][0] // (pw / 3) if pw else 0,
            b["bbox"][1]
        ))
    else:"""
code = code.replace(order_old, order_new)

# Let's fix the caller of _sort_reading_order which hardcodes strategy if not modified.
caller_old = """    # Reading Order
    valid_blocks = _sort_reading_order(valid_blocks, pw, page_layout_hint)
    
    # Store Debug Info
    parser_debug["quality_notes"] = quality_notes
    parser_debug["merge_events"] = merge_events
    parser_debug["classification_overrides"] = classification_overrides
    parser_debug["dedup_stats"] = dedup_stats
    parser_debug["text_merge_stats"] = text_merge_stats
    parser_debug["page_layout_hint"] = page_layout_hint
    parser_debug["reading_order_strategy"] = "column_aware_top_down" if page_layout_hint == "multi_column" else "y_bucket_sorted"
"""

caller_new = """    # Reading Order
    valid_blocks = _sort_reading_order(valid_blocks, pw, page_layout_hint)
    
    # Strategy assignment
    if page_layout_hint == "slide_like":
        strategy = "slide_title_priority"
    elif page_layout_hint == "multi_column":
        strategy = "column_aware_top_down"
    else:
        strategy = "y_bucket_sorted"

    if not quality_notes:
        quality_notes.append("no_major_adjustments")
        
    # Store Debug Info
    parser_debug["quality_notes"] = quality_notes
    parser_debug["merge_events"] = merge_events
    parser_debug["classification_overrides"] = classification_overrides
    parser_debug["dedup_stats"] = dedup_stats
    parser_debug["text_merge_stats"] = text_merge_stats
    parser_debug["page_layout_hint"] = page_layout_hint
    parser_debug["reading_order_strategy"] = strategy
"""
code = code.replace(caller_old, caller_new)

with open(r"c:\Users\jihyeon\Desktop\my\parsers\pdf_parser.py", "w", encoding="utf-8") as f:
    f.write(code)

with open(r"c:\Users\jihyeon\Desktop\my\static\index.html", "r", encoding="utf-8") as f:
    html = f.read()

# 3. Update index.html Block Cards
# We want "Class: ${b.meta?.classification_reason || '-'}" and "Score: ${b.meta?.score_reason || '-'}"
block_card_rx = re.compile(
    r'<div class="text-\[10px\] text-gray-500 mt-1 uppercase whitespace-nowrap overflow-hidden text-ellipsis">ID: \$\{b\.id\} \| TYPE: \$\{b\.type\} \| SRC: \$\{b\.source\}</div>.*?Score: \$\{b\.meta\.score_reason\}</div>` : \'\'\}',
    re.DOTALL
)

new_block_card = """<div class="text-[10px] text-gray-500 mt-1 uppercase whitespace-nowrap overflow-hidden text-ellipsis">ID: ${b.id} | TYPE: ${b.type} | SRC: ${b.source}</div>
          <div class="text-[10px] text-blue-400 mt-0.5 whitespace-nowrap overflow-hidden text-ellipsis" title="${b.meta?.classification_reason || '-'}">Class: ${b.meta?.classification_reason || '-'}</div>
          <div class="text-[10px] text-green-400 mt-0.5 whitespace-nowrap overflow-hidden text-ellipsis" title="${b.meta?.score_reason || '-'}">Score: ${b.meta?.score_reason || '-'}</div>"""
html = block_card_rx.sub(new_block_card, html)

# 4. Update renderParserDebug
func_regex = re.compile(r'function renderParserDebug\(debug\)\s*\{.*?\n  \}\n', re.DOTALL)

new_func = """function renderParserDebug(debug) {
  if (!debug) return '';
  function renderList(title, items) {
    if (!items || items.length === 0) return '';
    return `
      <div style="margin-top:8px;" class="text-xs text-gray-300">
        <b>${title}</b>
        <ul style="padding-left:16px; list-style-type:disc;">
          ${items.map(i => `<li>${typeof i === 'string' ? i : JSON.stringify(i)}</li>`).join('')}
        </ul>
      </div>
    `;
  }
  
  let html = `<div class="p-4 mb-4 bg-gray-800 rounded text-xs text-gray-300 overflow-x-auto whitespace-pre-wrap">` + JSON.stringify(debug, null, 2) + `</div>`;
  
  let listHtml = `<div class="p-4 mb-4 bg-gray-900 rounded border border-gray-700">`;
  listHtml += renderList("quality_notes", debug.quality_notes);
  listHtml += renderList("merge_events", debug.merge_events);
  listHtml += renderList("classification_overrides", debug.classification_overrides);
  listHtml += `</div>`;
  
  return listHtml + html;
}
"""
html = func_regex.sub(new_func, html)

with open(r"c:\Users\jihyeon\Desktop\my\static\index.html", "w", encoding="utf-8") as f:
    f.write(html)

print("Patch applied.")
