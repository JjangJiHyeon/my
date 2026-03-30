import re

with open(r"c:\Users\jihyeon\Desktop\my\parsers\pdf_parser.py", "r", encoding="utf-8") as f:
    code = f.read()

# 1. reading_order logic
order_def_old = """def _sort_reading_order(blocks: list[dict], hint: str, pw: float, ph: float) -> list[dict]:"""
order_def_new = """def _sort_reading_order(blocks: list[dict], hint: str, pw: float, ph: float) -> tuple[list[dict], str, str]:"""
code = code.replace(order_def_old, order_def_new)

order_multi_old = """        blocks_sorted = sorted(blocks, key=lambda b: (
            0 if b["type"] == "title" else 1,
            b["bbox"][0] // (pw / 3) if pw else 0,
            b["bbox"][1]
        ))
        return blocks_sorted"""
order_multi_new = """        blocks_sorted = sorted(blocks, key=lambda b: (
            0 if b["type"] == "title" else 1,
            b["bbox"][0] // (pw / 3) if pw else 0,
            b["bbox"][1]
        ))
        return blocks_sorted, strategy, "title prioritized, x_group column logic, then y" """
code = code.replace(order_multi_old, order_multi_new)

order_slide_old = """        blocks_sorted = sorted(blocks, key=lambda b: (
            0 if b["type"] == "title" else 1,
            b["bbox"][1]
        ))
        # Important: must set reading_order_strategy back on the caller via returning it, but wait, _sort_reading_order doesn't return strategy!
        # Ah, the caller does `parser_debug["reading_order_strategy"] = "y_bucket_sorted"` manually right now.
        pass"""
order_slide_new = """        blocks_sorted = sorted(blocks, key=lambda b: (
            0 if b["type"] == "title" else 1,
            b["bbox"][1]
        ))
        return blocks_sorted, strategy, "title first, then absolute vertical sort" """
code = code.replace(order_slide_old, order_slide_new)

order_default_old = """    blocks_sorted = sorted(blocks, key=lambda b: (
        0 if b["type"] == "title" else 2 if b["type"] == "footer" else 1,
        b["bbox"][1] // 20, 
        b["bbox"][0]
    ))
    return blocks_sorted"""
order_default_new = """    blocks_sorted = sorted(blocks, key=lambda b: (
        0 if b["type"] == "title" else 2 if b["type"] == "footer" else 1,
        b["bbox"][1] // 20, 
        b["bbox"][0]
    ))
    return blocks_sorted, strategy, "title first, footer last, 20px grouping, then x" """
code = code.replace(order_default_old, order_default_new)

caller_old = """    # Reading Order
    valid_blocks = _sort_reading_order(valid_blocks, page_layout_hint, pw, ph)
    
    # Strategy assignment
    if page_layout_hint == "slide_like":
        strategy = "slide_title_priority"
    elif page_layout_hint == "multi_column":
        strategy = "column_aware_top_down"
    else:
        strategy = "y_bucket_sorted" """

caller_new = """    # Reading Order
    valid_blocks, actual_strategy, strategy_basis = _sort_reading_order(valid_blocks, page_layout_hint, pw, ph) """
code = code.replace(caller_old, caller_new)

assign_old = """    parser_debug["reading_order_strategy"] = strategy"""
assign_new = """    parser_debug["reading_order_strategy"] = actual_strategy
    parser_debug["reading_order_basis"] = strategy_basis"""
code = code.replace(assign_old, assign_new)

with open(r"c:\Users\jihyeon\Desktop\my\parsers\pdf_parser.py", "w", encoding="utf-8") as f:
    f.write(code)

with open(r"c:\Users\jihyeon\Desktop\my\static\index.html", "r", encoding="utf-8") as f:
    html = f.read()

# Overlay title
ov_old = """div.title = `ID: ${b.id}\\nType: ${b.type}\\nSource: ${b.source}\\nScore: ${b.score}\\nBBox: ${b.bbox.join(', ')}`;"""
ov_new = """div.title = `ID: ${b.id}\\nType: ${b.type}\\nSource: ${b.source}\\nScore: ${b.score}\\nBBox: ${b.bbox.join(', ')}` + 
            (b.meta?.classification_reason ? `\\nClass Reason: ${b.meta.classification_reason}` : '') + 
            (b.meta?.score_reason ? `\\nScore Reason: ${b.meta.score_reason}` : '');"""
html = html.replace(ov_old, ov_new)

# renderBlocks item
block_old = r'<div class="text-\[10px\] text-gray-500 mt-1 uppercase whitespace-nowrap overflow-hidden text-ellipsis">ID: \$\{b\.id\} \| TYPE: \$\{b\.type\} \| SRC: \$\{b\.source\}</div>\s*<div class="text-\[10px\] text-blue-400 mt-0\.5 whitespace-nowrap overflow-hidden text-ellipsis".*?</div>\s*<div class="text-\[10px\] text-green-400 mt-0\.5 whitespace-nowrap overflow-hidden text-ellipsis".*?</div>'
block_new = """<div class="text-[10px] text-gray-500 mt-1 uppercase whitespace-nowrap overflow-hidden text-ellipsis">ID: ${b.id} | TYPE: ${b.type} | SRC: ${b.source}</div>
          ${b.meta?.classification_reason ? `<div class="text-[10px] text-blue-400 mt-0.5 whitespace-nowrap overflow-hidden text-ellipsis" title="${b.meta.classification_reason}">Class: ${b.meta.classification_reason}</div>` : ''}
          ${b.meta?.score_reason ? `<div class="text-[10px] text-green-400 mt-0.5 whitespace-nowrap overflow-hidden text-ellipsis" title="${b.meta.score_reason}">Score: ${b.meta.score_reason}</div>` : ''}"""
html = re.sub(block_old, block_new, html, flags=re.DOTALL)

# renderParserDebug
func_regex = re.compile(r'function renderParserDebug\(debug\)\s*\{.*?\n  \}\n', re.DOTALL)
new_func = """function renderParserDebug(debug) {
  if (!debug) return '';
  
  function renderList(title, items) {
    if (!items || items.length === 0) return '';
    return `
      <div class="mt-3">
        <b class="text-xs text-gray-400 font-bold">\u25b8 ${title}</b>
        <ul class="list-disc pl-5 mt-1 space-y-1 text-xs text-gray-300">
          ${items.map(i => `<li>${typeof i === 'string' ? i : JSON.stringify(i)}</li>`).join('')}
        </ul>
      </div>
    `;
  }

  function renderDict(title, dict) {
    if (!dict || Object.keys(dict).length === 0) return '';
    return `
      <div class="mt-3">
        <b class="text-xs text-gray-400 font-bold">\u25b8 ${title}</b>
        <ul class="list-disc pl-5 mt-1 space-y-1 text-xs text-gray-300">
          ${Object.entries(dict).map(([k, v]) => `<li><b>${k}</b>: ${v}</li>`).join('')}
        </ul>
      </div>
    `;
  }
  
  let layout = `
  <div class="p-4 mb-4 bg-gray-900 rounded border border-gray-700">
      <div class="text-sm text-gray-200 uppercase mb-3 font-bold border-b border-gray-700 pb-2">Parser Debug Details</div>
      
      <div class="mt-2 text-xs text-indigo-300">
          <b>Reading Order</b>
          <div class="pl-3 mt-1 text-gray-300">
             <div>Layout Hint: <span class="text-emerald-400">${debug.page_layout_hint || '-'}</span></div>
             <div>Strategy: <span class="text-blue-400">${debug.reading_order_strategy || '-'}</span></div>
             ${debug.reading_order_basis ? `<div>Basis: ${debug.reading_order_basis}</div>` : ''}
          </div>
      </div>
      
      ${renderList("Quality Notes", debug.quality_notes)}
      ${renderList("Merge Events", debug.merge_events)}
      ${renderList("Classification Overrides", debug.classification_overrides)}
      ${renderDict("Dedup / Merge Stats", {
          ...(debug.dedup_stats && debug.dedup_stats.dropped_duplicates > 0 ? {"Dropped Duplicates": debug.dedup_stats.dropped_duplicates} : {}),
          ...(debug.text_merge_stats && debug.text_merge_stats.merged_pairs > 0 ? {"Merged Paragraphs": debug.text_merge_stats.merged_pairs} : {})
      })}
      
      ${(debug.parse_warnings && debug.parse_warnings.length > 0) || (debug.bbox_warnings && debug.bbox_warnings.length > 0) || (debug.dropped_blocks && debug.dropped_blocks.length > 0) ? `
        <div class="mt-3">
            <b class="text-xs text-red-400 font-bold">\u25b8 Warnings & Drops</b>
            <div class="pl-3 mt-1 space-y-2">
                ${renderList("Parse Warnings", debug.parse_warnings)}
                ${renderList("BBox Warnings", debug.bbox_warnings)}
                ${renderList("Dropped Blocks", debug.dropped_blocks)}
            </div>
        </div>
      ` : ''}
      
      <div class="mt-4 pt-3 border-t border-gray-700">
        <details>
            <summary class="cursor-pointer text-xs text-gray-500 hover:text-gray-300 transition-colors">View Raw JSON / Misc</summary>
            <div class="mt-2 p-2 bg-gray-950 rounded text-xs text-gray-400 overflow-x-auto whitespace-pre-wrap">${JSON.stringify(debug, null, 2)}</div>
        </details>
      </div>
  </div>`;
  
  return layout;
}
"""
html = func_regex.sub(new_func, html)

with open(r"c:\Users\jihyeon\Desktop\my\static\index.html", "w", encoding="utf-8") as f:
    f.write(html)

print("Patch applied.")
