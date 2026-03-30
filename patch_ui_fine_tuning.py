import re

with open(r"c:\Users\jihyeon\Desktop\my\static\index.html", "r", encoding="utf-8") as f:
    html = f.read()

# 1. Update Block Card
old_block_card = """          <div class="text-[10px] text-gray-500 mt-1 uppercase">ID: ${b.id} | TYPE: ${b.type} | SRC: ${b.source}</div>
          ${b.meta?.classification_reason ? `<div class="text-[10px] text-blue-400 mt-0.5">Class Reason: ${b.meta.classification_reason}</div>` : ''}
          ${b.meta?.score_reason ? `<div class="text-[10px] text-green-400 mt-0.5">Score Reason: ${b.meta.score_reason}</div>` : ''}"""

new_block_card = """          <div class="text-[10px] text-gray-500 mt-1 uppercase whitespace-nowrap overflow-hidden text-ellipsis">ID: ${b.id} | TYPE: ${b.type} | SRC: ${b.source}</div>
          ${b.meta?.classification_reason ? `<div class="text-[10px] text-blue-400 mt-0.5 whitespace-nowrap overflow-hidden text-ellipsis" title="${b.meta.classification_reason}">Class: ${b.meta.classification_reason}</div>` : ''}
          ${b.meta?.score_reason ? `<div class="text-[10px] text-green-400 mt-0.5 whitespace-nowrap overflow-hidden text-ellipsis" title="${b.meta.score_reason}">Score: ${b.meta.score_reason}</div>` : ''}"""

if old_block_card in html:
    html = html.replace(old_block_card, new_block_card)
else:
    # Just in case it wasn't matched exactly
    print("Warning: old_block_card not found. Checking fallback...")

# 2. Update renderParserDebug
# We replace the entire renderParserDebug function
func_regex = re.compile(r'function renderParserDebug\(debug\)\s*\{.*?\n  \}\n', re.DOTALL)

new_func = """function renderParserDebug(debug) {
  if (!debug) return '';
  
  function renderList(title, items) {
    if (!items || items.length === 0) return '';
    return `
      <div class="mt-3 pt-3 border-t border-gray-700">
        <b class="text-xs text-gray-400">${title}</b>
        <ul class="list-disc pl-4 mt-1 space-y-1 text-xs text-gray-300">
          ${items.map(i => `<li>${typeof i === 'string' ? i : JSON.stringify(i)}</li>`).join('')}
        </ul>
      </div>
    `;
  }

  let out = `<div class="p-4 mb-4 bg-gray-800 rounded">
      <div class="text-xs text-gray-400 uppercase mb-2 font-bold flex justify-between">
          <span>Parser Debug Info</span>
          <span class="text-emerald-500">${debug.page_layout_hint || 'unknown_layout'} | ${debug.reading_order_strategy || 'unknown_order'}</span>
      </div>
      <table class="w-full text-xs text-gray-300">
          <tr><td class="py-1 text-gray-500 w-1/3">OCR Used:</td><td><span class="${debug.ocr_used ? 'text-yellow-400' : ''}">${debug.ocr_used}</span>
              ${debug.ocr_used ? `<span class="text-gray-500 ml-2">(${debug.ocr_engine_used}, ${debug.ocr_trigger_reason})</span>` : ''}
          </td></tr>
          <tr><td class="py-1 text-gray-500">Dedup Drops:</td><td class="text-blue-300 font-bold">${debug.dedup_stats?.dropped_duplicates || 0}</td></tr>
          <tr><td class="py-1 text-gray-500">Paragraph Merges:</td><td class="text-green-300 font-bold">${debug.text_merge_stats?.merged_pairs || 0}</td></tr>
      </table>`;
      
  out += renderList("Quality Notes", debug.quality_notes);
  out += renderList("Merge Events", debug.merge_events);
  out += renderList("Classification Overrides", debug.classification_overrides);
  out += renderList("Parse Warnings", debug.parse_warnings);
  
  if (debug.block_type_counts) {
      out += `<div class="mt-3 pt-3 border-t border-gray-700">
          <div class="text-xs text-gray-500 mb-1">Final Block Types:</div>
          <div class="flex flex-wrap gap-2 text-xs">`;
      for (const [t, c] of Object.entries(debug.block_type_counts)) {
          if (c > 0) out += `<span class="bg-gray-700 px-2 py-0.5 rounded text-gray-300">${t}: ${c}</span>`;
      }
      out += `</div></div>`;
  }
  
  out += `</div>`;
  return out;
}
"""
html = func_regex.sub(new_func, html)

with open(r"c:\Users\jihyeon\Desktop\my\static\index.html", "w", encoding="utf-8") as f:
    f.write(html)

print("UI fine tuning complete.")
