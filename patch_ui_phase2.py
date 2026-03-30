import re

with open(r"c:\Users\jihyeon\Desktop\my\static\index.html", "r", encoding="utf-8") as f:
    html = f.read()

# 1. Update Block Click info
old_block_info = """          <div class="text-[10px] text-gray-500 mt-1 uppercase">ID: ${b.id} | TYPE: ${b.type}</div>"""
new_block_info = """          <div class="text-[10px] text-gray-500 mt-1 uppercase">ID: ${b.id} | TYPE: ${b.type} | SRC: ${b.source}</div>
          ${b.meta?.classification_reason ? `<div class="text-[10px] text-blue-400 mt-0.5">Class Reason: ${b.meta.classification_reason}</div>` : ''}
          ${b.meta?.score_reason ? `<div class="text-[10px] text-green-400 mt-0.5">Score Reason: ${b.meta.score_reason}</div>` : ''}"""
html = html.replace(old_block_info, new_block_info)

# 2. Update renderParserDebug to include new dict entries
old_debug = """  let out = `<div class="p-4 mb-4 bg-gray-800 rounded">
      <div class="text-xs text-gray-400 uppercase mb-2 font-bold">Parser Debug Info</div>
      <table class="w-full text-xs text-gray-300">
          <tr><td class="py-1 w-1/3 text-gray-500">Preview Generated:</td><td>${debug.preview_generated}</td></tr>
          <tr><td class="py-1 text-gray-500">Native Text Chars:</td><td>${debug.native_text_chars}</td></tr>
          <tr><td class="py-1 text-gray-500">OCR Used:</td><td><span class="${debug.ocr_used ? 'text-yellow-400' : ''}">${debug.ocr_used}</span>
              ${debug.ocr_used ? `<span class="text-gray-500 ml-2">(${debug.ocr_engine_used}, ${debug.ocr_trigger_reason})</span>` : ''}
          </td></tr>
          <tr><td class="py-1 text-gray-500">Merge Strategy:</td><td>${debug.merge_strategy}</td></tr>
      </table>`"""

new_debug = """  let out = `<div class="p-4 mb-4 bg-gray-800 rounded">
      <div class="text-xs text-gray-400 uppercase mb-2 font-bold flex justify-between">
          <span>Parser Debug Info</span>
          <span class="text-emerald-500">${debug.page_layout_hint || 'unknown_layout'} | ${debug.reading_order_strategy || 'unknown_order'}</span>
      </div>
      <table class="w-full text-xs text-gray-300">
          <tr><td class="py-1 w-1/3 text-gray-500">Preview Generated:</td><td>${debug.preview_generated}</td></tr>
          <tr><td class="py-1 text-gray-500">Native Text Chars:</td><td>${debug.native_text_chars}</td></tr>
          <tr><td class="py-1 text-gray-500">OCR Used:</td><td><span class="${debug.ocr_used ? 'text-yellow-400' : ''}">${debug.ocr_used}</span>
              ${debug.ocr_used ? `<span class="text-gray-500 ml-2">(${debug.ocr_engine_used}, ${debug.ocr_trigger_reason})</span>` : ''}
          </td></tr>
          <tr><td class="py-1 text-gray-500">Merge / Drops:</td><td class="text-gray-400">
              ${debug.dedup_stats?.dropped_duplicates ? `Dedup Dropped: ${debug.dedup_stats.dropped_duplicates}` : 'Dedup: 0'} 
              | ${debug.text_merge_stats?.merged_pairs ? `Paragraph Merges: ${debug.text_merge_stats.merged_pairs}` : 'Merges: 0'}
          </td></tr>
      </table>`"""
html = html.replace(old_debug, new_debug)

# Add rendering for Arrays: quality_notes, merge_events
old_counts = """  if (debug.block_type_counts) {
      out += `<div class="mt-3 pt-3 border-t border-gray-700">
          <div class="text-xs text-gray-500 mb-1">Final Block Types:</div>
          <div class="flex flex-wrap gap-2 text-xs">`"""

new_counts = """
  // Phase 2 Arrays
  if (debug.quality_notes && debug.quality_notes.length > 0) {
      out += `<div class="mt-3 pt-3 border-t border-gray-700 text-xs text-indigo-300">
          <div class="text-gray-500 mb-1 font-bold">Quality Notes:</div>
          <ul class="list-disc pl-4 space-y-0.5">`;
      debug.quality_notes.forEach(note => { out += `<li>${note}</li>`; });
      out += `</ul></div>`;
  }
  if (debug.merge_events && debug.merge_events.length > 0) {
      out += `<div class="mt-2 text-xs text-teal-300">
          <div class="text-gray-500 mb-1 font-bold">Merge Events:</div>
          <ul class="list-disc pl-4 space-y-0.5">`;
      debug.merge_events.forEach(evt => { out += `<li>KEPT ${evt.kept} &larr; DROPPED ${evt.dropped} [${evt.reason}]</li>`; });
      out += `</ul></div>`;
  }
  if (debug.classification_overrides && debug.classification_overrides.length > 0) {
      out += `<div class="mt-2 text-xs text-purple-300">
          <div class="text-gray-500 mb-1 font-bold">Classification Overrides:</div>
          <ul class="list-disc pl-4 space-y-0.5">`;
      debug.classification_overrides.forEach(ov => { out += `<li>${ov.id}: ${ov.from} &rarr; ${ov.to} [${ov.reason}]</li>`; });
      out += `</ul></div>`;
  }

  if (debug.block_type_counts) {
      out += `<div class="mt-3 pt-3 border-t border-gray-700">
          <div class="text-xs text-gray-500 mb-1">Final Block Types:</div>
          <div class="flex flex-wrap gap-2 text-xs">`"""
html = html.replace(old_counts, new_counts)

with open(r"c:\Users\jihyeon\Desktop\my\static\index.html", "w", encoding="utf-8") as f:
    f.write(html)

print("index.html Phase 2 UI patches applied!")
