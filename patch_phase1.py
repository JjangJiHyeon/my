import os
import re

base_dir = r"c:\Users\jihyeon\Desktop\my"
pdf_parser_path = os.path.join(base_dir, "parsers", "pdf_parser.py")
index_path = os.path.join(base_dir, "static", "index.html")

# --- 1. pdf_parser.py Updates ---
with open(pdf_parser_path, "r", encoding="utf-8") as f:
    pdf_code = f.read()

# 1.1 Update _should_ocr logic
should_ocr_old = """def _should_ocr(native_text: str, image_count: int) -> bool:
    if len(native_text) >= OCR_TEXT_THRESHOLD:
        return False
    if image_count >= OCR_IMAGE_MIN:
        return True
    if len(native_text) == 0:
        return True
    return False"""

should_ocr_new = """def _should_ocr(native_text: str, image_count: int) -> tuple[bool, str]:
    if len(native_text) >= OCR_TEXT_THRESHOLD:
        return False, "ocr_not_needed"
    if image_count >= OCR_IMAGE_MIN:
        return True, "images_present_and_text_weak"
    if len(native_text) == 0:
        return True, "native_text_empty"
    return False, "native_text_below_threshold_but_no_images" """
pdf_code = pdf_code.replace(should_ocr_old, should_ocr_new)

# 1.2 Update _process_page scale outputs and ocr trigger reason
need_ocr_old = "need_ocr = _should_ocr(native_text, image_count)"
need_ocr_new = "need_ocr, ocr_reason = _should_ocr(native_text, image_count)"
pdf_code = pdf_code.replace(need_ocr_old, need_ocr_new)

trigger_reason_old = '"ocr_trigger_reason": "native_text_below_threshold_or_images_present" if need_ocr else "none",'
trigger_reason_new = '"ocr_trigger_reason": ocr_reason,'
pdf_code = pdf_code.replace(trigger_reason_old, trigger_reason_new)

page_return_old = """        "page_width": pw,
        "page_height": ph,
        "preview_width": preview_w,
        "preview_height": preview_h,
        "coord_space": "page_points","""
page_return_new = """        "page_width": pw,
        "page_height": ph,
        "preview_width": preview_w,
        "preview_height": preview_h,
        "preview_scale_x": scale_x,
        "preview_scale_y": scale_y,
        "coord_space": "page_points","""
pdf_code = pdf_code.replace(page_return_old, page_return_new)

# 1.3 Update parse_pdf exception block
except_old_pattern = re.compile(r'except Exception as exc:\n\s+logger\.warning\("Page %d failed.*?\n\s+\}\)', re.DOTALL)

except_new = """except Exception as exc:
            logger.warning("Page %d failed: %s", idx + 1, exc)
            pw, ph = 0.0, 0.0
            try:
                rect = doc[idx].rect
                pw, ph = rect.width, rect.height
            except:
                pass
            pages.append({
                "page_num": idx + 1,
                "page_width": pw,
                "page_height": ph,
                "preview_width": 0,
                "preview_height": 0,
                "preview_scale_x": 1.0,
                "preview_scale_y": 1.0,
                "coord_space": "page_points",
                "preview_image": None,
                "text": "",
                "tables": [],
                "blocks": [],
                "image_count": 0,
                "text_source": "error",
                "ocr_applied": False,
                "ocr_confidence": 0.0,
                "parser_debug": {
                    "preview_generated": False,
                    "preview_error": str(exc),
                    "fallback_reason": "page_processing_exception",
                    "parse_warnings": [f"Page {idx+1} error: {exc}"]
                },
                "error": str(exc),
            })"""
pdf_code = except_old_pattern.sub(except_new, pdf_code)

with open(pdf_parser_path, "w", encoding="utf-8") as f:
    f.write(pdf_code)

# --- 2. static/index.html Updates ---
with open(index_path, "r", encoding="utf-8") as f:
    html = f.read()

# 2.1 Split UI into 2 columns
# We look for where `2. Preview & Overlay Layer` starts, and we will wrap it and the block list in a flex container.
# Currently switchTab adds: html += `<div class="mb-4">`
# We'll replace the Preview Layer construction.
prev_layer_regex = re.compile(r'// 2\. Preview & Overlay Layer\s*if \(page\.preview_image\) \{.*?\} else \{.*?\}', re.DOTALL)

prev_layer_new = """// 2. Preview & Overlay Layer + Block List Panel
  if (page.preview_image) {
      html += `<div class="mb-4 flex gap-4 h-[70vh]">
          <!-- Left: Preview (60%) -->
          <div class="w-[60%] flex flex-col h-full bg-gray-900 border border-gray-700 rounded overflow-hidden relative">
              <div class="p-2 border-b border-gray-700 bg-gray-800 flex justify-between items-center text-xs text-gray-300">
                  <span class="uppercase font-semibold text-gray-400">Preview Overlay</span>
                  <div class="flex gap-3 pointer-events-auto" id="blockToggles">
                      <label><input type="checkbox" checked onchange="toggleBlocks('text')" class="mr-1">Text</label>
                      <label><input type="checkbox" checked onchange="toggleBlocks('title')" class="mr-1">Title</label>
                      <label><input type="checkbox" checked onchange="toggleBlocks('table')" class="mr-1">Table</label>
                      <label><input type="checkbox" checked onchange="toggleBlocks('image')" class="mr-1">Image</label>
                  </div>
              </div>
              <div class="flex-1 overflow-auto p-4 flex justify-center bg-gray-900">
                  <div class="preview-container" id="previewContainer">
                      <img src="${page.preview_image}" style="display:block; max-width:100%; height:auto;" onload="drawOverlays(${idx})" id="previewImg"/>
                      <div id="overlaysLayer" class="absolute inset-0 pointer-events-none overflow-hidden"></div>
                  </div>
              </div>
              <div id="hoverInfo" class="p-2 bg-gray-800 border-t border-gray-700 font-mono text-xs text-yellow-300 min-h-[2rem]">Hover over a block to see details...</div>
          </div>
          
          <!-- Right: Block List (40%) -->
          <div class="w-[40%] flex flex-col h-full bg-gray-900 border border-gray-700 rounded overflow-hidden">
              <div class="p-2 border-b border-gray-700 bg-gray-800 text-xs text-gray-400 uppercase font-semibold flex justify-between">
                  <span>Block List (${page.blocks ? page.blocks.length : 0})</span>
                  <span>(Scrollable)</span>
              </div>
              <div class="flex-1 overflow-y-auto p-2 space-y-2" id="blockListPanel">
                  ${generateBlockListHtml(page.blocks)}
              </div>
          </div>
      </div>`;
  } else {
      html += `<div class="p-4 mb-4 bg-red-900 bg-opacity-30 border border-red-800 text-red-300 text-sm rounded">Preview generation failed: ${page.parser_debug?.preview_error || 'Unknown'}</div>`;
  }"""

html = prev_layer_regex.sub(prev_layer_new, html)

# 2.2 Add generateBlockListHtml and interaction functions
# Add them before window.drawOverlays
func_injection_point = "window.drawOverlays = function(pageIdx) {"

func_new = """window.generateBlockListHtml = function(blocks) {
    if (!blocks || blocks.length === 0) return '<div class="text-xs text-gray-500 text-center mt-4">No blocks</div>';
    
    return blocks.map(b => {
        let typeColor = '#3b82f6';
        if (b.type === 'table') typeColor = '#10b981';
        else if (b.type === 'image' || b.type === 'chart') typeColor = '#8b5cf6';
        else if (b.type === 'title') typeColor = '#f59e0b';
        
        const shortText = b.text ? (b.text.length > 80 ? b.text.substring(0, 80) + '...' : b.text) : '';
        const confText = typeof b.score === 'number' ? (b.score * 100).toFixed(1) + '%' : '-';
        
        return `<div id="list-item-${b.id}" class="bg-gray-800 border-l-4 border-gray-600 rounded p-2 text-xs transition-colors cursor-pointer hover:bg-gray-700" 
            style="border-left-color: ${typeColor};"
            onmouseenter="highlightOverlay('${b.id}')"
            onmouseleave="resetOverlay('${b.id}')"
            onclick="focusOverlay('${b.id}')">
            <div class="flex justify-between items-start mb-1">
                <span class="font-bold font-mono text-gray-300">${b.id}</span>
                <span class="text-gray-500">${b.source} | ${confText}</span>
            </div>
            <div class="text-blue-300 mb-1">${b.type.toUpperCase()} <span class="text-gray-500 ml-2">BBox: [${b.bbox.join(', ')}]</span></div>
            <div class="text-gray-400 whitespace-pre-wrap break-all">${escapeHtml(shortText)}</div>
        </div>`;
    }).join('');
};

window.highlightOverlay = function(id) {
    const el = document.getElementById('overlay-' + id);
    if (el) {
        el.style.backgroundColor = 'rgba(59, 130, 246, 0.4)';
        el.style.zIndex = '20';
        el.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'nearest' });
    }
}
window.resetOverlay = function(id) {
    const el = document.getElementById('overlay-' + id);
    if (el) {
        el.style.backgroundColor = 'rgba(255, 255, 255, 0.1)';
        el.style.zIndex = '1';
    }
}
window.highlightList = function(id) {
    const el = document.getElementById('list-item-' + id);
    if (el) {
        el.style.backgroundColor = '#374151';
        el.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
}
window.resetList = function(id) {
    const el = document.getElementById('list-item-' + id);
    if (el) el.style.backgroundColor = '';
}
window.focusOverlay = function(id) {
    highlightOverlay(id);
    // Add brief flash
    const el = document.getElementById('overlay-' + id);
    if (el) {
        el.style.borderWidth = '4px';
        setTimeout(() => { if (el) el.style.borderWidth = '2px'; }, 300);
    }
}

window.drawOverlays = function(pageIdx) {"""
html = html.replace(func_injection_point, func_new)

# 2.3 Add id to overlays and hover inter-link inside drawOverlays loop
html = html.replace(
    'html += `<div class="bbox-overlay ${typeClass}', 
    'html += `<div id="overlay-${b.id}" class="bbox-overlay ${typeClass} transition-all duration-200"'
)
# Update the onmouseenter/mouseleave to also highlight list
html = html.replace(
    '''onmouseenter="showHoverInfo('${escapeHtml(info)}')"''',
    '''onmouseenter="showHoverInfo('${escapeHtml(info)}'); highlightList('${b.id}')"'''
)
html = html.replace(
    '''onmouseleave="clearHoverInfo()"></div>`;''',
    '''onmouseleave="clearHoverInfo(); resetList('${b.id}')"></div>`;'''
)

with open(index_path, "w", encoding="utf-8") as f:
    f.write(html)
print("patch_phase1 completed.")
