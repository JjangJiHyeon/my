import os
import re

base_dir = r'c:\Users\jihyeon\Desktop\my'
index_path = os.path.join(base_dir, 'static', 'index.html')

with open(index_path, 'r', encoding='utf-8') as f:
    html = f.read()

# 1. Add CSS
css_old = '</style>'
css_new = '''
    .preview-container { position: relative; display: inline-block; border: 1px solid #475569; background: white; max-width: 100%; }
    .bbox-overlay { position: absolute; border: 2px solid; background: rgba(255, 255, 255, 0.1); box-sizing: border-box; cursor: pointer; transition: background 0.2s; }
    .bbox-overlay:hover { background: rgba(59, 130, 246, 0.4); z-index: 10; border-color: #2563eb !important; }
    .bbox-text { border-color: #3b82f6; }
    .bbox-table { border-color: #10b981; }
    .bbox-image { border-color: #8b5cf6; }
    .bbox-chart { border-color: #d946ef; }
    .bbox-title { border-color: #f59e0b; }
    .bbox-footer { border-color: #64748b; }
    .bbox-unknown { border-color: #ef4444; }
</style>'''
html = html.replace(css_old, css_new)

# 2. replace switchTab entirely up to renderBlocks
switch_tab_pattern = re.compile(r'function switchTab\(idx\).*?function renderBlocks\(blocks\) \{', re.DOTALL)

switch_tab_new = '''function switchTab(idx) {
  const pages = activeDoc.pages || [];
  const page = pages[idx];
  if (!page) return;

  document.querySelectorAll('.tab-btn').forEach((b, i) => {
    b.classList.toggle('active', i === idx);
  });

  const content = document.getElementById('tabContent');
  let html = '';

  // 1. Diagnostics Header
  if (page.page_width && page.page_height) {
      let stats = [
          `Page: ${Math.round(page.page_width)}x${Math.round(page.page_height)}`,
          `Preview: ${Math.round(page.preview_width||0)}x${Math.round(page.preview_height||0)}`,
          `Space: ${page.coord_space}`,
          `Scale: ${(page.preview_width/page.page_width).toFixed(2)}x`,
          `Total Blocks: ${(page.blocks||[]).length}`,
          `Dropped: ${(page.parser_debug?.dropped_blocks||[]).length}`,
          `OCR: ${page.ocr_applied ? 'Yes' : 'No'}`
      ];
      html += `<div class="p-3 mb-4 bg-gray-800 rounded font-mono text-xs text-gray-300 flex flex-wrap gap-4 border border-gray-700">
          ${stats.map(s => `<div>${escapeHtml(s)}</div>`).join('')}
      </div>`;
  }

  // 2. Preview & Overlay Layer
  if (page.preview_image) {
      html += `<div class="mb-4">
          <div class="flex justify-between items-end mb-2">
              <div class="text-xs text-gray-500 uppercase">Page Preview (Overlay)</div>
              <div class="text-xs flex gap-3 pointer-events-auto" id="blockToggles">
                  <label><input type="checkbox" checked onchange="toggleBlocks('text')" class="mr-1">Text/Footer</label>
                  <label><input type="checkbox" checked onchange="toggleBlocks('title')" class="mr-1">Title</label>
                  <label><input type="checkbox" checked onchange="toggleBlocks('table')" class="mr-1">Table</label>
                  <label><input type="checkbox" checked onchange="toggleBlocks('image')" class="mr-1">Image/Chart</label>
              </div>
          </div>
          <div class="preview-container" id="previewContainer">
              <img src="${page.preview_image}" style="display:block; max-width:100%; height:auto;" onload="drawOverlays(${idx})" id="previewImg"/>
              <div id="overlaysLayer" class="absolute inset-0 pointer-events-none overflow-hidden"></div>
          </div>
          <div id="hoverInfo" class="mt-2 min-h-[1.5rem] font-mono text-xs text-yellow-300">Hover over a block to see details...</div>
      </div>`;
  } else {
      html += `<div class="p-4 mb-4 bg-red-900 bg-opacity-30 border border-red-800 text-red-300 text-sm rounded">Preview generation failed: ${page.parser_debug?.preview_error || 'Unknown'}</div>`;
  }

  // 3. Raw Text fallback
  if (page.text && !page.preview_image) {
      html += `<div class="mb-4"><div class="text-xs text-gray-500 uppercase mb-2">Raw Text</div><pre class="parsed-text">${escapeHtml(page.text)}</pre></div>`;
  }

  // 4. Parser Debug Summaries
  if (page.parser_debug) {
      html += renderParserDebug(page.parser_debug);
  }

  content.innerHTML = html;
}

window.drawOverlays = function(pageIdx) {
    const page = activeDoc.pages[pageIdx];
    if (!page || !page.blocks) return;
    
    const img = document.getElementById('previewImg');
    const layer = document.getElementById('overlaysLayer');
    if (!img || !layer) return;

    const currentImgWidth = img.clientWidth;
    const currentImgHeight = img.clientHeight;
    if (currentImgWidth === 0) return;
    
    const scaleX = currentImgWidth / page.page_width;
    const scaleY = currentImgHeight / page.page_height;
    
    let html = '';
    page.blocks.forEach(b => {
        let typeClass = 'bbox-' + (b.type || 'unknown');
        
        const [x0, y0, x1, y1] = b.bbox;
        const left = x0 * scaleX;
        const top = y0 * scaleY;
        const width = (x1 - x0) * scaleX;
        const height = (y1 - y0) * scaleY;
        
        const confText = typeof b.score === 'number' ? (b.score * 100).toFixed(1) + '%' : '-';
        const info = `ID: ${b.id} | Type: ${b.type} | Src: ${b.source} | Score: ${confText} | BBox: [${b.bbox.join(', ')}]`;
        
        html += `<div class="bbox-overlay ${typeClass} pointer-events-auto" data-type="${b.type}" 
            style="left:${left}px; top:${top}px; width:${width}px; height:${height}px;"
            onmouseenter="showHoverInfo('${escapeHtml(info)}')"
            onmouseleave="clearHoverInfo()"></div>`;
    });
    layer.innerHTML = html;
}

window.showHoverInfo = function(text) {
    const info = document.getElementById('hoverInfo');
    if (info) info.textContent = text;
}
window.clearHoverInfo = function() {
    const info = document.getElementById('hoverInfo');
    if (info) info.textContent = 'Hover over a block to see details...';
}

window.toggleBlocks = function(cls) {
    const layer = document.getElementById('overlaysLayer');
    if (!layer) return;
    
    const showText = document.querySelector('input[onchange*="text"]').checked;
    const showTitle = document.querySelector('input[onchange*="title"]').checked;
    const showTable = document.querySelector('input[onchange*="table"]').checked;
    const showImage = document.querySelector('input[onchange*="image"]').checked;
    
    layer.querySelectorAll('.bbox-overlay').forEach(o => {
        const t = o.getAttribute('data-type');
        let show = true;
        if (t === 'text' || t === 'footer') show = showText;
        if (t === 'title') show = showTitle;
        if (t === 'table') show = showTable;
        if (t === 'image' || t === 'chart') show = showImage;
        o.style.display = show ? 'block' : 'none';
    });
}

function renderBlocks(blocks) {'''

html = switch_tab_pattern.sub(switch_tab_new, html)

# 3. replace renderParserDebug
debug_old_pattern = re.compile(r'function renderParserDebug\(debug\) \{.*?return `<div class="mb-4 mt-6">', re.DOTALL)

debug_new = '''function renderParserDebug(debug) {
    const strategy = debug.merge_strategy || '-';
    const ocrEngine = debug.ocr_engine_used || '-';
    const fallbackReason = debug.fallback_reason || '-';
    const ocrTrigger = debug.ocr_trigger_reason || '-';
    
    let candidates = debug.candidate_counts ? Object.entries(debug.candidate_counts).map(([k,v]) => `${k}:${v}`).join(', ') : '-';
    let typeCounts = debug.block_type_counts ? Object.entries(debug.block_type_counts).map(([k,v]) => `${k}:${v}`).join(', ') : '-';
    
    let warningsHtml = '-';
    if (Array.isArray(debug.parse_warnings) && debug.parse_warnings.length > 0) {
        warningsHtml = `<ul class="list-disc pl-4 space-y-1">` + 
            debug.parse_warnings.map(w => `<li>${escapeHtml(w)}</li>`).join('') + 
            `</ul>`;
    }

    return `<div class="mb-4 mt-6">
        <div class="text-xs text-gray-500 uppercase mb-2">Parser Debug Details</div>
        <div class="debug-box space-y-2">
            <div><span class="font-bold text-gray-400">OCR Used:</span> <span class="text-yellow-300 font-bold">${debug.ocr_used ? 'YES' : 'NO'}</span></div>
            <div><span class="font-bold text-gray-400">OCR Trigger:</span> <span class="text-gray-200">${escapeHtml(ocrTrigger)}</span></div>
            <div><span class="font-bold text-gray-400">Candidates:</span> <span class="text-gray-200">${escapeHtml(candidates)}</span></div>
            <div><span class="font-bold text-gray-400">Types:</span> <span class="text-gray-200">${escapeHtml(typeCounts)}</span></div>
            <div><span class="font-bold text-gray-400">Strategy:</span> <span class="text-gray-200">${escapeHtml(strategy)}</span></div>
            <div><span class="font-bold text-gray-400">Warnings:</span> <div class="text-red-300 mt-1">${warningsHtml}</div></div>
            
            <details class="mt-4 pt-2 border-t border-gray-700">
                <summary class="cursor-pointer text-gray-500 hover:text-gray-300 transition-colors">Raw Config & Logs (click to expand)</summary>
                <pre class="mt-2 text-xs overflow-x-auto">${escapeHtml(JSON.stringify(debug, null, 2))}</pre>
            </details>
        </div>
    </div>`;
}

// Dummy to avoid breaking the regex pattern entirely
function renderParserDebugOLD(debug) {
    return `<div class="mb-4 mt-6">'''

html = debug_old_pattern.sub(debug_new, html)

with open(index_path, 'w', encoding='utf-8') as f:
    f.write(html)
print("HTML patched.")
