
let documents = [];
let activeDocId = null;
let activeDoc = null;

async function loadDocuments() {
  const res = await fetch('/api/documents');
  documents = await res.json();
  renderSidebar();
}

function renderSidebar() {
  const el = document.getElementById('docList');
  el.innerHTML = documents.map(d => `
    <div class="doc-card flex items-center gap-3 px-3 py-2.5 rounded-lg cursor-pointer ${d.id === activeDocId ? 'active' : ''}"
         onclick="selectDoc('${d.id}')">
      <span class="status-dot status-${d.status}"></span>
      <div class="flex-1 min-w-0">
        <div class="text-sm text-gray-200 truncate">${d.filename}</div>
        <div class="flex items-center gap-2 mt-0.5">
          <span class="badge badge-${d.file_type}">${d.file_type}</span>
          <span class="text-xs text-gray-500">${formatSize(d.file_size)}</span>
        </div>
      </div>
      ${d.status === 'pending' ? `<button onclick="event.stopPropagation();parseSingle('${d.id}')" class="text-xs text-blue-400 hover:text-blue-300 whitespace-nowrap">Parse</button>` : ''}
    </div>
  `).join('');
}

function formatSize(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / 1048576).toFixed(1) + ' MB';
}

async function parseSingle(docId) {
  const doc = documents.find(d => d.id === docId);
  if (!doc) return;
  doc.status = 'parsing';
  renderSidebar();

  try {
    const res = await fetch(`/api/documents/${docId}/parse`, { method: 'POST' });
    const data = await res.json();
    doc.status = data.status || 'error';
    doc.parsed = true;
    renderSidebar();
    if (activeDocId === docId) showDetail(data);
  } catch (e) {
    doc.status = 'error';
    renderSidebar();
  }
}

async function parseAll() {
  const btn = document.getElementById('parseAllBtn');
  const statusEl = document.getElementById('globalStatus');
  btn.disabled = true;
  btn.innerHTML = '<span class="loading-spinner mr-2"></span>Parsing...';
  statusEl.textContent = '';

  try {
    const res = await fetch('/api/parse-all', { method: 'POST' });
    const results = await res.json();
    const success = results.filter(r => r.status === 'success').length;
    const total = results.length;
    statusEl.textContent = `${success}/${total} parsed successfully`;

    await loadDocuments();
    if (activeDocId) {
      const r = results.find(r => r.id === activeDocId);
      if (r) showDetail(r);
    }
  } catch (e) {
    statusEl.textContent = 'Parse failed: ' + e.message;
  } finally {
    btn.disabled = false;
    btn.innerHTML = 'Parse All';
  }
}

async function selectDoc(docId) {
  activeDocId = docId;
  renderSidebar();

  const doc = documents.find(d => d.id === docId);
  if (!doc) return;

  if (doc.parsed) {
    const res = await fetch(`/api/documents/${docId}`);
    const data = await res.json();
    showDetail(data);
  } else {
    await parseSingle(docId);
  }
}

function showDetail(data) {
  activeDoc = data;
  document.getElementById('emptyState').style.display = 'none';
  document.getElementById('detailView').style.display = 'flex';

  document.getElementById('detailBadge').className = `badge badge-${data.file_type}`;
  document.getElementById('detailBadge').textContent = data.file_type;
  document.getElementById('detailFilename').textContent = data.filename;

  const meta = data.metadata || {};
  const stats = document.getElementById('detailStats');
  const statItems = [
    { label: 'Status', value: data.status, color: data.status === 'success' ? 'text-green-400' : 'text-red-400' },
    { label: 'Pages', value: (data.pages || []).length },
    { label: 'Size', value: formatSize(data.file_size) },
    { label: 'Total Chars', value: (meta.total_chars || 0).toLocaleString() },
  ];
  if (meta.parse_time_sec != null) statItems.push({ label: 'Parse Time', value: meta.parse_time_sec + 's' });
  if (meta.text_quality) statItems.push({ label: 'Quality', value: meta.text_quality });
  if (meta.parser_used) statItems.push({ label: 'Parser', value: meta.parser_used });

  stats.innerHTML = statItems.map(s => `
    <div class="stat-card">
      <div class="text-xs text-gray-500">${s.label}</div>
      <div class="text-sm font-semibold ${s.color || 'text-white'}">${s.value}</div>
    </div>
  `).join('');

  if (data.error) {
    stats.innerHTML += `<div class="stat-card bg-red-900 bg-opacity-30">
      <div class="text-xs text-red-400">Error</div>
      <div class="text-sm text-red-300">${data.error}</div>
    </div>`;
  }

  renderTabs(data);
}

function renderTabs(data) {
  const pages = data.pages || [];
  const tabBar = document.getElementById('tabBar');

  if (pages.length === 0) {
    tabBar.innerHTML = '';
    document.getElementById('tabContent').innerHTML = '<p class="text-gray-500 text-sm">No content extracted.</p>';
    return;
  }

  tabBar.innerHTML = pages.map((p, i) => {
    const label = p.sheet_name || `Page ${p.page_num}`;
    return `<button class="tab-btn ${i === 0 ? 'active' : ''}" onclick="switchTab(${i})">${label}</button>`;
  }).join('');

  switchTab(0);
}

function switchTab(idx) {
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

  // 2. Preview & Overlay Layer + Block List Panel
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

window.generateBlockListHtml = function(blocks) {
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
        
        html += `<div id="overlay-${b.id}" class="bbox-overlay ${typeClass} transition-all duration-200" pointer-events-auto" data-type="${b.type}" 
            style="left:${left}px; top:${top}px; width:${width}px; height:${height}px;"
            onmouseenter="showHoverInfo('${escapeHtml(info)}'); highlightList('${b.id}')"
            onmouseleave="clearHoverInfo(); resetList('${b.id}')"></div>`;
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

function renderBlocks(blocks) {
    if (!blocks || blocks.length === 0) return '';
    let bHtml = `<div class="mb-6"><div class="text-xs text-gray-500 uppercase mb-2">Extracted Blocks (${blocks.length})</div><div class="space-y-2">`;
    blocks.forEach(b => {
        let typeColor = '#3b82f6'; // blue (default)
        if (b.type === 'table') typeColor = '#10b981'; // green
        else if (b.type === 'image' || b.type === 'chart_like') typeColor = '#8b5cf6'; // purple
        else if (b.type === 'title') typeColor = '#f59e0b'; // orange
        
        const confText = typeof b.confidence === 'number' ? `${(b.confidence * 100).toFixed(1)}%` : '-';
        const bboxText = Array.isArray(b.bbox) ? b.bbox.map(v => Math.round(v)).join(', ') : '-';
        
        bHtml += `
        <div class="block-card">
            <div class="block-header">
                <div>
                    <span class="block-type" style="background:${typeColor}">${b.type}</span>
                    <span class="block-meta ml-2 truncate">ID: ${b.id} | Src: ${b.source}</span>
                </div>
                <div class="block-meta">Conf: ${confText} | BBox: [${bboxText}]</div>
            </div>`;
        if (b.text) {
            bHtml += `<div class="block-text">${escapeHtml(b.text)}</div>`;
        }
        bHtml += `</div>`;
    });
    bHtml += `</div></div>`;
    return bHtml;
}

function renderParserDebug(debug) {
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
    return `<div class="mb-4 mt-6">
        <div class="text-xs text-gray-500 uppercase mb-2">Parser Debug Info</div>
        <div class="debug-box space-y-2">
            <div><span class="font-bold text-gray-400">Strategy:</span> <span class="text-gray-200">${escapeHtml(strategy)}</span></div>
            <div><span class="font-bold text-gray-400">OCR Engine:</span> <span class="text-gray-200">${escapeHtml(ocrEngine)}</span></div>
            <div><span class="font-bold text-gray-400">Fallback Reason:</span> <span class="text-gray-200">${escapeHtml(fallbackReason)}</span></div>
            <div><span class="font-bold text-gray-400">Extraction Order:</span> <span class="text-gray-200">${escapeHtml(order)}</span></div>
            <div><span class="font-bold text-gray-400">Warnings:</span> <div class="text-red-300 mt-1">${warningsHtml}</div></div>
            
            <details class="mt-4 pt-2 border-t border-gray-700">
                <summary class="cursor-pointer text-gray-500 hover:text-gray-300 transition-colors">Raw JSON</summary>
                <pre class="mt-2 text-xs overflow-x-auto">${escapeHtml(JSON.stringify(debug, null, 2))}</pre>
            </details>
        </div>
    </div>`;
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

loadDocuments();
