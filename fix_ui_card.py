import re

with open(r"c:\Users\jihyeon\Desktop\my\static\index.html", "r", encoding="utf-8") as f:
    html = f.read()

html = re.sub(
    r'<div class="text-\[10px\] text-gray-500 mt-1 uppercase">ID: \$\{b\.id\} \| TYPE: \$\{b\.type\} \| SRC: \$\{b\.source\}</div>\s*\$\{b\.meta\?\.classification_reason \? `<div class="text-\[10px\] text-blue-400 mt-0\.5">Class Reason: \$\{b\.meta\.classification_reason\}</div>` : \'\'\}\s*\$\{b\.meta\?\.score_reason \? `<div class="text-\[10px\] text-green-400 mt-0\.5">Score Reason: \$\{b\.meta\.score_reason\}</div>` : \'\'\}',
    r'''<div class="text-[10px] text-gray-500 mt-1 uppercase whitespace-nowrap overflow-hidden text-ellipsis">ID: ${b.id} | TYPE: ${b.type} | SRC: ${b.source}</div>
          ${b.meta?.classification_reason ? `<div class="text-[10px] text-blue-400 mt-0.5 whitespace-nowrap overflow-hidden text-ellipsis" title="${b.meta.classification_reason}">Class: ${b.meta.classification_reason}</div>` : ''}
          ${b.meta?.score_reason ? `<div class="text-[10px] text-green-400 mt-0.5 whitespace-nowrap overflow-hidden text-ellipsis" title="${b.meta.score_reason}">Score: ${b.meta.score_reason}</div>` : ''}''',
    html
)

with open(r"c:\Users\jihyeon\Desktop\my\static\index.html", "w", encoding="utf-8") as f:
    f.write(html)
