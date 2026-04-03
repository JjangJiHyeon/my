import os
import sys
import json
from parsers import parse_document

test_files = [
    "미래에셋증권 1분기 실적보고서.pdf",
    "미래에셋증권 4분기 실적보고서.pdf",
    "한화투자증권 두산밥캣 기업분석 리포트.pdf"
]

docs_dir = "./documents"

print(f"{'Filename':<40} | {'Type':<15} | {'Pipeline':<25} | {'Blocks'}")
print("-" * 100)

for filename in test_files:
    filepath = os.path.join(docs_dir, filename)
    if not os.path.exists(filepath):
        continue
        
    try:
        parsed = parse_document(filepath)
        meta = parsed.get("metadata", {})
        pages = parsed.get("pages", [])
        
        doc_type = meta.get("document_type", "unknown")
        # Check first page's debug info
        pipeline = pages[0].get("parser_debug", {}).get("pipeline_used", "N/A")
        block_count = sum(len(p.get("blocks", [])) for p in pages)
        
        print(f"{filename[:40]:<40} | {doc_type:<15} | {pipeline:<25} | {block_count}")
        
        # Detailed Check for the first page
        p1 = pages[0]
        debug = p1.get("parser_debug", {})
        print(f"  > P1 Role: {debug.get('page_role', 'N/A')}")
        print(f"  > P1 Signals: {debug.get('slide_like_signals', debug.get('column_mode', 'N/A'))}")
        
    except Exception as e:
        print(f"Error parsing {filename}: {e}")
