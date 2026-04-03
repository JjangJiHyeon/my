import os
import sys
import json
from parsers import parse_document
from parsers.pdf_parser import set_result_dir

# Set temporary result dir for previews
set_result_dir("./temp_verify_results")

test_files = [
    # PDF
    "DS투자증권 시황분석 리포트.pdf",
    "미래에셋증권 1분기 실적보고서.pdf",
    "한화투자증권 두산밥캣 기업분석 리포트.pdf",
    # DOC
    "금융감독원 251125_(보도자료) 25.10월중 기업의 직접금융 조달실적.doc",
    # HWP
    "농협 2022년 9월말 기준 사업보고서.hwp"
]

docs_dir = "./documents"

print(f"{'Filename':<50} | {'Type':<15} | {'Ext':<5} | {'Reason'}")
print("-" * 120)

results = []

for filename in test_files:
    filepath = os.path.join(docs_dir, filename)
    if not os.path.exists(filepath):
        print(f"File not found: {filename}")
        continue
        
    try:
        # Run universal parser (parse_document in parsers/__init__.py)
        parsed = parse_document(filepath)
        meta = parsed.get("metadata", {})
        doc_type = meta.get("document_type", "unknown")
        reason = meta.get("routing_reason", "N/A")
        signals = meta.get("routing_signals", {})
        ext = parsed.get("file_type", "???")
        
        print(f"{filename[:50]:<50} | {doc_type:<15} | {ext:<5} | {reason}")
        
        results.append({
            "filename": filename,
            "document_type": doc_type,
            "extension": ext,
            "reason": reason,
            "signals": signals
        })
    except Exception as e:
        print(f"Error parsing {filename}: {e}")

# Result details
print("\n" + "="*80)
print("Routing Signal Summary")
print("="*80)
for r in results:
    s = r["signals"]
    print(f"\n[ {r['filename']} ] ({r['extension']})")
    print(f"  - Final Type: {r['document_type']}")
    print(f"  - Signals: {s}")
