import os
import sys
import json
from parsers.pdf_parser import parse_pdf, set_result_dir

# Set temporary result dir for previews
set_result_dir("./temp_verify_results")

test_files = [
    "DS투자증권 시황분석 리포트.pdf",
    "미래에셋증권 1분기 실적보고서.pdf",
    "미래에셋증권 2분기 실적보고서.pdf",
    "미래에셋증권 3분기 실적보고서.pdf",
    "미래에셋증권 4분기 실적보고서.pdf",
    "한화투자증권 두산밥캣 기업분석 리포트.pdf"
]

docs_dir = "./documents"

print(f"{'Filename':<40} | {'Document Type':<15} | {'Reason'}")
print("-" * 100)

results = []

for filename in test_files:
    filepath = os.path.join(docs_dir, filename)
    if not os.path.exists(filepath):
        print(f"File not found: {filename}")
        continue
        
    try:
        # Run parser
        parsed = parse_pdf(filepath)
        meta = parsed.get("metadata", {})
        doc_type = meta.get("document_type", "unknown")
        reason = meta.get("routing_reason", "N/A")
        signals = meta.get("routing_signals", {})
        
        print(f"{filename[:40]:<40} | {doc_type:<15} | {reason}")
        
        results.append({
            "filename": filename,
            "document_type": doc_type,
            "reason": reason,
            "signals": signals
        })
    except Exception as e:
        print(f"Error parsing {filename}: {e}")

# Detailed summary
print("\n" + "="*50)
print("Detailed Feature Summary")
print("="*50)
for r in results:
    s = r["signals"]
    print(f"\n[ {r['filename']} ]")
    print(f"  - Type: {r['document_type']}")
    print(f"  - Pages: {s.get('page_count')}")
    print(f"  - Landscape Ratio: {s.get('landscape_ratio')}")
    print(f"  - Avg Text Density: {s.get('avg_text_density')}")
    print(f"  - Avg Table Density: {s.get('avg_table_density')}")
    print(f"  - Avg Visual Density: {s.get('avg_visual_density')}")
    print(f"  - Page Type Dist: {s.get('page_type_distribution')}")
