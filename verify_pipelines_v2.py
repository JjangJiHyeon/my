import os
import sys
import json

# Try to find documents directory in common locations
base_dir = os.path.dirname(os.path.abspath(__file__))
docs_dir = os.path.join(base_dir, "documents")

if not os.path.exists(docs_dir):
    parent_docs = os.path.join(os.path.dirname(base_dir), "documents")
    if os.path.exists(parent_docs):
        docs_dir = parent_docs

test_files = [
    "DS투자증권 시황분석 리포트.pdf",
    "미래에셋증권 1분기 실적보고서.pdf",
    "미래에셋증권 2분기 실적보고서.pdf",
    "미래에셋증권 3분기 실적보고서.pdf",
    "미래에셋증권 4분기 실적보고서.pdf",
    "한화투자증권 두산밥캣 기업분석 리포트.pdf"
]

def run_verification():
    print(f"{'Filename':<40} | {'Type':<15} | {'Pipeline':<25} | {'Blocks'} | {'Empty/Salvage'}")
    print("-" * 125)
    
    try:
        from parsers import parse_document
    except Exception as e:
        print(f"CRITICAL: Failed to import parsers. Reason: {e}")
        return

    for filename in test_files:
        filepath = os.path.join(docs_dir, filename)
        if not os.path.exists(filepath):
            print(f"{filename[:37]+'...':<40} | {'MISSING':<15}")
            continue
            
        try:
            parsed = parse_document(filepath)
            meta = parsed.get("metadata", {})
            pages = parsed.get("pages", [])
            
            doc_type = meta.get("document_type", "unknown")
            pipeline = meta.get("pipeline_used", "N/A")
            total_blocks = sum(len(p.get("blocks", [])) for p in pages)
            
            # Slide specific metrics
            empty_count = sum(1 for p in pages if p.get("parser_debug", {}).get("empty_page_flag", False))
            salvage_count = sum(1 for p in pages if p.get("parser_debug", {}).get("salvage_applied", False))
            
            mismatch = meta.get("routing_mismatch_flag", False)
            status_str = f"{empty_count} emp / {salvage_count} slv"
            
            print(f"{filename[:40]:<40} | {doc_type:<15} | {pipeline:<25} | {total_blocks:<6} | {status_str}")
            
            if mismatch:
                init = meta.get("initial_routing_type", "?")
                ref = meta.get("refined_routing_type", "?")
                print(f"  [!] Mismatch: Predicted {init} -> Refined {ref} (Used {pipeline})")
                
            if meta.get("pipeline_fallback_reason"):
                print(f"  [!] Fallback: {meta['pipeline_fallback_reason']}")
                
        except Exception as e:
            print(f"{filename[:40]:<40} | ERROR: {e}")

if __name__ == "__main__":
    run_verification()
