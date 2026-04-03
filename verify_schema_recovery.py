import json
import os
import sys
from parsers import parse_document

test_docs = [
    "documents/DS투자증권 시황분석 리포트.pdf",
    "documents/미래에셋증권 4분기 실적보고서.pdf",
    "documents/한화투자증권 두산밥캣 기업분석 리포트.pdf"
]

def verify_all():
    print("=== Schema Recovery Verification ===")
    all_pass = True
    
    for doc_path in test_docs:
        if not os.path.exists(doc_path):
            print(f"[SKIP] {doc_path} not found")
            continue
            
        print(f"\nProcessing: {doc_path}")
        try:
            result = parse_document(doc_path)
            
            # 1. Global metadata check
            meta = result.get("metadata", {})
            total_chars = meta.get("total_chars", 0)
            status = result.get("status", "error")
            pipeline = meta.get("pipeline_used", "unknown")
            
            print(f"  Pipeline: {pipeline}")
            print(f"  Status: {status}")
            print(f"  Total Chars: {total_chars}")
            
            if total_chars == 0:
                print("  [FAIL] total_chars is 0")
                all_pass = False
            if status == "partial" and total_chars > 0:
                 # This shouldn't happen now if total_chars is > 0 and status is success
                 pass
            
            # 2. Page level check
            pages = result.get("pages", [])
            if not pages:
                print("  [FAIL] No pages returned")
                all_pass = False
                continue
                
            for i, p in enumerate(pages):
                page_num = p.get("page_num")
                preview = p.get("preview_image")
                text = p.get("text", "")
                blocks = p.get("blocks", [])
                
                print(f"  Page {page_num}: text_len={len(text)}, blocks={len(blocks)}, preview={preview}")
                
                if not preview or not preview.startswith("/api/"):
                    print(f"    [FAIL] Missing or invalid preview_image on page {page_num}")
                    all_pass = False
                if not text and blocks:
                    print(f"    [FAIL] text field is empty despite having blocks on page {page_num}")
                    all_pass = False
                if not p.get("page_width") or not p.get("page_height"):
                    print(f"    [FAIL] Missing page dimensions on page {page_num}")
                    all_pass = False
                    
        except Exception as e:
            print(f"  [ERROR] {e}")
            all_pass = False
            
    if all_pass:
        print("\n[SUCCESS] All schema recovery checks passed!")
    else:
        print("\n[FAILURE] Some schema recovery checks failed.")
    return all_pass

if __name__ == "__main__":
    verify_all()
