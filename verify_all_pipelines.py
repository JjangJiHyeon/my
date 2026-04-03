import sys
import os
import json
import traceback

# 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    from parsers.pdf_parser import parse_pdf
    
    # 테스트용 PDF 경로 (리포트 예시)
    pdf_path = os.path.join(current_dir, "documents", "한화투자증권 두산밥캣 기업분석 리포트.pdf")
    
    if not os.path.exists(pdf_path):
        print(f"Error: {pdf_path} not found.")
        sys.exit(0)

    print(f"--- Parsing Test: {os.path.basename(pdf_path)} ---")
    result = parse_pdf(pdf_path)

    if result.get("status") == "success":
        print("\n[V] Status: SUCCESS (1)")
        metadata = result.get("metadata", {})
        print(f"Document Type: {metadata.get('document_type')}")
        print(f"Pipeline Used: {metadata.get('pipeline_used')}")
        
        pages = result.get("pages", [])
        if pages:
            p = pages[0]
            debug = p.get("parser_debug", {})
            print(f"\n--- Page 1 Debug Details ---")
            print(f"Layout Hint: {debug.get('page_layout_hint', 'N/A')}")
            print(f"Reading Strategy: {debug.get('reading_order_strategy', 'N/A')}")
            print(f"Reading Basis: {debug.get('reading_order_basis', 'N/A')}")
            print(f"Quality Notes: {debug.get('quality_notes', [])}")
            print(f"Merge Events: {len(debug.get('merge_events', []))}")
            print(f"Classification Overrides: {len(debug.get('classification_overrides', []))}")
            
            # 필수 필드 체크
            required = ["page_layout_hint", "reading_order_strategy", "quality_notes"]
            missing = [f for f in required if f not in debug]
            if not missing:
                print("\n[V] Verification complete: All unified debug fields present.")
            else:
                print(f"\n[!] Verification failed: Missing fields {missing}")
        else:
            print("[!] No pages extracted.")
    else:
        print(f"\n[!] Status: FAILURE (0). Error: {result.get('error')}")

except Exception as e:
    print(f"\n[!] Critical Exception: {e}")
    traceback.print_exc()
