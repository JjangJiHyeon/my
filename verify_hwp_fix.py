import sys
import os
import traceback

# 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    from parsers.hwp_parser import parse_hwp
    
    # 문제가 되었던 HWP 파일 경로
    hwp_path = os.path.join(current_dir, "documents", "농협 2022년 9월말 기준 사업보고서.hwp")
    
    if not os.path.exists(hwp_path):
        print(f"Error: {hwp_path} not found.")
        sys.exit(0)

    print(f"--- HWP Parsing Test: {os.path.basename(hwp_path)} ---")
    result = parse_hwp(hwp_path)

    if result.get("status") == "success":
        print("\n[V] Status: SUCCESS (1)")
        metadata = result.get("metadata", {})
        print(f"Document Type: {metadata.get('document_type')}")
        print(f"Page Count: {metadata.get('page_count')}")
        
        pages = result.get("pages", [])
        if pages:
            # 첫 페이지 요약
            p = pages[0]
            debug = p.get("parser_debug", {})
            print(f"Text length: {len(p.get('text', ''))} chars")
            print(f"Block count: {len(p.get('blocks', []))}")
            print(f"Narrative Analysis: {debug.get('narrative_analysis', {})}")
            
            if len(p.get('text', '')) > 0:
                print("\n[V] Verification complete: HWP text successfully extracted.")
            else:
                print("\n[!] Warning: HWP parsed but text is empty.")
        else:
            print("[!] No pages extracted.")
    else:
        print(f"\n[!] Status: FAILURE (0). Error: {result.get('error')}")

except Exception as e:
    print(f"\n[!] Critical Exception: {e}")
    traceback.print_exc()
