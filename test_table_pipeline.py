import os
import json
from parsers.pdf_parser import parse_pdf, set_result_dir

def main():
    set_result_dir("tests_run")
    os.makedirs("tests_run", exist_ok=True)
    
    docs_to_test = [
        "documents/미래에셋증권 1분기 실적보고서.pdf",
        "documents/DS투자증권 시황분석 리포트.pdf",
        "documents/한화투자증권 두산밥캣 기업분석 리포트.pdf"
    ]
    
    for doc_path in docs_to_test:
        if not os.path.exists(doc_path):
            print(f"File not found: {doc_path}")
            continue
            
        print(f"\n--- Testing {os.path.basename(doc_path)} ---")
        try:
            res = parse_pdf(doc_path)
            for page in res.get("pages", []):
                p_num = page.get("page_num")
                print(f"Page {p_num}:")
                for sb in page.get("summary_blocks", []):
                    role = sb.get("meta", {}).get("summary_generation_method")
                    if role == "table_compaction":
                        score = sb.get("meta", {}).get("table_quality", 0)
                        print(f"  [TABLE] Quality: {score:.2f} | Rows: {sb.get('meta', {}).get('num_rows')} | Cols: {sb.get('meta', {}).get('num_cols')}")
        except Exception as e:
            print(f"Error parsing {doc_path}: {e}")

if __name__ == "__main__":
    main()
