import sys
import json
import logging
import fitz
from parsers.pdf_parser import parse_pdf

logging.basicConfig(level=logging.INFO)

def run_test(file_path):
    print(f"\n{'='*60}\nTesting {file_path}\n{'='*60}")
    try:
        result = parse_pdf(file_path)
        pages = result.get("pages", [])
    except Exception as e:
        print("Parse error:", e)
        return

    table_summaries = 0
    chart_summaries = 0
    total_summaries = 0
    used_ocr_count = 0
    
    for p in pages:
        sblocks = p.get("summary_blocks", [])
        total_summaries += len(sblocks)
        raw_tables = sum(1 for b in p.get("blocks", []) if b.get("meta", {}).get("summary_role") == "table_like")
        raw_visuals = sum(1 for b in p.get("blocks", []) if b.get("meta", {}).get("summary_role") in ("chart_like", "image_like"))
        print(f"Page {p['page_num']}: raw tables={raw_tables}, raw visuals={raw_visuals}, sblocks={len(sblocks)}")
        if not sblocks:
            continue
            
        print(f"\n--- Page {p['page_num']} ---")
        for sb in sblocks:
            t = sb.get("type")
            if t == "table_like_summary":
                table_summaries += 1
            elif t in ("chart_like_summary", "image_like_summary"):
                chart_summaries += 1
                
            if sb.get("meta", {}).get("used_ocr"):
                used_ocr_count += 1
                
            print(f"[{t}] id={sb['id']} src={sb['source_block_id']} conf={sb.get('confidence', 0):.2f}")
            print(f"  TEXT:\n{sb.get('text', '')}")
            meta = sb.get("meta", {})
            print(f"  META: compaction={meta.get('table_compaction')}, used_ocr={meta.get('used_ocr')}, method={meta.get('summary_generation_method')}")
            print("-" * 40)
            
    print(f"\nStats for {file_path}:")
    print(f"Total Summary Blocks: {total_summaries}")
    print(f"  Table Summaries:    {table_summaries}")
    print(f"  Chart/Image Summaries: {chart_summaries}")
    print(f"  Used OCR:           {used_ocr_count}")

if __name__ == "__main__":
    docs = [
        r"C:\Users\jihyeon\Desktop\my\documents\미래에셋증권 1분기 실적보고서.pdf",
        r"C:\Users\jihyeon\Desktop\my\documents\DS투자증권 시황분석 리포트.pdf", 
    ]
    for d in docs:
        run_test(d)
