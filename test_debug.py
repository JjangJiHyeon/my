import sys
import logging
from parsers.pdf_parser import parse_pdf
logging.basicConfig(level=logging.WARNING)
filepath = r"C:\Users\jihyeon\Desktop\my\documents\DS투자증권 시황분석 리포트.pdf"
pages = parse_pdf(filepath).get("pages", [])
for p in pages:
    if p["page_num"] == 1:
        blocks = p.get("blocks", [])
        raw_vis = [b for b in blocks if b.get("meta", {}).get("summary_role") in ("table_like", "chart_like", "image_like")]
        print(f"Page 1 has {len(raw_vis)} raw visual blocks.")
        if len(raw_vis) > 0:
            print("Types:", [b["type"] for b in raw_vis])
            print("Roles:", [b["meta"]["summary_role"] for b in raw_vis])
        sblocks = p.get("summary_blocks", [])
        print(f"Page 1 has {len(sblocks)} summary blocks.")
        break
