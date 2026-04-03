import os
import json
from parsers.pdf_parser import parse_pdf, set_result_dir

set_result_dir("./results")

docs = [f for f in os.listdir("documents") if f.endswith(".pdf") and ("미래" in f or "실적" in f)]
if not docs:
    docs = [f for f in os.listdir("documents") if f.endswith(".pdf")]

test_doc = os.path.join("documents", docs[0])
print(f"Testing with document: {test_doc}")

result = parse_pdf(test_doc)

for i, page in enumerate(result.get("pages", [])[:3]):
    print(f"\n--- PAGE {i+1} ---")
    blocks = page.get("blocks", [])
    
    # Print some block data
    for b in blocks:
        meta = b.get("meta", {})
        role = meta.get("summary_role", b.get("type"))
        text = b.get("text", "").replace("\n", " ")[:60]
        exclude = meta.get("summary_exclude", False)
        reason = meta.get("summary_exclude_reason")
        priority = meta.get("summary_priority")
        src = meta.get("recovery_source")
        
        ex_str = f"[EXCLUDE: {reason}]" if exclude else ""
        print(f"[{role:^14}] (P:{priority}, Src:{src}) {text:<50} {ex_str}")
        
        # Check visual subtypes and nearby text
        if role in ("chart_like", "table_like", "image"):
            if "associated_title" in meta:
                print(f"   => Assoc Title: {meta['associated_title']}")
            if "nearby_text_before" in meta:
                print(f"   => Text Before: {meta['nearby_text_before'][:40]}")
            if "nearby_text_after" in meta:
                print(f"   => Text After : {meta['nearby_text_after'][:40]}")

    print("\n[Parser Debug]")
    debug = page.get("parser_debug", {})
    print(f"OCR Used: {debug.get('ocr_used')} ({debug.get('ocr_trigger_reason')})")
    print(f"Native Text Chars: {debug.get('native_text_chars')}")
