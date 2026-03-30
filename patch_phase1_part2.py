import os
import re

base_dir = r"c:\Users\jihyeon\Desktop\my"
index_path = os.path.join(base_dir, "static", "index.html")
doc_parser_path = os.path.join(base_dir, "parsers", "doc_parser.py")
hwp_parser_path = os.path.join(base_dir, "parsers", "hwp_parser.py")

# 1. Fix static/index.html syntax error
with open(index_path, "r", encoding="utf-8") as f:
    html = f.read()

# Replace dangling '}</div>`;' at line 335 with just '}'
html = html.replace("}</div>`;\n  }\n\n  // 3. Raw Text fallback", "}\n\n  // 3. Raw Text fallback")

with open(index_path, "w", encoding="utf-8") as f:
    f.write(html)

# 2. Fix doc_parser.py
with open(doc_parser_path, "r", encoding="utf-8") as f:
    doc_code = f.read()

def inject_doc_schema(match):
    tables_val = match.group(1)
    return f"""pages = [{{
        "page_num": 1,
        "page_width": 0,
        "page_height": 0,
        "preview_width": 0,
        "preview_height": 0,
        "preview_scale_x": 1.0,
        "preview_scale_y": 1.0,
        "coord_space": "page_points",
        "preview_image": None,
        "text": full_text,
        "tables": {tables_val},
        "blocks": [{{
            "id": "p1_b0",
            "type": "text",
            "bbox": [0, 0, 0, 0],
            "text": full_text,
            "page_num": 1,
            "source": "doc_parser",
            "score": 1.0,
            "meta": {{}}
        }}] if full_text else [],
        "image_count": 0,
        "text_source": "native",
        "ocr_applied": False,
        "ocr_confidence": 0.0,
        "parser_debug": {{
            "preview_generated": False,
            "preview_error": None,
            "native_text_chars": len(full_text),
            "ocr_used": False,
            "ocr_trigger_reason": "ocr_not_needed",
            "candidate_counts": {{
                "raw_text_blocks": 1 if full_text else 0,
                "final_blocks": 1 if full_text else 0
            }},
            "block_type_counts": {{
                "text": 1 if full_text else 0
            }},
            "dropped_blocks": [],
            "bbox_warnings": []
        }}
    }}]"""

doc_code = re.sub(r'pages\s*=\s*\[\{"page_num": 1, "text": full_text, "tables": (.*?)\}\]', inject_doc_schema, doc_code)

with open(doc_parser_path, "w", encoding="utf-8") as f:
    f.write(doc_code)

# 3. Fix hwp_parser.py
with open(hwp_parser_path, "r", encoding="utf-8") as f:
    hwp_code = f.read()

schema_hwp = """pages.append({
            "page_num": i + 1,
            "page_width": 0,
            "page_height": 0,
            "preview_width": 0,
            "preview_height": 0,
            "preview_scale_x": 1.0,
            "preview_scale_y": 1.0,
            "coord_space": "page_points",
            "preview_image": None,
            "text": text,
            "tables": [],
            "blocks": [{
                "id": f"p{i+1}_b0",
                "type": "text",
                "bbox": [0, 0, 0, 0],
                "text": text,
                "page_num": i + 1,
                "source": "hwp_parser",
                "score": 1.0,
                "meta": {}
            }] if text else [],
            "image_count": 0,
            "text_source": "native",
            "ocr_applied": False,
            "ocr_confidence": 0.0,
            "parser_debug": {
                "preview_generated": False,
                "preview_error": None,
                "native_text_chars": len(text),
                "ocr_used": False,
                "ocr_trigger_reason": "ocr_not_needed",
                "candidate_counts": {
                    "raw_text_blocks": 1 if text else 0,
                    "final_blocks": 1 if text else 0
                },
                "block_type_counts": {
                    "text": 1 if text else 0
                },
                "dropped_blocks": [],
                "bbox_warnings": []
            }
        })"""

hwp_code = hwp_code.replace('pages.append({"page_num": i + 1, "text": text, "tables": []})', schema_hwp)

with open(hwp_parser_path, "w", encoding="utf-8") as f:
    f.write(hwp_code)

print("Patch applied successfully.")
