import sys
import os
import json

base_dir = r"c:\Users\jihyeon\Desktop\my"
sys.path.insert(0, base_dir)

from app import _doc_id, DOC_DIR, RESULT_DIR

hwp_filename = "DS투자증권_미국주식.hwp"
hwp_path = os.path.join(DOC_DIR, hwp_filename)
if os.path.exists(hwp_path):
    hwp_id = _doc_id(hwp_path)
    print("HWP JSON Example:")
    try:
        with open(os.path.join(RESULT_DIR, hwp_id + ".json"), "r", encoding="utf-8") as f:
            d = json.load(f)
            p = d["pages"][0]
            if p.get("text"): p["text"] = p["text"][:50] + "..."
            if p.get("blocks") and p["blocks"][0].get("text"):
                p["blocks"][0]["text"] = p["blocks"][0]["text"][:50] + "..."
            print(json.dumps(p, indent=2))
    except Exception as e:
        print("HWP Error", e)

print("Done")
