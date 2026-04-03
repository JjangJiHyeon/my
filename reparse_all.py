import os
import sys

# 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from parsers import parse_document
import json

DOC_DIR = os.path.join(current_dir, "documents")
RESULT_DIR = os.path.join(current_dir, "parsed_results")
os.makedirs(RESULT_DIR, exist_ok=True)

import hashlib
def _doc_id(filepath):
    return hashlib.md5(filepath.encode()).hexdigest()

print("Clearing results dir...")
for fname in os.listdir(RESULT_DIR):
    if fname.endswith(".json"):
        os.remove(os.path.join(RESULT_DIR, fname))

print("Parsing documents...")
for fname in sorted(os.listdir(DOC_DIR)):
    ext = os.path.splitext(fname)[1].lower()
    if ext not in [".pdf", ".hwp", ".doc", ".docx"]: continue
    
    fpath = os.path.join(DOC_DIR, fname)
    did = _doc_id(fpath)
    print(f"Parsing: {fname}")
    try:
        res = parse_document(fpath)
        res["filename"] = fname
        out_path = os.path.join(RESULT_DIR, did+".json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False)
        print(" -> Success (Parsing)")
        
        # Run export_to_gpt immediately
        import subprocess
        print(f" -> Exporting: {fname}")
        subprocess.run([sys.executable, "export_to_gpt.py", out_path], capture_output=True)
        print(" -> Success (Export)")
        
    except Exception as e:
        print(f" -> Error: {e}")
        
print("Done.")
