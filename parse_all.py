import sys
import os
import shutil
import json

base_dir = r"c:\Users\jihyeon\Desktop\my"
sys.path.insert(0, base_dir)

from app import parsed_cache, DOC_DIR, RESULT_DIR, _doc_id, _save_result
from parsers import parse_document, SUPPORTED_EXTENSIONS
from parsers.pdf_parser import set_result_dir

set_result_dir(RESULT_DIR)

# Clear old cache files
print("Clearing old parsed results...")
for f in os.listdir(RESULT_DIR):
    fpath = os.path.join(RESULT_DIR, f)
    if os.path.isfile(fpath) and f.endswith(".json"):
        os.remove(fpath)
    elif os.path.isdir(fpath):
        import shutil
        shutil.rmtree(fpath, ignore_errors=True)

parsed_cache.clear()

results = []
print("Starting parse-all...")
for fname in sorted(os.listdir(DOC_DIR)):
    ext = os.path.splitext(fname)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        continue
    fpath = os.path.join(DOC_DIR, fname)
    if not os.path.isfile(fpath):
        continue
        
    did = _doc_id(fpath)
    print(f"Parsing: {fname}")
    try:
        result = parse_document(fpath)
    except Exception as exc:
        result = {
            "id": did,
            "filename": fname,
            "status": "error",
            "error": str(exc),
            "pages": [],
            "metadata": {},
        }
    parsed_cache[did] = result
    _save_result(did, result)
    results.append(result)

print(f"Parsed {len(results)} documents.")

for res in results:
    if res['status'] == 'success' and res.get('pages'):
        print('\n--- SUCCESS PAGE JSON ---')
        p = dict(res['pages'][0])
        p['blocks'] = [p['blocks'][0]] if p.get('blocks') else []
        print(json.dumps(p, indent=2))
        break

print('\n--- SIMULATED ERROR PAGE JSON ---')
from parsers.pdf_parser import _process_page, parse_pdf
import fitz
# Mocking a failed doc[idx] scenario would create the exception we wrote in parse_pdf
# Let's just create a dummy one that parse_pdf outputs natively if we pass an invalid path? No, invalid path raises FileNotFoundError
# We can just manually trigger the catch block by patching a method dynamically
doc = fitz.open()
doc.insert_page(0) # page 0

def mock_process_page(*args, **kwargs):
    raise RuntimeError("Intentional UI Error Test")

import parsers.pdf_parser
parsers.pdf_parser._process_page = mock_process_page

pdf_path = os.path.join(RESULT_DIR, 'dummy.pdf')
doc.save(pdf_path)

try:
    err_res = parsers.pdf_parser.parse_pdf(pdf_path)
    print(json.dumps(err_res['pages'][0], indent=2))
finally:
    doc.close()
    if os.path.exists(pdf_path):
        os.remove(pdf_path)

print("Done")
