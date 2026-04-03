import os
import sys
from parsers import parse_document

docs_dir = "./documents"
filename = "DS투자증권 시황분석 리포트.pdf"
filepath = os.path.join(docs_dir, filename)

print(f"Testing: {filepath}")
if not os.path.exists(filepath):
    print("File not found")
    sys.exit(1)

try:
    result = parse_document(filepath)
    if result.get("status") == "error":
        print(f"Error: {result.get('error')}")
    else:
        print("Success")
        print(f"Type: {result.get('metadata', {}).get('document_type')}")
        print(f"Pipeline: {result.get('metadata', {}).get('pipeline_used')}")
except Exception as e:
    print(f"Exception: {e}")
