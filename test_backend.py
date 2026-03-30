import sys
import os
import json
import traceback

sys.path.insert(0, r"c:\Users\jihyeon\Desktop\my")
from app import _warm_cache, _scan_documents, parsed_cache

print("Warming cache...")
try:
    _warm_cache()
    print("Cache loaded. Keys:", list(parsed_cache.keys()))
except Exception as e:
    print("Error in _warm_cache:")
    traceback.print_exc()

print("\nScanning documents...")
try:
    docs = _scan_documents()
    print("Docs scanned:", len(docs))
    if len(docs) > 0:
        print("First doc:", docs[0])
except Exception as e:
    print("Error in _scan_documents:")
    traceback.print_exc()
