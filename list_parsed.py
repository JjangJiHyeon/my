import json
import os

parsed_dir = "parsed_results"
if not os.path.exists(parsed_dir):
    print("No parsed_results found.")
else:
    print(f"{'DOC_ID':<35} | {'TITLE'}")
    print("-" * 60)
    for f in os.listdir(parsed_dir):
        if f.endswith(".json"):
            try:
                with open(os.path.join(parsed_dir, f), "r", encoding="utf-8") as j:
                    # Optimized: only read first 2048 bytes for metadata
                    chunk = j.read(10000)
                    # Rough search for title if json loading fails for 100MB+ files
                    metadata_start = chunk.find('"metadata":')
                    if metadata_start != -1:
                        meta_chunk = chunk[metadata_start:metadata_start+2000]
                        title_start = meta_chunk.find('"title":')
                        if title_start != -1:
                            title_val = meta_chunk[title_start+8:].split('"')[1]
                            print(f"{f[:-5]:<35} | {title_val}")
                            continue
                # Fallback full load for small files
                with open(os.path.join(parsed_dir, f), "r", encoding="utf-8") as j:
                    data = json.load(j)
                    title = data.get("metadata", {}).get("title", "Untitled")
                    print(f"{f[:-5]:<35} | {title}")
            except Exception:
                print(f"{f[:-5]:<35} | [Error reading metadata]")
