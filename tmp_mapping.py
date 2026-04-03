import os
import json

parsed_dir = "parsed_results"
results = []
if os.path.exists(parsed_dir):
    for f in os.listdir(parsed_dir):
        if f.endswith(".json"):
            path = os.path.join(parsed_dir, f)
            try:
                with open(path, "r", encoding="utf-8") as j:
                    data = json.load(j)
                    results.append(f"{f[:-5]} | {data.get('filename')}")
            except:
                pass
print("\n".join(sorted(results)))
