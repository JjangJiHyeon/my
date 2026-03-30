import os
import json

res_dir = r"c:\Users\jihyeon\Desktop\my\parsed_results"
files = [f for f in os.listdir(res_dir) if f.endswith('.json')]

total_dedup = 0
total_merges = 0
total_reclassified = 0

for f in files:
    with open(os.path.join(res_dir, f), 'r', encoding='utf-8') as file:
        data = json.load(file)
        for p in data['pages']:
            debug = p.get('parser_debug', {})
            total_dedup += debug.get('dedup_stats', {}).get('dropped_duplicates', 0)
            total_merges += debug.get('text_merge_stats', {}).get('merged_pairs', 0)
            total_reclassified += len(debug.get('classification_overrides', []))

print(f"Total Dedup Dropped: {total_dedup}")
print(f"Total Text Merged: {total_merges}")
print(f"Total Images Reclassified to Text: {total_reclassified}")
