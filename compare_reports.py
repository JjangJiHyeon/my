import json

with open('quality_report_before.json', 'r', encoding='utf-8') as f:
    b = json.load(f)
with open('quality_report_after.json', 'r', encoding='utf-8') as f:
    a = json.load(f)

m = {x['file']: x for x in b}

print("--- Comparison ---")
for x in a:
    fname = x['file']
    if fname not in m: continue
    
    s_b = m[fname]['singleton_text_ratio']
    s_a = x['singleton_text_ratio']
    
    blk_b = m[fname]['avg_blocks_per_page']
    blk_a = x['avg_blocks_per_page']
    
    tin_b = m[fname]['tiny_text_ratio']
    tin_a = x['tiny_text_ratio']
    
    sc_b = m[fname]['summary_ready_score']
    sc_a = x['summary_ready_score']
    
    print(f"{fname}:")
    print(f"  Singleton Ratio: {s_b:.3f} -> {s_a:.3f}")
    print(f"  AvgBlocks/Page:  {blk_b:.1f} -> {blk_a:.1f}")
    print(f"  Tiny Ratio:      {tin_b:.3f} -> {tin_a:.3f}")
    print(f"  Score:           {sc_b} -> {sc_a}")
    print()
