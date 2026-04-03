import json, glob

files = glob.glob("parsed_results/*.json")
for f in sorted(files):
    data = json.load(open(f, encoding="utf-8"))
    for page in data.get("pages", []):
        debug = page.get("parser_debug", {}).get("summary_debug", {})
        hint = debug.get("_page_layout_hint", "")
        if not hint:
            continue
        sblocks = page.get("summary_blocks", [])
        pn = page.get("page_num", "?")
        print(f"\n=== {f} p{pn} | layout={hint} ===")
        print(f"  summary_blocks={len(sblocks)}")
        for s in sblocks:
            bid = s.get("id", "?")
            role = s.get("meta", {}).get("summary_role", "?")
            print(f"    {bid}  role={role}")
        
        # skip stats
        cover_skip = debug.get("cover_visual_noise_skips", 0)
        dash_skip = debug.get("dashboard_final_gate_skips", 0)
        dash_noise = debug.get("dashboard_visual_noise_skips", 0)
        print(f"  Skips: cover_visual_noise={cover_skip}, dashboard_final_gate={dash_skip}, dashboard_visual_noise={dash_noise}")
