import os
import json

def calculate_summary_ready_score(stats):
    score = 100
    
    # 감점: 너무 많은 파편화
    if stats["singleton_text_ratio"] > 0.6: score -= 20
    if stats["tiny_text_ratio"] > 0.4: score -= 15
    if stats["avg_blocks_per_page"] > 80: score -= 10
    
    # 가점: 의미 단위로 병합됨
    if stats["singleton_text_ratio"] < 0.3: score += 10
    if stats["avg_blocks_per_page"] < 30 and stats["avg_blocks_per_page"] > 5: score += 15

    return max(0, min(100, score))

def _is_singleton(t):
    return len(t.split('\n')) == 1

def generate_report(output_file="quality_report.json"):
    results_dir = os.path.join(os.path.dirname(__file__), "parsed_results")
    if not os.path.exists(results_dir):
        print("No parsed_results dir.")
        return
        
    report = []
    
    for fname in os.listdir(results_dir):
        if not fname.endswith(".json"): continue
        with open(os.path.join(results_dir, fname), "r", encoding="utf-8") as f:
            data = json.load(f)
            
        if data.get("status") != "success":
            continue
            
        meta = data.get("metadata", {})
        pages = data.get("pages", [])
        
        doc_type = meta.get("document_type", "unknown")
        pipeline = pages[0].get("parser_debug", {}).get("pipeline_used", "unknown") if pages else "unknown"
        
        total_blocks = 0
        total_text_blocks = 0
        singleton_text_blocks = 0
        tiny_text_blocks = 0
        
        title_count = 0
        table_count = 0
        image_count = 0
        
        salvage_applied = False
        
        page_block_counts = []
        
        for p in pages:
            blocks = p.get("blocks", [])
            b_cnt = len(blocks)
            page_block_counts.append((p["page_num"], b_cnt))
            total_blocks += b_cnt
            
            dbg = p.get("parser_debug", {})
            if dbg.get("salvage_applied"):
                salvage_applied = True
            
            for b in blocks:
                t = b["type"]
                text = b.get("text", "").strip()
                if t == "title": title_count += 1
                elif t == "table": table_count += 1
                elif t in ("image", "chart"): image_count += 1
                elif t == "text":
                    total_text_blocks += 1
                    if _is_singleton(text):
                        singleton_text_blocks += 1
                    if len(text) < 30:
                        tiny_text_blocks += 1

        avg_blocks = round(total_blocks / max(1, len(pages)), 2)
        r_singleton = round(singleton_text_blocks / max(1, total_text_blocks), 3)
        r_tiny = round(tiny_text_blocks / max(1, total_text_blocks), 3)
        
        page_block_counts.sort(key=lambda x: x[1], reverse=True)
        worst_pages = [pc[0] for pc in page_block_counts[:3]]
        
        stats = {
            "file": data.get("filename", fname),
            "doc_type": doc_type,
            "pipeline": pipeline,
            "pages": len(pages),
            "total_blocks": total_blocks,
            "avg_blocks_per_page": avg_blocks,
            "singleton_text_ratio": r_singleton,
            "tiny_text_ratio": r_tiny,
            "title_count": title_count,
            "text_count": total_text_blocks,
            "table_count": table_count,
            "image_count": image_count,
            "salvage_applied": salvage_applied,
            "worst_pages": worst_pages
        }
        
        stats["summary_ready_score"] = calculate_summary_ready_score(stats)
        report.append(stats)
        
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        
    print(f"Generated report at {output_file} ({len(report)} docs)")

if __name__ == "__main__":
    generate_report()
