import os
import json
from typing import Any, Dict

def calculate_quality_score(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate a quality score (0-100) and grade (A-F) for a parsing result.
    
    Metrics:
    - Fragmentation (Singleton Ratio): Too many 1-line blocks is bad.
    - Density: Too high block count per page is bad for RAG.
    - Structure: Presence of titles and table summaries is good.
    - Content: Empty documents get 0.
    """
    pages = result.get("pages", [])
    if not pages:
        return {"score": 0, "grade": "F", "reason": "No pages extracted"}

    total_blocks = 0
    total_text_blocks = 0
    singleton_text_blocks = 0
    title_count = 0
    table_count = 0
    total_chars = 0
    
    for p in pages:
        blocks = p.get("blocks", [])
        total_blocks += len(blocks)
        total_chars += len(p.get("text", ""))
        
        for b in blocks:
            btype = b.get("type", "text")
            text = b.get("text", "").strip()
            
            if btype == "title":
                title_count += 1
            elif btype == "table":
                table_count += 1
            elif btype == "text":
                total_text_blocks += 1
                if "\n" not in text: # Singleton
                    singleton_text_blocks += 1

    if total_chars == 0:
        return {"score": 0, "grade": "F", "reason": "Empty text content"}

    # Base Score
    score = 80 
    
    # 1. Fragmentation Penalty
    singleton_ratio = singleton_text_blocks / max(1, total_text_blocks)
    if singleton_ratio > 0.7: score -= 30
    elif singleton_ratio > 0.5: score -= 15
    elif singleton_ratio < 0.2: score += 10 # Good merging
    
    # 2. Block Density Penalty
    avg_blocks = total_blocks / len(pages)
    if avg_blocks > 100: score -= 20 # Too fragmented
    elif avg_blocks > 50: score -= 10
    elif avg_blocks < 20: score += 5 # Lean & clean
    
    # 3. Structure Bonus
    if title_count > 0: score += 5
    if table_count > 0: score += 5 # Detected tables
    
    # 4. RAG-Ready Bonus (Check if rag_text exists and is meaningful)
    has_rag = any(p.get("rag_text") for p in pages)
    if has_rag: score += 5

    # Final clamp
    final_score = max(0, min(100, score))
    
    # Grade assignment
    if final_score >= 90: grade = "A"
    elif final_score >= 70: grade = "B"
    elif final_score >= 50: grade = "C"
    else: grade = "F"
    
    return {
        "score": final_score,
        "grade": grade,
        "metrics": {
            "singleton_ratio": round(singleton_ratio, 2),
            "avg_blocks_per_page": round(avg_blocks, 1),
            "title_count": title_count,
            "table_count": table_count,
            "has_rag": has_rag
        }
    }
