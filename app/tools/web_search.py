# app/tools/web_search.py
from __future__ import annotations
from ddgs import DDGS   # 👈 use the new package
from typing import List, Dict

def perform_web_search(query: str, num_results: int = 5) -> List[Dict]:
    """
    Perform a web search using DuckDuckGo (ddgs).
    Returns a list of title, snippet, url.
    """
    print("query-----------------------", query)
    results = []
    try:
        with DDGS() as ddgs:
            # The new API returns different keys: "title", "href", "body"
            for r in ddgs.text(query, max_results=num_results):
                print("result -------------------", r)
                results.append(
                    {
                        "title": r.get("title", ""),
                        "snippet": r.get("body", ""),   # summary/snippet
                        "url": r.get("href", ""),       # link
                    }
                )
    except Exception as e:
        results.append({"title": "Error", "snippet": str(e), "url": ""})
    return results
