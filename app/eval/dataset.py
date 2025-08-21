from __future__ import annotations
from typing import List, Dict, Any
import yaml

def load_dataset(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    items = y.get("items", [])
    assert isinstance(items, list) and items, "Dataset has no items"
    # minimal normalization
    for it in items:
        it.setdefault("gold_citations", [])
        it.setdefault("notes", "")
    return items
