from __future__ import annotations

from typing import Any

import yaml


def load_dataset(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        y = yaml.safe_load(f)
    items = y.get("items", [])
    assert isinstance(items, list) and items, "Dataset has no items"
    # minimal normalization
    for it in items:
        it.setdefault("gold_citations", [])
        it.setdefault("notes", "")
    return items
