# app/corpus/clean.py
import re

_WS = re.compile(r"[ \t]+")
_MULTI_NL = re.compile(r"\n{3,}")

def normalize_text(t: str) -> str:
    # strip BOM & normalize line endings
    t = t.replace("\r\n", "\n").replace("\r", "\n").replace("\ufeff", "")
    # collapse excessive inline whitespace
    t = _WS.sub(" ", t)
    # trim lines
    t = "\n".join(line.strip() for line in t.split("\n"))
    # collapse too many blank lines
    t = _MULTI_NL.sub("\n\n", t)
    return t.strip()
