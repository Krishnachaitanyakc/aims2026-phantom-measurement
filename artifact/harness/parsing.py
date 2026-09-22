"""Answer parsing, numeric equivalence, and refusal detection.

The parser intentionally reproduces the original pipeline's behavior (a `####`
marker extraction with a last-numeric-token fallback) so that the campaign
measures the same instrument the historical pipeline used, while additionally
*tagging* refusal-shaped outputs before parsing so their contribution can be
quantified. The phrase list is a heuristic diagnostic without independent
annotation-based calibration. Flags can include false positives and false
negatives; their frequency is not a validated refusal rate or lower bound.
"""
from __future__ import annotations

import re

# `#### 42`, `#### $1,234.50`, `#### -3` — take the LAST occurrence.
_HASH_RE = re.compile(r"####\s*\$?\s*(-?[\d,]+(?:\.\d+)?)")
# Fallback: any numeric token; we take the LAST one (original pipeline behavior).
_NUM_RE = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?%?")

REFUSAL_PATTERNS = [
    "don't see the problem", "do not see the problem",
    "don't see the original", "do not see the original",
    "don't have the problem", "do not have the problem",
    "no problem statement", "problem statement is missing",
    "missing problem statement", "need the problem statement",
    "need the original problem", "need the actual problem",
    "provide the problem", "share the problem", "without the problem statement",
    "cannot see the problem", "can't see the problem",
    "don't see the actual", "do not see the actual",
    "there is no problem", "no original problem",
    "don't have enough context", "do not have enough context",
    "missing context", "context is missing",
    "previous steps aren't shown", "previous analysis is missing",
    "what problem are you", "which problem you", "the problem you're referring",
    "i don't see any", "i do not see any",
]


def is_refusal(text: str) -> bool:
    low = text.lower()
    return any(p in low for p in REFUSAL_PATTERNS)


def _clean_token(tok: str) -> str:
    return tok.replace("$", "").replace(",", "").rstrip("%").strip()


def extract_answer(text: str) -> tuple[str, str]:
    """Return (answer, path) where path is 'hash', 'fallback', or 'none'."""
    if not text:
        return "", "none"
    hash_matches = _HASH_RE.findall(text)
    if hash_matches:
        return _clean_token(hash_matches[-1]), "hash"
    num_matches = _NUM_RE.findall(text)
    if num_matches:
        return _clean_token(num_matches[-1]), "fallback"
    return "", "none"


def parse_output(text: str) -> dict:
    ans, path = extract_answer(text)
    return {"answer": ans, "parse_path": path, "refusal": is_refusal(text)}


def _to_float(s: str):
    try:
        return float(_clean_token(s))
    except (ValueError, AttributeError):
        return None


def answers_match(a: str, b: str) -> bool:
    """Numeric equivalence with tolerance; empty answers never match anything."""
    a = (a or "").strip()
    b = (b or "").strip()
    if not a or not b:
        return False
    fa, fb = _to_float(a), _to_float(b)
    if fa is not None and fb is not None:
        return abs(fa - fb) <= 1e-6 * max(1.0, abs(fa), abs(fb))
    return a == b


def gold_from_gsm8k(answer_field: str) -> str:
    """GSM8K gold: text after the final '#### '."""
    m = re.search(r"####\s*(.+?)\s*$", answer_field.strip())
    if not m:
        raise ValueError("gold answer marker '####' not found")
    return _clean_token(m.group(1))
