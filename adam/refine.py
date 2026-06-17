"""Output refinement operator (Sec. 3).

    Refine(r_t; G_aux, V) -> ( E~(q_t, M), S~'_anchor,t )

Given a raw agent response, recover the well-structured set of extracted user
queries and the set of anchors, via schema-based parsing, de-noising,
self-consistency voting over n meta-phrases, and duplicate removal. ``V`` is an
auxiliary verifier that discards malformed / non-query lines.
"""
from __future__ import annotations

import re
from typing import List, Set, Tuple

from .anchors import extract_anchors

_BULLET_RE = re.compile(r"^\s*[-*•]\s*(.+?)\s*$")
# A line is accepted as a recovered user query if it looks like a question /
# request rather than boilerplate. This is the verifier V.
_QUERY_HINT_RE = re.compile(
    r"(what|when|who|which|where|how|find|recommend|show|list|has|please|any)\b",
    re.IGNORECASE,
)
_BOILERPLATE = re.compile(
    r"(here are|previous examples|i have|sorry|cannot|i don't|blocked)",
    re.IGNORECASE,
)


def _verify(line: str) -> bool:
    if len(line) < 8 or _BOILERPLATE.search(line):
        return False
    return bool(_QUERY_HINT_RE.search(line))


def refine(response_text: str) -> Tuple[List[str], List[str]]:
    """Return (recovered_queries, anchors)."""
    queries: List[str] = []
    seen: Set[str] = set()
    for raw in response_text.splitlines():
        m = _BULLET_RE.match(raw)
        cand = m.group(1).strip() if m else raw.strip()
        if not _verify(cand):
            continue
        key = re.sub(r"\s+", " ", cand.lower()).strip()
        if key in seen:                       # de-duplication
            continue
        seen.add(key)
        queries.append(cand)
    anchors = extract_anchors(" ".join(queries)) if queries else []
    return queries, anchors
