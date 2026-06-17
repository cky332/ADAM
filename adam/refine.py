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


def refine(response_text: str, retrieved=None) -> Tuple[List[str], List[str]]:
    """Return (recovered_queries, anchors).

    Two modes:
      * structured response (bulleted / listed) -- parse lines verbatim;
      * if ``retrieved`` is given (real-LLM agents), additionally treat a record
        as recovered when a substantial overlap of its query text appears in the
        response (the LLM may have paraphrased or quoted partially).
    """
    queries: List[str] = []
    seen: Set[str] = set()
    for raw in response_text.splitlines():
        m = _BULLET_RE.match(raw)
        cand = m.group(1).strip() if m else raw.strip()
        if not _verify(cand):
            continue
        key = re.sub(r"\s+", " ", cand.lower()).strip()
        if key in seen:
            continue
        seen.add(key)
        queries.append(cand)

    if retrieved is not None:
        body = re.sub(r"\s+", " ", response_text.lower())
        for rec in retrieved:
            q = re.sub(r"\s+", " ", rec.query.strip().lower())
            if q in body:                                       # verbatim quote
                if q not in seen:
                    seen.add(q); queries.append(rec.query)
                continue
            # paraphrase heuristic: most distinctive content tokens reappear
            toks = [t for t in re.findall(r"[a-z]+", q) if len(t) > 3]
            if len(toks) >= 4:
                hits = sum(1 for t in toks if t in body)
                if hits / len(toks) >= 0.65:
                    if q not in seen:
                        seen.add(q); queries.append(rec.query)

    anchors = extract_anchors(" ".join(queries) + " " + response_text) if queries else \
              extract_anchors(response_text)
    return queries, anchors
