"""Pure eval metrics: retrieval ranks, hit@k, MRR, refusal and answer checks."""

import re

from app.chain import is_refusal


def first_hit_rank(
    retrieved_pages_per_rank: list[list[int]], expected_pages: list[int]
) -> int | None:
    """1-based rank of the first chunk whose pages intersect `expected_pages`, or None."""
    expected = set(expected_pages)
    for rank, pages in enumerate(retrieved_pages_per_rank, start=1):
        if expected & set(pages):
            return rank
    return None


def hit_at_k(ranks: list[int | None], k: int) -> float:
    if not ranks:
        return 0.0
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks)


def mrr(ranks: list[int | None]) -> float:
    if not ranks:
        return 0.0
    return sum(1 / r for r in ranks if r is not None) / len(ranks)


def refusal_correct(answer: str, question_type: str) -> bool:
    refused = is_refusal(answer)
    return refused if question_type == "unanswerable" else not refused


# Hyphen and dash variants models use in ranges ("36‑42", "36–42").
_DASHES = str.maketrans(dict.fromkeys("\u2010\u2011\u2012\u2013\u2014\u2212", "-"))


def _squash(text: str) -> str:
    return re.sub(r"\s+", "", text).translate(_DASHES).casefold()


def contains_expected(answer: str, expected_answer: str) -> bool:
    """Cheap signal: case-insensitive substring match that ignores all whitespace.

    Whitespace is dropped entirely because Docling renders `mg/m2` as `mg/m 2`.
    """
    return _squash(expected_answer) in _squash(answer)
