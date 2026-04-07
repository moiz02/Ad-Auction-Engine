"""In-memory inverted index for keyword-based candidate retrieval."""

from __future__ import annotations

import re
from collections import defaultdict

from ad_auction_engine.schemas import AdRecord

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def normalize_text(text: str) -> list[str]:
    """Tokenize query/keyword text into normalized terms."""
    return _TOKEN_PATTERN.findall(text.lower())


def build_inverted_index(ads: list[AdRecord]) -> dict[str, set[str]]:
    """Build a keyword -> ad_ids map."""
    index: dict[str, set[str]] = defaultdict(set)

    for ad in ads:
        for keyword in ad.keywords:
            for token in normalize_text(keyword):
                index[token].add(ad.ad_id)

    return dict(index)


def get_match_counts(
    query: str,
    inverted_index: dict[str, set[str]],
) -> dict[str, int]:
    """Return per-ad overlap counts for query tokens found in the index."""
    counts: dict[str, int] = defaultdict(int)

    unique_tokens = set(normalize_text(query))
    for token in unique_tokens:
        for ad_id in inverted_index.get(token, set()):
            counts[ad_id] += 1

    return dict(counts)
