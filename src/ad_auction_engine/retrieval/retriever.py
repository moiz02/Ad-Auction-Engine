"""Candidate retriever built on top of an inverted keyword index."""

from __future__ import annotations

from dataclasses import dataclass

from ad_auction_engine.data.inventory import load_ad_inventory_csv
from ad_auction_engine.retrieval.index import build_inverted_index, get_match_counts, normalize_text
from ad_auction_engine.schemas import AdRecord, CandidateRecord


@dataclass(slots=True)
class CandidateRetriever:
    """Fetch and score relevant ads for a search query."""

    ads_by_id: dict[str, AdRecord]
    inverted_index: dict[str, set[str]]

    @classmethod
    def from_ads(cls, ads: list[AdRecord]) -> "CandidateRetriever":
        return cls(
            ads_by_id={ad.ad_id: ad for ad in ads},
            inverted_index=build_inverted_index(ads),
        )

    @classmethod
    def from_csv(cls, ads_path: str) -> "CandidateRetriever":
        ads = load_ad_inventory_csv(ads_path)
        return cls.from_ads(ads)

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        min_overlap: int = 1,
    ) -> list[CandidateRecord]:
        """Return top-k candidate ads for the query ranked by retrieval score."""
        query_tokens = set(normalize_text(query))
        if not query_tokens or top_k <= 0:
            return []

        match_counts = get_match_counts(query=query, inverted_index=self.inverted_index)
        candidates: list[CandidateRecord] = []

        for ad_id, overlap_count in match_counts.items():
            if overlap_count < min_overlap:
                continue

            ad = self.ads_by_id[ad_id]
            ad_tokens = set()
            for keyword in ad.keywords:
                ad_tokens.update(normalize_text(keyword))

            matched_keywords = sorted(query_tokens.intersection(ad_tokens))
            if not matched_keywords:
                continue

            coverage = len(matched_keywords) / len(query_tokens)
            exact_match = len(matched_keywords) == len(query_tokens)
            keyword_match_type = "exact" if exact_match else "broad"

            historical_ctr = ad.historical_clicks / ad.historical_impressions
            quality_norm = ad.quality_score / 10.0
            exact_bonus = 0.1 if exact_match else 0.0

            retrieval_score = (0.6 * coverage) + (0.25 * quality_norm) + (0.15 * historical_ctr)
            retrieval_score += exact_bonus

            candidates.append(
                CandidateRecord(
                    ad_id=ad.ad_id,
                    advertiser_name=ad.advertiser_name,
                    keywords=ad.keywords,
                    bid_price=ad.bid_price,
                    quality_score=ad.quality_score,
                    historical_ctr=round(historical_ctr, 6),
                    keyword_match_type=keyword_match_type,
                    matched_keywords=matched_keywords,
                    retrieval_score=round(retrieval_score, 6),
                )
            )

        candidates.sort(
            key=lambda record: (
                record.retrieval_score,
                record.quality_score,
                record.bid_price,
                record.ad_id,
            ),
            reverse=True,
        )
        return candidates[:top_k]
