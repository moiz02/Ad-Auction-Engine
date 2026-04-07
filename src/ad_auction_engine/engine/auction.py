"""Generalized second-price auction pricing logic."""

from __future__ import annotations

from ad_auction_engine.engine.ranker import build_auction_candidates, rank_ads
from ad_auction_engine.schemas import CandidateRecord, RankedAd, SearchResult


def apply_gsp_pricing(
    ranked_ads: list[RankedAd],
    reserve_price: float = 0.05,
    epsilon: float = 0.01,
) -> list[SearchResult]:
    """Apply generalized second-price clearing rules to ranked ads."""
    if not ranked_ads:
        return []

    results: list[SearchResult] = []
    for index, ad in enumerate(ranked_ads):
        if index < len(ranked_ads) - 1:
            next_ad_rank = ranked_ads[index + 1].ad_rank
            denominator = max(ad.predicted_ctr * ad.quality_score, 1e-9)
            clearing_price = (next_ad_rank / denominator) + epsilon
            # In practical sponsored search systems, CPC is capped by submitted bid.
            clearing_price = min(clearing_price, ad.bid_price)
        else:
            clearing_price = min(ad.bid_price, max(reserve_price, epsilon))

        results.append(
            SearchResult(
                ad_id=ad.ad_id,
                advertiser_name=ad.advertiser_name,
                bid_price=ad.bid_price,
                predicted_ctr=ad.predicted_ctr,
                quality_score=ad.quality_score,
                ad_rank=ad.ad_rank,
                clearing_price=round(float(max(clearing_price, 0.0)), 6),
            )
        )

    return results


def run_gsp_auction(
    candidates: list[CandidateRecord],
    predicted_ctrs: list[float],
    top_k: int,
    reserve_price: float = 0.05,
    epsilon: float = 0.01,
) -> list[SearchResult]:
    """Run ranking plus pricing end-to-end for top-k winners."""
    if top_k <= 0:
        return []

    auction_candidates = build_auction_candidates(candidates, predicted_ctrs)
    ranked_ads = rank_ads(auction_candidates)[:top_k]
    return apply_gsp_pricing(ranked_ads, reserve_price=reserve_price, epsilon=epsilon)
