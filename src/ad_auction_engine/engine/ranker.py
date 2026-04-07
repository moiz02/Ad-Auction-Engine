"""Ad ranking logic for sponsored search auctions."""

from __future__ import annotations

from ad_auction_engine.schemas import AuctionCandidate, CandidateRecord, RankedAd


def compute_ad_rank(bid_price: float, predicted_ctr: float, quality_score: float) -> float:
    """Compute AdRank using bid * predicted_ctr * quality_score."""
    return round(float(bid_price * predicted_ctr * quality_score), 6)


def build_auction_candidates(
    candidates: list[CandidateRecord],
    predicted_ctrs: list[float],
) -> list[AuctionCandidate]:
    """Combine retrieval candidates and predicted CTR values."""
    if len(candidates) != len(predicted_ctrs):
        raise ValueError("candidates and predicted_ctrs must have the same length")

    result: list[AuctionCandidate] = []
    for candidate, predicted_ctr in zip(candidates, predicted_ctrs, strict=True):
        bounded_ctr = min(max(float(predicted_ctr), 0.0), 1.0)
        result.append(
            AuctionCandidate(
                ad_id=candidate.ad_id,
                advertiser_name=candidate.advertiser_name,
                bid_price=candidate.bid_price,
                quality_score=candidate.quality_score,
                predicted_ctr=bounded_ctr,
            )
        )

    return result


def rank_ads(auction_candidates: list[AuctionCandidate]) -> list[RankedAd]:
    """Sort ads by AdRank descending with deterministic tie-breakers."""
    ranked = [
        RankedAd(
            ad_id=ad.ad_id,
            advertiser_name=ad.advertiser_name,
            bid_price=ad.bid_price,
            quality_score=ad.quality_score,
            predicted_ctr=ad.predicted_ctr,
            ad_rank=compute_ad_rank(ad.bid_price, ad.predicted_ctr, ad.quality_score),
        )
        for ad in auction_candidates
    ]

    ranked.sort(
        key=lambda row: (row.ad_rank, row.quality_score, row.bid_price, row.ad_id),
        reverse=True,
    )
    return ranked
