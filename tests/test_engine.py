import pytest

from ad_auction_engine.engine.auction import apply_gsp_pricing, run_gsp_auction
from ad_auction_engine.engine.ranker import (
    build_auction_candidates,
    compute_ad_rank,
    rank_ads,
)
from ad_auction_engine.schemas import CandidateRecord


def _candidates() -> list[CandidateRecord]:
    return [
        CandidateRecord(
            ad_id="ad-1",
            advertiser_name="Alpha",
            keywords=["running", "shoes"],
            bid_price=2.0,
            quality_score=8.0,
            historical_ctr=0.08,
            keyword_match_type="exact",
            matched_keywords=["running", "shoes"],
            retrieval_score=0.9,
        ),
        CandidateRecord(
            ad_id="ad-2",
            advertiser_name="Beta",
            keywords=["running"],
            bid_price=1.8,
            quality_score=7.0,
            historical_ctr=0.07,
            keyword_match_type="broad",
            matched_keywords=["running"],
            retrieval_score=0.8,
        ),
        CandidateRecord(
            ad_id="ad-3",
            advertiser_name="Gamma",
            keywords=["fitness"],
            bid_price=1.2,
            quality_score=6.0,
            historical_ctr=0.06,
            keyword_match_type="broad",
            matched_keywords=["fitness"],
            retrieval_score=0.7,
        ),
    ]


def test_compute_ad_rank_formula() -> None:
    assert compute_ad_rank(2.0, 0.3, 8.0) == pytest.approx(4.8)


def test_build_auction_candidates_validates_lengths() -> None:
    with pytest.raises(ValueError):
        build_auction_candidates(_candidates(), [0.2, 0.3])


def test_rank_ads_orders_descending_by_ad_rank() -> None:
    candidates = _candidates()
    predicted_ctrs = [0.25, 0.2, 0.3]

    auction_candidates = build_auction_candidates(candidates, predicted_ctrs)
    ranked = rank_ads(auction_candidates)

    assert [row.ad_id for row in ranked] == ["ad-1", "ad-2", "ad-3"]
    assert ranked[0].ad_rank >= ranked[1].ad_rank >= ranked[2].ad_rank


def test_apply_gsp_pricing_uses_next_ad_rank_and_reserve_for_last() -> None:
    candidates = _candidates()
    predicted_ctrs = [0.25, 0.2, 0.3]
    ranked = rank_ads(build_auction_candidates(candidates, predicted_ctrs))

    priced = apply_gsp_pricing(ranked, reserve_price=0.1, epsilon=0.01)

    assert len(priced) == 3

    expected_first = (ranked[1].ad_rank / (ranked[0].predicted_ctr * ranked[0].quality_score)) + 0.01
    expected_second = (ranked[2].ad_rank / (ranked[1].predicted_ctr * ranked[1].quality_score)) + 0.01

    assert priced[0].clearing_price == pytest.approx(min(expected_first, ranked[0].bid_price), rel=1e-6)
    assert priced[1].clearing_price == pytest.approx(min(expected_second, ranked[1].bid_price), rel=1e-6)
    assert priced[2].clearing_price == pytest.approx(0.1)


def test_run_gsp_auction_respects_top_k() -> None:
    results = run_gsp_auction(_candidates(), predicted_ctrs=[0.25, 0.2, 0.3], top_k=2)

    assert len(results) == 2
    assert all(0.0 <= row.clearing_price <= row.bid_price for row in results)
