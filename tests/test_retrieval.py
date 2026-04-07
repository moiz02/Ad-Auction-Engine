from pathlib import Path

from ad_auction_engine.data.inventory import generate_ad_inventory, save_ad_inventory_csv
from ad_auction_engine.retrieval.index import build_inverted_index, get_match_counts, normalize_text
from ad_auction_engine.retrieval.retriever import CandidateRetriever
from ad_auction_engine.schemas import AdRecord


def _fixture_ads() -> list[AdRecord]:
    return [
        AdRecord(
            ad_id="ad-1",
            advertiser_name="Runner One",
            keywords=["running", "shoes", "sneakers"],
            bid_price=2.2,
            quality_score=9.2,
            historical_clicks=180,
            historical_impressions=1200,
        ),
        AdRecord(
            ad_id="ad-2",
            advertiser_name="Budget Sports",
            keywords=["running", "fitness"],
            bid_price=1.8,
            quality_score=7.4,
            historical_clicks=90,
            historical_impressions=1200,
        ),
        AdRecord(
            ad_id="ad-3",
            advertiser_name="City Travel",
            keywords=["travel", "hotel"],
            bid_price=3.1,
            quality_score=8.5,
            historical_clicks=130,
            historical_impressions=1300,
        ),
    ]


def test_normalize_text_splits_and_lowercases() -> None:
    assert normalize_text("Running SHOES!!!") == ["running", "shoes"]


def test_build_inverted_index_contains_expected_ad_ids() -> None:
    index = build_inverted_index(_fixture_ads())

    assert index["running"] == {"ad-1", "ad-2"}
    assert index["hotel"] == {"ad-3"}


def test_get_match_counts_tracks_overlap_per_ad() -> None:
    index = build_inverted_index(_fixture_ads())
    counts = get_match_counts(query="running shoes", inverted_index=index)

    assert counts["ad-1"] == 2
    assert counts["ad-2"] == 1
    assert "ad-3" not in counts


def test_retriever_returns_relevant_ranked_candidates() -> None:
    retriever = CandidateRetriever.from_ads(_fixture_ads())

    results = retriever.retrieve(query="running shoes", top_k=5, min_overlap=1)

    assert len(results) == 2
    assert results[0].ad_id == "ad-1"
    assert results[0].keyword_match_type == "exact"
    assert results[1].ad_id == "ad-2"
    assert results[1].keyword_match_type == "broad"
    assert results[0].retrieval_score >= results[1].retrieval_score


def test_retriever_respects_top_k_and_noisy_query() -> None:
    ads = generate_ad_inventory(count=80, seed=321)
    retriever = CandidateRetriever.from_ads(ads)

    results = retriever.retrieve(query="running shoes extra words", top_k=7, min_overlap=1)

    assert len(results) <= 7
    assert all(result.matched_keywords for result in results)


def test_retriever_returns_empty_for_no_match() -> None:
    retriever = CandidateRetriever.from_ads(_fixture_ads())

    results = retriever.retrieve(query="quantum biology", top_k=10)

    assert results == []


def test_retriever_can_be_built_from_csv(tmp_path: Path) -> None:
    ads_path = tmp_path / "ads.csv"
    save_ad_inventory_csv(_fixture_ads(), ads_path)

    retriever = CandidateRetriever.from_csv(str(ads_path))
    results = retriever.retrieve(query="travel hotel", top_k=3)

    assert len(results) == 1
    assert results[0].ad_id == "ad-3"
