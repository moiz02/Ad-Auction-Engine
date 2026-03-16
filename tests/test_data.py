from pathlib import Path

from ad_auction_engine.data.generate import generate_datasets
from ad_auction_engine.data.interactions import (
    generate_interactions,
    load_interactions_csv,
    save_interactions_csv,
)
from ad_auction_engine.data.inventory import (
    generate_ad_inventory,
    load_ad_inventory_csv,
    save_ad_inventory_csv,
)


def test_generate_ad_inventory_is_reproducible() -> None:
    first = generate_ad_inventory(count=25, seed=101)
    second = generate_ad_inventory(count=25, seed=101)

    assert [item.model_dump() for item in first] == [item.model_dump() for item in second]


def test_ad_inventory_has_expected_ranges() -> None:
    records = generate_ad_inventory(count=50, seed=9)

    assert len(records) == 50
    assert all(0.5 <= row.bid_price <= 5.0 for row in records)
    assert all(1.0 <= row.quality_score <= 10.0 for row in records)
    assert all(1 <= len(row.keywords) <= 5 for row in records)
    assert all(row.historical_clicks <= row.historical_impressions for row in records)


def test_interactions_are_reproducible_for_same_seed() -> None:
    ads = generate_ad_inventory(count=30, seed=7)

    first = generate_interactions(ads=ads, count=100, seed=31)
    second = generate_interactions(ads=ads, count=100, seed=31)

    assert [item.model_dump() for item in first] == [item.model_dump() for item in second]
    assert set(record.keyword_match_type for record in first).issubset({"exact", "broad"})
    assert all(0.0 <= record.historical_ctr <= 1.0 for record in first)


def test_save_and_load_csv_round_trip(tmp_path: Path) -> None:
    ads = generate_ad_inventory(count=20, seed=88)
    interactions = generate_interactions(ads=ads, count=200, seed=89)

    ads_file = tmp_path / "ads.csv"
    interactions_file = tmp_path / "interactions.csv"

    save_ad_inventory_csv(ads, ads_file)
    save_interactions_csv(interactions, interactions_file)

    loaded_ads = load_ad_inventory_csv(ads_file)
    loaded_interactions = load_interactions_csv(interactions_file)

    assert [item.model_dump() for item in ads] == [item.model_dump() for item in loaded_ads]
    assert [item.model_dump() for item in interactions] == [item.model_dump() for item in loaded_interactions]


def test_generate_datasets_writes_expected_files(tmp_path: Path) -> None:
    ads_output = tmp_path / "ads.csv"
    interactions_output = tmp_path / "interactions.csv"

    ads_path, interactions_path = generate_datasets(
        ads_count=15,
        interactions_count=120,
        seed=123,
        ads_output_path=str(ads_output),
        interactions_output_path=str(interactions_output),
    )

    assert ads_path == ads_output
    assert interactions_path == interactions_output
    assert ads_output.exists()
    assert interactions_output.exists()

    loaded_ads = load_ad_inventory_csv(ads_output)
    loaded_interactions = load_interactions_csv(interactions_output)

    assert len(loaded_ads) == 15
    assert len(loaded_interactions) == 120
