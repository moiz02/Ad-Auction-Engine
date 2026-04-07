from pathlib import Path

from fastapi.testclient import TestClient

import api.main as api_main
from ad_auction_engine.data.interactions import generate_interactions, save_interactions_csv
from ad_auction_engine.data.inventory import generate_ad_inventory, save_ad_inventory_csv
from ad_auction_engine.ml.trainer import train_and_save_ctr_model
from api.main import app, get_search_components


def test_search_endpoint_runs_end_to_end(tmp_path: Path, monkeypatch) -> None:
    ads = generate_ad_inventory(count=120, seed=501)
    interactions = generate_interactions(ads=ads, count=2500, seed=502)

    ads_path = tmp_path / "ads.csv"
    interactions_path = tmp_path / "interactions.csv"
    model_path = tmp_path / "ctr_model.pkl"

    save_ad_inventory_csv(ads, ads_path)
    save_interactions_csv(interactions, interactions_path)
    train_and_save_ctr_model(interactions_path=interactions_path, model_output_path=model_path)

    monkeypatch.setattr(api_main.settings, "ads_output_path", str(ads_path))
    monkeypatch.setattr(api_main.settings, "ctr_model_path", str(model_path))
    monkeypatch.setattr(api_main.settings, "top_k_results", 3)
    monkeypatch.setattr(api_main.settings, "top_k_candidates", 10)
    monkeypatch.setattr(api_main.settings, "retrieval_min_overlap", 1)
    get_search_components.cache_clear()

    query = " ".join(ads[0].keywords[:2])
    client = TestClient(app)
    response = client.post("/search", json={"query": query, "user_id": "test-user"})

    assert response.status_code == 200
    payload = response.json()

    assert payload["query"] == query
    assert len(payload["results"]) <= 3
    assert len(payload["results"]) > 0
    assert "Returned" in payload["message"]

    top = payload["results"][0]
    assert 0.0 <= top["predicted_ctr"] <= 1.0
    assert top["clearing_price"] <= top["bid_price"]
    get_search_components.cache_clear()


def test_search_endpoint_handles_missing_artifacts_gracefully(tmp_path: Path, monkeypatch) -> None:
    missing_ads_path = tmp_path / "missing_ads.csv"
    missing_model_path = tmp_path / "missing_model.pkl"

    monkeypatch.setattr(api_main.settings, "ads_output_path", str(missing_ads_path))
    monkeypatch.setattr(api_main.settings, "ctr_model_path", str(missing_model_path))
    get_search_components.cache_clear()

    client = TestClient(app)
    response = client.post("/search", json={"query": "running shoes"})

    assert response.status_code == 200
    payload = response.json()

    assert payload["results"] == []
    assert "Search is unavailable" in payload["message"]
    get_search_components.cache_clear()
