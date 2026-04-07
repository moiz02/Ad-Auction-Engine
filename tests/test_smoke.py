from fastapi.testclient import TestClient

from ad_auction_engine.config import get_settings
from api.main import app


def test_health_endpoint_returns_ok() -> None:
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_settings_defaults_are_loaded() -> None:
    settings = get_settings()

    assert settings.app_name == "Ad Auction Engine"
    assert settings.top_k_results == 3
    assert settings.top_k_candidates == 20
