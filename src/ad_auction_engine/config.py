"""Application configuration shared across API and UI surfaces."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Configuration for the local demo environment."""

    app_name: str = "Ad Auction Engine"
    environment: str = "development"
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    top_k_results: int = 3
    data_dir: str = "data"
    ads_output_path: str = "data/ads.csv"
    interactions_output_path: str = "data/interactions.csv"
    ads_count: int = 3000
    interactions_count: int = 50000
    random_seed: int = 42

    model_config = SettingsConfigDict(
        env_prefix="AD_AUCTION_",
        env_file=".env",
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings()
