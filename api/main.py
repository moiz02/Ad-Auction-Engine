"""Compatibility wrapper for uvicorn entrypoint imports."""

from ad_auction_engine.api_app import app, get_search_components, run_search_flow, settings

__all__ = ["app", "settings", "get_search_components", "run_search_flow"]
