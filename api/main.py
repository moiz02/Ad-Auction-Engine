"""FastAPI entrypoint for the ad auction demo."""

from fastapi import FastAPI

from ad_auction_engine.config import get_settings
from ad_auction_engine.schemas import HealthResponse, SearchRequest, SearchResponse

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    summary="Sponsored search ad auction simulator",
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        environment=settings.environment,
    )


@app.post("/search", response_model=SearchResponse)
def search(payload: SearchRequest) -> SearchResponse:
    return SearchResponse(
        query=payload.query,
        results=[],
        message="Search flow is not implemented yet. Phase 2 will add retrieval and ranking.",
    )
