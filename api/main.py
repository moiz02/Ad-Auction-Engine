"""FastAPI entrypoint for the ad auction demo."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI

from ad_auction_engine.config import get_settings
from ad_auction_engine.engine import run_gsp_auction
from ad_auction_engine.ml import load_ctr_model, predict_ctr
from ad_auction_engine.retrieval import CandidateRetriever
from ad_auction_engine.schemas import HealthResponse, SearchRequest, SearchResponse

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    summary="Sponsored search ad auction simulator",
)


@dataclass(slots=True)
class SearchComponents:
    retriever: CandidateRetriever
    model: Any


@lru_cache(maxsize=1)
def get_search_components() -> SearchComponents:
    ads_path = Path(settings.ads_output_path)
    model_path = Path(settings.ctr_model_path)

    if not ads_path.exists():
        raise FileNotFoundError(f"Ads dataset not found at {ads_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"CTR model not found at {model_path}")

    retriever = CandidateRetriever.from_csv(str(ads_path))
    model, _feature_names = load_ctr_model(str(model_path))
    return SearchComponents(retriever=retriever, model=model)


def run_search_flow(query: str) -> SearchResponse:
    components = get_search_components()

    candidates = components.retriever.retrieve(
        query=query,
        top_k=settings.top_k_candidates,
        min_overlap=settings.retrieval_min_overlap,
    )
    if not candidates:
        return SearchResponse(
            query=query,
            results=[],
            message="No matching ads found for this query.",
        )

    predicted_ctrs = predict_ctr(model=components.model, candidates=candidates, query=query)
    results = run_gsp_auction(
        candidates=candidates,
        predicted_ctrs=predicted_ctrs,
        top_k=settings.top_k_results,
    )

    return SearchResponse(
        query=query,
        results=results,
        message=f"Returned {len(results)} ads.",
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
    try:
        return run_search_flow(payload.query)
    except FileNotFoundError as exc:
        return SearchResponse(
            query=payload.query,
            results=[],
            message=(
                f"Search is unavailable: {exc}. "
                "Generate data and train a model before calling /search."
            ),
        )
