"""Shared domain and transport models for the auction demo."""

from typing import Literal

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User search query.")
    user_id: str | None = Field(default=None, description="Optional caller identifier.")


class SearchResult(BaseModel):
    ad_id: str
    advertiser_name: str
    bid_price: float = Field(..., ge=0)
    predicted_ctr: float = Field(..., ge=0, le=1)
    quality_score: float = Field(..., ge=0)
    ad_rank: float = Field(..., ge=0)
    clearing_price: float = Field(..., ge=0)


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    message: str


class HealthResponse(BaseModel):
    status: str
    app_name: str
    environment: str


class AdRecord(BaseModel):
    ad_id: str
    advertiser_name: str
    keywords: list[str] = Field(..., min_length=1)
    bid_price: float = Field(..., ge=0)
    quality_score: float = Field(..., ge=1, le=10)
    historical_clicks: int = Field(..., ge=0)
    historical_impressions: int = Field(..., gt=0)


class InteractionRecord(BaseModel):
    interaction_id: str
    user_id: str
    query: str = Field(..., min_length=1)
    query_length: int = Field(..., ge=1)
    ad_id: str
    keyword_match_type: Literal["exact", "broad"]
    quality_score: float = Field(..., ge=1, le=10)
    historical_ctr: float = Field(..., ge=0, le=1)
    clicked: int = Field(..., ge=0, le=1)
