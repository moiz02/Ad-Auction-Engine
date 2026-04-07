"""CTR inference utilities for retrieval candidates."""

from __future__ import annotations

from typing import Any

import numpy as np

from ad_auction_engine.ml.trainer import FEATURE_NAMES, load_ctr_model
from ad_auction_engine.schemas import CandidateRecord


def _candidate_features(candidates: list[CandidateRecord], query: str) -> np.ndarray:
    query_length = max(1, len(query.split()))
    rows = [
        [
            candidate.quality_score,
            candidate.historical_ctr,
            float(query_length),
            1.0 if candidate.keyword_match_type == "exact" else 0.0,
        ]
        for candidate in candidates
    ]
    return np.array(rows, dtype=np.float64)


def predict_ctr(
    model: Any,
    candidates: list[CandidateRecord],
    query: str,
) -> list[float]:
    """Predict click probability for candidates in their current ranked order."""
    if not candidates:
        return []

    X = _candidate_features(candidates, query=query)
    probabilities = model.predict_proba(X)[:, 1]
    return [float(np.clip(prob, 0.0, 1.0)) for prob in probabilities]


def load_model_and_predict(
    model_path: str,
    candidates: list[CandidateRecord],
    query: str,
) -> list[float]:
    """Load trained model artifact and predict CTR for candidate list."""
    model, feature_names = load_ctr_model(model_path)
    if tuple(feature_names) != FEATURE_NAMES:
        raise ValueError("Model feature names do not match predictor expectations.")

    return predict_ctr(model=model, candidates=candidates, query=query)
