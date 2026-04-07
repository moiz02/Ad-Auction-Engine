"""CTR baseline model training and persistence utilities."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ad_auction_engine.data.interactions import load_interactions_csv
from ad_auction_engine.schemas import InteractionRecord

FEATURE_NAMES: tuple[str, ...] = (
    "quality_score",
    "historical_ctr",
    "query_length",
    "is_exact_match",
)


def _encode_features(records: list[InteractionRecord]) -> np.ndarray:
    """Convert interaction records into the model's numeric feature matrix."""
    matrix = np.array(
        [
            [
                record.quality_score,
                record.historical_ctr,
                float(record.query_length),
                1.0 if record.keyword_match_type == "exact" else 0.0,
            ]
            for record in records
        ],
        dtype=np.float64,
    )
    return matrix


def _encode_labels(records: list[InteractionRecord]) -> np.ndarray:
    return np.array([record.clicked for record in records], dtype=np.int64)


def train_ctr_model(
    records: list[InteractionRecord],
    test_size: float = 0.2,
    random_seed: int = 42,
    max_iter: int = 500,
) -> tuple[Pipeline, dict[str, float]]:
    """Train a baseline logistic regression model for click prediction."""
    if len(records) < 20:
        raise ValueError("At least 20 interaction records are required for training.")

    X = _encode_features(records)
    y = _encode_labels(records)

    unique_labels = np.unique(y)
    stratify = y if len(unique_labels) > 1 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_seed,
        stratify=stratify,
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=max_iter,
                    random_state=random_seed,
                    solver="liblinear",
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "sample_size": float(len(records)),
        "positive_rate": float(np.mean(y)),
    }

    # ROC AUC is only defined if both classes are represented in the test split.
    if len(np.unique(y_test)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_test, probabilities))

    return model, metrics


def save_ctr_model(model: Any, output_path: str | Path) -> Path:
    """Persist trained model artifact with feature contract metadata."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": model,
        "feature_names": FEATURE_NAMES,
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle)

    return path


def load_ctr_model(input_path: str | Path) -> tuple[Any, tuple[str, ...]]:
    """Load a trained model artifact and its feature metadata."""
    path = Path(input_path)
    with path.open("rb") as handle:
        payload = pickle.load(handle)

    return payload["model"], tuple(payload["feature_names"])


def train_and_save_ctr_model(
    interactions_path: str | Path,
    model_output_path: str | Path,
    test_size: float = 0.2,
    random_seed: int = 42,
    max_iter: int = 500,
) -> dict[str, float]:
    """Train on interactions CSV and persist model artifact."""
    records = load_interactions_csv(interactions_path)
    model, metrics = train_ctr_model(
        records=records,
        test_size=test_size,
        random_seed=random_seed,
        max_iter=max_iter,
    )
    save_ctr_model(model, model_output_path)
    return metrics
