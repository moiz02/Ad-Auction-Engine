from pathlib import Path

import numpy as np

from ad_auction_engine.data.interactions import generate_interactions, save_interactions_csv
from ad_auction_engine.data.inventory import generate_ad_inventory, save_ad_inventory_csv
from ad_auction_engine.ml.predictor import load_model_and_predict, predict_ctr
from ad_auction_engine.ml.trainer import (
    FEATURE_NAMES,
    load_ctr_model,
    save_ctr_model,
    train_and_save_ctr_model,
    train_ctr_model,
)
from ad_auction_engine.retrieval.retriever import CandidateRetriever


def _training_records() -> tuple[list, list]:
    ads = generate_ad_inventory(count=120, seed=200)
    interactions = generate_interactions(ads=ads, count=2500, seed=201)
    return ads, interactions


def test_train_ctr_model_returns_metrics_and_fitted_model() -> None:
    _, interactions = _training_records()

    model, metrics = train_ctr_model(interactions, test_size=0.25, random_seed=99)

    assert hasattr(model, "predict_proba")
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["positive_rate"] <= 1.0
    assert metrics["sample_size"] == float(len(interactions))
    if "roc_auc" in metrics:
        assert 0.0 <= metrics["roc_auc"] <= 1.0


def test_model_save_load_round_trip_predicts_same_values(tmp_path: Path) -> None:
    _, interactions = _training_records()
    model, _ = train_ctr_model(interactions)

    output_path = tmp_path / "ctr_model.pkl"
    save_ctr_model(model, output_path)
    loaded_model, feature_names = load_ctr_model(output_path)

    assert feature_names == FEATURE_NAMES

    features = np.array(
        [
            [8.1, 0.09, 2.0, 1.0],
            [4.3, 0.04, 3.0, 0.0],
        ],
        dtype=np.float64,
    )

    original = model.predict_proba(features)[:, 1]
    restored = loaded_model.predict_proba(features)[:, 1]
    assert np.allclose(original, restored)


def test_predict_ctr_for_candidates_returns_probabilities() -> None:
    ads, interactions = _training_records()
    model, _ = train_ctr_model(interactions)

    retriever = CandidateRetriever.from_ads(ads)
    candidates = retriever.retrieve(query="running shoes", top_k=8)

    predictions = predict_ctr(model=model, candidates=candidates, query="running shoes")

    assert len(predictions) == len(candidates)
    assert all(0.0 <= value <= 1.0 for value in predictions)


def test_train_and_save_then_load_and_predict(tmp_path: Path) -> None:
    ads, interactions = _training_records()

    ads_path = tmp_path / "ads.csv"
    interactions_path = tmp_path / "interactions.csv"
    model_path = tmp_path / "ctr_model.pkl"

    save_ad_inventory_csv(ads, ads_path)
    save_interactions_csv(interactions, interactions_path)

    metrics = train_and_save_ctr_model(
        interactions_path=interactions_path,
        model_output_path=model_path,
        test_size=0.2,
        random_seed=44,
    )

    assert model_path.exists()
    assert 0.0 <= metrics["accuracy"] <= 1.0

    retriever = CandidateRetriever.from_csv(str(ads_path))
    candidates = retriever.retrieve(query="fitness workout", top_k=5)
    predictions = load_model_and_predict(str(model_path), candidates=candidates, query="fitness workout")

    assert len(predictions) == len(candidates)
    assert all(0.0 <= value <= 1.0 for value in predictions)
