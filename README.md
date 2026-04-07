<!-- # Ad Auction Engine

This repository is a recruiter-facing sponsored-search auction simulator. The goal is to show clean system structure, ML workflow discipline, and a small interactive product surface without over-claiming production realism.

## Phase 1 Status

Phase 1 establishes the project foundation:

- package-style Python layout under `src/`
- bootable FastAPI and Streamlit entrypoints
- shared schemas and configuration
- smoke tests and CI

Business logic for data generation, retrieval, ranking, auction pricing, and CTR modeling will land in later phases. -->

## Project Structure

```text
ad-auction-engine/
├── api/
│   └── main.py
├── data/
├── models/
├── src/
│   └── ad_auction_engine/
│       ├── config.py
│       ├── schemas.py
## Progress Status

### Phase 1: Foundation (Completed)

Phase 1 established the project foundation:

- package-style Python layout under `src/`
- bootable FastAPI and Streamlit entrypoints
- shared schemas and configuration
- smoke tests and CI

Business logic for data generation, retrieval, ranking, auction pricing, and CTR modeling will land in later phases.

### Phase 2: Data Generation (Completed)

Synthetic data generation is now implemented:

- deterministic ad inventory generation
- deterministic interaction/click history generation
- CSV dataset outputs at `data/ads.csv` and `data/interactions.csv`

Generate datasets with:

```bash
PYTHONPATH=src python -m ad_auction_engine.data.generate --ads-count 3000 --interactions-count 50000 --seed 42
```

### Phase 3: Candidate Retrieval (Completed)

Keyword retrieval is now implemented with an in-memory inverted index:

- query token normalization and keyword matching
- top-K candidate retrieval with deterministic scoring
- candidate metadata including `keyword_match_type`, `matched_keywords`, and `retrieval_score`

Business logic for CTR modeling, ad ranking, auction pricing, and full API/UI integration will land in later phases.

### Phase 4: CTR Baseline ML (Completed)

Baseline CTR prediction is now implemented with Logistic Regression:

- training utilities for interaction features (`quality_score`, `historical_ctr`, `query_length`, `keyword_match_type`)
- model persistence and loading (`models/ctr_model.pkl`)
- inference utilities that return `predict_proba` CTR values in `[0, 1]`

Train the baseline model with:

```bash
PYTHONPATH=src python -m ad_auction_engine.ml.train --interactions-path data/interactions.csv --model-output models/ctr_model.pkl
```

### Phase 5: Ranking + GSP Auction (Completed)

Auction engine logic is now implemented:

- `AdRank = bid_price * predicted_ctr * quality_score`
- deterministic descending ranking with tie-breakers
- generalized second-price (GSP) clearing calculation
- reserve pricing behavior for the final shown ad

### Phase 6: End-to-End Integration (Completed)

The full local flow is now wired in both API and UI:

- `POST /search` executes retrieval -> CTR prediction -> ranking -> GSP pricing
- Streamlit UI executes the same search flow and renders winners as a table
- integration tests verify happy-path and missing-artifact behavior

Run the apps locally:

```bash
uvicorn api.main:app --reload
streamlit run ui/app.py
```

│       ├── engine/
│       ├── ml/
1. Upgrade baseline CTR model to XGBoost and keep interface compatibility.
└── pyproject.toml
```

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest
uvicorn api.main:app --reload
streamlit run ui/app.py
```

## Architecture Direction

- `src/ad_auction_engine/data`: synthetic ad inventory and interaction generation
- `src/ad_auction_engine/retrieval`: keyword retrieval and inverted index
- `src/ad_auction_engine/ml`: training and inference utilities for CTR prediction
- `src/ad_auction_engine/engine`: ranking and generalized second-price auction logic
- `api/main.py`: API surface for search requests
- `ui/app.py`: demo UI for local exploration

## Near-Term Milestones

1. Generate reproducible synthetic inventory and click-history datasets.
2. Implement candidate retrieval and match metadata.
3. Train an initial CTR model and persist inference artifacts.
4. Add ranking and GSP pricing.
5. Wire the full search flow through the API and UI.
