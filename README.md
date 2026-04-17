# Sponsored Search Ad Auction Engine

A modular, production-style simulation of a sponsored search advertising system demonstrating end-to-end ML pipeline design, retrieval, ranking, and auction-based pricing.

The project focuses on system architecture, feature engineering, and ML integration in a ranking/auction context, rather than optimizing for a single model metric.

## Problem Statement

Sponsored search platforms (Google Ads, Bing Ads) must rank and price ads in milliseconds using:
1. **Relevance signals** — does this ad match the query?
2. **User engagement** — how likely will this ad be clicked (CTR)?
3. **Business signals** — what did the advertiser bid?

This project implements a complete end-to-end auction system that:
- Retrieves relevant ads from a keyword-based index
- Predicts click-through rate (CTR) using machine learning
- Ranks ads by `AdRank = bid × CTR × quality`
- Prices ads using generalized second-price (GSP) auction rules


## Architecture: Modular & Testable

```
Query → Retrieval → CTR Prediction → Ranking → GSP Pricing → Results
         (in-mem      (LogisticReg      (AdRank     (clearing
          index)       or XGBoost)       formula)    prices)
```

### Components

| Module | Responsibility | Key Files |
|--------|-----------------|-----------|
| **Retrieval** | Keyword-based candidate search with inverted index | `retrieval/` |
| **CTR Prediction** | Train both Logistic Regression and XGBoost models | `ml/trainer.py`, `ml/predictor.py` |
| **Ranking + Pricing** | AdRank calculation and GSP auction logic | `engine/` |
| **Data** | Synthetic inventory + click-through history generation | `data/` |
| **API** | FastAPI REST endpoint for search requests | `api/main.py` |
| **UI** | Streamlit dashboard for interactive exploration | `ui/app.py` |

## Feature Engineering:

### Features

**Continuous Features**:
- `quality_score` — ad quality (0-10)
- `historical_ctr` — past click rate
- `query_length` — number of query tokens
- `bid_price` — advertiser bid amount
- `keyword_count` — ad's keyword count
- `matched_keyword_count` — keywords matching query
- `matched_keyword_ratio` — coverage [0, 1]
- `retrieval_score` — match quality from retrieval
- `bid_quality` — bid × quality interaction term

**Derived Features (Expriemental)**
- match strength indicators (exact / partial match signals)
- bid-quality interactions
- coverage ratios (keyword overlap)
- threshold-based engagement signals

**These features are shared consistently across:**
- data generation
- training pipeline
- inference pipeline

This ensures no train/serving skew.


## Quick Start

### 1. Setup Environment

```bash
# Clone and navigate
cd ~/AI\ Ad\ Auction\ Engine

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install package with dev dependencies
pip install -e .[dev]
```

### 2. Generate Synthetic Data

```bash
PYTHONPATH=src python -m ad_auction_engine.data.generate \
  --ads-count 3000 \
  --interactions-count 50000 \
  --seed 42
```

**Output**:
- `data/ads.csv` — 3000 ads with quality_score, historical_ctr, keywords
- `data/interactions.csv` — 50000 user interactions (ad shown, clicked or not)

### 3. Train CTR Models

```bash
# Train Logistic Regression
PYTHONPATH=src python -m ad_auction_engine.ml.train \
  --interactions-path data/interactions.csv \
  --model-output models/ctr_model.pkl \
  --model-type logistic

# OR train XGBoost
PYTHONPATH=src python -m ad_auction_engine.ml.train \
  --interactions-path data/interactions.csv \
  --model-output models/ctr_model.pkl \
  --model-type xgboost

# Benchmark both models on same data
PYTHONPATH=src python -m ad_auction_engine.ml.train \
  --interactions-path data/interactions.csv \
  --benchmark
```

### 4. Run Tests

```bash
# Test suite (28 tests)
pytest -q

# Code quality
ruff check .
```

### 5. Demo: API + UI

**Terminal 1 — API**:
```bash
source .venv/bin/activate
cd /Users/moiz/AI\ Ad\ Auction\ Engine
PYTHONPATH=src uvicorn api.main:app --reload
```

**Terminal 2 — UI**:
```bash
source .venv/bin/activate
cd /Users/moiz/AI\ Ad\ Auction\ Engine
PYTHONPATH=src streamlit run ui/app.py
```

Visit:
- **API**: http://localhost:8000 (OpenAPI docs at `/docs`)
- **UI**: http://localhost:8501

**Example API Call**:
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "machine learning tutorial"}'
```

## Testing & Quality Assurance

### Test Coverage

```bash
# Run all tests
pytest -q
# Output: 28 passed in 2.80s
```

**Test Categories**:
- **Unit Tests** — ML training, feature encoding, data loading
- **Integration Tests** — Retrieval → CTR → Ranking → Pricing pipeline
- **Artifact Tests** — Model persistence and cross-version compatibility
- **Smoke Tests** — API and UI bootability

### Code Quality

```bash
# Lint check (zero tolerance)
ruff check .
# Output: All checks passed!
```

## Project Structure

```
ad-auction-engine/
├── api/
│   └── main.py                    # FastAPI endpoint wrapper
├── ui/
│   └── app.py                     # Streamlit interface
├── src/
│   └── ad_auction_engine/
│       ├── __init__.py
│       ├── config.py              # Global constants & configuration
│       ├── schemas.py             # Pydantic domain models
│       ├── api_app.py             # FastAPI app factory
│       ├── data/
│       │   ├── __init__.py
│       │   ├── generate.py        # CLI: data generation orchestration
│       │   ├── inventory.py       # Ad inventory generation logic
│       │   └── interactions.py    # Click history generation logic
│       ├── retrieval/
│       │   ├── __init__.py
│       │   ├── index.py           # Inverted index data structure
│       │   └── retriever.py       # Keyword-based candidate search
│       ├── ml/
│       │   ├── __init__.py
│       │   ├── train.py           # CLI: model training orchestration
│       │   ├── trainer.py         # Model training utilities
│       │   └── predictor.py       # CTR inference utilities
│       └── engine/
│           ├── __init__.py
│           ├── auction.py         # GSP auction & preprocessing
│           └── ranker.py          # AdRank sorting & ranking
├── tests/
│   ├── test_smoke.py              # API/UI bootability checks
│   ├── test_data.py               # Data generation tests
│   ├── test_retrieval.py          # Keyword matching tests
│   ├── test_ml.py                 # Model training/inference tests
│   └── test_engine.py             # Ranking/pricing tests
├── data/                          # Synthetic datasets (generated at runtime)
│   ├── ads.csv
│   └── interactions.csv
├── models/                        # Model artifacts
│   └── ctr_model.pkl              # Trained CTR model (Logistic or XGBoost)
├── pyproject.toml                 # Package config, dependencies, build info
├── requirements.txt               # Pinned production dependencies
├── requirements-dev.txt           # Dev dependencies (pytest, ruff)
└── README.md                      # This file
```


## Dependencies

**Core Libraries**:
- `fastapi` — REST API framework
- `streamlit` — Interactive UI
- `scikit-learn` — Logistic Regression baseline
- `xgboost` — Gradient boosting alternative
- `pydantic` — Type-safe schemas
- `numpy`, `pandas` — Data handling

**Dev Dependencies**:
- `pytest` — Testing framework
- `ruff` — Code linting

See `pyproject.toml` for version pinning.

## Key Design Decisions

### 1. Modular System Design

Each component (retrieval, ML, ranking, auction) is independently testable and replaceable.

### 2. Synthetic Data Generation

The system uses synthetic ad and interaction data to:

ensure reproducibility
allow controlled experimentation
avoid dependency on private datasets

### 3. Model Abstraction

CTR models share a unified interface, allowing seamless swapping between:

linear models
tree-based models

### 4. Lightweight Retrieval Layer

An in-memory inverted index is used for simplicity and clarity. This can be replaced with scalable systems (e.g., Elasticsearch or vector search) for production use.


### Key Metrics to Highlight

- **Accuracy**: 96.03% on balanced train/test split
- **Test Coverage**: 28 tests, all passing, zero lint violations
- **Latency**: Sub-millisecond ranking for 50+ candidates (in-memory)


### Extensions

Potential production upgrades:

- Distributed retrieval layer (e.g., Elasticsearch / FAISS)
- Online learning for CTR models
- A/B testing framework for ranking models
- Calibration layer for probability adjustment
- Real-time feature store integration


## Conclusion

This project demonstrates a full sponsored search ranking system architecture, combining retrieval, machine learning, and auction theory into a unified pipeline.

The emphasis is on system design, modular ML engineering, and ranking logic integration, rather than maximizing a single offline metric.
