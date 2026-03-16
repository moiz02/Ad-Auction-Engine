# Ad Auction Engine

This repository is a recruiter-facing sponsored-search auction simulator. The goal is to show clean system structure, ML workflow discipline, and a small interactive product surface without over-claiming production realism.

## Phase 1 Status

Phase 1 establishes the project foundation:

- package-style Python layout under `src/`
- bootable FastAPI and Streamlit entrypoints
- shared schemas and configuration
- smoke tests and CI

Business logic for data generation, retrieval, ranking, auction pricing, and CTR modeling will land in later phases.

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
│       ├── data/
│       ├── engine/
│       ├── ml/
│       └── retrieval/
├── tests/
│   └── test_smoke.py
├── ui/
│   └── app.py
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
