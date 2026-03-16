"""Synthetic ad inventory generation for Phase 2."""

from __future__ import annotations

import csv
from pathlib import Path
from uuid import NAMESPACE_DNS, uuid5

import numpy as np
from faker import Faker

from ad_auction_engine.schemas import AdRecord

KEYWORD_UNIVERSE: tuple[str, ...] = (
    "running",
    "shoes",
    "sneakers",
    "fitness",
    "workout",
    "yoga",
    "gym",
    "protein",
    "electronics",
    "laptop",
    "smartphone",
    "headphones",
    "gaming",
    "travel",
    "flights",
    "hotel",
    "vacation",
    "fashion",
    "jeans",
    "jackets",
    "beauty",
    "skincare",
    "books",
    "education",
    "finance",
    "insurance",
)


def _historical_clicks(impressions: int, quality_score: float, rng: np.random.Generator) -> int:
    """Build a plausible click count using quality-informed CTR priors."""
    quality_factor = quality_score / 10.0
    ctr = 0.01 + (0.14 * quality_factor) + rng.normal(0.0, 0.015)
    ctr = float(np.clip(ctr, 0.005, 0.35))
    return int(round(impressions * ctr))


def generate_ad_inventory(
    count: int,
    seed: int,
    min_keywords: int = 2,
    max_keywords: int = 5,
) -> list[AdRecord]:
    """Generate deterministic synthetic ad records."""
    if count <= 0:
        return []

    faker = Faker()
    Faker.seed(seed)
    rng = np.random.default_rng(seed)

    records: list[AdRecord] = []
    for index in range(count):
        keyword_count = int(rng.integers(min_keywords, max_keywords + 1))
        keywords = list(rng.choice(KEYWORD_UNIVERSE, size=keyword_count, replace=False))

        quality_score = round(float(rng.uniform(1.0, 10.0)), 2)
        bid_price = round(float(rng.uniform(0.5, 5.0)), 2)
        historical_impressions = int(rng.integers(300, 20000))
        historical_clicks = _historical_clicks(historical_impressions, quality_score, rng)

        records.append(
            AdRecord(
                ad_id=str(uuid5(NAMESPACE_DNS, f"ad-{seed}-{index}")),
                advertiser_name=faker.company(),
                keywords=sorted(keywords),
                bid_price=bid_price,
                quality_score=quality_score,
                historical_clicks=min(historical_clicks, historical_impressions),
                historical_impressions=historical_impressions,
            )
        )

    return records


def save_ad_inventory_csv(records: list[AdRecord], output_path: str | Path) -> Path:
    """Persist ad inventory to CSV using pipe-delimited keywords."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "ad_id",
                "advertiser_name",
                "keywords",
                "bid_price",
                "quality_score",
                "historical_clicks",
                "historical_impressions",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "ad_id": record.ad_id,
                    "advertiser_name": record.advertiser_name,
                    "keywords": "|".join(record.keywords),
                    "bid_price": f"{record.bid_price:.2f}",
                    "quality_score": f"{record.quality_score:.2f}",
                    "historical_clicks": record.historical_clicks,
                    "historical_impressions": record.historical_impressions,
                }
            )

    return path


def load_ad_inventory_csv(input_path: str | Path) -> list[AdRecord]:
    """Load ad inventory records from CSV."""
    path = Path(input_path)
    records: list[AdRecord] = []

    with path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            records.append(
                AdRecord(
                    ad_id=row["ad_id"],
                    advertiser_name=row["advertiser_name"],
                    keywords=[part for part in row["keywords"].split("|") if part],
                    bid_price=float(row["bid_price"]),
                    quality_score=float(row["quality_score"]),
                    historical_clicks=int(row["historical_clicks"]),
                    historical_impressions=int(row["historical_impressions"]),
                )
            )

    return records
