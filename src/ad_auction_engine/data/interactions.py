"""Synthetic interaction generation for CTR training bootstrap."""

from __future__ import annotations

import csv
from pathlib import Path
from uuid import NAMESPACE_DNS, uuid5

import numpy as np

from ad_auction_engine.schemas import AdRecord, InteractionRecord


def _interaction_query(
    ad_keywords: list[str],
    global_keywords: list[str],
    match_type: str,
    rng: np.random.Generator,
) -> str:
    """Create exact or broad query strings for an ad impression."""
    if match_type == "exact":
        words = list(rng.choice(ad_keywords, size=min(len(ad_keywords), 2), replace=False))
        return " ".join(words)

    match_count = 1 if len(ad_keywords) == 1 else 2
    ad_words = list(rng.choice(ad_keywords, size=match_count, replace=False))
    noise_word = str(rng.choice(global_keywords))
    words = sorted(set([*ad_words, noise_word]))
    return " ".join(words)


def generate_interactions(
    ads: list[AdRecord],
    count: int,
    seed: int,
    user_pool_size: int = 5000,
) -> list[InteractionRecord]:
    """Generate deterministic synthetic interactions with click labels."""
    if count <= 0 or not ads:
        return []

    rng = np.random.default_rng(seed)
    global_keywords = sorted({keyword for ad in ads for keyword in ad.keywords})

    weights = np.array(
        [max(ad.bid_price * ad.quality_score, 0.01) for ad in ads],
        dtype=np.float64,
    )
    weights = weights / np.sum(weights)

    records: list[InteractionRecord] = []
    for index in range(count):
        ad = ads[int(rng.choice(len(ads), p=weights))]

        match_type = "exact" if rng.random() < 0.4 else "broad"
        query = _interaction_query(ad.keywords, global_keywords, match_type, rng)
        query_length = len(query.split())

        historical_ctr = ad.historical_clicks / ad.historical_impressions
        quality_factor = ad.quality_score / 10.0
        match_bonus = 0.05 if match_type == "exact" else -0.01

        click_probability = 0.02 + (0.45 * historical_ctr) + (0.18 * quality_factor) + match_bonus
        click_probability += float(rng.normal(0.0, 0.02))
        click_probability = float(np.clip(click_probability, 0.005, 0.95))
        clicked = int(rng.random() < click_probability)

        records.append(
            InteractionRecord(
                interaction_id=str(uuid5(NAMESPACE_DNS, f"interaction-{seed}-{index}")),
                user_id=f"user_{int(rng.integers(1, user_pool_size + 1))}",
                query=query,
                query_length=query_length,
                ad_id=ad.ad_id,
                keyword_match_type=match_type,
                quality_score=ad.quality_score,
                historical_ctr=round(historical_ctr, 6),
                clicked=clicked,
            )
        )

    return records


def save_interactions_csv(records: list[InteractionRecord], output_path: str | Path) -> Path:
    """Persist interaction records to CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "interaction_id",
                "user_id",
                "query",
                "query_length",
                "ad_id",
                "keyword_match_type",
                "quality_score",
                "historical_ctr",
                "clicked",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(record.model_dump())

    return path


def load_interactions_csv(input_path: str | Path) -> list[InteractionRecord]:
    """Load interaction records from CSV."""
    path = Path(input_path)
    records: list[InteractionRecord] = []

    with path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            records.append(
                InteractionRecord(
                    interaction_id=row["interaction_id"],
                    user_id=row["user_id"],
                    query=row["query"],
                    query_length=int(row["query_length"]),
                    ad_id=row["ad_id"],
                    keyword_match_type=row["keyword_match_type"],
                    quality_score=float(row["quality_score"]),
                    historical_ctr=float(row["historical_ctr"]),
                    clicked=int(row["clicked"]),
                )
            )

    return records
