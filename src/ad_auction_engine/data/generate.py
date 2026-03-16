"""CLI helper to generate Phase 2 synthetic datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

from ad_auction_engine.config import get_settings
from ad_auction_engine.data.interactions import generate_interactions, save_interactions_csv
from ad_auction_engine.data.inventory import generate_ad_inventory, save_ad_inventory_csv


def generate_datasets(
    ads_count: int,
    interactions_count: int,
    seed: int,
    ads_output_path: str,
    interactions_output_path: str,
) -> tuple[Path, Path]:
    """Generate and persist ad inventory plus interactions datasets."""
    ads = generate_ad_inventory(count=ads_count, seed=seed)
    interactions = generate_interactions(ads=ads, count=interactions_count, seed=seed + 1)

    ads_path = save_ad_inventory_csv(ads, ads_output_path)
    interactions_path = save_interactions_csv(interactions, interactions_output_path)

    return ads_path, interactions_path


def _build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Generate synthetic ad auction datasets")
    parser.add_argument("--ads-count", type=int, default=settings.ads_count)
    parser.add_argument("--interactions-count", type=int, default=settings.interactions_count)
    parser.add_argument("--seed", type=int, default=settings.random_seed)
    parser.add_argument("--ads-output", type=str, default=settings.ads_output_path)
    parser.add_argument("--interactions-output", type=str, default=settings.interactions_output_path)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    ads_path, interactions_path = generate_datasets(
        ads_count=args.ads_count,
        interactions_count=args.interactions_count,
        seed=args.seed,
        ads_output_path=args.ads_output,
        interactions_output_path=args.interactions_output,
    )

    print(f"Saved ads dataset to: {ads_path}")
    print(f"Saved interactions dataset to: {interactions_path}")


if __name__ == "__main__":
    main()
