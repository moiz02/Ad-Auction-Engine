"""CLI entrypoint for training the baseline CTR model."""

from __future__ import annotations

import argparse

from ad_auction_engine.config import get_settings
from ad_auction_engine.ml.trainer import train_and_save_ctr_model


def _build_parser() -> argparse.ArgumentParser:
    settings = get_settings()

    parser = argparse.ArgumentParser(
        description="Train baseline CTR model from interactions dataset"
    )
    parser.add_argument("--interactions-path", type=str, default=settings.interactions_output_path)
    parser.add_argument("--model-output", type=str, default=settings.ctr_model_path)
    parser.add_argument("--test-size", type=float, default=settings.ctr_model_test_size)
    parser.add_argument("--seed", type=int, default=settings.random_seed)
    parser.add_argument("--max-iter", type=int, default=settings.ctr_model_max_iter)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    metrics = train_and_save_ctr_model(
        interactions_path=args.interactions_path,
        model_output_path=args.model_output,
        test_size=args.test_size,
        random_seed=args.seed,
        max_iter=args.max_iter,
    )

    print(f"Saved model to: {args.model_output}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    if "roc_auc" in metrics:
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"Positive rate: {metrics['positive_rate']:.4f}")
    print(f"Sample size: {int(metrics['sample_size'])}")


if __name__ == "__main__":
    main()
