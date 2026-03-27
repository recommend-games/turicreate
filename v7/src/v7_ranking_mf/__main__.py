"""CLI smoke runner for NDJSON/JSONL training data."""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from v7_ranking_mf import create


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train/test v7_ranking_mf from NDJSON/JSONL data."
    )
    p.add_argument("--data", required=True, help="Path to .jsonl/.ndjson file.")
    p.add_argument("--user-col", default="user_id")
    p.add_argument("--item-col", default="item_id")
    p.add_argument("--target-col", default="rating")
    p.add_argument(
        "--invalid-target-action",
        default="drop",
        choices=["drop", "error"],
        help="How to handle null/non-finite/non-castable target values.",
    )
    p.add_argument("--num-factors", type=int, default=32)
    p.add_argument("--regularization", type=float, default=1e-9)
    p.add_argument("--linear-regularization", type=float, default=1e-9)
    p.add_argument("--ranking-regularization", type=float, default=0.25)
    p.add_argument("--unobserved-rating-value", type=float, default=None)
    p.add_argument("--num-sampled-negative-examples", type=int, default=4)
    p.add_argument("--max-iterations", type=int, default=25)
    p.add_argument("--sgd-step-size", type=float, default=0.0)
    p.add_argument("--random-seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers. Try 4-12 for large datasets.",
    )
    p.add_argument(
        "--accelerator",
        default="auto",
        choices=["auto", "cpu", "mps", "gpu"],
        help="Lightning accelerator backend.",
    )
    p.add_argument("--devices", default=1, help="Lightning devices argument.")
    p.add_argument("--k", type=int, default=10, help="Top-k recommendations per user.")
    p.add_argument(
        "--no-exclude-known",
        action="store_true",
        help="Do not filter training interactions from recommendations.",
    )
    p.add_argument(
        "--predictions-out",
        default=None,
        help="Optional output path for predictions (.csv or .ndjson).",
    )
    p.add_argument(
        "--recommendations-out",
        default=None,
        help="Optional output path for recommendations (.csv or .ndjson).",
    )
    p.add_argument("--quiet", action="store_true", help="Disable progress output.")
    return p.parse_args()


def _write_frame(df: pl.DataFrame, path: str) -> None:
    out = Path(path)
    if out.suffix.lower() == ".csv":
        df.write_csv(out)
    else:
        # Default to NDJSON for jsonl/ndjson/unknown.
        df.write_ndjson(out)


def main() -> None:
    args = _parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Input file not found: {data_path}")

    df = pl.read_ndjson(data_path)
    required = {args.user_col, args.item_col, args.target_col}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    before_rows = df.height
    cast_expr = pl.col(args.target_col).cast(pl.Float64, strict=False).alias(args.target_col)
    df = df.with_columns(cast_expr)
    invalid_mask = ~pl.col(args.target_col).is_finite() | pl.col(args.target_col).is_null()
    invalid_rows = df.filter(invalid_mask).height
    if args.invalid_target_action == "drop" and invalid_rows > 0:
        df = df.filter(~invalid_mask)
        print(f"Dropped {invalid_rows} rows with invalid target values.")
    elif args.invalid_target_action == "error" and invalid_rows > 0:
        raise ValueError(
            f"Target column {args.target_col!r} contains {invalid_rows} null/non-finite values."
        )

    model = create(
        df,
        user_id=args.user_col,
        item_id=args.item_col,
        target=args.target_col,
        num_factors=args.num_factors,
        regularization=args.regularization,
        linear_regularization=args.linear_regularization,
        ranking_regularization=args.ranking_regularization,
        unobserved_rating_value=args.unobserved_rating_value,
        num_sampled_negative_examples=args.num_sampled_negative_examples,
        max_iterations=args.max_iterations,
        sgd_step_size=args.sgd_step_size,
        random_seed=args.random_seed,
        verbose=not args.quiet,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        invalid_target_action=args.invalid_target_action,
        accelerator=args.accelerator,
        devices=args.devices,
    )

    preds = model.predict(df)
    pred_df = df.with_columns(preds.alias("prediction"))
    rec_df = model.recommend(
        k=args.k,
        exclude_known=not args.no_exclude_known,
    )

    print("Training complete.")
    print(
        f"rows={df.height} (input_rows={before_rows}), "
        f"users={df[args.user_col].n_unique()}, items={df[args.item_col].n_unique()}"
    )
    print("hyperparams:", model.hyperparams)
    print("\nPredictions sample:")
    print(pred_df.head(10))
    print("\nRecommendations sample:")
    print(rec_df.head(10))

    if args.predictions_out:
        _write_frame(pred_df, args.predictions_out)
        print(f"\nWrote predictions to {args.predictions_out}")
    if args.recommendations_out:
        _write_frame(rec_df, args.recommendations_out)
        print(f"Wrote recommendations to {args.recommendations_out}")


if __name__ == "__main__":
    main()
