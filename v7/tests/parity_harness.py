"""Side-by-side parity harness for Turi C++ and v7."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import numpy as np
import polars as pl

from v7_ranking_mf import create as create_v7

from parity_metrics import prediction_parity_report, topk_quality_and_overlap_report


def _to_sframe(df: pl.DataFrame):
    import turicreate as tc

    data = {c: df[c].to_list() for c in df.columns}
    return tc.SFrame(data)


@dataclass
class SplitData:
    train: pl.DataFrame
    test: pl.DataFrame


def split_rows(df: pl.DataFrame, test_frac: float, seed: int) -> SplitData:
    if not (0 < test_frac < 1):
        raise ValueError("test_frac must be in (0,1).")
    n = df.height
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test = int(round(n * test_frac))
    test_idx = set(perm[:n_test].tolist())
    row_idx = pl.Series("_row_idx", np.arange(n, dtype=np.int64))
    with_idx = df.with_columns(row_idx)
    test = with_idx.filter(pl.col("_row_idx").is_in(list(test_idx))).drop("_row_idx")
    train = with_idx.filter(~pl.col("_row_idx").is_in(list(test_idx))).drop("_row_idx")
    return SplitData(train=train, test=test)


def _clean_target(df: pl.DataFrame, target_col: str, drop_invalid: bool) -> pl.DataFrame:
    df = df.with_columns(pl.col(target_col).cast(pl.Float64, strict=False).alias(target_col))
    bad = ~pl.col(target_col).is_finite() | pl.col(target_col).is_null()
    if drop_invalid:
        return df.filter(~bad)
    bad_rows = df.filter(bad).height
    if bad_rows:
        raise ValueError(f"Found {bad_rows} invalid target rows.")
    return df


def train_turi_model(
    train_df: pl.DataFrame,
    user_col: str,
    item_col: str,
    target_col: str,
    params: Dict[str, Any],
):
    import turicreate as tc

    sf = _to_sframe(train_df.select([user_col, item_col, target_col]))
    return tc.ranking_factorization_recommender.create(
        sf,
        user_id=user_col,
        item_id=item_col,
        target=target_col,
        num_factors=params["num_factors"],
        regularization=params["regularization"],
        linear_regularization=params["linear_regularization"],
        ranking_regularization=params["ranking_regularization"],
        num_sampled_negative_examples=params["num_sampled_negative_examples"],
        max_iterations=params["max_iterations"],
        random_seed=params["random_seed"],
        binary_target=False,
        verbose=params["verbose"],
    )


def train_v7_model(
    train_df: pl.DataFrame,
    user_col: str,
    item_col: str,
    target_col: str,
    params: Dict[str, Any],
):
    return create_v7(
        train_df,
        user_id=user_col,
        item_id=item_col,
        target=target_col,
        num_factors=params["num_factors"],
        regularization=params["regularization"],
        linear_regularization=params["linear_regularization"],
        ranking_regularization=params["ranking_regularization"],
        num_sampled_negative_examples=params["num_sampled_negative_examples"],
        max_iterations=params["max_iterations"],
        random_seed=params["random_seed"],
        verbose=params["verbose"],
        accelerator=params["accelerator"],
        devices=params["devices"],
        batch_size=params["batch_size"],
        num_workers=params["num_workers"],
        invalid_target_action="drop",
    )


def _predict_turi(model, test_df: pl.DataFrame, user_col: str, item_col: str):
    sf = _to_sframe(test_df.select([user_col, item_col]))
    pred = model.predict(sf)
    return np.asarray(list(pred), dtype=np.float64)


def _predict_v7(model, test_df: pl.DataFrame):
    pred = model.predict(test_df)
    return np.asarray(pred.to_list(), dtype=np.float64)


def _recommend_turi(model, users: Sequence, user_col: str, k: int):
    import turicreate as tc

    user_sa = tc.SArray(list(users))
    rec = model.recommend(users=user_sa, k=k, exclude_known=True)
    return {
        "user": list(rec[user_col]),
        "item": list(rec["item_id" if "item_id" in rec.column_names() else rec.column_names()[1]]),
        "rank": list(rec["rank"]),
    }


def _recommend_v7(model, users: Sequence, k: int, user_col: str, item_col: str):
    rec = model.recommend(users=list(users), k=k, exclude_known=True)
    return {
        "user": rec[user_col].to_list(),
        "item": rec[item_col].to_list(),
        "rank": rec["rank"].to_list(),
    }


def run_parity(
    data: pl.DataFrame,
    user_col: str,
    item_col: str,
    target_col: str,
    params: Dict[str, Any],
    test_frac: float = 0.2,
    k_values: Iterable[int] = (5, 10, 20),
    user_eval_cap: int = 2000,
) -> Dict[str, Any]:
    data = _clean_target(data, target_col=target_col, drop_invalid=True)
    split = split_rows(data, test_frac=test_frac, seed=params["random_seed"])

    turi = train_turi_model(split.train, user_col, item_col, target_col, params)
    v7 = train_v7_model(split.train, user_col, item_col, target_col, params)

    y_true = np.asarray(split.test[target_col].to_list(), dtype=np.float64)
    turi_pred = _predict_turi(turi, split.test, user_col, item_col)
    v7_pred = _predict_v7(v7, split.test.select([user_col, item_col]))
    pred_report = prediction_parity_report(turi_pred=turi_pred, v7_pred=v7_pred, target=y_true)

    users = split.test[user_col].unique(maintain_order=True)
    if user_eval_cap > 0:
        users = users.head(user_eval_cap)
    users_list = users.to_list()

    topk_reports: Dict[str, float] = {}
    for k in k_values:
        tr = _recommend_turi(turi, users_list, user_col=user_col, k=int(k))
        vr = _recommend_v7(v7, users_list, k=int(k), user_col=user_col, item_col=item_col)
        rep = topk_quality_and_overlap_report(
            turi_recs=tr,
            v7_recs=vr,
            truth_user=split.test[user_col].to_list(),
            truth_item=split.test[item_col].to_list(),
            k_values=(int(k),),
        )
        topk_reports.update(rep)

    return {
        "dataset": {
            "n_rows": data.height,
            "n_train": split.train.height,
            "n_test": split.test.height,
            "n_users_test_eval": len(users_list),
        },
        "params": params,
        "prediction_report": pred_report,
        "topk_report": topk_reports,
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Turi-v7 parity benchmark.")
    p.add_argument("--data", required=True, help="CSV or NDJSON input path.")
    p.add_argument("--user-col", default="user_id")
    p.add_argument("--item-col", default="item_id")
    p.add_argument("--target-col", default="rating")
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--out", default="v7/tests/baselines/latest_parity_metrics.json")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--user-eval-cap", type=int, default=2000)
    p.add_argument("--k-values", default="5,10,20")
    p.add_argument("--num-factors", type=int, default=32)
    p.add_argument("--regularization", type=float, default=1e-9)
    p.add_argument("--linear-regularization", type=float, default=1e-9)
    p.add_argument("--ranking-regularization", type=float, default=0.25)
    p.add_argument("--num-sampled-negative-examples", type=int, default=4)
    p.add_argument("--max-iterations", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--accelerator", default="cpu")
    p.add_argument("--devices", default=1)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def _read_table(path: str) -> pl.DataFrame:
    p = Path(path)
    if p.suffix.lower() in {".ndjson", ".jsonl", ".jl"}:
        return pl.read_ndjson(p)
    return pl.read_csv(p)


def main() -> None:
    args = _parse_args()
    try:
        import turicreate  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "turicreate import failed. Parity harness requires Turi Create runtime."
        ) from e

    df = _read_table(args.data)
    params = {
        "num_factors": args.num_factors,
        "regularization": args.regularization,
        "linear_regularization": args.linear_regularization,
        "ranking_regularization": args.ranking_regularization,
        "num_sampled_negative_examples": args.num_sampled_negative_examples,
        "max_iterations": args.max_iterations,
        "random_seed": args.seed,
        "verbose": args.verbose,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "accelerator": args.accelerator,
        "devices": args.devices,
    }
    k_values = [int(x) for x in args.k_values.split(",") if x.strip()]
    result = run_parity(
        df,
        user_col=args.user_col,
        item_col=args.item_col,
        target_col=args.target_col,
        params=params,
        test_frac=args.test_frac,
        k_values=k_values,
        user_eval_cap=args.user_eval_cap,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"Wrote parity report to {out_path}")


if __name__ == "__main__":
    main()
