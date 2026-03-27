"""Smoke tests and determinism for v7_ranking_mf."""

from __future__ import annotations

import polars as pl
import pytest

from v7_ranking_mf import create


@pytest.fixture
def tiny_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "user_id": ["a", "a", "b", "b", "c", "c"],
            "item_id": ["x", "y", "x", "z", "y", "z"],
            "rating": [1.0, 5.0, 2.0, 4.0, 3.0, 2.5],
        }
    )


def test_create_predict_shape(tiny_df: pl.DataFrame) -> None:
    m = create(
        tiny_df,
        target="rating",
        max_iterations=2,
        verbose=False,
        random_seed=42,
    )
    pred = m.predict(tiny_df)
    assert len(pred) == tiny_df.height
    assert pred.dtype == pl.Float32 or pred.dtype == pl.Float64


def test_determinism(tiny_df: pl.DataFrame) -> None:
    m1 = create(
        tiny_df,
        target="rating",
        max_iterations=2,
        verbose=False,
        random_seed=123,
    )
    m2 = create(
        tiny_df,
        target="rating",
        max_iterations=2,
        verbose=False,
        random_seed=123,
    )
    p1 = m1.predict(tiny_df).to_numpy()
    p2 = m2.predict(tiny_df).to_numpy()
    assert (p1 == p2).all()


def test_unknown_user_predict_raises(tiny_df: pl.DataFrame) -> None:
    m = create(tiny_df, target="rating", max_iterations=1, verbose=False, random_seed=0)
    bad = pl.DataFrame({"user_id": ["nope"], "item_id": ["x"], "rating": [1.0]})
    with pytest.raises(ValueError, match="Unknown user"):
        m.predict(bad)


def test_recommend_columns(tiny_df: pl.DataFrame) -> None:
    m = create(tiny_df, target="rating", max_iterations=2, verbose=False, random_seed=7)
    rec = m.recommend(k=2, exclude_known=True)
    assert set(rec.columns) == {"user_id", "item_id", "score", "rank"}
    assert rec.height == 3 * 2  # 3 users, k=2


def test_default_unobserved_value_documented(tiny_df: pl.DataFrame) -> None:
    m = create(tiny_df, target="rating", max_iterations=1, verbose=False, random_seed=0)
    hp = m.hyperparams
    assert "unobserved_rating_value" in hp
    assert isinstance(hp["unobserved_rating_value"], float)


def test_target_required() -> None:
    df = pl.DataFrame({"user_id": [1], "item_id": [2], "rating": [3.0]})
    with pytest.raises(ValueError, match="target"):
        create(df, target=None)


def test_binary_target_rejected() -> None:
    df = pl.DataFrame({"user_id": [1], "item_id": [2], "rating": [3.0]})
    with pytest.raises(NotImplementedError):
        create(df, target="rating", binary_target=True, max_iterations=1, verbose=False)
