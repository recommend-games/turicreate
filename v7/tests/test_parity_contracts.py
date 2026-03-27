from __future__ import annotations

import polars as pl
import pytest

from v7_ranking_mf import create as create_v7


tc = pytest.importorskip("turicreate")


def _to_sframe(df: pl.DataFrame):
    return tc.SFrame({c: df[c].to_list() for c in df.columns})


@pytest.fixture
def tiny_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u2", "u3"],
            "item_id": ["i1", "i2", "i1", "i3", "i2"],
            "rating": [1.0, 4.0, 2.0, 5.0, 3.0],
        }
    )


def test_recommend_schema_and_count_parity(tiny_df: pl.DataFrame) -> None:
    params = dict(
        user_id="user_id",
        item_id="item_id",
        target="rating",
        num_factors=8,
        max_iterations=2,
        random_seed=0,
        ranking_regularization=0.25,
        num_sampled_negative_examples=2,
        verbose=False,
    )
    m_v7 = create_v7(tiny_df, **params)
    m_tc = tc.ranking_factorization_recommender.create(_to_sframe(tiny_df), **params)

    users = ["u1", "u2"]
    k = 2
    r_v7 = m_v7.recommend(users=users, k=k, exclude_known=True)
    r_tc = m_tc.recommend(users=users, k=k, exclude_known=True)

    assert set(r_v7.columns) == {"user_id", "item_id", "score", "rank"}
    assert set(r_tc.column_names()) >= {"user_id", "item_id", "score", "rank"}
    assert r_v7.height == len(users) * k
    assert r_tc.num_rows() == len(users) * k


def test_unknown_user_contract_difference_documented(tiny_df: pl.DataFrame) -> None:
    params = dict(
        user_id="user_id",
        item_id="item_id",
        target="rating",
        num_factors=8,
        max_iterations=1,
        random_seed=0,
        ranking_regularization=0.25,
        num_sampled_negative_examples=2,
        verbose=False,
    )
    m_v7 = create_v7(tiny_df, **params)
    m_tc = tc.ranking_factorization_recommender.create(_to_sframe(tiny_df), **params)

    with pytest.raises(ValueError):
        m_v7.recommend(users=["does-not-exist"], k=1)

    # Turi behavior may differ (often returns recommendations for known/new users differently).
    # Ensure at least that it does not crash unexpectedly for this query.
    _ = m_tc.recommend(users=["does-not-exist"], k=1)
