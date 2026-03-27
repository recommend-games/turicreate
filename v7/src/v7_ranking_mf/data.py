"""Polars → index maps, user→seen items, PyTorch DataLoader."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Set, Tuple, Union

import polars as pl
import torch
from torch.utils.data import DataLoader, TensorDataset


def load_observation_frame(
    observation_data: Union[pl.DataFrame, str, Path],
) -> pl.DataFrame:
    if isinstance(observation_data, (str, Path)):
        return pl.read_csv(observation_data)
    return observation_data


def default_unobserved_rating_value(df: pl.DataFrame, target: str) -> float:
    """Turi docstring: ~5% quantile via mean - 1.96 * std on observed targets."""
    s = df.select(
        pl.col(target).mean().alias("m"),
        pl.col(target).std(ddof=0).alias("s"),
    )
    m = float(s["m"][0])
    sd = float(s["s"][0])
    if sd != sd:  # NaN
        sd = 0.0
    return m - 1.96 * sd


@dataclass
class TrainingMatrices:
    """Integer indices and ratings for training."""

    user_idx: torch.Tensor  # (N,) int64
    item_idx: torch.Tensor
    rating: torch.Tensor  # (N,) float32
    n_users: int
    n_items: int
    user_seen: List[Set[int]]  # per user index, set of item indices


def build_training_matrices(
    df: pl.DataFrame,
    user_col: str,
    item_col: str,
    target: str,
) -> Tuple[TrainingMatrices, dict, dict]:
    """Returns matrices, user_original_to_idx, item_original_to_idx (raw -> int)."""
    u_ids = df[user_col].to_list()
    i_ids = df[item_col].to_list()

    u_unique = df[user_col].unique(maintain_order=True).to_list()
    i_unique = df[item_col].unique(maintain_order=True).to_list()

    u_map = {u: i for i, u in enumerate(u_unique)}
    i_map = {it: j for j, it in enumerate(i_unique)}

    n_users = len(u_map)
    n_items = len(i_map)

    user_idx = torch.tensor([u_map[u] for u in u_ids], dtype=torch.long)
    item_idx = torch.tensor([i_map[it] for it in i_ids], dtype=torch.long)
    ratings = df[target].cast(pl.Float64).to_numpy()
    rating = torch.tensor(ratings, dtype=torch.float32)

    user_seen: List[Set[int]] = [set() for _ in range(n_users)]
    for u, it in zip(user_idx.tolist(), item_idx.tolist()):
        user_seen[u].add(it)

    return (
        TrainingMatrices(
            user_idx=user_idx,
            item_idx=item_idx,
            rating=rating,
            n_users=n_users,
            n_items=n_items,
            user_seen=user_seen,
        ),
        u_map,
        i_map,
    )


def make_train_dataloader(
    matrices: TrainingMatrices,
    batch_size: int,
    random_seed: int,
    num_workers: int = 0,
) -> DataLoader:
    ds = TensorDataset(matrices.user_idx, matrices.item_idx, matrices.rating)
    g = torch.Generator()
    g.manual_seed(random_seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        generator=g,
        num_workers=num_workers,
        drop_last=False,
    )
