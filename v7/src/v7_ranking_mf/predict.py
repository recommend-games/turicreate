"""Inference: predict rows and top-k recommend."""

from __future__ import annotations

from typing import Dict, List, Set

import polars as pl
import torch

from v7_ranking_mf.model import BiasedMF


def _map_ids(
    raw_ids: List,
    id_map: Dict,
    name: str,
) -> List[int]:
    out = []
    for x in raw_ids:
        if x not in id_map:
            raise ValueError(f"Unknown {name} id: {x!r}")
        out.append(id_map[x])
    return out


def predict_scores(
    model: BiasedMF,
    user_idx: torch.Tensor,
    item_idx: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return model(user_idx.to(device), item_idx.to(device)).cpu()


def predict_dataframe(
    model: BiasedMF,
    df: pl.DataFrame,
    user_col: str,
    item_col: str,
    user_map: Dict,
    item_map: Dict,
    device: torch.device,
    batch_size: int = 4096,
) -> pl.Series:
    u_raw = df[user_col].to_list()
    i_raw = df[item_col].to_list()
    u_idx = _map_ids(u_raw, user_map, "user")
    i_idx = _map_ids(i_raw, item_map, "item")
    n = len(u_idx)
    scores: List[float] = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        tu = torch.tensor(u_idx[start:end], dtype=torch.long)
        ti = torch.tensor(i_idx[start:end], dtype=torch.long)
        s = predict_scores(model, tu, ti, device)
        scores.extend(s.tolist())
    return pl.Series("prediction", scores)


def recommend_top_k(
    model: BiasedMF,
    user_indices: List[int],
    user_seen: List[Set[int]],
    idx_to_item: List,
    item_col_name: str,
    k: int,
    exclude_known: bool,
    device: torch.device,
    batch_items: int = 2048,
) -> pl.DataFrame:
    """Returns frame with user index column 'user_idx_internal', item id, score, rank."""
    n_items = model.n_items
    rows_user: List = []
    rows_item: List = []
    rows_score: List[float] = []
    rows_rank: List[int] = []

    model.eval()
    with torch.no_grad():
        for u in user_indices:
            seen = user_seen[u] if exclude_known else set()
            # scores for all items
            all_scores = torch.empty(n_items, dtype=torch.float32)
            u_tensor = torch.full((batch_items,), u, dtype=torch.long, device=device)
            for start in range(0, n_items, batch_items):
                end = min(start + batch_items, n_items)
                bs = end - start
                u_batch = u_tensor[:bs]
                items = torch.arange(start, end, dtype=torch.long, device=device)
                all_scores[start:end] = model(u_batch, items).cpu()

            if exclude_known:
                for j in seen:
                    if j < n_items:
                        all_scores[j] = float("-inf")

            topk = torch.topk(all_scores, k=min(k, n_items)).indices.tolist()
            topk_scores = all_scores[topk].tolist()
            for rank, (it_idx, sc) in enumerate(zip(topk, topk_scores), start=1):
                rows_user.append(u)
                rows_item.append(idx_to_item[it_idx])
                rows_score.append(float(sc))
                rows_rank.append(rank)

    return pl.DataFrame(
        {
            "_user_idx": rows_user,
            item_col_name: rows_item,
            "score": rows_score,
            "rank": rows_rank,
        }
    )
