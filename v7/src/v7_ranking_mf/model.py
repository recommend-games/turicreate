"""Biased matrix factorization: global + user/item biases + dot-product factors."""

from __future__ import annotations

import torch
from torch import nn


class BiasedMF(nn.Module):
    """Score(i,j) = μ + w_i + w_j + u_i · v_j."""

    def __init__(self, n_users: int, n_items: int, n_factors: int) -> None:
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.global_bias = nn.Parameter(torch.zeros(1))
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.user_factors = nn.Embedding(n_users, n_factors)
        self.item_factors = nn.Embedding(n_items, n_factors)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
        nn.init.normal_(self.user_factors.weight, std=0.05)
        nn.init.normal_(self.item_factors.weight, std=0.05)

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        """user_idx, item_idx: LongTensor of same shape (any rank); returns same shape."""
        ub = self.user_bias(user_idx).squeeze(-1)
        ib = self.item_bias(item_idx).squeeze(-1)
        dot = (self.user_factors(user_idx) * self.item_factors(item_idx)).sum(dim=-1)
        return self.global_bias.squeeze() + ub + ib + dot
