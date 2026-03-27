"""Lightning module: MSE + ranking regularization + L2."""

from __future__ import annotations

from typing import List, Literal, Set

import numpy as np
import torch
import torch.nn.functional as F
from lightning import LightningModule
from torch.optim import Adam

from v7_ranking_mf.model import BiasedMF


class BiasedMFLightningModule(LightningModule):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        num_factors: int,
        user_seen: List[Set[int]],
        unobserved_rating_value: float,
        ranking_regularization: float,
        num_sampled_negative_examples: int,
        regularization: float,
        linear_regularization: float,
        learning_rate: float,
        random_seed: int,
        observation_loss_weight: float = 1.0,
        ranking_loss_weight: float = 1.0,
        l2_loss_weight: float = 1.0,
        ranking_loss_reduce: Literal["valid_mean", "batch_mean"] = "valid_mean",
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["user_seen"])
        self.user_seen = user_seen
        self.model = BiasedMF(n_users, n_items, num_factors)
        self._sample_rng = np.random.default_rng(seed=random_seed + 17)

    def forward(self, u: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        return self.model(u, i)

    def _l2_penalty(self) -> torch.Tensor:
        lr_lin = self.hparams.linear_regularization
        lr_fac = self.hparams.regularization
        bias_sq = self.model.global_bias.pow(2).sum()
        bias_sq = bias_sq + self.model.user_bias.weight.pow(2).sum()
        bias_sq = bias_sq + self.model.item_bias.weight.pow(2).sum()
        fac_sq = self.model.user_factors.weight.pow(2).sum()
        fac_sq = fac_sq + self.model.item_factors.weight.pow(2).sum()
        return lr_lin * bias_sq + lr_fac * fac_sq

    def _sample_negative_candidates(
        self, user_idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns candidates (B, K) on same device as user_idx, and valid mask (B,)."""
        device = user_idx.device
        B = user_idx.size(0)
        K = self.hparams.num_sampled_negative_examples
        n_items = self.model.n_items
        out = torch.zeros(B, K, dtype=torch.long, device=device)
        valid = torch.zeros(B, dtype=torch.bool, device=device)

        u_list = user_idx.detach().cpu().numpy()
        for b, u in enumerate(u_list):
            seen = self.user_seen[int(u)]
            if len(seen) >= n_items:
                continue
            draws: list[int] = []
            draw_set: set[int] = set()
            max_tries = max(16 * K, 64)
            tries = 0
            while len(draws) < K and tries < max_tries:
                tries += 1
                cand = int(self._sample_rng.integers(0, n_items))
                if cand in seen or cand in draw_set:
                    continue
                draw_set.add(cand)
                draws.append(cand)

            # Rare fallback for dense users if rejection sampling did not fill K.
            if len(draws) < K:
                start = int(self._sample_rng.integers(0, n_items))
                cur = start
                for _ in range(n_items):
                    if cur not in seen and cur not in draw_set:
                        draw_set.add(cur)
                        draws.append(cur)
                        if len(draws) == K:
                            break
                    cur += 1
                    if cur == n_items:
                        cur = 0

            if not draws:
                continue
            valid[b] = True
            if len(draws) < K:
                # Pad by repeating a valid sampled item to keep shape (B, K).
                draws.extend([draws[-1]] * (K - len(draws)))
            out[b] = torch.tensor(draws[:K], dtype=torch.long, device=device)

        return out, valid

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        user_idx, item_idx, rating = batch
        device = user_idx.device

        pred = self(user_idx, item_idx)
        loss_obs = F.mse_loss(pred, rating)

        w_obs = self.hparams.observation_loss_weight
        w_rank = self.hparams.ranking_loss_weight
        w_l2 = self.hparams.l2_loss_weight
        reduce_mode = self.hparams.ranking_loss_reduce

        rr = self.hparams.ranking_regularization
        if rr > 0:
            cand, valid = self._sample_negative_candidates(user_idx)
            if valid.any():
                B, K = cand.shape
                u_exp = user_idx.unsqueeze(1).expand(B, K).reshape(-1)
                c_flat = cand.reshape(-1)
                scores = self(u_exp, c_flat).view(B, K)
                max_scores, _ = scores.max(dim=1)
                target = torch.full_like(max_scores, self.hparams.unobserved_rating_value)
                per_row = F.mse_loss(max_scores, target, reduction="none")
                if reduce_mode == "valid_mean":
                    loss_neg = per_row[valid].mean()
                else:
                    # Mean over full batch; invalid rows contribute 0 (no gradient).
                    masked = per_row * valid.to(dtype=per_row.dtype)
                    loss_neg = masked.sum() / float(B)
                loss_rank = rr * loss_neg
            else:
                loss_rank = torch.zeros((), device=device)
        else:
            loss_rank = torch.zeros((), device=device)

        l2 = self._l2_penalty()
        loss = w_obs * loss_obs + w_rank * loss_rank + w_l2 * l2

        self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train/loss_obs", loss_obs, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train/loss_rank", loss_rank, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train/loss_l2", l2, prog_bar=False, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.learning_rate)
