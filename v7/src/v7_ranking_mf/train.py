"""Public `create()` API and trained model wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Union

import lightning as L
import polars as pl
import torch

from v7_ranking_mf.data import (
    build_training_matrices,
    default_unobserved_rating_value,
    load_observation_frame,
    make_train_dataloader,
)
from v7_ranking_mf.lightning_module import BiasedMFLightningModule
from v7_ranking_mf.predict import predict_dataframe, recommend_top_k


def _invert_map(m: Dict[Any, int]) -> List[Any]:
    out: List[Any] = [None] * len(m)  # type: ignore[assignment]
    for k, v in m.items():
        out[v] = k
    return out


class RankingFactorizationRecommender:
    """Trained biased MF with ranking regularization (Polars I/O)."""

    def __init__(
        self,
        model: torch.nn.Module,
        user_col: str,
        item_col: str,
        target_col: str,
        user_map: Dict[Any, int],
        item_map: Dict[Any, int],
        user_seen: List[Set[int]],
        unobserved_rating_value: float,
        hyperparams: Dict[str, Any],
    ) -> None:
        self._model = model
        self._user_col = user_col
        self._item_col = item_col
        self._target_col = target_col
        self._user_map = user_map
        self._item_map = item_map
        self._idx_to_user = _invert_map(user_map)
        self._idx_to_item = _invert_map(item_map)
        self._user_seen = user_seen
        self._unobserved_rating_value = unobserved_rating_value
        self._hyperparams = hyperparams
        self._device = torch.device("cpu")

    @property
    def hyperparams(self) -> Dict[str, Any]:
        return dict(self._hyperparams)

    def to(self, device: Union[str, torch.device]) -> "RankingFactorizationRecommender":
        self._device = torch.device(device)
        self._model.to(self._device)
        return self

    def predict(
        self,
        dataset: Union[pl.DataFrame, str, Path],
    ) -> pl.Series:
        df = load_observation_frame(dataset)
        return predict_dataframe(
            self._model,
            df,
            self._user_col,
            self._item_col,
            self._user_map,
            self._item_map,
            self._device,
        )

    def recommend(
        self,
        k: int = 10,
        users: Optional[Union[pl.Series, List]] = None,
        exclude_known: bool = True,
    ) -> pl.DataFrame:
        if users is None:
            user_indices = list(range(len(self._user_map)))
        else:
            if isinstance(users, pl.Series):
                u_list = users.to_list()
            else:
                u_list = list(users)
            user_indices = []
            for u in u_list:
                if u not in self._user_map:
                    raise ValueError(f"Unknown user id for recommend: {u!r}")
                user_indices.append(self._user_map[u])

        rec = recommend_top_k(
            self._model,
            user_indices,
            self._user_seen,
            self._idx_to_item,
            self._item_col,
            k=k,
            exclude_known=exclude_known,
            device=self._device,
        )
        u_ids = [self._idx_to_user[i] for i in rec["_user_idx"].to_list()]
        return pl.DataFrame(
            {
                self._user_col: u_ids,
                self._item_col: rec[self._item_col],
                "score": rec["score"],
                "rank": rec["rank"],
            }
        )


def create(
    observation_data: Union[pl.DataFrame, str, Path],
    user_id: str = "user_id",
    item_id: str = "item_id",
    target: Optional[str] = None,
    num_factors: int = 32,
    regularization: float = 1e-9,
    linear_regularization: float = 1e-9,
    ranking_regularization: float = 0.25,
    unobserved_rating_value: Optional[float] = None,
    num_sampled_negative_examples: int = 4,
    max_iterations: int = 25,
    sgd_step_size: float = 0.0,
    random_seed: int = 0,
    binary_target: bool = False,
    solver: str = "auto",
    verbose: bool = True,
    batch_size: Optional[int] = None,
    num_workers: int = 0,
    invalid_target_action: str = "error",
    accelerator: str = "auto",
    devices: Union[int, str] = 1,
    observation_loss_weight: float = 1.0,
    ranking_loss_weight: float = 1.0,
    l2_loss_weight: float = 1.0,
    ranking_loss_reduce: Literal["valid_mean", "batch_mean"] = "valid_mean",
) -> RankingFactorizationRecommender:
    """
    Train a biased matrix factorization model with ranking regularization.

    ``binary_target`` and ``solver`` are accepted for API symmetry with Turi Create
    but are ignored (only non-binary explicit ratings are supported).

    Notes
    -----
    **Approximating Turi Create's RankingFactorizationRecommender (explicit ratings,
    user/item/target only).** The model and ranking term (hard negative among K
    unseen items, squared to ``unobserved_rating_value``) match the same intent as
    Turi, but optimization differs (Adam vs SGD/Adagrad, how L2 couples to ranking
    steps, batch scaling). Identical weights or rankings should not be expected.
    For behaviorally similar results, tune on a validation metric (RMSE, overlap@K,
    etc.): learning rate (``sgd_step_size``), ``ranking_regularization``, both L2
    settings, ``num_sampled_negative_examples``, and ``max_iterations``. Use
    ``observation_loss_weight``, ``ranking_loss_weight``, and ``l2_loss_weight`` to
    rebalance terms if needed. ``ranking_loss_reduce='batch_mean'`` averages the
    ranking MSE over the full mini-batch (zero contribution for rows with no valid
    negatives), which can change the effective strength of ``ranking_regularization``
    relative to the observation term compared to ``'valid_mean'`` (default).
    """
    _ = solver  # API compatibility; training always uses Adam
    if binary_target:
        raise NotImplementedError("binary_target=True is not supported in v7.")

    if observation_loss_weight < 0 or ranking_loss_weight < 0 or l2_loss_weight < 0:
        raise ValueError("Loss weights must be non-negative.")
    if ranking_loss_reduce not in ("valid_mean", "batch_mean"):
        raise ValueError("ranking_loss_reduce must be 'valid_mean' or 'batch_mean'.")

    if target is None:
        raise ValueError("target must be the name of the explicit rating column.")

    df = load_observation_frame(observation_data)
    for col in (user_id, item_id, target):
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col!r}")
    if invalid_target_action not in {"error", "drop"}:
        raise ValueError("invalid_target_action must be one of {'error', 'drop'}.")

    # Ensure explicit ratings are numeric and finite to avoid NaN losses.
    cast_target = pl.col(target).cast(pl.Float64, strict=False).alias(target)
    df = df.with_columns(cast_target)
    bad_mask = ~pl.col(target).is_finite() | pl.col(target).is_null()
    bad_target_rows = df.filter(bad_mask).height
    if bad_target_rows > 0 and invalid_target_action == "error":
        raise ValueError(
            f"Target column {target!r} contains {bad_target_rows} null/non-finite values."
        )
    if invalid_target_action == "drop":
        df = df.filter(~bad_mask)
        if df.height == 0:
            raise ValueError(
                f"All rows were dropped because target column {target!r} had no finite values."
            )

    if unobserved_rating_value is None:
        v_ur = default_unobserved_rating_value(df, target)
    else:
        v_ur = float(unobserved_rating_value)

    matrices, u_map, i_map = build_training_matrices(df, user_id, item_id, target)

    lr = 1e-3 if sgd_step_size == 0 else float(sgd_step_size)
    bs = batch_size if batch_size is not None else min(4096, max(32, matrices.user_idx.size(0) // 10))

    L.seed_everything(random_seed, workers=True)

    lit = BiasedMFLightningModule(
        n_users=matrices.n_users,
        n_items=matrices.n_items,
        num_factors=num_factors,
        user_seen=matrices.user_seen,
        unobserved_rating_value=v_ur,
        ranking_regularization=ranking_regularization,
        num_sampled_negative_examples=num_sampled_negative_examples,
        regularization=regularization,
        linear_regularization=linear_regularization,
        learning_rate=lr,
        random_seed=random_seed,
        observation_loss_weight=observation_loss_weight,
        ranking_loss_weight=ranking_loss_weight,
        l2_loss_weight=l2_loss_weight,
        ranking_loss_reduce=ranking_loss_reduce,
    )

    train_loader = make_train_dataloader(
        matrices,
        bs,
        random_seed,
        num_workers=num_workers,
    )

    trainer = L.Trainer(
        max_epochs=max_iterations,
        enable_progress_bar=verbose,
        enable_model_summary=verbose,
        logger=verbose if verbose else False,
        enable_checkpointing=False,
        accelerator=accelerator,
        devices=devices,
        deterministic=True,
    )
    trainer.fit(lit, train_loader)

    model = lit.model.cpu().eval()
    hyperparams = {
        "user_id": user_id,
        "item_id": item_id,
        "target": target,
        "num_factors": num_factors,
        "regularization": regularization,
        "linear_regularization": linear_regularization,
        "ranking_regularization": ranking_regularization,
        "unobserved_rating_value": v_ur,
        "num_sampled_negative_examples": num_sampled_negative_examples,
        "max_iterations": max_iterations,
        "sgd_step_size": sgd_step_size,
        "learning_rate_used": lr,
        "random_seed": random_seed,
        "observation_loss_weight": observation_loss_weight,
        "ranking_loss_weight": ranking_loss_weight,
        "l2_loss_weight": l2_loss_weight,
        "ranking_loss_reduce": ranking_loss_reduce,
    }

    return RankingFactorizationRecommender(
        model=model,
        user_col=user_id,
        item_col=item_id,
        target_col=target,
        user_map=u_map,
        item_map=i_map,
        user_seen=matrices.user_seen,
        unobserved_rating_value=v_ur,
        hyperparams=hyperparams,
    )
