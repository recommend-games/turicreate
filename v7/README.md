# v7 — Ranking-style matrix factorization (PyTorch Lightning + Polars)

Standalone trainer inspired by Turi Create’s `ranking_factorization_recommender` for **explicit ratings** only (`user_id`, `item_id`, target column).

## Install

```bash
cd v7
pip install -e ".[dev]"
```

## Quick start

```python
import polars as pl
from v7_ranking_mf import create

df = pl.DataFrame({
    "user_id": ["a", "a", "b", "b", "c"],
    "item_id": ["x", "y", "x", "z", "y"],
    "rating": [1.0, 5.0, 2.0, 4.0, 3.0],
})
model = create(df, target="rating", max_iterations=10)
scores = model.predict(df)
recs = model.recommend(k=2)
```

## Defaults (match `ranking_factorization_recommender.create`)

| Parameter | Default |
|-----------|---------|
| `user_id` | `"user_id"` |
| `item_id` | `"item_id"` |
| `num_factors` | `32` |
| `regularization` | `1e-9` |
| `linear_regularization` | `1e-9` |
| `ranking_regularization` | `0.25` |
| `unobserved_rating_value` | `None` → `mean(rating) - 1.96 * std(rating)` on training targets |
| `num_sampled_negative_examples` | `4` |
| `max_iterations` | `25` (Lightning `max_epochs`) |
| `sgd_step_size` | `0` → Adam `lr=1e-3` (not Turi’s internal line search) |
| `random_seed` | `0` |
| `verbose` | `True` |

`binary_target`, `solver`, and side-feature options are not implemented.

## Model and loss

**Score** (no side features):

\[
\hat{r}_{ij} = \mu + w_i + w_j + \mathbf{u}_i^\top \mathbf{v}_j
\]

**Training loss** (conceptually):

1. **Observed**: mean squared error between \(\hat{r}_{ij}\) and the rating on training rows.
2. **Ranking regularization** (when `ranking_regularization > 0`): for each training row, sample `num_sampled_negative_examples` items the user did not observe, score them with the **current** model, take the candidate with the **highest** predicted score, and add squared error pushing that score toward `unobserved_rating_value`, scaled by `ranking_regularization`. Rows with no unseen items skip this term.
3. **L2**: `linear_regularization` on global bias and all user/item bias embeddings; `regularization` on all factor embeddings.

Exact scaling may differ from Turi’s C++ implementation; this package does not claim bitwise parity.

## API

- **`create(observation_data, target, ...)`** — `observation_data` is a `polars.DataFrame` or path to CSV. Returns a `RankingFactorizationRecommender` instance.
- **`predict(dataset)`** — same schema as training (user, item columns). Returns `pl.Series` of scores. Unknown user or item ids raise `ValueError`.
- **`recommend(k=10, users=None, exclude_known=True)`** — returns `pl.DataFrame` with user column, `item_id` column (name matches training), `score`, `rank`. Unknown users raise `ValueError`.

## Tests

```bash
cd v7 && pytest
```
