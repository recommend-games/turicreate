# Parity Protocol: Turi C++ vs v7 PyTorch

This document defines how to validate agreement between:
- Turi Create `ranking_factorization_recommender` (C++ implementation)
- `v7_ranking_mf` (PyTorch/Lightning implementation)

The goal is **behavioral/statistical parity**, not exact coefficient equality.

## Scope

- Explicit rating only (`user_id`, `item_id`, `target`)
- No implicit-only mode, no binary target, no side features
- Shared defaults mirroring Turi `create()` where implemented:
  - `num_factors=32`
  - `regularization=1e-9`
  - `linear_regularization=1e-9`
  - `ranking_regularization=0.25`
  - `num_sampled_negative_examples=4`
  - `max_iterations=25`
  - `random_seed=0`

## Deterministic data protocol

1. Read the same raw dataset once.
2. Apply the same target cleaning policy to both systems.
3. Split rows deterministically using a shared RNG seed:
   - `train`
   - `test`
4. Evaluate both systems on exactly the same:
   - prediction query table (test rows)
   - recommendation user cohort

## Agreement checks

### 1) API/contract parity

- Required columns accepted/rejected consistently.
- Unknown user/item behavior documented and validated.
- Recommendation schema and ranking columns valid.
- `exclude_known` semantics validated.

### 2) Prediction agreement

For matched `(user,item)` rows:
- Inter-model:
  - Spearman correlation
  - Pearson correlation
  - Mean/std of score deltas
- Quality:
  - RMSE and MAE vs ground truth for each model
  - Relative RMSE drift: `abs(rmse_v7 - rmse_turi) / max(rmse_turi, eps)`

Suggested initial acceptance targets:
- Spearman >= 0.70
- Relative RMSE drift <= 0.10
- Absolute mean score delta <= dataset-defined tolerance

### 3) Top-k recommendation agreement

For a fixed user cohort and each `k in {5,10,20}`:
- Precision@k, Recall@k, NDCG@k (against held-out positives)
- Inter-model Jaccard@k between top-k sets

Suggested initial acceptance targets:
- Jaccard@10 >= 0.20 (depends strongly on catalog size/sparsity)
- Metric drift in Precision/Recall/NDCG within dataset-specific tolerance

## Known expected non-parities

- Different optimizer internals and step-size behavior
- Different negative sampling implementation details
- Thread scheduling and hardware parallelism differences
- Floating point accumulation/order differences

These are acceptable as long as agreement metrics stay above thresholds.
