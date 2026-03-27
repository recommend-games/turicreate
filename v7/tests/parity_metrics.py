"""Metrics for cross-implementation parity checks."""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


def _safe_mean(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.mean(x))


def _rankdata_average_ties(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_x = x[order]
    n = len(sorted_x)
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        avg_rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return 0.0
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return 0.0
    rx = _rankdata_average_ties(x)
    ry = _rankdata_average_ties(y)
    return pearson_corr(rx, ry)


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    if pred.size == 0:
        return 0.0
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    if pred.size == 0:
        return 0.0
    return float(np.mean(np.abs(pred - target)))


def prediction_parity_report(
    turi_pred: Sequence[float],
    v7_pred: Sequence[float],
    target: Sequence[float],
) -> Dict[str, float]:
    tp = np.asarray(turi_pred, dtype=np.float64)
    vp = np.asarray(v7_pred, dtype=np.float64)
    yt = np.asarray(target, dtype=np.float64)
    delta = vp - tp
    rmse_turi = rmse(tp, yt)
    rmse_v7 = rmse(vp, yt)
    return {
        "n": float(len(tp)),
        "spearman_turi_vs_v7": spearman_corr(tp, vp),
        "pearson_turi_vs_v7": pearson_corr(tp, vp),
        "rmse_turi": rmse_turi,
        "rmse_v7": rmse_v7,
        "mae_turi": mae(tp, yt),
        "mae_v7": mae(vp, yt),
        "rmse_relative_drift": abs(rmse_v7 - rmse_turi) / max(rmse_turi, 1e-12),
        "delta_mean_v7_minus_turi": _safe_mean(delta),
        "delta_std_v7_minus_turi": float(np.std(delta)) if delta.size else 0.0,
    }


def _group_truth_by_user(
    user_ids: Sequence,
    item_ids: Sequence,
) -> Dict[object, set]:
    out: Dict[object, set] = {}
    for u, i in zip(user_ids, item_ids):
        out.setdefault(u, set()).add(i)
    return out


def _recs_by_user(
    rec_user: Sequence,
    rec_item: Sequence,
    rec_rank: Sequence[int],
    k: int,
) -> Dict[object, List[object]]:
    out: Dict[object, List[Tuple[int, object]]] = {}
    for u, i, r in zip(rec_user, rec_item, rec_rank):
        if int(r) <= k:
            out.setdefault(u, []).append((int(r), i))
    return {u: [it for _, it in sorted(v)] for u, v in out.items()}


def _dcg(recommended: List[object], truth_set: set) -> float:
    if not recommended:
        return 0.0
    s = 0.0
    for idx, item in enumerate(recommended, start=1):
        if item in truth_set:
            s += 1.0 / np.log2(idx + 1.0)
    return s


def topk_quality_and_overlap_report(
    turi_recs: Mapping[str, Sequence],
    v7_recs: Mapping[str, Sequence],
    truth_user: Sequence,
    truth_item: Sequence,
    k_values: Iterable[int] = (5, 10, 20),
) -> Dict[str, float]:
    truth = _group_truth_by_user(truth_user, truth_item)
    result: Dict[str, float] = {}

    for k in k_values:
        tr = _recs_by_user(
            turi_recs["user"],
            turi_recs["item"],
            turi_recs["rank"],
            k,
        )
        vr = _recs_by_user(
            v7_recs["user"],
            v7_recs["item"],
            v7_recs["rank"],
            k,
        )
        users = list(set(tr.keys()) | set(vr.keys()))
        if not users:
            result[f"precision_at_{k}_turi"] = 0.0
            result[f"precision_at_{k}_v7"] = 0.0
            result[f"recall_at_{k}_turi"] = 0.0
            result[f"recall_at_{k}_v7"] = 0.0
            result[f"ndcg_at_{k}_turi"] = 0.0
            result[f"ndcg_at_{k}_v7"] = 0.0
            result[f"jaccard_at_{k}_turi_vs_v7"] = 0.0
            continue

        p_t, p_v, r_t, r_v, n_t, n_v, j = [], [], [], [], [], [], []
        for u in users:
            gt = truth.get(u, set())
            t_items = tr.get(u, [])
            v_items = vr.get(u, [])
            set_t, set_v = set(t_items), set(v_items)

            hit_t = len(set_t & gt)
            hit_v = len(set_v & gt)
            denom_gt = max(len(gt), 1)
            p_t.append(hit_t / max(len(t_items), 1))
            p_v.append(hit_v / max(len(v_items), 1))
            r_t.append(hit_t / denom_gt)
            r_v.append(hit_v / denom_gt)

            dcg_t = _dcg(t_items, gt)
            dcg_v = _dcg(v_items, gt)
            ideal_len = min(k, len(gt))
            if ideal_len == 0:
                ndcg_t = 0.0
                ndcg_v = 0.0
            else:
                ideal = float(np.sum([1.0 / np.log2(i + 1.0) for i in range(2, ideal_len + 2)]))
                ndcg_t = dcg_t / max(ideal, 1e-12)
                ndcg_v = dcg_v / max(ideal, 1e-12)
            n_t.append(ndcg_t)
            n_v.append(ndcg_v)

            union = set_t | set_v
            j.append(len(set_t & set_v) / max(len(union), 1))

        result[f"precision_at_{k}_turi"] = float(np.mean(p_t))
        result[f"precision_at_{k}_v7"] = float(np.mean(p_v))
        result[f"recall_at_{k}_turi"] = float(np.mean(r_t))
        result[f"recall_at_{k}_v7"] = float(np.mean(r_v))
        result[f"ndcg_at_{k}_turi"] = float(np.mean(n_t))
        result[f"ndcg_at_{k}_v7"] = float(np.mean(n_v))
        result[f"jaccard_at_{k}_turi_vs_v7"] = float(np.mean(j))

    return result
