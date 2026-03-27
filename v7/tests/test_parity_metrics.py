from __future__ import annotations

import numpy as np

from parity_metrics import prediction_parity_report, topk_quality_and_overlap_report


def test_prediction_parity_report_basic() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    t = np.array([1.1, 1.9, 2.8, 4.2])
    v = np.array([1.0, 2.1, 3.2, 3.9])
    rep = prediction_parity_report(turi_pred=t, v7_pred=v, target=y)
    assert rep["n"] == 4.0
    assert -1.0 <= rep["spearman_turi_vs_v7"] <= 1.0
    assert rep["rmse_turi"] >= 0.0
    assert rep["rmse_v7"] >= 0.0


def test_topk_report_basic() -> None:
    turi = {
        "user": ["u1", "u1", "u2", "u2"],
        "item": ["i1", "i2", "i1", "i3"],
        "rank": [1, 2, 1, 2],
    }
    v7 = {
        "user": ["u1", "u1", "u2", "u2"],
        "item": ["i2", "i3", "i1", "i2"],
        "rank": [1, 2, 1, 2],
    }
    truth_user = ["u1", "u2"]
    truth_item = ["i2", "i1"]
    rep = topk_quality_and_overlap_report(
        turi_recs=turi,
        v7_recs=v7,
        truth_user=truth_user,
        truth_item=truth_item,
        k_values=(2,),
    )
    assert "precision_at_2_turi" in rep
    assert "precision_at_2_v7" in rep
    assert 0.0 <= rep["jaccard_at_2_turi_vs_v7"] <= 1.0
