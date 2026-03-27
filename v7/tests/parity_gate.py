"""CI gate for parity report thresholds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate parity metrics against thresholds.")
    p.add_argument("--report", required=True, help="Path to parity report JSON.")
    p.add_argument(
        "--thresholds",
        default="v7/tests/baselines/thresholds.json",
        help="Path to threshold JSON.",
    )
    return p.parse_args()


def _fail(msg: str) -> None:
    raise SystemExit(msg)


def main() -> None:
    args = _parse_args()
    report = json.loads(Path(args.report).read_text())
    thresholds = json.loads(Path(args.thresholds).read_text())

    pr = report.get("prediction_report", {})
    tr = report.get("topk_report", {})
    t_pr = thresholds.get("prediction_report", {})
    t_tr = thresholds.get("topk_report", {})

    if "spearman_turi_vs_v7_min" in t_pr:
        v = float(pr.get("spearman_turi_vs_v7", -1.0))
        if v < float(t_pr["spearman_turi_vs_v7_min"]):
            _fail(f"FAIL: spearman_turi_vs_v7={v:.4f} below threshold.")

    if "rmse_relative_drift_max" in t_pr:
        v = float(pr.get("rmse_relative_drift", 1e9))
        if v > float(t_pr["rmse_relative_drift_max"]):
            _fail(f"FAIL: rmse_relative_drift={v:.4f} above threshold.")

    if "delta_mean_v7_minus_turi_abs_max" in t_pr:
        v = abs(float(pr.get("delta_mean_v7_minus_turi", 1e9)))
        if v > float(t_pr["delta_mean_v7_minus_turi_abs_max"]):
            _fail(f"FAIL: |delta_mean_v7_minus_turi|={v:.4f} above threshold.")

    if "jaccard_at_10_turi_vs_v7_min" in t_tr:
        v = float(tr.get("jaccard_at_10_turi_vs_v7", -1.0))
        if v < float(t_tr["jaccard_at_10_turi_vs_v7_min"]):
            _fail(f"FAIL: jaccard_at_10_turi_vs_v7={v:.4f} below threshold.")

    print("PASS: parity thresholds satisfied.")


if __name__ == "__main__":
    main()
