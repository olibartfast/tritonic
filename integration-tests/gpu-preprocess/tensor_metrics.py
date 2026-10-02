"""Tensor parity metrics shared by the GPU preprocessing checks (spec D-5).

The thresholds are the ones the DALI path was accepted with in
integration-tests/yolo-ensemble/compare_dali.py. Do not loosen them per variant.
"""

from __future__ import annotations

import numpy as np

MAX_ABS_LIMIT = 0.15
MEAN_ABS_LIMIT = 0.003
P99_ABS_LIMIT = 0.016
EXPECTED_SHAPE = [3, 640, 640]


def tensor_metrics(expected: np.ndarray, actual: np.ndarray) -> dict:
    difference = np.abs(actual.astype(np.float32) - expected.astype(np.float32))
    return {
        "shape": list(actual.shape),
        "max_abs": float(difference.max()),
        "mean_abs": float(difference.mean()),
        "p99_abs": float(np.quantile(difference, 0.99)),
        "different_elements": int(np.count_nonzero(difference)),
    }


def passes(metrics: dict) -> bool:
    return (
        metrics["shape"] == EXPECTED_SHAPE
        and metrics["max_abs"] <= MAX_ABS_LIMIT
        and metrics["mean_abs"] <= MEAN_ABS_LIMIT
        and metrics["p99_abs"] <= P99_ABS_LIMIT
    )
