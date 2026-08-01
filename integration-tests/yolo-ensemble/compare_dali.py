#!/usr/bin/env python3
"""Compare DALI GPU preprocessing with the checked CPU baseline."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

MAX_ABS_LIMIT = 0.15
MEAN_ABS_LIMIT = 0.003
P99_ABS_LIMIT = 0.016


def load_pipeline_factory(repo: Path):
    path = (
        repo
        / "deploy/object_detection/yolo/ensemble/dali/generate_pipeline.py"
    )
    spec = importlib.util.spec_from_file_location("yolo_dali_pipeline", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import DALI pipeline from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.yolo_preprocess_pipeline


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("/workspace"))
    parser.add_argument("--baseline", type=Path, default=Path("/baseline"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    expected_manifest = json.loads(
        (
            args.repo
            / "tests/data/ensemble/yolo/cpu_preprocess_sha256.json"
        ).read_text(encoding="utf-8")
    )
    pipeline_factory = load_pipeline_factory(args.repo)
    pipeline = pipeline_factory(
        batch_size=1,
        num_threads=2,
        device_id=0,
        exec_async=False,
        exec_pipelined=False,
        prefetch_queue_depth=1,
    )
    pipeline.build()

    results = []
    failed = False
    for fixture in expected_manifest["fixtures"]:
        source = args.repo / fixture["source"]
        encoded = np.fromfile(source, dtype=np.uint8)
        pipeline.feed_input("IMAGE", [encoded])
        (output,) = pipeline.run()
        actual = np.asarray(output.as_cpu().at(0), dtype=np.float32)
        expected = np.fromfile(
            args.baseline / fixture["tensor"], dtype=np.float32
        ).reshape(3, 640, 640)
        difference = np.abs(actual - expected)
        result = {
            "source": fixture["source"],
            "shape": list(actual.shape),
            "max_abs": float(difference.max()),
            "mean_abs": float(difference.mean()),
            "p99_abs": float(np.quantile(difference, 0.99)),
            "different_elements": int(np.count_nonzero(difference)),
        }
        result["passed"] = (
            result["shape"] == [3, 640, 640]
            and result["max_abs"] <= MAX_ABS_LIMIT
            and result["mean_abs"] <= MEAN_ABS_LIMIT
            and result["p99_abs"] <= P99_ABS_LIMIT
        )
        failed = failed or not result["passed"]
        results.append(result)
        print(json.dumps(result, sort_keys=True))

    report = {
        "limits": {
            "max_abs": MAX_ABS_LIMIT,
            "mean_abs": MEAN_ABS_LIMIT,
            "p99_abs": P99_ABS_LIMIT,
        },
        "fixtures": results,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
