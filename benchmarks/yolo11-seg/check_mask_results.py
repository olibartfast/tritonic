#!/usr/bin/env python3
"""Validate Tritonic YOLO11-seg mask semantics and summarize timings.

The previous version of this file validated nothing at all: it defined only timing
helpers, so a GPU path that dropped detections still produced a clean run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.agreement import (  # noqa: E402
    compare,
    require_model_family,
    timing_summary,
    timing_summary_many,
)
from common.masks import load_mask, mask_overlap_for  # noqa: E402

PATHS = ("cpu_pre_cpu_post", "gpu_pre_cpu_post", "gpu_pre_gpu_post")

# "yolo11seg" is what the binary emitted before the model_family fix.
ACCEPTED_FAMILIES = {"yolo11-seg", "yolo11seg"}

# Masks amplify the DALI-versus-OpenCV resize difference far more than boxes do: on
# the stock fixtures the measured minimum box IoU is 0.975 while mask IoU drops to
# 0.877 (bus). The bound is set just under that, so it still catches a real
# regression without failing on ordinary resampling drift.
PRE_TOLERANCE = (0.95, 0.85, 0.30)

# Given identical DALI-preprocessed tensors, GPU postprocessing reproduces the CPU
# result exactly -- measured mask IoU 1.0 and box IoU 1.0 on every stock fixture --
# so this stays strict.
POST_TOLERANCE = (1.0, 1.0, 1e-6)

# See check_results.py: the dense synthetic fixture exists to exercise the detection
# cap, so only its preprocessing tolerance is relaxed (measured mask IoU 0.761).
DENSE_FIXTURES = {"crowd"}
DENSE_PRE_TOLERANCE = (0.90, 0.70, 0.30)


def compare_masks(reference_doc, reference_path, candidate_doc, candidate_path,
                  tolerance):
    min_box_iou, min_mask_iou, max_score_delta = tolerance
    return compare(
        reference_doc,
        candidate_doc,
        min_box_iou,
        max_score_delta,
        overlap=mask_overlap_for(reference_path, candidate_path),
        min_overlap=min_mask_iou,
        overlap_name="mask_iou",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {"schema_version": 1, "model_family": "yolo11-seg", "fixtures": {}}
    all_documents = {name: [] for name in PATHS}

    fixture_dirs = []
    for candidate in sorted(path for path in args.results_dir.iterdir() if path.is_dir()):
        present = [(candidate / f"{path_name}.json").is_file() for path_name in PATHS]
        if not any(present):
            continue
        if not all(present):
            raise ValueError(f"incomplete benchmark fixture directory: {candidate}")
        fixture_dirs.append(candidate)
    if not fixture_dirs:
        raise ValueError(f"no benchmark fixtures found in {args.results_dir}")

    for fixture_dir in fixture_dirs:
        documents = {}
        paths = {}
        for path_name in PATHS:
            paths[path_name] = fixture_dir / f"{path_name}.json"
            documents[path_name] = json.loads(paths[path_name].read_text())
            require_model_family(
                documents[path_name], ACCEPTED_FAMILIES, paths[path_name]
            )
            if documents[path_name].get("segmentation_output") != "mask":
                raise ValueError(f"non-mask benchmark output in {paths[path_name]}")
            all_documents[path_name].append(documents[path_name])
            for detection in documents[path_name]["detections"]:
                load_mask(paths[path_name], detection)
        pre_tolerance = (
            DENSE_PRE_TOLERANCE if fixture_dir.name in DENSE_FIXTURES else PRE_TOLERANCE
        )
        fixture = {
            "timings": {name: timing_summary(documents[name]) for name in PATHS},
            "cpu_vs_dali_pre": compare_masks(
                documents["cpu_pre_cpu_post"],
                paths["cpu_pre_cpu_post"],
                documents["gpu_pre_cpu_post"],
                paths["gpu_pre_cpu_post"],
                pre_tolerance,
            ),
            "cpu_post_vs_dali_post": compare_masks(
                documents["gpu_pre_cpu_post"],
                paths["gpu_pre_cpu_post"],
                documents["gpu_pre_gpu_post"],
                paths["gpu_pre_gpu_post"],
                POST_TOLERANCE,
            ),
            "status": "pass",
        }
        report["fixtures"][fixture_dir.name] = fixture

    report["aggregate"] = {
        name: timing_summary_many(all_documents[name]) for name in PATHS
    }
    cpu_total = report["aggregate"]["cpu_pre_cpu_post"]["total"]["median_ms"]
    gpu_cpu_total = report["aggregate"]["gpu_pre_cpu_post"]["total"]["median_ms"]
    gpu_total = report["aggregate"]["gpu_pre_gpu_post"]["total"]["median_ms"]
    report["speedup"] = {
        "dali_gpu_pre_post_vs_cpu_pre_post_median": cpu_total / gpu_total,
        "dali_gpu_pre_post_vs_gpu_pre_cpu_post_median": gpu_cpu_total / gpu_total,
    }
    report["status"] = "pass"
    output = args.output or args.results_dir / "summary.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
