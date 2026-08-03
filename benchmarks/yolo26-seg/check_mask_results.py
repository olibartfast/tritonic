#!/usr/bin/env python3
"""Validate Tritonic YOLO26-seg mask semantics and summarize timings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.agreement import (  # noqa: E402
    compare,
    require_model_family,
    timing_summary_many,
)
from common.masks import load_mask, mask_overlap_for  # noqa: E402

PATHS = ("cpu_pre_cpu_post", "gpu_pre_cpu_post", "gpu_pre_gpu_post")

# Result documents written before the family naming was corrected say "yolo26m-seg",
# where the m is the model size rather than part of the family.
ACCEPTED_FAMILIES = {"yolo26-seg", "yolo26m-seg"}


def compare_masks(
    reference_doc,
    reference_path,
    candidate_doc,
    candidate_path,
    min_box_iou,
    min_mask_iou,
    max_score_delta,
):
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
    report = {"schema_version": 1, "model_family": "yolo26-seg", "fixtures": {}}
    all_documents = {name: [] for name in PATHS}
    fixture_dirs = []
    for candidate in sorted(
        path for path in args.results_dir.iterdir() if path.is_dir()
    ):
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
        fixture = {
            "cpu_vs_dali_pre": compare_masks(
                documents["cpu_pre_cpu_post"],
                paths["cpu_pre_cpu_post"],
                documents["gpu_pre_cpu_post"],
                paths["gpu_pre_cpu_post"],
                0.95,
                0.90,
                0.30,
            ),
            "cpu_post_vs_dali_post": compare_masks(
                documents["gpu_pre_cpu_post"],
                paths["gpu_pre_cpu_post"],
                documents["gpu_pre_gpu_post"],
                paths["gpu_pre_gpu_post"],
                1.0,
                1.0,
                1e-6,
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
