#!/usr/bin/env python3
"""Validate Tritonic YOLO26m-seg mask semantics and summarize timings."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import cv2
import numpy as np

PATHS = ("cpu_pre_cpu_post", "gpu_pre_cpu_post", "gpu_pre_gpu_post")


def bbox_iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union else 0.0


def canonical(detections):
    kept = []
    for detection in sorted(detections, key=lambda item: item["score"], reverse=True):
        if any(
            detection["class_id"] == prior["class_id"]
            and bbox_iou(detection["bbox"], prior["bbox"]) >= 0.9
            for prior in kept
        ):
            continue
        kept.append(detection)
    return kept


def load_mask(json_path, detection, frame_shape=None):
    mask_path = json_path.parent / detection["mask_file"]
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"could not read mask artifact: {mask_path}")
    mask = mask != 0
    if not np.count_nonzero(mask):
        raise ValueError(f"empty mask artifact: {mask_path}")
    x, y, width, height = detection["bbox"]
    if mask.shape == (height, width):
        if frame_shape is None:
            return mask
        expanded = np.zeros(frame_shape, dtype=bool)
        if y + height > frame_shape[0] or x + width > frame_shape[1]:
            raise ValueError(f"bbox-local mask exceeds frame: {mask_path}")
        expanded[y : y + height, x : x + width] = mask
        return expanded
    if frame_shape is not None and mask.shape != frame_shape:
        raise ValueError(f"full-frame mask shape mismatch: {mask_path}")
    return mask


def compare(
    reference_doc,
    reference_path,
    candidate_doc,
    candidate_path,
    min_box_iou,
    min_mask_iou,
    max_score_delta,
):
    reference = canonical(reference_doc["detections"])
    candidate = canonical(candidate_doc["detections"])
    if len(reference) != len(candidate):
        raise ValueError(
            f"canonical detection count mismatch: {len(reference)} != {len(candidate)}"
        )
    remaining = set(range(len(candidate)))
    matches = []
    for ref in reference:
        choices = [
            index
            for index in remaining
            if candidate[index]["class_id"] == ref["class_id"]
        ]
        if not choices:
            raise ValueError(f"missing class {ref['class_id']}")
        index = max(
            choices, key=lambda item: bbox_iou(ref["bbox"], candidate[item]["bbox"])
        )
        cand = candidate[index]
        remaining.remove(index)
        box = bbox_iou(ref["bbox"], cand["bbox"])
        score_delta = abs(ref["score"] - cand["score"])
        ref_mask = load_mask(reference_path, ref)
        cand_mask = load_mask(candidate_path, cand, ref_mask.shape)
        intersection = int(np.count_nonzero(ref_mask & cand_mask))
        union = int(np.count_nonzero(ref_mask | cand_mask))
        mask_iou = intersection / union if union else 0.0
        if (
            box < min_box_iou
            or mask_iou < min_mask_iou
            or score_delta > max_score_delta
        ):
            raise ValueError(
                f"semantic mismatch class={ref['class_id']} box_iou={box:.6f} "
                f"mask_iou={mask_iou:.6f} score_delta={score_delta:.6f}"
            )
        matches.append(
            {
                "class_id": ref["class_id"],
                "box_iou": box,
                "mask_iou": mask_iou,
                "score_delta": score_delta,
            }
        )
    return matches


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)]


def timing_summary_many(documents):
    result = {}
    for key in ("preprocess", "infer", "postprocess", "total"):
        values = [
            sample[key] for document in documents for sample in document["samples_ms"]
        ]
        result[key] = {
            "mean_ms": statistics.fmean(values),
            "median_ms": statistics.median(values),
            "p95_ms": percentile(values, 0.95),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {"schema_version": 1, "model_family": "yolo26m-seg", "fixtures": {}}
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
            if documents[path_name].get("model_family") != "yolo26m-seg":
                raise ValueError(f"wrong model family in {paths[path_name]}")
            if documents[path_name].get("segmentation_output") != "mask":
                raise ValueError(f"non-mask benchmark output in {paths[path_name]}")
            all_documents[path_name].append(documents[path_name])
            for detection in documents[path_name]["detections"]:
                load_mask(paths[path_name], detection)
        fixture = {
            "cpu_vs_dali_pre": compare(
                documents["cpu_pre_cpu_post"],
                paths["cpu_pre_cpu_post"],
                documents["gpu_pre_cpu_post"],
                paths["gpu_pre_cpu_post"],
                0.95,
                0.90,
                0.30,
            ),
            "cpu_post_vs_dali_post": compare(
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
