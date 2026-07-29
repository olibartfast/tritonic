#!/usr/bin/env python3
"""Validate Tritonic YOLO26m-seg benchmark semantics and summarize timings."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

PATHS = ("cpu_pre_cpu_post", "gpu_pre_cpu_post", "gpu_pre_gpu_post")


def read_pgm(path: Path):
    with path.open("rb") as stream:
        if stream.readline().strip() != b"P5":
            raise ValueError(f"not a binary PGM: {path}")
        width, height = map(int, stream.readline().split())
        if stream.readline().strip() != b"255":
            raise ValueError(f"unsupported PGM range: {path}")
        data = stream.read()
    if len(data) != width * height or any(value not in (0, 255) for value in data):
        raise ValueError(f"invalid binary mask: {path}")
    return width, height, data


def bbox_iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union else 0.0


def mask_pixels(base: Path, detection):
    width, height, data = read_pgm(base / detection["mask_file"])
    x, y, box_width, box_height = detection["bbox"]
    if (height, width) == (box_height, box_width):
        ox, oy = x, y
    else:
        ox, oy = 0, 0
    points = {
        (ox + index % width, oy + index // width)
        for index, value in enumerate(data)
        if value
    }
    if len(points) != detection["mask_nonzero"]:
        raise ValueError("mask_nonzero does not match PGM data")
    fill_ratio = len(points) / (width * height)
    if not points or fill_ratio >= 0.98:
        raise ValueError(f"empty or solid garbage mask: {base / detection['mask_file']}")
    return points


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


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)]


def timing_summary(document):
    return timing_summary_many([document])


def timing_summary_many(documents):
    result = {}
    for key in ("preprocess", "infer", "postprocess", "total"):
        values = [sample[key] for document in documents for sample in document["samples_ms"]]
        result[key] = {
            "mean_ms": statistics.fmean(values),
            "median_ms": statistics.median(values),
            "p95_ms": percentile(values, 0.95),
        }
    return result


def compare(reference_doc, candidate_doc, reference_base, candidate_base,
            min_box_iou, min_mask_iou, max_score_delta):
    reference = canonical(reference_doc["detections"])
    candidate = canonical(candidate_doc["detections"])
    if len(reference) != len(candidate):
        raise ValueError(f"canonical detection count mismatch: {len(reference)} != {len(candidate)}")
    remaining = set(range(len(candidate)))
    matches = []
    for ref in reference:
        choices = [index for index in remaining if candidate[index]["class_id"] == ref["class_id"]]
        if not choices:
            raise ValueError(f"missing class {ref['class_id']}")
        index = max(choices, key=lambda item: bbox_iou(ref["bbox"], candidate[item]["bbox"]))
        cand = candidate[index]
        remaining.remove(index)
        box = bbox_iou(ref["bbox"], cand["bbox"])
        score_delta = abs(ref["score"] - cand["score"])
        ref_mask = mask_pixels(reference_base, ref)
        cand_mask = mask_pixels(candidate_base, cand)
        mask = len(ref_mask & cand_mask) / len(ref_mask | cand_mask)
        if box < min_box_iou or mask < min_mask_iou or score_delta > max_score_delta:
            raise ValueError(
                f"semantic mismatch class={ref['class_id']} box_iou={box:.6f} "
                f"mask_iou={mask:.6f} score_delta={score_delta:.6f}"
            )
        matches.append({"class_id": ref["class_id"], "box_iou": box,
                        "mask_iou": mask, "score_delta": score_delta})
    return matches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {"schema_version": 1, "model_family": "yolo26m-seg", "fixtures": {}}
    all_documents = {name: [] for name in PATHS}
    for fixture_dir in sorted(path for path in args.results_dir.iterdir() if path.is_dir()):
        documents = {}
        for path_name in PATHS:
            json_path = fixture_dir / f"{path_name}.json"
            documents[path_name] = json.loads(json_path.read_text())
            if documents[path_name].get("model_family") != "yolo26m-seg":
                raise ValueError(f"wrong model family in {json_path}")
            all_documents[path_name].append(documents[path_name])
            for detection in documents[path_name]["detections"]:
                mask_pixels(fixture_dir, detection)
        fixture = {"timings": {name: timing_summary(documents[name]) for name in PATHS}}
        fixture["cpu_vs_dali_pre"] = compare(
            documents["cpu_pre_cpu_post"], documents["gpu_pre_cpu_post"],
            fixture_dir, fixture_dir, 0.95, 0.90, 0.30)
        fixture["cpu_post_vs_dali_post"] = compare(
            documents["gpu_pre_cpu_post"], documents["gpu_pre_gpu_post"],
            fixture_dir, fixture_dir, 1.0, 1.0, 1e-6)
        fixture["status"] = "pass"
        report["fixtures"][fixture_dir.name] = fixture
    report["aggregate"] = {
        name: timing_summary_many(all_documents[name]) for name in PATHS
    }
    cpu_total = report["aggregate"]["cpu_pre_cpu_post"]["total"]
    dali_total = report["aggregate"]["gpu_pre_gpu_post"]["total"]
    gpu_cpu_total = report["aggregate"]["gpu_pre_cpu_post"]["total"]
    report["speedup"] = {
        "dali_gpu_pre_post_vs_cpu_pre_post_median":
            cpu_total["median_ms"] / dali_total["median_ms"],
        "dali_gpu_pre_post_vs_gpu_pre_cpu_post_median":
            gpu_cpu_total["median_ms"] / dali_total["median_ms"],
    }
    report["status"] = "pass"
    output = args.output or args.results_dir / "summary.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
