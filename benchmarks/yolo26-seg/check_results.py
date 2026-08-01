#!/usr/bin/env python3
"""Validate Tritonic YOLO26m-seg polygon semantics and summarize timings."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.agreement import (  # noqa: E402
    bbox_iou,
    compare,
    require_model_family,
    timing_summary,
    timing_summary_many,
)

PATHS = ("cpu_pre_cpu_post", "gpu_pre_cpu_post", "gpu_pre_gpu_post")

# Result documents written before the family naming was corrected say "yolo26m-seg",
# where the m is the model size rather than part of the family.
ACCEPTED_FAMILIES = {"yolo26-seg", "yolo26m-seg"}


def signed_area(ring):
    return 0.5 * sum(
        x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(ring, ring[1:] + ring[:1])
    )


def point_in_ring(point, ring):
    px, py = point
    inside = False
    previous = ring[-1]
    for current in ring:
        ax, ay = current
        bx, by = previous
        if (ay > py) != (by > py) and px < (bx - ax) * (py - ay) / (by - ay) + ax:
            inside = not inside
        previous = current
    return inside


def validate_ring(ring, bbox, expected_sign, name):
    if len(ring) < 3:
        raise ValueError(f"{name} has fewer than three points")
    x, y, width, height = bbox
    for point in ring:
        if len(point) != 2 or not all(math.isfinite(value) for value in point):
            raise ValueError(f"{name} contains an invalid point")
        px, py = point
        if px < x or px > x + width or py < y or py > y + height:
            raise ValueError(f"{name} point lies outside its bounding box")
    area = signed_area(ring)
    if area == 0 or math.copysign(1.0, area) != expected_sign:
        raise ValueError(f"{name} has invalid winding or zero area")
    for index, current in enumerate(ring):
        previous = ring[index - 1]
        following = ring[(index + 1) % len(ring)]
        turn = (current[0] - previous[0]) * (following[1] - current[1]) - (
            current[1] - previous[1]
        ) * (following[0] - current[0])
        if turn * expected_sign < 0:
            raise ValueError(f"{name} is not convex")


def polygon_pixels(detection):
    bbox = detection["bbox"]
    x, y, width, height = bbox
    if width <= 0 or height <= 0:
        raise ValueError("invalid detection bounding box")
    polygons = detection.get("polygons", [])
    if not polygons or detection.get("polygon_count") != len(polygons):
        raise ValueError("missing or inconsistent polygon output")

    actual_point_count = 0
    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon_index, polygon in enumerate(polygons):
        exterior = polygon.get("exterior", [])
        holes = polygon.get("holes", [])
        validate_ring(exterior, bbox, 1.0, f"polygon {polygon_index} exterior")
        actual_point_count += len(exterior)
        for hole_index, hole in enumerate(holes):
            validate_ring(
                hole, bbox, -1.0, f"polygon {polygon_index} hole {hole_index}"
            )
            if not point_in_ring(hole[0], exterior):
                raise ValueError("polygon hole is not contained by its exterior")
            actual_point_count += len(hole)

        exterior_points = np.asarray(
            [[point[0] - x, point[1] - y] for point in exterior], dtype=np.int32
        )
        cv2.fillPoly(mask, [exterior_points], 1)
        for hole in holes:
            hole_points = np.asarray(
                [[point[0] - x, point[1] - y] for point in hole], dtype=np.int32
            )
            cv2.fillPoly(mask, [hole_points], 0)

    if detection.get("polygon_point_count") != actual_point_count:
        raise ValueError("polygon_point_count does not match polygon data")
    filled = int(np.count_nonzero(mask))
    fill_ratio = filled / (width * height)
    if not filled or fill_ratio >= 0.98:
        raise ValueError("empty or near-solid garbage polygon output")
    ys, xs = np.nonzero(mask)
    return set(zip(xs + x, ys + y))


def polygon_overlap(ref, cand):
    ref_pixels = polygon_pixels(ref)
    cand_pixels = polygon_pixels(cand)
    return len(ref_pixels & cand_pixels) / len(ref_pixels | cand_pixels)


def compare_polygons(reference_doc, candidate_doc, min_box_iou, min_polygon_iou,
                     max_score_delta):
    return compare(
        reference_doc,
        candidate_doc,
        min_box_iou,
        max_score_delta,
        overlap=polygon_overlap,
        min_overlap=min_polygon_iou,
        overlap_name="polygon_iou",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {"schema_version": 2, "model_family": "yolo26m-seg", "fixtures": {}}
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
        for path_name in PATHS:
            json_path = fixture_dir / f"{path_name}.json"
            documents[path_name] = json.loads(json_path.read_text())
            require_model_family(documents[path_name], ACCEPTED_FAMILIES, json_path)
            if documents[path_name].get("segmentation_output") != "polygon":
                raise ValueError(f"non-polygon benchmark output in {json_path}")
            all_documents[path_name].append(documents[path_name])
            for detection in documents[path_name]["detections"]:
                polygon_pixels(detection)
        fixture = {"timings": {name: timing_summary(documents[name]) for name in PATHS}}
        fixture["cpu_vs_dali_pre"] = compare_polygons(
            documents["cpu_pre_cpu_post"],
            documents["gpu_pre_cpu_post"],
            0.95,
            0.90,
            0.30,
        )
        fixture["cpu_post_vs_dali_post"] = compare_polygons(
            documents["gpu_pre_cpu_post"], documents["gpu_pre_gpu_post"], 1.0, 1.0, 1e-6
        )
        fixture["status"] = "pass"
        report["fixtures"][fixture_dir.name] = fixture
    report["aggregate"] = {
        name: timing_summary_many(all_documents[name]) for name in PATHS
    }
    cpu_total = report["aggregate"]["cpu_pre_cpu_post"]["total"]
    dali_total = report["aggregate"]["gpu_pre_gpu_post"]["total"]
    gpu_cpu_total = report["aggregate"]["gpu_pre_cpu_post"]["total"]
    report["speedup"] = {
        "dali_gpu_pre_post_vs_cpu_pre_post_median": cpu_total["median_ms"]
        / dali_total["median_ms"],
        "dali_gpu_pre_post_vs_gpu_pre_cpu_post_median": gpu_cpu_total["median_ms"]
        / dali_total["median_ms"],
    }
    report["status"] = "pass"
    output = args.output or args.results_dir / "summary.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
