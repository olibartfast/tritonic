#!/usr/bin/env python3
"""Validate Tritonic YOLO11-seg polygon semantics and summarize timings."""

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
    for px, py in ring:
        if not (x <= px <= x + width and y <= py <= y + height):
            raise ValueError(
                f"{name} has a point ({px},{py}) outside its bounding box "
                f"[{x},{y},{width},{height}]"
            )
    area = signed_area(ring)
    if area == 0:
        raise ValueError(f"{name} has zero area (collinear points)")
    if (area < 0) != (expected_sign < 0):
        raise ValueError(
            f"{name} has wrong winding (area {area:.1f}, expected sign {expected_sign})"
        )


def polygon_iou(rings_a, rings_b, image_shape):
    h, w = int(image_shape[0]), int(image_shape[1])
    canvas_a = np.zeros((h, w), dtype=np.uint8)
    canvas_b = np.zeros((h, w), dtype=np.uint8)
    for ring in rings_a:
        if len(ring) >= 3:
            pts = np.array(ring, dtype=np.int32)
            cv2.fillPoly(canvas_a, [pts], 255)
    for ring in rings_b:
        if len(ring) >= 3:
            pts = np.array(ring, dtype=np.int32)
            cv2.fillPoly(canvas_b, [pts], 255)
    intersection = np.sum(np.logical_and(canvas_a, canvas_b))
    union = np.sum(np.logical_or(canvas_a, canvas_b))
    return intersection / union if union else 0.0


def extract_polygons(det):
    image_w, image_h = det.get("image_width", 0), det.get("image_height", 0)
    rings = []
    bbox = det["box"]
    ring_offsets = det.get("instance_ring_offsets", [])
    ring_point_offsets = det.get("ring_point_offsets", [])
    polygon_points = det.get("polygon_points", [])
    if not ring_offsets or not ring_point_offsets:
        return rings, (image_h, image_w)
    for ri in range(len(ring_offsets) - 1):
        ring_start = ring_offsets[ri]
        ring_end = ring_offsets[ri + 1]
        instance = []
        for rj in range(int(ring_start), int(ring_end)):
            ps = int(ring_point_offsets[rj])
            pe = int(ring_point_offsets[rj + 1])
            ring = [tuple(polygon_points[k]) for k in range(ps, pe)]
            instance.append(ring)
        rings.append(instance)
    return rings, (image_h, image_w)


def validate_detection(ref, det, image_shape):
    if det.get("class_id", -1) != ref.get("class_id", -2):
        return False
    ref_box = ref["box"]
    det_box = det["box"]
    iou = bbox_iou(ref_box, det_box)
    if iou < 0.95:
        return False
    return True


def pct(values, p):
    k = (len(values) - 1) * p
    f = int(k)
    c = k - f
    return values[f] * (1 - c) + values[f + 1] * c if f + 1 < len(values) else values[f]


def aggregate(samples):
    pre = sorted(s["preprocess"] for s in samples)
    inf = sorted(s["infer"] for s in samples)
    post = sorted(s["postprocess"] for s in samples)
    total = sorted(s["total"] for s in samples)
    return {
        "preprocess": {"mean": statistics.mean(pre), "median": statistics.median(pre), "p95": pct(pre, 0.95)},
        "infer": {"mean": statistics.mean(inf), "median": statistics.median(inf), "p95": pct(inf, 0.95)},
        "postprocess": {"mean": statistics.mean(post), "median": statistics.median(post), "p95": pct(post, 0.95)},
        "total": {"mean": statistics.mean(total), "median": statistics.median(total), "p95": pct(total, 0.95)},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", help="Benchmark results directory (e.g. results/2026-07-31_yolo11m-seg_rtx3060)")
    parser.add_argument("--reference", help="Reference results directory for validation", default=None)
    args = parser.parse_args()
    results_dir = Path(args.results_dir)

    summary = {"model_family": "yolo11-seg", "fixtures": {}}
    for fixture_dir in sorted(results_dir.iterdir()):
        if not fixture_dir.is_dir():
            continue
        fixture = fixture_dir.name
        fixture_data = {}
        for path_label in PATHS:
            json_file = fixture_dir / f"{path_label}.json"
            if not json_file.exists():
                continue
            data = json.loads(json_file.read_text())
            samples = data.get("samples_ms", [])
            timings = aggregate(samples)
            fixture_data[path_label] = {
                "timings": timings,
                "detection_count": len(data.get("detections", [])),
            }
            print(f"  {fixture}/{path_label}: "
                  f"total={timings['total']['median']:.1f}ms, "
                  f"detections={fixture_data[path_label]['detection_count']}")
        if fixture_data:
            summary["fixtures"][fixture] = fixture_data

    out_path = results_dir / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nSummary written to {out_path}")

    # Print overview
    print()
    header = f"{'Path':<28} {'Pre':>8} {'Infer':>8} {'Post':>8} {'Total':>8}"
    print(header)
    print("-" * len(header))
    agg = {}
    for path in PATHS:
        medians = []
        for f in summary["fixtures"]:
            if path in summary["fixtures"][f]:
                medians.append(summary["fixtures"][f][path]["timings"]["total"]["median"])
        if medians:
            agg[path] = statistics.mean(medians)
            p = {}
            for stage in ("preprocess", "infer", "postprocess", "total"):
                p[stage] = statistics.mean(
                    summary["fixtures"][fix][path]["timings"][stage]["median"]
                    for fix in summary["fixtures"]
                    if path in summary["fixtures"][fix]
                )
            print(f"{path:<28} {p['preprocess']:>7.1f}ms {p['infer']:>7.1f}ms "
                  f"{p['postprocess']:>7.1f}ms {p['total']:>8.1f}ms")
    if "cpu_pre_cpu_post" in agg and "gpu_pre_gpu_post" in agg:
        speedup = agg["cpu_pre_cpu_post"] / agg["gpu_pre_gpu_post"]
        print(f"\nDALI pre/post is {speedup:.2f}x faster than CPU pre/post")


if __name__ == "__main__":
    main()
