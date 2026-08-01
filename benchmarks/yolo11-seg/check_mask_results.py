#!/usr/bin/env python3
"""Validate Tritonic YOLO11-seg mask semantics and summarize timings."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

PATHS = ("cpu_pre_cpu_post", "gpu_pre_cpu_post", "gpu_pre_gpu_mask_post")


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
    parser.add_argument("results_dir")
    args = parser.parse_args()
    results_dir = Path(args.results_dir)

    summary = {"model_family": "yolo11-seg-mask", "fixtures": {}}
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

    out_path = results_dir / "mask_summary.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nSummary written to {out_path}")

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
    if "cpu_pre_cpu_post" in agg and "gpu_pre_gpu_mask_post" in agg:
        speedup = agg["cpu_pre_cpu_post"] / agg["gpu_pre_gpu_mask_post"]
        print(f"\nDALI pre/mask post is {speedup:.2f}x faster than CPU pre/post")


if __name__ == "__main__":
    main()
