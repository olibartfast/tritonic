#!/usr/bin/env python3
"""Run YOLO11-seg benchmark: 4 paths, 4 fixtures, aggregate timings."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

FIXTURES = [
    "data/images/bus.jpg",
    "data/images/horses.jpg",
    "data/images/person.jpg",
    "data/images/mug.jpg",
]

PATHS = {
    "cpu_pre_cpu_post": {
        "model": "yolo11seg_trt",
        "model_type": "yolo11seg",
        "input_mode": "preprocessed",
        "postprocess_mode": "cpu",
        "task_model": None,
        "segmentation_output": None,
    },
    "gpu_pre_cpu_post": {
        "model": "yolo11seg_gpu_pre_cpu_post",
        "model_type": "yolo11seg",
        "input_mode": "encoded-image",
        "postprocess_mode": "cpu",
        "task_model": "yolo11seg_trt",
        "segmentation_output": None,
    },
    "gpu_pre_gpu_mask_post": {
        "model": "yolo11seg_gpu_pre_gpu_mask_post",
        "model_type": "yolo11seg",
        "input_mode": "encoded-image",
        "postprocess_mode": "gpu",
        "task_model": "yolo11seg_trt",
        "segmentation_output": "mask",
    },
    "gpu_pre_gpu_post": {
        "model": "yolo11seg_gpu_pre_gpu_post",
        "model_type": "yolo11seg",
        "input_mode": "encoded-image",
        "postprocess_mode": "gpu",
        "task_model": "yolo11seg_trt",
        "segmentation_output": "polygon",
    },
}
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TRITONIC = REPO_ROOT / "build" / "tritonic"


def pct(values, p):
    k = (len(values) - 1) * p
    f = int(k)
    c = k - f
    if f + 1 < len(values):
        return values[f] * (1 - c) + values[f + 1] * c
    return values[f]


def aggregate(samples):
    pre = sorted(s["preprocess"] for s in samples)
    inf = sorted(s["infer"] for s in samples)
    post = sorted(s["postprocess"] for s in samples)
    total = sorted(s["total"] for s in samples)
    return {
        "preprocess": {
            "mean_ms": statistics.mean(pre),
            "median_ms": statistics.median(pre),
            "p95_ms": pct(pre, 0.95),
        },
        "infer": {
            "mean_ms": statistics.mean(inf),
            "median_ms": statistics.median(inf),
            "p95_ms": pct(inf, 0.95),
        },
        "postprocess": {
            "mean_ms": statistics.mean(post),
            "median_ms": statistics.median(post),
            "p95_ms": pct(post, 0.95),
        },
        "total": {
            "mean_ms": statistics.mean(total),
            "median_ms": statistics.median(total),
            "p95_ms": pct(total, 0.95),
        },
    }


def run_benchmark(args, path_label, params, fixture, output):
    cmd = [
        str(TRITONIC),
        f"--source={fixture}",
        f"--model_type={params['model_type']}",
        f"--model={params['model']}",
        f"--labelsFile={args.labels}",
        f"--protocol={args.protocol}",
        f"--serverAddress={args.server}",
        f"--port={args.port}",
        f"--input_mode={params['input_mode']}",
        f"--postprocess_mode={params['postprocess_mode']}",
        f"--benchmark_warmup={args.warmup}",
        f"--benchmark_iterations={args.iterations}",
        f"--benchmark_output={str(output)}",
        "--write_frame=false",
    ]
    if params["task_model"]:
        cmd.insert(5, f"--task_model={params['task_model']}")
    if params["segmentation_output"]:
        cmd.append(f"--segmentation_output={params['segmentation_output']}")
    print(f"  [{path_label}] {Path(fixture).name} ...", end=" ", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        print("FAILED")
        print("STDERR:", result.stderr)
        sys.exit(1)
    print("OK")
    return json.loads(output.read_text())


def main():
    parser = argparse.ArgumentParser(description="YOLO11-seg benchmark")
    parser.add_argument("--labels", default="labels/coco.txt")
    parser.add_argument("--protocol", default="grpc", choices=["http", "grpc"])
    parser.add_argument("--server", default="localhost")
    parser.add_argument("--port", default="8001")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    if not TRITONIC.exists():
        sys.exit(f"tritonic not found at {TRITONIC}")

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    label = f"{timestamp}_yolo11m-seg_mask_rtx3060"
    if args.output_dir:
        out = Path(args.output_dir)
    else:
        out = REPO_ROOT / "benchmarks" / "yolo11-seg" / "results" / label
    out.mkdir(parents=True, exist_ok=True)

    print(f"Output: {out}")
    print(f"Warmup: {args.warmup}, Iterations: {args.iterations}")
    print()

    summary = {"schema_version": 3, "model_family": "yolo11-seg", "fixtures": {}}
    for fixture in FIXTURES:
        fixture_name = Path(fixture).stem
        fixture_dir = out / fixture_name
        fixture_dir.mkdir(parents=True, exist_ok=True)
        fixture_summary = {}
        for path_label, params in PATHS.items():
            output_file = fixture_dir / f"{path_label}.json"
            data = run_benchmark(args, path_label, params, fixture, output_file)
            fixture_summary[path_label] = aggregate(data["samples_ms"])
        summary["fixtures"][fixture_name] = {"timings": fixture_summary}

    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nSummary: {summary_path}")

    overview = aggregate_global(summary)
    print_overview(overview)


def aggregate_global(summary):
    paths = ("cpu_pre_cpu_post", "gpu_pre_cpu_post", "gpu_pre_gpu_mask_post", "gpu_pre_gpu_post")
    stages = ("preprocess", "infer", "postprocess", "total")
    result = {}
    for path in paths:
        if not any(path in summary["fixtures"][f]["timings"] for f in summary["fixtures"]):
            continue
        result[path] = {}
        for stage in stages:
            medians = [
                summary["fixtures"][f]["timings"][path][stage]["median_ms"]
                for f in summary["fixtures"]
            ]
            result[path][stage] = statistics.mean(medians)
    return result


def print_overview(agg):
    print()
    header = f"{'Path':<30} {'Pre median':>10} {'Infer median':>12} {'Post median':>11} {'Total median':>12}"
    print(header)
    print("-" * len(header))
    for label in ("cpu_pre_cpu_post", "gpu_pre_cpu_post", "gpu_pre_gpu_mask_post", "gpu_pre_gpu_post"):
        if label not in agg:
            continue
        p = agg[label]
        print(
            f"{label:<30}"
            f" {p['preprocess']:>9.2f} ms"
            f" {p['infer']:>11.2f} ms"
            f" {p['postprocess']:>10.2f} ms"
            f" {p['total']:>11.2f} ms"
        )
    if "cpu_pre_cpu_post" in agg and "gpu_pre_gpu_mask_post" in agg:
        print(f"\nDALI pre/mask post is {agg['cpu_pre_cpu_post']['total'] / agg['gpu_pre_gpu_mask_post']['total']:.2f}x vs CPU")
    if "cpu_pre_cpu_post" in agg and "gpu_pre_gpu_post" in agg:
        print(f"DALI pre/polygon post is {agg['cpu_pre_cpu_post']['total'] / agg['gpu_pre_gpu_post']['total']:.2f}x vs CPU")


if __name__ == "__main__":
    main()
