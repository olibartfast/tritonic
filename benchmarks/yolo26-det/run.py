#!/usr/bin/env python3
"""Run YOLO26 detection benchmark: 3 paths, 4 fixtures, aggregate timings."""

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
        "model": "yolo26",
        "input_mode": "preprocessed",
        "postprocess_mode": "cpu",
        "task_model": None,
    },
    "gpu_pre_cpu_post": {
        "model": "yolo26det_gpu_pre_cpu_post",
        "input_mode": "encoded-image",
        "postprocess_mode": "cpu",
        "task_model": "yolo26",
    },
    "gpu_pre_gpu_post": {
        "model": "yolo26det_gpu_pre_gpu_post",
        "input_mode": "encoded-image",
        "postprocess_mode": "gpu",
        "task_model": "yolo26",
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


def run_benchmark(args, path_label, model, input_mode, postprocess_mode, task_model, fixture, output):
    cmd = [
        str(TRITONIC),
        f"--source={fixture}",
        f"--model_type={args.model_type}",
        f"--model={model}",
        f"--labelsFile={args.labels}",
        f"--protocol={args.protocol}",
        f"--serverAddress={args.server}",
        f"--port={args.port}",
        f"--input_mode={input_mode}",
        f"--postprocess_mode={postprocess_mode}",
        f"--benchmark_warmup={args.warmup}",
        f"--benchmark_iterations={args.iterations}",
        f"--benchmark_output={str(output)}",
        "--write_frame=false",
    ]
    if task_model:
        cmd.insert(5, f"--task_model={task_model}")
    print(f"  [{path_label}] {Path(fixture).name} ...", end=" ", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        print("FAILED")
        print("STDERR:", result.stderr)
        sys.exit(1)
    print("OK")
    return json.loads(output.read_text())


def main():
    parser = argparse.ArgumentParser(description="YOLO26 detection benchmark")
    parser.add_argument("--model-type", default="yolo", help="Model type tag")
    parser.add_argument("--labels", default="labels/coco.txt", help="Labels file")
    parser.add_argument("--protocol", default="grpc", choices=["http", "grpc"])
    parser.add_argument("--server", default="localhost")
    parser.add_argument("--port", default="8001")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--output-dir", default=None, help="Output directory")
    args = parser.parse_args()

    if not TRITONIC.exists():
        sys.exit(f"tritonic not found at {TRITONIC}")

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    label = f"{timestamp}_yolo26-det_rtx3060"
    if args.output_dir:
        out = Path(args.output_dir)
    else:
        out = REPO_ROOT / "benchmarks" / "yolo26-det" / "results" / label
    out.mkdir(parents=True, exist_ok=True)

    print(f"Output: {out}")
    print(f"Warmup: {args.warmup}, Iterations: {args.iterations}")
    print()

    summary = {"schema_version": 3, "model_family": "yolo26-det", "fixtures": {}}
    for fixture in FIXTURES:
        fixture_name = Path(fixture).stem
        fixture_dir = out / fixture_name
        fixture_dir.mkdir(parents=True, exist_ok=True)
        fixture_summary = {}
        for path_label, params in PATHS.items():
            output_file = fixture_dir / f"{path_label}.json"
            data = run_benchmark(
                args,
                path_label,
                params["model"],
                params["input_mode"],
                params["postprocess_mode"],
                params["task_model"],
                fixture,
                output_file,
            )
            fixture_summary[path_label] = aggregate(data["samples_ms"])
        summary["fixtures"][fixture_name] = {"timings": fixture_summary}

    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nSummary: {summary_path}")

    overview = aggregate_global(summary)
    print_overview(overview)


def aggregate_global(summary):
    paths = ("cpu_pre_cpu_post", "gpu_pre_cpu_post", "gpu_pre_gpu_post")
    stages = ("preprocess", "infer", "postprocess", "total")
    result = {}
    for path in paths:
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
    header = f"{'Path':<28} {'Pre median':>10} {'Infer median':>12} {'Post median':>11} {'Total median':>12} {'Total p95':>9}"
    print(header)
    print("-" * len(header))
    for label in ("cpu_pre_cpu_post", "gpu_pre_cpu_post", "gpu_pre_gpu_post"):
        p = agg[label]
        print(
            f"{label:<28}"
            f" {p['preprocess']:>9.2f} ms"
            f" {p['infer']:>11.2f} ms"
            f" {p['postprocess']:>10.2f} ms"
            f" {p['total']:>11.2f} ms"
        )
    gpu_pre_post = agg["gpu_pre_gpu_post"]["total"]
    cpu_pre_post = agg["cpu_pre_cpu_post"]["total"]
    gpu_pre_cpu_post = agg["gpu_pre_cpu_post"]["total"]
    if cpu_pre_post > 0:
        print(f"\nDALI pre/post is {cpu_pre_post / gpu_pre_post:.2f}x faster than CPU pre/post")
        print(
            f"DALI pre/cpu post is "
            f"{gpu_pre_cpu_post / gpu_pre_post:.2f}x vs DALI pre/post"
        )


if __name__ == "__main__":
    main()
