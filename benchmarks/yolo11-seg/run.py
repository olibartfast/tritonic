#!/usr/bin/env python3
"""Run the YOLO11-seg benchmark: 3 paths x 4 fixtures for one segmentation mode.

Pass --segmentation-output to pick the mode; run it twice (mask and polygon) to
cover both, then validate each result directory with the matching checker.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.fixtures import make_crowded_fixture  # noqa: E402

# The ensemble is size-agnostic -- every YOLO11-seg scale shares the same tensor
# contract -- but a benchmark is meaningless without knowing which engine produced
# it, so the label comes from the engine that setup_model_repository.sh staged
# rather than being hardcoded.
DEFAULT_MANIFEST = (
    Path(__file__).resolve().parents[2]
    / "deploy/instance_segmentation/yolo11/ensemble/model_repository/reference_model.yaml"
)


def deployed_engine(manifest):
    """Return the staged engine's stem, e.g. 'yolo11m-seg', or None if unknown."""
    try:
        for line in Path(manifest).read_text().splitlines():
            if line.startswith("engine_file:"):
                return Path(line.split(":", 1)[1].strip()).stem
    except OSError:
        pass
    return None

FIXTURES = [
    "data/images/bus.jpg",
    "data/images/horses.jpg",
    "data/images/person.jpg",
    "data/images/mug.jpg",
]

# One campaign produces one segmentation mode across all three paths. Mixing modes
# within a run leaves the GPU polygon path without a polygon CPU reference, so the
# checkers cannot compare it against anything -- which is how a truncating GPU path
# previously passed as a pure timing win.
def build_paths(segmentation_output):
    gpu_post_model = (
        "yolo11seg_gpu_pre_gpu_post"
        if segmentation_output == "polygon"
        else "yolo11seg_gpu_pre_gpu_mask_post"
    )
    return {
        "cpu_pre_cpu_post": {
            "model": "yolo11seg_trt",
            "model_type": "yolo11seg",
            "input_mode": "preprocessed",
            "postprocess_mode": "cpu",
            "task_model": None,
            "segmentation_output": segmentation_output,
        },
        "gpu_pre_cpu_post": {
            "model": "yolo11seg_gpu_pre_cpu_post",
            "model_type": "yolo11seg",
            "input_mode": "encoded-image",
            "postprocess_mode": "cpu",
            "task_model": "yolo11seg_trt",
            "segmentation_output": segmentation_output,
        },
        "gpu_pre_gpu_post": {
            "model": gpu_post_model,
            "model_type": "yolo11seg",
            "input_mode": "encoded-image",
            "postprocess_mode": "gpu",
            "task_model": "yolo11seg_trt",
            "segmentation_output": segmentation_output,
        },
    }


PATH_ORDER = ("cpu_pre_cpu_post", "gpu_pre_cpu_post", "gpu_pre_gpu_post")
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
    parser.add_argument(
        "--segmentation-output", default="mask", choices=["mask", "polygon"]
    )
    parser.add_argument(
        "--engine-label",
        default=None,
        help="Engine identity recorded in results; defaults to the staged engine",
    )
    parser.add_argument(
        "--no-crowded",
        action="store_true",
        help="skip the synthetic dense fixture that exercises the detection cap",
    )
    args = parser.parse_args()
    paths = build_paths(args.segmentation_output)

    if not TRITONIC.exists():
        sys.exit(f"tritonic not found at {TRITONIC}")

    engine_label = args.engine_label or deployed_engine(DEFAULT_MANIFEST)
    if engine_label is None:
        sys.exit(
            "could not determine which engine is deployed; pass --engine-label "
            f"(looked for engine_file in {DEFAULT_MANIFEST})"
        )
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    label = f"{timestamp}_{engine_label}_{args.segmentation_output}_rtx3060"
    if args.output_dir:
        out = Path(args.output_dir)
    else:
        out = REPO_ROOT / "benchmarks" / "yolo11-seg" / "results" / label
    out.mkdir(parents=True, exist_ok=True)

    print(f"Output: {out}")
    print(f"Warmup: {args.warmup}, Iterations: {args.iterations}")
    print()

    fixtures = list(FIXTURES)
    if not args.no_crowded:
        # Generated into the results dir so it is reproducible but never committed.
        fixtures.append(
            str(make_crowded_fixture("data/images/bus.jpg", out / "crowd.jpg"))
        )

    summary = {
        "schema_version": 3,
        "model_family": "yolo11-seg",
        "engine": engine_label,
        "fixtures": {},
    }
    for fixture in fixtures:
        fixture_name = Path(fixture).stem
        fixture_dir = out / fixture_name
        fixture_dir.mkdir(parents=True, exist_ok=True)
        fixture_summary = {}
        for path_label, params in paths.items():
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
    paths = PATH_ORDER
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
    for label in PATH_ORDER:
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
    if "cpu_pre_cpu_post" in agg and "gpu_pre_gpu_post" in agg:
        speedup = agg["cpu_pre_cpu_post"]["total"] / agg["gpu_pre_gpu_post"]["total"]
        print(f"\nDALI GPU pre/post is {speedup:.2f}x vs CPU pre/post")


if __name__ == "__main__":
    main()
