#!/usr/bin/env python3
"""Run the YOLO26 detection preprocessing benchmark (summary schema v4).

Every selected variant (see variants.py) runs on every fixture over every
selected protocol. Client timings come from Tritonic's --benchmark_output.
Server-side cost comes from Triton per-model statistics, read before and after
each run. The environment identity from env_probe.sh is embedded, so results
from different GPUs or images are never silently mixed.

    python3 benchmarks/yolo26-det/run.py --variants B0,B1 --label rtx3060-2608
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from common.fixtures import RESOLUTION_FIXTURES, make_resolution_fixture  # noqa: E402
from variants import select  # noqa: E402

REPO_ROOT = HERE.parents[1]
TRITONIC = REPO_ROOT / "build" / "tritonic"
STOCK_FIXTURES = [
    "data/images/bus.jpg",
    "data/images/horses.jpg",
    "data/images/person.jpg",
    "data/images/mug.jpg",
]
STAGES = ("preprocess", "infer", "postprocess", "total")
SERVER_STAGES = ("queue", "compute_input", "compute_infer", "compute_output")


def pct(values, p):
    k = (len(values) - 1) * p
    f = int(k)
    c = k - f
    if f + 1 < len(values):
        return values[f] * (1 - c) + values[f + 1] * c
    return values[f]


def aggregate(samples):
    result = {}
    for stage in STAGES:
        values = sorted(sample[stage] for sample in samples)
        result[stage] = {
            "mean_ms": statistics.mean(values),
            "median_ms": statistics.median(values),
            "p95_ms": pct(values, 0.95),
            "p99_ms": pct(values, 0.99),
        }
    return result


def model_stats(http_url, model):
    with urllib.request.urlopen(f"{http_url}/v2/models/{model}/stats", timeout=10) as response:
        stats = json.load(response)["model_stats"][0]["inference_stats"]
    return {stage: (int(stats[stage]["count"]), int(stats[stage]["ns"]))
            for stage in ("success", *SERVER_STAGES)}


def server_breakdown(before, after):
    """Per-model mean milliseconds per successful inference between two snapshots."""
    result = {}
    for model, start in before.items():
        end = after[model]
        count = end["success"][0] - start["success"][0]
        result[model] = {"count": count}
        for stage in SERVER_STAGES:
            ns = end[stage][1] - start[stage][1]
            result[model][f"{stage}_ms"] = ns / count / 1e6 if count else None
    return result


def prepare_fixtures(work_dir):
    fixtures = [REPO_ROOT / path for path in STOCK_FIXTURES]
    for name, size in RESOLUTION_FIXTURES.items():
        fixtures.append(
            make_resolution_fixture(REPO_ROOT / "data/images/bus.jpg", work_dir / f"{name}.jpg", size)
        )
    return fixtures


def run_benchmark(args, variant, protocol, fixture, output):
    port = args.http_port if protocol == "http" else args.grpc_port
    cmd = [
        str(args.tritonic),
        f"--source={fixture}",
        f"--model_type={args.model_type}",
        f"--labelsFile={args.labels}",
        f"--protocol={protocol}",
        f"--serverAddress={args.server}",
        f"--port={port}",
        f"--benchmark_warmup={args.warmup}",
        f"--benchmark_iterations={args.iterations}",
        f"--benchmark_output={output}",
        "--write_frame=false",
        *variant.client_args(),
    ]
    print(f"  [{protocol} {variant.key}] {Path(fixture).name} ...", end=" ", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        print("FAILED")
        return {"error": (result.stdout + result.stderr)[-2000:]}
    print("OK")
    return json.loads(Path(output).read_text())


def environment(path):
    probe = HERE / "env_probe.sh"
    completed = subprocess.run([str(probe), str(path)], capture_output=True, text=True)
    if completed.returncode != 0:
        sys.exit(f"env_probe.sh failed:\n{completed.stderr}")
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description="YOLO26 detection preprocessing benchmark")
    parser.add_argument("--variants", default=None, help="comma-separated keys from variants.py")
    parser.add_argument("--protocols", default="grpc,http")
    parser.add_argument("--label", required=True, help="hardware/image label, e.g. rtx3060-2608")
    parser.add_argument("--model-type", default="yolo26")
    parser.add_argument("--labels", default="labels/coco.txt")
    parser.add_argument("--server", default="localhost")
    parser.add_argument("--http-port", type=int, default=8000)
    parser.add_argument("--grpc-port", type=int, default=8001)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--tritonic", type=Path, default=TRITONIC)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--environment", type=Path, default=None,
                        help="env_probe.sh JSON collected beforehand (e.g. on the Docker host)")
    args = parser.parse_args()

    if not args.tritonic.exists():
        sys.exit(f"tritonic not found at {args.tritonic}")
    variants = select(args.variants)
    protocols = [protocol.strip() for protocol in args.protocols.split(",")]
    http_url = f"http://{args.server}:{args.http_port}"

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out = args.output_dir or HERE / "results" / f"{timestamp}_yolo26-det_{args.label}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out}\nWarmup: {args.warmup}, Iterations: {args.iterations}\n")

    summary = {
        "schema_version": 4,
        "model_family": "yolo26-det",
        "label": args.label,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": json.loads(args.environment.read_text()) if args.environment
        else environment(out / "environment.json"),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "variants": {variant.key: {"title": variant.title, "model": variant.model,
                                   "payload": variant.payload} for variant in variants},
        "protocols": {},
    }

    with tempfile.TemporaryDirectory() as work:
        fixtures = prepare_fixtures(Path(work))
        for protocol in protocols:
            protocol_summary = {}
            for fixture in fixtures:
                fixture_name = fixture.stem
                fixture_dir = out / protocol / fixture_name
                fixture_dir.mkdir(parents=True, exist_ok=True)
                jpeg_bytes = fixture.stat().st_size
                entries = {}
                for variant in variants:
                    before = {model: model_stats(http_url, model) for model in variant.server_models}
                    data = run_benchmark(
                        args, variant, protocol, fixture, fixture_dir / f"{variant.key}.json"
                    )
                    after = {model: model_stats(http_url, model) for model in variant.server_models}
                    entry = {"request_bytes": variant.request_bytes(jpeg_bytes)}
                    if "error" in data:
                        entry["error"] = data["error"]
                    else:
                        entry["timings"] = aggregate(data["samples_ms"])
                        entry["server"] = server_breakdown(before, after)
                    entries[variant.key] = entry
                protocol_summary[fixture_name] = entries
            summary["protocols"][protocol] = protocol_summary

    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nSummary: {summary_path}")
    print_overview(summary)


def print_overview(summary):
    for protocol, fixtures in summary["protocols"].items():
        print(f"\n[{protocol}] mean of per-fixture medians")
        header = f"{'Variant':<14} {'Pre':>9} {'Infer':>9} {'Post':>9} {'Total':>9} {'Total p95':>10}"
        print(header)
        print("-" * len(header))
        for key in summary["variants"]:
            runs = [fixture[key] for fixture in fixtures.values() if "timings" in fixture[key]]
            if not runs:
                print(f"{key:<14} FAILED")
                continue
            values = {stage: statistics.mean(run["timings"][stage]["median_ms"] for run in runs)
                      for stage in STAGES}
            p95 = statistics.mean(run["timings"]["total"]["p95_ms"] for run in runs)
            print(f"{key:<14} {values['preprocess']:>7.2f}ms {values['infer']:>7.2f}ms "
                  f"{values['postprocess']:>7.2f}ms {values['total']:>7.2f}ms {p95:>8.2f}ms")


if __name__ == "__main__":
    main()
