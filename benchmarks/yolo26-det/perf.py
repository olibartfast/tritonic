#!/usr/bin/env python3
"""Server throughput per preprocessing variant with perf_analyzer (spec V-2, plan T-6).

perf_analyzer runs from the Triton SDK image on the host network, at each
concurrency level, with the 1080p fixture as real JPEG input. Triton's metrics
endpoint is sampled during each run for GPU utilization and memory.

perf_analyzer measures the server and transport only. V2 is skipped because its
preprocessing happens in the client, which perf_analyzer bypasses; its
throughput comes from the client benchmark (run.py) instead.

    python3 benchmarks/yolo26-det/perf.py --variants B0,B1 --label rtx3060-2608
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from common.fixtures import RESOLUTION_FIXTURES, make_resolution_fixture  # noqa: E402
from variants import PREPROCESSED_BYTES, select  # noqa: E402

REPO_ROOT = HERE.parents[1]

def _ngc_tag():
    """Read NGC_CONTAINER_TAG from environment (if set) or versions.env."""
    if "NGC_CONTAINER_TAG" in os.environ:
        return os.environ["NGC_CONTAINER_TAG"]
    versions_env = REPO_ROOT / "versions.env"
    if not versions_env.exists():
        raise RuntimeError(f"versions.env not found at {versions_env}")
    with open(versions_env) as f:
        for line in f:
            match = re.match(r'^NGC_CONTAINER_TAG=(.+)$', line.strip())
            if match:
                return match.group(1)
    raise RuntimeError("NGC_CONTAINER_TAG not found in versions.env")

SDK_IMAGE = os.environ.get("TRITON_SDK_IMAGE") or f"nvcr.io/nvidia/tritonserver:{_ngc_tag()}-py3-sdk"
METRIC = re.compile(r'^(nv_gpu_utilization|nv_gpu_memory_used_bytes)\{[^}]*\} ([0-9.e+]+)$')


class MetricsSampler(threading.Thread):
    def __init__(self, url, interval=0.25):
        super().__init__(daemon=True)
        self.url, self.interval = url, interval
        self.samples = {"nv_gpu_utilization": [], "nv_gpu_memory_used_bytes": []}
        self.stop_event = threading.Event()

    def run(self):
        while not self.stop_event.is_set():
            try:
                with urllib.request.urlopen(self.url, timeout=2) as response:
                    totals = {}
                    for line in response.read().decode().splitlines():
                        match = METRIC.match(line)
                        if match:
                            totals[match[1]] = totals.get(match[1], 0.0) + float(match[2])
                    for name, value in totals.items():
                        self.samples[name].append(value)
            except OSError:
                pass
            self.stop_event.wait(self.interval)

    def summary(self):
        util = self.samples["nv_gpu_utilization"]
        memory = self.samples["nv_gpu_memory_used_bytes"]
        return {
            "gpu_utilization_mean": sum(util) / len(util) if util else None,
            "gpu_memory_peak_mib": max(memory) / 2**20 if memory else None,
            "samples": len(util),
        }


def input_data(variant, jpeg: Path, work: Path) -> Path:
    path = work / f"{variant.key}.json"
    if variant.payload == "encoded":
        raw = jpeg.read_bytes()
        entry = {"IMAGE": {"b64": base64.b64encode(raw).decode(), "shape": [len(raw)]}}
    else:
        entry = {"images": {"b64": base64.b64encode(bytes(PREPROCESSED_BYTES)).decode(),
                            "shape": [3, 640, 640]}}
    path.write_text(json.dumps({"data": [entry]}))
    return path


def perf_analyzer(args, variant, data: Path, concurrency: int, work: Path):
    report = work / f"{variant.key}_c{concurrency}.csv"
    # Inside the SDK image perf_analyzer is on PATH; elsewhere run it from that image.
    prefix = [] if shutil.which("perf_analyzer") else [
        "docker", "run", "--rm", "--network", "host", "-v", f"{work}:{work}", args.sdk_image]
    command = [
        *prefix,
        "perf_analyzer",
        "-m", variant.model,
        "-u", f"{args.server}:{args.grpc_port}", "-i", "grpc",
        "--input-data", str(data),
        "--concurrency-range", f"{concurrency}:{concurrency}",
        "--measurement-interval", str(args.measurement_ms),
        "--percentile", "95",
        "-f", str(report),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    # perf_analyzer can exit 0 without a report, e.g. when no measurement window is stable.
    if completed.returncode != 0 or not report.exists():
        return {"error": (completed.stdout + completed.stderr)[-2000:]}
    with report.open() as stream:
        row = next(csv.DictReader(stream))
    return {
        "inferences_per_second": float(row["Inferences/Second"]),
        "p50_us": float(row["p50 latency"]),
        "p95_us": float(row["p95 latency"]),
        "p99_us": float(row["p99 latency"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variants", default=None)
    parser.add_argument("--concurrency", default="1,4,8")
    parser.add_argument("--label", required=True)
    parser.add_argument("--server", default="localhost")
    parser.add_argument("--grpc-port", type=int, default=8001)
    parser.add_argument("--metrics-port", type=int, default=8002)
    parser.add_argument("--measurement-ms", type=int, default=5000)
    parser.add_argument("--sdk-image", default=SDK_IMAGE)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out = args.output_dir or HERE / "results" / f"{timestamp}_yolo26-det_{args.label}"
    out.mkdir(parents=True, exist_ok=True)
    levels = [int(level) for level in args.concurrency.split(",")]
    metrics_url = f"http://{args.server}:{args.metrics_port}/metrics"

    result = {"schema_version": 1, "label": args.label, "fixture": "bus_1080p",
              "sdk_image": args.sdk_image, "variants": {}}
    # perf_analyzer runs in a container, so the work directory must be bind-mountable.
    with tempfile.TemporaryDirectory(dir=out) as directory:
        work = Path(directory)
        width, height = RESOLUTION_FIXTURES["bus_1080p"]
        jpeg = make_resolution_fixture(REPO_ROOT / "data/images/bus.jpg",
                                       work / "bus_1080p.jpg", (width, height))
        for variant in select(args.variants):
            if not variant.supports_perf_analyzer:
                result["variants"][variant.key] = {"skipped": "client-side preprocessing"}
                continue
            data = input_data(variant, jpeg, work)
            levels_result = {}
            for level in levels:
                print(f"  [{variant.key}] concurrency {level} ...", end=" ", flush=True)
                sampler = MetricsSampler(metrics_url)
                sampler.start()
                started = time.monotonic()
                measured = perf_analyzer(args, variant, data, level, work)
                sampler.stop_event.set()
                sampler.join()
                measured.update(sampler.summary())
                measured["wall_s"] = round(time.monotonic() - started, 1)
                levels_result[str(level)] = measured
                print("FAILED" if "error" in measured else "OK")
            result["variants"][variant.key] = levels_result

    path = out / "perf.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
