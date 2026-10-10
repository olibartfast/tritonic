#!/usr/bin/env python3
"""Stability soak for the server-side variants (spec R-11, V-13).

Per variant: warm up, record GPU memory, send N sequential requests and then M
requests at concurrency C (both protocols are covered by the parity checks, so this
uses gRPC only). It then checks for zero errors, a server success count equal to the
requests sent (a restart resets it), and GPU memory within 5% of the post-warmup value.

Needs: pip install "tritonclient[grpc,http]" numpy
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import tritonclient.grpc as grpcclient

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "benchmarks" / "yolo26-det"))
from variants import select  # noqa: E402

MEMORY = re.compile(r"^nv_gpu_memory_used_bytes\{[^}]*\} ([0-9.e+]+)$")
MEMORY_GROWTH_LIMIT = 1.05


def gpu_memory(metrics_url: str) -> float:
    with urllib.request.urlopen(metrics_url, timeout=5) as response:
        return sum(float(m[1]) for line in response.read().decode().splitlines()
                   if (m := MEMORY.match(line)))


def success_count(http_url: str, model: str) -> int:
    with urllib.request.urlopen(f"{http_url}/v2/models/{model}/stats", timeout=5) as response:
        return int(json.load(response)["model_stats"][0]["inference_stats"]["success"]["count"])


_local = threading.local()


def request(url: str, model: str, payload: np.ndarray) -> bool:
    # One client per thread: reusing the channel keeps connection setup out of the soak.
    if getattr(_local, "url", None) != url:
        _local.client, _local.url = grpcclient.InferenceServerClient(url), url
    client = _local.client
    tensor = grpcclient.InferInput("IMAGE", list(payload.shape), "UINT8")
    tensor.set_data_from_numpy(payload)
    try:
        client.infer(model, [tensor])
        return True
    except Exception:  # noqa: BLE001 - counted as an error
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variants", default="V12,V13")
    parser.add_argument("--sequential", type=int, default=10000)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--concurrent-requests", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--grpc-url", default="localhost:8001")
    parser.add_argument("--http-url", default="http://localhost:8000")
    parser.add_argument("--metrics-url", default="http://localhost:8002/metrics")
    parser.add_argument("--output", type=Path, default=Path("/tmp/tritonic-gpu-preprocess/soak.json"))
    args = parser.parse_args()

    payload = np.fromfile(REPO / "data/images/bus.jpg", dtype=np.uint8).reshape(1, -1)
    results = []
    for variant in select(args.variants):
        if variant.payload != "encoded":
            print(f"skip {variant.key}: soak covers server-side preprocessing variants")
            continue
        for _ in range(args.warmup):
            request(args.grpc_url, variant.model, payload)
        memory_after_warmup = gpu_memory(args.metrics_url)
        count_before = success_count(args.http_url, variant.model)

        errors = sum(not request(args.grpc_url, variant.model, payload)
                     for _ in range(args.sequential))
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            outcomes = list(pool.map(lambda _: request(args.grpc_url, variant.model, payload),
                                     range(args.concurrent_requests)))
        errors += outcomes.count(False)

        memory_after = gpu_memory(args.metrics_url)
        expected = args.sequential + args.concurrent_requests
        served = success_count(args.http_url, variant.model) - count_before
        entry = {
            "variant": variant.key, "model": variant.model, "requests": expected,
            "errors": errors, "served": served,
            "gpu_memory_after_warmup_mib": memory_after_warmup / 2**20,
            "gpu_memory_after_mib": memory_after / 2**20,
        }
        entry["passed"] = (errors == 0 and served == expected
                           and memory_after <= MEMORY_GROWTH_LIMIT * memory_after_warmup)
        results.append(entry)
        print(json.dumps(entry))

    report = {"results": results, "passed": all(item["passed"] for item in results)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
