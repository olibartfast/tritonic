#!/usr/bin/env python3
"""Compare direct and DALI-ensemble detections through the Tritonic client."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import tempfile
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

DETECTION = re.compile(
    r"Detection: class_id=(?P<class>\d+) confidence=(?P<confidence>[0-9.]+) "
    r"bbox=\[(?P<x>-?\d+),(?P<y>-?\d+),(?P<w>-?\d+),(?P<h>-?\d+)\]"
)
INFER_TIME = re.compile(r"Infer time for 1 image: (?P<ms>\d+) ms")


def run_client(
    executable: Path,
    fixture: Path,
    labels: Path,
    protocol: str,
    encoded: bool,
) -> tuple[list[dict[str, object]], float]:
    command = [
        str(executable),
        f"--source={fixture}",
        "--model_type=yolo",
        f"--model={'yolo_dali_ensemble' if encoded else 'yolo_trt'}",
        f"--labelsFile={labels}",
        f"--protocol={protocol}",
        "--serverAddress=localhost",
        f"--port={8000 if protocol == 'http' else 8001}",
        "--write_frame=false",
        "--log_level=debug",
    ]
    if encoded:
        command.extend(["--input_mode=encoded-image", "--task_model=yolo_trt"])

    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    output = completed.stdout + completed.stderr
    detections = [
        {
            "class": int(match["class"]),
            "confidence": float(match["confidence"]),
            "box": [int(match[key]) for key in ("x", "y", "w", "h")],
        }
        for match in DETECTION.finditer(output)
    ]
    timing = INFER_TIME.search(output)
    if timing is None:
        raise AssertionError(f"missing inference timing in output:\n{output}")
    return detections, float(timing["ms"])


def box_error(one: list[int], two: list[int]) -> int:
    return max(abs(left - right) for left, right in zip(one, two))


def compare(reference: list[dict[str, object]], candidate: list[dict[str, object]]) -> None:
    if len(reference) != len(candidate):
        raise AssertionError(f"detection count mismatch: {len(reference)} != {len(candidate)}")

    unmatched = list(candidate)
    for expected in reference:
        same_class = [item for item in unmatched if item["class"] == expected["class"]]
        if not same_class:
            raise AssertionError(f"missing class {expected['class']}")
        match = min(same_class, key=lambda item: box_error(expected["box"], item["box"]))
        unmatched.remove(match)
        confidence_error = abs(float(expected["confidence"]) - float(match["confidence"]))
        coordinate_error = box_error(expected["box"], match["box"])
        # Resize differences are bounded separately by the tensor parity gate.
        # Allow 12 original-image pixels here for edge boxes.
        if confidence_error > 0.03 or coordinate_error > 12:
            raise AssertionError(
                f"detection mismatch: confidence_error={confidence_error:.6f}, "
                f"coordinate_error={coordinate_error}"
            )


def invalid_http_request() -> None:
    body = json.dumps(
        {
            "inputs": [
                {
                    "name": "IMAGE",
                    "shape": [1, 4],
                    "datatype": "UINT8",
                    "data": [0, 1, 2, 3],
                }
            ],
            "outputs": [{"name": "output0"}],
        }
    ).encode()
    request = urllib.request.Request(
        "http://localhost:8000/v2/models/yolo_dali_ensemble/infer",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        urllib.request.urlopen(request, timeout=10)
    except urllib.error.HTTPError as error:
        if error.code < 400:
            raise
    else:
        raise AssertionError("invalid encoded bytes unexpectedly succeeded")

    with urllib.request.urlopen("http://localhost:8000/v2/health/ready", timeout=10) as response:
        if response.status != 200:
            raise AssertionError("Triton was not ready after invalid encoded input")


def invalid_grpc_request(executable: Path, labels: Path) -> None:
    # Valid SOF dimensions but no JPEG scan data: the client accepts the
    # transport contract and DALI must return a controlled decode error.
    malformed = bytes(
        [
            0xFF, 0xD8, 0xFF, 0xC0, 0x00, 0x11, 0x08, 0x00, 0x02, 0x00, 0x03, 0x03,
            0x01, 0x11, 0x00, 0x02, 0x11, 0x00, 0x03, 0x11, 0x00, 0xFF, 0xD9,
        ]
    )
    with tempfile.NamedTemporaryFile(suffix=".jpg") as fixture:
        fixture.write(malformed)
        fixture.flush()
        completed = subprocess.run(
            [
                str(executable),
                f"--source={fixture.name}",
                "--model_type=yolo",
                "--model=yolo_dali_ensemble",
                "--task_model=yolo_trt",
                "--input_mode=encoded-image",
                f"--labelsFile={labels}",
                "--protocol=grpc",
                "--port=8001",
                "--write_frame=false",
                "--log_level=error",
            ],
            text=True,
            capture_output=True,
        )
    if completed.returncode == 0:
        raise AssertionError("invalid gRPC encoded bytes unexpectedly succeeded")

    with urllib.request.urlopen("http://localhost:8000/v2/health/ready", timeout=10) as response:
        if response.status != 200:
            raise AssertionError("Triton was not ready after invalid gRPC encoded input")


def percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    return ordered[min(int(round((len(ordered) - 1) * fraction)), len(ordered) - 1)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--executable", default="build/tritonic")
    parser.add_argument("--labels", default="labels/coco.txt")
    parser.add_argument(
        "--output", default="/tmp/tritonic-yolo-baseline/yolo_ensemble_report.json"
    )
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    executable = (repo / args.executable).resolve()
    labels = (repo / args.labels).resolve()
    fixtures = [
        repo / "data/images/bus.jpg",
        repo / "data/images/horses.jpg",
        repo / "data/images/person.jpg",
        repo / "data/images/mug.jpg",
    ]

    report: dict[str, object] = {}
    for protocol in ("http", "grpc"):
        run_client(executable, fixtures[0], labels, protocol, False)
        run_client(executable, fixtures[0], labels, protocol, True)
        direct_times = []
        ensemble_times = []
        fixture_reports = []
        for fixture in fixtures:
            direct, direct_ms = run_client(executable, fixture, labels, protocol, False)
            ensemble, ensemble_ms = run_client(executable, fixture, labels, protocol, True)
            compare(direct, ensemble)
            direct_times.append(direct_ms)
            ensemble_times.append(ensemble_ms)
            fixture_reports.append(
                {
                    "source": str(fixture.relative_to(repo)),
                    "detections": len(direct),
                    "direct_ms": direct_ms,
                    "ensemble_ms": ensemble_ms,
                }
            )
        report[protocol] = {
            "fixtures": fixture_reports,
            "latency_ms": {
                "direct_mean": statistics.mean(direct_times),
                "direct_p95": percentile(direct_times, 0.95),
                "ensemble_mean": statistics.mean(ensemble_times),
                "ensemble_p95": percentile(ensemble_times, 0.95),
            },
        }

    invalid_http_request()
    invalid_grpc_request(executable, labels)
    report["invalid_encoded_http"] = "controlled error; server remained ready"
    report["invalid_encoded_grpc"] = "controlled error; server remained ready"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
