#!/usr/bin/env python3
"""Detection parity of a preprocessing variant against DALI (spec V-7, V-8, V-10).

Runs Tritonic on every fixture for the reference variant and each candidate over
HTTP and gRPC, then applies the tolerance used to accept the DALI path
(integration-tests/yolo-ensemble/compare_inference.py): same count and classes,
confidence within 0.03, boxes within 12 px. A candidate with GPU postprocessing
is compared against the DALI GPU-postprocess reference.

    compare_detections.py --candidates V12,V12-gpupost --protocols http,grpc
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "benchmarks"))
sys.path.insert(0, str(REPO / "benchmarks" / "yolo26-det"))
sys.path.insert(0, str(REPO / "integration-tests" / "yolo-ensemble"))
from common.fixtures import RESOLUTION_FIXTURES, make_resolution_fixture  # noqa: E402
from compare_inference import compare  # noqa: E402
from variants import VARIANTS, select  # noqa: E402

DETECTION = re.compile(
    r"Detection: class_id=(?P<class>\d+) confidence=(?P<confidence>[0-9.]+) "
    r"bbox=\[(?P<x>-?\d+),(?P<y>-?\d+),(?P<w>-?\d+),(?P<h>-?\d+)\]"
)
STOCK = ["bus.jpg", "horses.jpg", "person.jpg", "mug.jpg"]


def detections(args, variant, protocol, fixture):
    port = 8000 if protocol == "http" else 8001
    command = [
        str(args.tritonic),
        f"--source={fixture}",
        f"--model_type={args.model_type}",
        f"--labelsFile={args.labels}",
        f"--protocol={protocol}",
        f"--serverAddress={args.server}",
        f"--port={port}",
        "--write_frame=false",
        "--log_level=debug",
        *variant.client_args(),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, cwd=str(REPO))
    output = completed.stdout + completed.stderr
    if completed.returncode != 0:
        raise RuntimeError(f"{variant.key} failed on {fixture.name}:\n{output[-2000:]}")
    return [
        {
            "class": int(match["class"]),
            "confidence": float(match["confidence"]),
            "box": [int(match[key]) for key in ("x", "y", "w", "h")],
        }
        for match in DETECTION.finditer(output)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--candidates", required=True, help="variant keys, e.g. V12,V13,V2")
    parser.add_argument("--protocols", default="http,grpc")
    parser.add_argument("--model-type", default="yolo")
    parser.add_argument("--labels", default="labels/coco.txt")
    parser.add_argument("--server", default="localhost")
    parser.add_argument("--tritonic", type=Path, default=REPO / "build" / "tritonic")
    parser.add_argument("--output", type=Path, default=Path("/tmp/tritonic-gpu-preprocess/detections.json"))
    args = parser.parse_args()

    report = {"results": []}
    failed = False
    with tempfile.TemporaryDirectory() as work:
        fixtures = [REPO / "data/images" / name for name in STOCK]
        fixtures += [make_resolution_fixture(REPO / "data/images/bus.jpg",
                                             Path(work) / f"{name}.jpg", size)
                     for name, size in RESOLUTION_FIXTURES.items()]
        for candidate in select(args.candidates):
            reference = VARIANTS["B1-gpupost" if "gpu-post" in candidate.tags else "B1"]
            for protocol in args.protocols.split(","):
                for fixture in fixtures:
                    entry = {"candidate": candidate.key, "reference": reference.key,
                             "protocol": protocol, "fixture": fixture.name}
                    try:
                        expected = detections(args, reference, protocol, fixture)
                        actual = detections(args, candidate, protocol, fixture)
                        compare(expected, actual)
                        entry.update(passed=True, detections=len(expected))
                    except (AssertionError, RuntimeError) as error:
                        entry.update(passed=False, error=str(error)[-1000:])
                        failed = True
                    print(f"{'PASS' if entry['passed'] else 'FAIL'} {candidate.key:<12} "
                          f"{protocol:<5} {fixture.name}")
                    report["results"].append(entry)

    report["passed"] = not failed
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
