#!/usr/bin/env python3
"""Invalid-input robustness of the preprocessing variants (spec R-11, V-11).

Server variants (V12, V13 and the B1 DALI reference) get raw requests over HTTP and
gRPC through tritonclient. Each case must produce the expected outcome, Triton must
stay ready, and a valid request must succeed afterwards. Cases whose behaviour R-11
does not fix (a JPEG truncated mid-scan) are recorded for every variant, including
B1, without gating.

V2 runs the Tritonic client: frames the CUDA contract rejects (degenerate aspect
ratio, side over the limit) must fail with a message, and a valid run must follow.

Needs: pip install "tritonclient[http,grpc]" opencv-python-headless numpy
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "benchmarks" / "yolo26-det"))
from variants import VARIANTS  # noqa: E402

# Valid SOF dimensions but no scan data (the case compare_inference.py uses for DALI).
HEADER_ONLY_JPEG = bytes([
    0xFF, 0xD8, 0xFF, 0xC0, 0x00, 0x11, 0x08, 0x00, 0x02, 0x00, 0x03, 0x03,
    0x01, 0x11, 0x00, 0x02, 0x11, 0x00, 0x03, 0x11, 0x00, 0xFF, 0xD9,
])
# Side limits from the model configs: the decode step is bounded by the TRT profile.
MAX_SIDE = {"V12": 8192, "V13": 4096, "B1": None}


def jpeg(image: np.ndarray) -> bytes:
    ok, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    assert ok
    return data.tobytes()


def server_cases(key: str):
    bus = (REPO / "data/images/bus.jpg").read_bytes()
    cases = [
        ("empty", b"", "error"),
        ("not_jpeg", b"\x89PNG\r\n\x1a\n" + bytes(64), "error"),
        ("header_only", HEADER_ONLY_JPEG, "error"),
        ("degenerate_1x1000", jpeg(np.full((1, 1000, 3), 128, np.uint8)), "error"),
        ("grayscale", jpeg(cv2.cvtColor(cv2.imread(str(REPO / "data/images/bus.jpg")),
                                        cv2.COLOR_BGR2GRAY)), "ok"),
        ("truncated_mid_scan", bus[: len(bus) // 2], "record"),
    ]
    if MAX_SIDE[key]:
        side = MAX_SIDE[key] + 1
        cases.append((f"over_limit_{side}x64", jpeg(np.zeros((64, side, 3), np.uint8)), "error"))
    return cases


def infer(protocol: str, model: str, payload: bytes):
    data = np.frombuffer(payload, dtype=np.uint8).reshape(1, -1)
    if protocol == "http":
        client = httpclient.InferenceServerClient("localhost:8000")
        tensor = httpclient.InferInput("IMAGE", list(data.shape), "UINT8")
    else:
        client = grpcclient.InferenceServerClient("localhost:8001")
        tensor = grpcclient.InferInput("IMAGE", list(data.shape), "UINT8")
    tensor.set_data_from_numpy(data)
    try:
        client.infer(model, [tensor])
        return "ok", ""
    except Exception as error:  # noqa: BLE001 - both clients raise InferenceServerException
        return "error", str(error)[:300]


def ready() -> bool:
    return httpclient.InferenceServerClient("localhost:8000").is_server_ready()


def check_server(key: str, protocols):
    variant = VARIANTS[key]
    valid = (REPO / "data/images/bus.jpg").read_bytes()
    results = []
    for protocol in protocols:
        for name, payload, expected in server_cases(key):
            outcome, message = infer(protocol, variant.model, payload)
            still_ready = ready()
            recovered, _ = infer(protocol, variant.model, valid)
            # B1 (DALI) is the reference, not under test: record its behaviour only.
            gated = expected != "record" and key != "B1"
            passed = still_ready and recovered == "ok" and (not gated or outcome == expected)
            results.append({"variant": key, "protocol": protocol, "case": name,
                            "expected": expected, "outcome": outcome, "message": message,
                            "ready_after": still_ready, "valid_after": recovered,
                            "gated": gated, "passed": passed})
            print(f"{'PASS' if passed else 'FAIL'} {key:<4} {protocol:<5} {name:<22} "
                  f"outcome={outcome}")
    return results


def check_client(tritonic: Path):
    """V2: the CUDA contract's rejections exit non-zero; a valid run still succeeds."""
    variant = VARIANTS["V2"]
    results = []
    with tempfile.TemporaryDirectory() as work:
        cases = {
            "degenerate_1x1000": np.full((1, 1000, 3), 128, np.uint8),
            "over_limit_8193x64": np.zeros((64, 8193, 3), np.uint8),
        }
        for name, image in cases.items():
            path = Path(work) / f"{name}.png"
            cv2.imwrite(str(path), image)
            completed = run_client(tritonic, variant, path)
            valid = run_client(tritonic, variant, REPO / "data/images/bus.jpg")
            passed = completed.returncode != 0 and valid.returncode == 0
            results.append({"variant": "V2", "case": name, "exit_code": completed.returncode,
                            "message": (completed.stdout + completed.stderr)[-300:],
                            "valid_after_exit_code": valid.returncode, "gated": True,
                            "passed": passed})
            print(f"{'PASS' if passed else 'FAIL'} V2   client {name:<22} "
                  f"exit={completed.returncode}")
    return results


def run_client(tritonic: Path, variant, source: Path):
    return subprocess.run(
        [str(tritonic), f"--source={source}", "--model_type=yolo", "--labelsFile=labels/coco.txt",
         "--protocol=grpc", "--port=8001", "--write_frame=false", *variant.client_args()],
        text=True, capture_output=True, cwd=str(REPO))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variants", default="B1,V12,V13,V2")
    parser.add_argument("--protocols", default="http,grpc")
    parser.add_argument("--tritonic", type=Path, default=REPO / "build" / "tritonic")
    parser.add_argument("--output", type=Path,
                        default=Path("/tmp/tritonic-gpu-preprocess/invalid_inputs.json"))
    args = parser.parse_args()

    results = []
    for key in args.variants.split(","):
        if key == "V2":
            results += check_client(args.tritonic)
        else:
            results += check_server(key, args.protocols.split(","))

    report = {"results": results, "passed": all(item["passed"] for item in results)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
