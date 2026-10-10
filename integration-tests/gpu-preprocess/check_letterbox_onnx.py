#!/usr/bin/env python3
"""CPU check: the V13 ONNX letterbox subgraph matches the NumPy R-2 reference.

Runs with onnxruntime on CPU, so it needs no GPU. It proves that the graph
encodes the contract. Whether TensorRT accepts and reproduces the graph is
spike T-2 (trt_letterbox_spike.sh).

    python3 integration-tests/gpu-preprocess/check_letterbox_onnx.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from letterbox_reference import letterbox  # noqa: E402
from tensor_metrics import passes, tensor_metrics  # noqa: E402

FIXTURES = ["bus.jpg", "horses.jpg", "person.jpg", "mug.jpg", "dog.jpg", "cat.jpeg"]
# Odd/even padding on both axes, extreme aspect ratios, and upscaling.
SYNTHETIC_SHAPES = [(479, 641), (481, 640), (640, 640), (5, 3), (1, 640), (2, 1279), (1080, 1920)]
# Aspect ratios beyond 640:1 collapse a side to zero; the graph must not
# silently produce a tensor for them (the decode step rejects them first).
DEGENERATE_SHAPES = [(1, 1000), (1000, 1)]


def load_builder():
    path = REPO / "deploy/object_detection/yolo26/ensemble/trt_pre/letterbox_onnx.py"
    spec = importlib.util.spec_from_file_location("letterbox_onnx", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_letterbox_graph


def cases():
    for name in FIXTURES:
        bgr = cv2.imread(str(REPO / "data/images" / name))
        if bgr is None:
            raise SystemExit(f"missing fixture data/images/{name}")
        yield name, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rng = np.random.default_rng(1234)
    for height, width in SYNTHETIC_SHAPES:
        yield f"random_{height}x{width}", rng.integers(0, 256, (height, width, 3), dtype=np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pad-rounding", choices=("floor", "ceil"), default="floor")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    model = load_builder()(pad_rounding=args.pad_rounding)
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])

    results = []
    for name, rgb in cases():
        (actual,) = session.run(None, {"image": rgb[None]})
        expected = letterbox(rgb, pad_rounding=args.pad_rounding)
        result = {"case": name, "input_hw": list(rgb.shape[:2])}
        result.update(tensor_metrics(expected, actual[0]))
        result["passed"] = passes(result)
        results.append(result)
        status = "PASS" if result["passed"] else "FAIL"
        print(
            f"{status} {name:<22} max={result['max_abs']:.6f} "
            f"mean={result['mean_abs']:.6f} p99={result['p99_abs']:.6f}"
        )

    for height, width in DEGENERATE_SHAPES:
        name = f"degenerate_{height}x{width}"
        try:
            session.run(None, {"image": np.zeros((1, height, width, 3), dtype=np.uint8)})
        except Exception:  # noqa: BLE001 - any runtime rejection is the expected outcome
            rejected = True
        else:
            rejected = False
        results.append({"case": name, "input_hw": [height, width], "passed": rejected})
        print(f"{'PASS' if rejected else 'FAIL'} {name:<22} rejected={rejected}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
    return 0 if all(item["passed"] for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
