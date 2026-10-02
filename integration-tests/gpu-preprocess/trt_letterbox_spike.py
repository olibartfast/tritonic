#!/usr/bin/env python3
"""Spike T-2 (assumption A-1): does TensorRT build and reproduce the letterbox graph?

Runs inside nvcr.io/nvidia/tensorrt:26.08-py3 on a GPU host (see
trt_letterbox_spike.sh). Steps:
  1. write the standalone letterbox ONNX graph
  2. build it with trtexec using one dynamic-shape profile
  3. run every case through trtexec and compare with the NumPy R-2 reference
     and, when --dali-dir is given, with the DALI tensors from spike T-3

The deliverable is the JSON report plus a pass/replan note in plan.md.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
from letterbox_reference import letterbox  # noqa: E402
from tensor_metrics import passes, tensor_metrics  # noqa: E402

FIXTURES = ["bus.jpg", "horses.jpg", "person.jpg", "mug.jpg"]
SYNTHETIC_SHAPES = [(479, 641), (481, 640), (5, 3), (1080, 1920), (2160, 3840)]
PROFILE = {"min": "1x32x32x3", "opt": "1x1080x1920x3", "max": "1x4096x4096x3"}


def trtexec() -> str:
    for candidate in (shutil.which("trtexec"), "/usr/src/tensorrt/bin/trtexec"):
        if candidate and Path(candidate).exists():
            return candidate
    raise SystemExit("trtexec not found; run inside nvcr.io/nvidia/tensorrt:26.08-py3")


def load_builder():
    path = REPO / "deploy/object_detection/yolo26/ensemble/trt_pre/letterbox_onnx.py"
    spec = importlib.util.spec_from_file_location("letterbox_onnx", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cases():
    for name in FIXTURES:
        bgr = cv2.imread(str(REPO / "data/images" / name))
        yield Path(name).stem, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rng = np.random.default_rng(1234)
    for height, width in SYNTHETIC_SHAPES:
        yield f"random_{height}x{width}", rng.integers(0, 256, (height, width, 3), dtype=np.uint8)


def run_case(binary: str, engine: Path, work: Path, name: str, rgb: np.ndarray) -> np.ndarray:
    height, width = rgb.shape[:2]
    source = work / f"{name}.u8"
    output = work / f"{name}.json"
    np.ascontiguousarray(rgb[None]).tofile(source)
    subprocess.run(
        [
            binary,
            f"--loadEngine={engine}",
            f"--shapes=image:1x{height}x{width}x3",
            f"--loadInputs=image:{source}",
            f"--exportOutput={output}",
            "--iterations=1",
            "--warmUp=0",
            "--duration=0",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    exported = json.loads(output.read_text())
    tensor = next(item for item in exported if item["name"] == "images")
    return np.asarray(tensor["values"], dtype=np.float32).reshape(3, 640, 640)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pad-rounding", choices=("floor", "ceil"), default="floor")
    parser.add_argument("--dali-dir", type=Path, help="DALI .fp32 tensors from spike T-3")
    parser.add_argument("--output", type=Path, default=Path("/tmp/tritonic-spikes/trt_letterbox.json"))
    args = parser.parse_args()

    binary = trtexec()
    module = load_builder()
    report = {"pad_rounding": args.pad_rounding, "profile": PROFILE, "cases": []}
    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        onnx_path = work / "letterbox.onnx"
        engine = work / "letterbox.plan"
        import onnx

        onnx.save(module.build_letterbox_graph(pad_rounding=args.pad_rounding), str(onnx_path))
        build = subprocess.run(
            [
                binary,
                f"--onnx={onnx_path}",
                f"--saveEngine={engine}",
                f"--minShapes=image:{PROFILE['min']}",
                f"--optShapes=image:{PROFILE['opt']}",
                f"--maxShapes=image:{PROFILE['max']}",
            ],
            capture_output=True,
            text=True,
        )
        report["build_ok"] = build.returncode == 0
        if not report["build_ok"]:
            report["build_log_tail"] = (build.stdout + build.stderr)[-4000:]
        else:
            for name, rgb in cases():
                actual = run_case(binary, engine, work, name, rgb)
                result = {"case": name, "input_hw": list(rgb.shape[:2])}
                result["vs_reference"] = tensor_metrics(
                    letterbox(rgb, pad_rounding=args.pad_rounding), actual
                )
                result["vs_reference"]["passed"] = passes(result["vs_reference"])
                dali = args.dali_dir / f"{name}.fp32" if args.dali_dir else None
                if dali and dali.exists():
                    expected = np.fromfile(dali, dtype=np.float32).reshape(3, 640, 640)
                    result["vs_dali"] = tensor_metrics(expected, actual)
                    result["vs_dali"]["passed"] = passes(result["vs_dali"])
                report["cases"].append(result)

    report["passed"] = report["build_ok"] and all(
        case["vs_reference"]["passed"] and case.get("vs_dali", {}).get("passed", True)
        for case in report["cases"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"build_ok": report["build_ok"], "passed": report["passed"]}))
    print(args.output)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
