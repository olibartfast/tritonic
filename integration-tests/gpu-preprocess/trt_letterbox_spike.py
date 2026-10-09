#!/usr/bin/env python3
"""Spike T-2 (assumption A-1): does TensorRT build and reproduce the letterbox graph?

Runs inside nvcr.io/nvidia/tensorrt:26.08-py3 on a GPU host (see run_spikes.sh),
or anywhere with the `tensorrt` and `cupy` Python packages and a GPU (Colab):
without trtexec it builds and runs the engine through the TensorRT Python API.
Steps:
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
PROFILE = {"min": "1x1x1x3", "opt": "1x1080x1920x3", "max": "1x4096x4096x3"}


def trtexec():
    for candidate in (shutil.which("trtexec"), "/usr/src/tensorrt/bin/trtexec"):
        if candidate and Path(candidate).exists():
            return candidate
    return None


class TrtexecRunner:
    def __init__(self, binary: str):
        self.binary = binary
        self.name = "trtexec"

    def build(self, onnx_path: Path, engine: Path):
        build = subprocess.run(
            [
                self.binary,
                f"--onnx={onnx_path}",
                f"--saveEngine={engine}",
                f"--minShapes=image:{PROFILE['min']}",
                f"--optShapes=image:{PROFILE['opt']}",
                f"--maxShapes=image:{PROFILE['max']}",
            ],
            capture_output=True,
            text=True,
        )
        return build.returncode == 0, (build.stdout + build.stderr)[-4000:]

    def run(self, engine: Path, work: Path, name: str, rgb: np.ndarray) -> np.ndarray:
        return run_case(self.binary, engine, work, name, rgb)


class PythonApiRunner:
    """TensorRT Python API + CuPy device buffers (pip wheels ship no trtexec).

    CuPy rather than torch: Colab's torch carries its own CUDA 12 runtime, while the
    TensorRT wheel may be CUDA 13; CuPy is installed for the same CUDA major.
    """

    def __init__(self):
        import cupy
        import tensorrt as trt

        self.trt, self.cupy = trt, cupy
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.name = f"python-api tensorrt {trt.__version__}"
        self.context = None

    def build(self, onnx_path: Path, engine: Path):
        trt = self.trt
        builder = trt.Builder(self.logger)
        try:
            network = builder.create_network(0)
        except TypeError:
            network = builder.create_network()
        parser = trt.OnnxParser(network, self.logger)
        if not parser.parse(onnx_path.read_bytes()):
            return False, "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        config = builder.create_builder_config()
        profile = builder.create_optimization_profile()
        shapes = [tuple(int(v) for v in PROFILE[key].split("x")) for key in ("min", "opt", "max")]
        profile.set_shape("image", *shapes)
        config.add_optimization_profile(profile)
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            return False, "build_serialized_network returned None"
        engine.write_bytes(bytes(serialized))
        return True, ""

    def run(self, engine: Path, work: Path, name: str, rgb: np.ndarray) -> np.ndarray:
        cupy = self.cupy
        if self.context is None:
            runtime = self.trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(engine.read_bytes())
            self.context = self.engine.create_execution_context()
        # Copies and inference share one stream: a separate non-blocking stream does not
        # wait for CuPy's upload, and TensorRT would read a half-written input.
        stream = cupy.cuda.Stream(non_blocking=True)
        with stream:
            source = cupy.asarray(np.ascontiguousarray(rgb[None]))
            output = cupy.empty((1, 3, 640, 640), dtype=cupy.float32)
            self.context.set_input_shape("image", tuple(source.shape))
            self.context.set_tensor_address("image", source.data.ptr)
            self.context.set_tensor_address("images", output.data.ptr)
            if not self.context.execute_async_v3(stream.ptr):
                raise RuntimeError("execute_async_v3 failed")
            result = cupy.asnumpy(output[0], stream=stream)
        stream.synchronize()
        return result


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
    runner = TrtexecRunner(binary) if binary else PythonApiRunner()
    module = load_builder()
    report = {"pad_rounding": args.pad_rounding, "profile": PROFILE, "runner": runner.name,
              "cases": []}
    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        onnx_path = work / "letterbox.onnx"
        engine = work / "letterbox.plan"
        import onnx

        onnx.save(module.build_letterbox_graph(pad_rounding=args.pad_rounding), str(onnx_path))
        report["build_ok"], build_log = runner.build(onnx_path, engine)
        if not report["build_ok"]:
            report["build_log_tail"] = build_log
        else:
            for name, rgb in cases():
                actual = runner.run(engine, work, name, rgb)
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
