#!/usr/bin/env python3
"""Convert a YOLO26 ONNX model to FP16 compute with FP32 inputs and outputs.

TensorRT 11 builds strongly typed networks only (trtexec has no --fp16), so FP16
precision must be expressed in the ONNX types. keep_io_types=True inserts Casts at the
graph boundary, so the Triton configs keep their FP32 `images` / `output0` contract.
For V13 this runs on the YOLO graph *before* the letterbox is prepended, which keeps
the letterbox arithmetic in FP32.

    to_fp16_onnx.py --input yolo26.onnx --output yolo26.fp16.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnxconverter_common import float16


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    model = onnx.load(str(args.input))
    converted = float16.convert_float_to_float16(model, keep_io_types=True)
    onnx.checker.check_model(converted)
    onnx.save(converted, str(args.output))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
