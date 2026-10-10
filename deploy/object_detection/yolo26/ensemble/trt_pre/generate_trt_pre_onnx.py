#!/usr/bin/env python3
"""Prepend the R-2 letterbox subgraph to a YOLO26 detection ONNX model (variant V13).

The result takes `image` UINT8 [1, H, W, 3] RGB and produces the YOLO26 outputs
unchanged, so the TensorRT engine built from it does preprocessing in-graph:

    generate_trt_pre_onnx.py --yolo-onnx yolo26s.onnx --output yolo26det_trt_pre.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import compose

from letterbox_onnx import build_letterbox_graph

YOLO_INPUT = "images"
LETTERBOX_OUTPUT = "lb_images"


def default_opset(model: onnx.ModelProto) -> int:
    for entry in model.opset_import:
        if entry.domain in ("", "ai.onnx"):
            return entry.version
    raise ValueError("YOLO model has no default-domain opset import")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--yolo-onnx", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pad-rounding", choices=("floor", "ceil"), default="floor")
    args = parser.parse_args()

    yolo = onnx.load(str(args.yolo_onnx))
    inputs = [value.name for value in yolo.graph.input]
    if inputs != [YOLO_INPUT]:
        raise SystemExit(f"expected a single YOLO input '{YOLO_INPUT}', found {inputs}")
    dims = [d.dim_value or d.dim_param for d in yolo.graph.input[0].type.tensor_type.shape.dim]
    if dims[1:] != [3, 640, 640]:
        raise SystemExit(f"expected YOLO input [N,3,640,640], found {dims}")

    letterbox = build_letterbox_graph(
        output_name=LETTERBOX_OUTPUT,
        pad_rounding=args.pad_rounding,
        opset=default_opset(yolo),
    )
    letterbox.ir_version = yolo.ir_version
    # merge_models rejects duplicate names, so a collision fails loudly instead of rewiring.
    merged = compose.merge_models(letterbox, yolo, io_map=[(LETTERBOX_OUTPUT, YOLO_INPUT)])
    merged.graph.name = "yolo26det_trt_pre"
    onnx.checker.check_model(merged)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(merged, str(args.output))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
