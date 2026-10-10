#!/usr/bin/env python3
"""ONNX letterbox subgraph for the TensorRT-graph preprocessing variant (V13).

Input  `image`  UINT8 [1, H, W, 3] RGB, H and W dynamic
Output `images` FP32  [1, 3, 640, 640], the YOLO26 TensorRT input

The graph computes the spec R-2 letterbox from the runtime input shape:
Shape -> size math -> Cast -> Transpose -> Resize (linear, half_pixel) -> Round ->
Clip -> Pad (constant 0, centered) -> Div 255.

Run standalone to write the spike graph (plan T-2):
    letterbox_onnx.py --output letterbox.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

CANVAS = 640
DEFAULT_OPSET = 17
IR_VERSION = 8


def _const(name: str, value) -> onnx.NodeProto:
    array = np.asarray(value)
    return helper.make_node(
        "Constant", [], [name], value=numpy_helper.from_array(array, name=f"{name}_value")
    )


def build_letterbox_graph(
    input_name: str = "image",
    output_name: str = "images",
    canvas: int = CANVAS,
    pad_rounding: str = "floor",
    opset: int = DEFAULT_OPSET,
) -> onnx.ModelProto:
    if pad_rounding not in ("floor", "ceil"):
        raise ValueError(f"unknown pad rounding: {pad_rounding}")

    nodes = [
        _const("lb_canvas", np.array([canvas], dtype=np.int64)),
        _const("lb_two", np.array([2], dtype=np.int64)),
        _const("lb_zero2", np.array([0, 0], dtype=np.int64)),
        _const("lb_batch_channels", np.array([1, 3], dtype=np.int64)),
        _const("lb_idx_h", np.array([1], dtype=np.int64)),
        _const("lb_idx_w", np.array([2], dtype=np.int64)),
        _const("lb_min", np.array(0.0, dtype=np.float32)),
        _const("lb_max", np.array(255.0, dtype=np.float32)),
        _const("lb_scale", np.array(255.0, dtype=np.float32)),
        _const("lb_fill", np.array(0.0, dtype=np.float32)),
        # Integer geometry from the runtime shape.
        helper.make_node("Shape", [input_name], ["lb_shape"]),
        helper.make_node("Gather", ["lb_shape", "lb_idx_h"], ["lb_h"], axis=0),
        helper.make_node("Gather", ["lb_shape", "lb_idx_w"], ["lb_w"], axis=0),
        helper.make_node("Max", ["lb_h", "lb_w"], ["lb_extent"]),
        helper.make_node("Mul", ["lb_h", "lb_canvas"], ["lb_h_scaled"]),
        helper.make_node("Mul", ["lb_w", "lb_canvas"], ["lb_w_scaled"]),
        helper.make_node("Div", ["lb_h_scaled", "lb_extent"], ["lb_rh"]),
        helper.make_node("Div", ["lb_w_scaled", "lb_extent"], ["lb_rw"]),
        helper.make_node("Sub", ["lb_canvas", "lb_rh"], ["lb_pad_h"]),
        helper.make_node("Sub", ["lb_canvas", "lb_rw"], ["lb_pad_w"]),
    ]
    if pad_rounding == "ceil":
        nodes += [
            _const("lb_one", np.array([1], dtype=np.int64)),
            helper.make_node("Add", ["lb_pad_h", "lb_one"], ["lb_pad_h_round"]),
            helper.make_node("Add", ["lb_pad_w", "lb_one"], ["lb_pad_w_round"]),
        ]
        pad_h_source, pad_w_source = "lb_pad_h_round", "lb_pad_w_round"
    else:
        pad_h_source, pad_w_source = "lb_pad_h", "lb_pad_w"
    nodes += [
        # Integer Div on non-negative values floors.
        helper.make_node("Div", [pad_h_source, "lb_two"], ["lb_top"]),
        helper.make_node("Div", [pad_w_source, "lb_two"], ["lb_left"]),
        helper.make_node("Sub", ["lb_pad_h", "lb_top"], ["lb_bottom"]),
        helper.make_node("Sub", ["lb_pad_w", "lb_left"], ["lb_right"]),
        helper.make_node(
            "Concat", ["lb_batch_channels", "lb_rh", "lb_rw"], ["lb_sizes"], axis=0
        ),
        helper.make_node(
            "Concat",
            ["lb_zero2", "lb_top", "lb_left", "lb_zero2", "lb_bottom", "lb_right"],
            ["lb_pads"],
            axis=0,
        ),
        # Pixel path: NHWC uint8 -> float -> NCHW -> resize -> uint8 rounding -> pad -> /255.
        # Cast comes first: TensorRT accepts UINT8 tensors only as Cast inputs (spike T-2,
        # TensorRT 11.2.1.2 rejected a Transpose on UINT8).
        helper.make_node("Cast", [input_name], ["lb_nhwc"], to=TensorProto.FLOAT),
        helper.make_node("Transpose", ["lb_nhwc"], ["lb_nchw"], perm=[0, 3, 1, 2]),
        helper.make_node(
            "Resize",
            ["lb_nchw", "", "", "lb_sizes"],
            ["lb_resized"],
            mode="linear",
            coordinate_transformation_mode="half_pixel",
        ),
        helper.make_node("Round", ["lb_resized"], ["lb_rounded"]),
        helper.make_node("Clip", ["lb_rounded", "lb_min", "lb_max"], ["lb_clipped"]),
        helper.make_node(
            "Pad", ["lb_clipped", "lb_pads", "lb_fill"], ["lb_padded"], mode="constant"
        ),
        helper.make_node("Div", ["lb_padded", "lb_scale"], [output_name]),
    ]

    graph = helper.make_graph(
        nodes,
        "yolo26_letterbox",
        [helper.make_tensor_value_info(input_name, TensorProto.UINT8, [1, "height", "width", 3])],
        [helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [1, 3, canvas, canvas])],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", opset)],
        producer_name="tritonic-letterbox",
    )
    # Pin the IR version: newer onnx packages default to an IR that older
    # TensorRT parsers and onnxruntime builds reject.
    model.ir_version = IR_VERSION
    onnx.checker.check_model(model)
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pad-rounding", choices=("floor", "ceil"), default="floor")
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    args = parser.parse_args()
    model = build_letterbox_graph(pad_rounding=args.pad_rounding, opset=args.opset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.output))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
