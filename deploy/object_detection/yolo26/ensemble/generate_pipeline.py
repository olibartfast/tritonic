#!/usr/bin/env python3
"""Serialize the YOLO26 detection DALI preprocessing component."""

from __future__ import annotations

import argparse
from pathlib import Path

from nvidia.dali import fn, pipeline_def, types


@pipeline_def
def pipeline():
    encoded = fn.external_source(name="IMAGE", device="cpu", ndim=1, dtype=types.UINT8)
    encoded_shape = fn.peek_image_shape(encoded)
    original_size = fn.slice(encoded_shape, start=[0], shape=[2], axes=[0])
    height = fn.slice(encoded_shape, start=[0], shape=[1], axes=[0])
    width = fn.slice(encoded_shape, start=[1], shape=[1], axes=[0])
    max_extent = fn.reductions.max(fn.cat(width, height, axis=0), axes=[0])
    resized_width = fn.cast(width * 640 // max_extent, dtype=types.FLOAT)
    resized_height = fn.cast(height * 640 // max_extent, dtype=types.FLOAT)
    decoded = fn.decoders.image(encoded, device="mixed", output_type=types.RGB)
    resized = fn.resize(
        decoded,
        device="gpu",
        resize_x=resized_width,
        resize_y=resized_height,
        interp_type=types.INTERP_LINEAR,
        antialias=False,
        subpixel_scale=False,
    )
    padded = fn.paste(
        resized,
        device="gpu",
        ratio=1.0,
        min_canvas_size=640,
        paste_x=0.5,
        paste_y=0.5,
        fill_value=[0, 0, 0],
    )
    normalized = fn.crop_mirror_normalize(
        padded,
        device="gpu",
        dtype=types.FLOAT,
        output_layout="CHW",
        mean=[0.0, 0.0, 0.0],
        std=[255.0, 255.0, 255.0],
    )
    return normalized, original_size


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pipeline(batch_size=1, num_threads=2, device_id=0).serialize(filename=str(output))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
