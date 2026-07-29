#!/usr/bin/env python3
"""Serialize the DALI-only YOLO26 polygon postprocessing component."""

from __future__ import annotations

import argparse
from pathlib import Path

from nvidia.dali import fn, pipeline_def, plugin_manager, types


@pipeline_def
def pipeline():
    detections = fn.external_source(
        name="DETECTIONS", device="gpu", ndim=2, dtype=types.FLOAT
    )
    prototypes = fn.external_source(
        name="PROTOTYPES", device="gpu", ndim=3, dtype=types.FLOAT
    )
    original_size = fn.external_source(
        name="ORIGINAL_SIZE", device="gpu", ndim=1, dtype=types.INT64
    )
    outputs = fn.yolo26_seg_postprocess(
        detections,
        prototypes,
        original_size,
        device="gpu",
        confidence_threshold=0.5,
        mask_threshold=0.5,
    )
    return tuple(outputs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plugin", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plugin_manager.load_library(args.plugin)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pipeline(batch_size=1, num_threads=1, device_id=0).serialize(filename=str(output))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
