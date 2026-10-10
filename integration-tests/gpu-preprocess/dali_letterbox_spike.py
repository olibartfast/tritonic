#!/usr/bin/env python3
"""Spike T-3 (assumption A-2): measure the DALI letterbox conventions.

Runs inside nvcr.io/nvidia/tritonserver:26.08-py3 on a GPU host (see
dali_letterbox_spike.sh). It answers three questions with evidence:
  1. Pad offset rule: is fn.paste(paste_x=0.5) floor((640-r)/2) or the ceiling?
     Measured on constant white JPEGs, whose content boundary is exact.
  2. Resampling: applying the NumPy reference to DALI's *own* decoded image
     isolates resize/pad from decoder differences (nvJPEG vs libjpeg).
  3. Degenerate geometry: what DALI does when one side resizes to 0 pixels.

It also writes the DALI tensors for the fixtures, which are the V-5 reference
(--tensor-dir).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from nvidia.dali import fn, pipeline_def, types

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
from letterbox_reference import letterbox, letterbox_geometry  # noqa: E402
from tensor_metrics import tensor_metrics  # noqa: E402

FIXTURES = ["bus.jpg", "horses.jpg", "person.jpg", "mug.jpg"]
GEOMETRY_SHAPES = [(479, 641), (481, 640), (640, 481), (641, 479), (5, 3), (3, 5), (333, 1000)]
DEGENERATE_SHAPES = [(1, 1000), (1000, 1)]


def load_pipeline():
    path = REPO / "deploy/object_detection/yolo26/ensemble/generate_pipeline.py"
    spec = importlib.util.spec_from_file_location("yolo26_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pipe = module.pipeline(batch_size=1, num_threads=2, device_id=0, exec_async=False,
                           exec_pipelined=False, prefetch_queue_depth=1)
    pipe.build()
    return pipe


@pipeline_def
def decode_only():
    encoded = fn.external_source(name="IMAGE", device="cpu", ndim=1, dtype=types.UINT8)
    return fn.decoders.image(encoded, device="mixed", output_type=types.RGB)


def run(pipe, encoded: np.ndarray):
    pipe.feed_input("IMAGE", [encoded])
    outputs = pipe.run()
    return [np.asarray(output.as_cpu().at(0)) for output in outputs]


def encode(image_rgb: np.ndarray, quality: int = 100) -> np.ndarray:
    ok, data = cv2.imencode(".jpg", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return data.reshape(-1)


def content_box(tensor: np.ndarray):
    rows = np.where(tensor.max(axis=(0, 2)) > 0.5)[0]
    cols = np.where(tensor.max(axis=(0, 1)) > 0.5)[0]
    return int(rows[0]), int(cols[0]), int(rows[-1] - rows[0] + 1), int(cols[-1] - cols[0] + 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor-dir", type=Path, default=Path("/tmp/tritonic-spikes/dali"))
    parser.add_argument("--output", type=Path, default=Path("/tmp/tritonic-spikes/dali_letterbox.json"))
    args = parser.parse_args()

    letterbox_pipe = load_pipeline()
    decode_pipe = decode_only(batch_size=1, num_threads=2, device_id=0, exec_async=False,
                              exec_pipelined=False, prefetch_queue_depth=1)
    decode_pipe.build()

    report = {"geometry": [], "resampling": [], "degenerate": []}

    for height, width in GEOMETRY_SHAPES:
        white = np.full((height, width, 3), 255, dtype=np.uint8)
        tensor, _ = run(letterbox_pipe, encode(white))
        top, left, box_h, box_w = content_box(tensor)
        entry = {"input_hw": [height, width], "measured": {"top": top, "left": left,
                                                          "h": box_h, "w": box_w}}
        for rule in ("floor", "ceil"):
            rh, rw, ptop, pleft = letterbox_geometry(height, width, pad_rounding=rule)
            entry[rule] = {"top": ptop, "left": pleft, "h": rh, "w": rw,
                           "matches": [ptop, pleft, rh, rw] == [top, left, box_h, box_w]}
        report["geometry"].append(entry)

    args.tensor_dir.mkdir(parents=True, exist_ok=True)
    for name in FIXTURES:
        encoded = np.fromfile(REPO / "data/images" / name, dtype=np.uint8)
        tensor, original_size = run(letterbox_pipe, encoded)
        (decoded,) = run(decode_pipe, encoded)
        tensor.astype(np.float32).tofile(args.tensor_dir / f"{Path(name).stem}.fp32")
        entry = {"case": name, "original_size": original_size.tolist()}
        for rule in ("floor", "ceil"):
            entry[rule] = tensor_metrics(letterbox(decoded, pad_rounding=rule), tensor)
        report["resampling"].append(entry)

    for height, width in DEGENERATE_SHAPES:
        entry = {"input_hw": [height, width]}
        try:
            tensor, _ = run(letterbox_pipe, encode(np.full((height, width, 3), 128, np.uint8)))
            entry["outcome"] = "tensor"
            entry["nonzero"] = int(np.count_nonzero(tensor))
        except Exception as error:  # noqa: BLE001 - recording DALI's behaviour is the point
            entry["outcome"] = "error"
            entry["message"] = str(error)[:500]
        report["degenerate"].append(entry)

    rules = [rule for rule in ("floor", "ceil")
             if all(item[rule]["matches"] for item in report["geometry"])]
    report["pad_rounding"] = rules[0] if len(rules) == 1 else (rules or "neither")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"pad_rounding": report["pad_rounding"]}))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
