#!/usr/bin/env python3
"""nvJPEG decode step parity against OpenCV (spec R-5, V-9).

Sends each fixture to the decode-only model and compares DECODED with OpenCV's
decode converted to RGB. nvJPEG and libjpeg-turbo use different IDCT and chroma
upsampling, so a few pixels differ by more than one level. The gate therefore
bounds the mean and p99 and records the max. Channel order is checked
separately: the error with R and B swapped must be clearly larger.

Needs: pip install "tritonclient[http]" opencv-python-headless numpy
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import tritonclient.http as httpclient

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "benchmarks"))
from common.fixtures import RESOLUTION_FIXTURES, make_resolution_fixture  # noqa: E402

MEAN_LIMIT = 1.0
P99_LIMIT = 3.0
STOCK = ["bus.jpg", "horses.jpg", "person.jpg", "mug.jpg"]


def decode(client, model, path: Path):
    data = np.fromfile(path, dtype=np.uint8).reshape(1, -1)
    tensor = httpclient.InferInput("IMAGE", list(data.shape), "UINT8")
    tensor.set_data_from_numpy(data)
    result = client.infer(model, [tensor], outputs=[
        httpclient.InferRequestedOutput("DECODED"),
        httpclient.InferRequestedOutput("ORIGINAL_SIZE"),
    ])
    return result.as_numpy("DECODED")[0], result.as_numpy("ORIGINAL_SIZE")[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="yolo26det_nvjpeg_decode")
    parser.add_argument("--url", default="localhost:8000")
    parser.add_argument("--output", type=Path,
                        default=Path("/tmp/tritonic-gpu-preprocess/decode.json"))
    args = parser.parse_args()

    client = httpclient.InferenceServerClient(args.url)
    results = []
    with tempfile.TemporaryDirectory() as work:
        paths = [REPO / "data/images" / name for name in STOCK]
        paths += [make_resolution_fixture(REPO / "data/images/bus.jpg", Path(work) / f"{n}.jpg", s)
                  for n, s in RESOLUTION_FIXTURES.items()]
        gray = Path(work) / "bus_gray.jpg"
        cv2.imwrite(str(gray), cv2.imread(str(paths[0]), cv2.IMREAD_GRAYSCALE))
        paths.append(gray)

        for path in paths:
            expected = cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            actual, original_size = decode(client, args.model, path)
            entry = {"fixture": path.name, "shape": list(actual.shape),
                     "original_size": original_size.tolist()}
            shape_ok = actual.shape == expected.shape
            size_ok = original_size.tolist() == list(expected.shape[:2])
            if shape_ok:
                diff = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
                swapped = np.abs(actual[..., ::-1].astype(np.int16) - expected.astype(np.int16))
                entry.update(max_abs=int(diff.max()), mean_abs=float(diff.mean()),
                             p99_abs=float(np.quantile(diff, 0.99)),
                             swapped_mean_abs=float(swapped.mean()))
                # A grayscale source has R = G = B, so swapping channels changes nothing there.
                order_ok = path is gray or entry["swapped_mean_abs"] > 2 * entry["mean_abs"]
                entry["passed"] = (size_ok and order_ok and entry["mean_abs"] <= MEAN_LIMIT
                                   and entry["p99_abs"] <= P99_LIMIT)
            else:
                entry["passed"] = False
            results.append(entry)
            print(f"{'PASS' if entry['passed'] else 'FAIL'} {path.name:<16} "
                  f"max={entry.get('max_abs')} mean={entry.get('mean_abs', 0):.3f} "
                  f"p99={entry.get('p99_abs')}")

    report = {"model": args.model, "limits": {"mean_abs": MEAN_LIMIT, "p99_abs": P99_LIMIT},
              "results": results, "passed": all(item["passed"] for item in results)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
