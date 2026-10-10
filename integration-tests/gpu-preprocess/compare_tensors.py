#!/usr/bin/env python3
"""Tensor parity between two directories of FP32 [3,640,640] tensors (spec V-5).

Pairs `<stem>.fp32` files by name and applies the spec D-5 thresholds, the same
ones the DALI path was accepted with. Typical uses:

    # kernel (V2/V12 shared code) against DALI, both from data/images fixtures
    compare_tensors.py --reference /tmp/tritonic-spikes/dali --actual /tmp/kernel

Exits non-zero when a reference tensor has no counterpart or any pair fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tensor_metrics import EXPECTED_SHAPE, passes, tensor_metrics  # noqa: E402


def load(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.float32)
    expected = int(np.prod(EXPECTED_SHAPE))
    if data.size != expected:
        raise ValueError(f"{path}: {data.size} floats, expected {expected}")
    return data.reshape(EXPECTED_SHAPE)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--actual", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    references = sorted(args.reference.glob("*.fp32"))
    if not references:
        raise SystemExit(f"no .fp32 tensors in {args.reference}")

    results = []
    for reference in references:
        actual = args.actual / reference.name
        if not actual.exists():
            results.append({"tensor": reference.name, "passed": False, "error": "missing"})
            print(f"FAIL {reference.name:<24} missing in {args.actual}")
            continue
        metrics = tensor_metrics(load(reference), load(actual))
        metrics["passed"] = passes(metrics)
        results.append({"tensor": reference.name, **metrics})
        print(
            f"{'PASS' if metrics['passed'] else 'FAIL'} {reference.name:<24} "
            f"max={metrics['max_abs']:.6f} mean={metrics['mean_abs']:.6f} "
            f"p99={metrics['p99_abs']:.6f}"
        )

    report = {"reference": str(args.reference), "actual": str(args.actual), "results": results,
              "passed": all(item["passed"] for item in results)}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
