#!/usr/bin/env python3
"""Capture the pinned neuriplo-tasks YOLO CPU preprocessing baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

FIXTURES = {
    "data/images/bus.jpg": "33b198a1d2839bb9ac4c65d61f9e852196793cae9a0781360859425f6022b69c",
    "data/images/horses.jpg": "c8f0a677a1356569e2ce71d2fa88c1030c0ae57ecf5e14170e02d9a86a20dcb4",
    "data/images/person.jpg": "cdcbab947e46110fc2b77784ac54ddbbab2640f1e44cb5e91fc8984a9793a7d1",
    "data/images/mug.jpg": "8461bab2b8c2ea98bb381ce4d69d86252aed3be7952c921147a0737bca1cd50c",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()


def capture(args: argparse.Namespace) -> int:
    repo = Path(__file__).resolve().parents[2]
    tool = (repo / args.tool).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    fixture_paths = []
    for relative, expected_hash in FIXTURES.items():
        path = repo / relative
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"fixture hash mismatch for {relative}: {actual_hash} != {expected_hash}"
            )
        fixture_paths.append(path)

    command = [str(tool), str(output_dir), *(str(path) for path in fixture_paths)]
    completed = subprocess.run(command, check=True, text=True, capture_output=True)

    records = []
    opencv_version = None
    for line in completed.stdout.splitlines():
        fields = line.split("\t")
        if fields[0] == "#opencv":
            opencv_version = fields[1]
            continue
        source, width, height, tensor_path, tensor_bytes = fields
        tensor = Path(tensor_path)
        records.append(
            {
                "source": str(Path(source).resolve().relative_to(repo)),
                "source_sha256": sha256(Path(source)),
                "width": int(width),
                "height": int(height),
                "tensor": tensor.name,
                "tensor_bytes": int(tensor_bytes),
                "tensor_sha256": sha256(tensor),
            }
        )

    manifest = {
        "contract": {
            "dependency": "neuriplo-tasks v0.6.0",
            "shape": [1, 3, 640, 640],
            "datatype": "FP32",
            "format": "NCHW",
            "color": "RGB",
            "scale": "1/255",
            "padding": 0,
        },
        "environment": {"opencv": opencv_version},
        "tritonic_commit": git_head(repo),
        "fixtures": records,
    }

    expected_path = repo / "tests/data/ensemble/yolo/cpu_preprocess_sha256.json"
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    comparable = {key: manifest[key] for key in ("contract", "environment", "fixtures")}
    if comparable != expected:
        raise RuntimeError(
            "CPU preprocessing baseline differs from "
            f"{expected_path.relative_to(repo)}"
        )

    manifest_path = output_dir / "cpu_preprocess_baseline.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(manifest_path)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tool",
        default="build/yolo_preprocess_baseline",
        help="path relative to the repository root",
    )
    parser.add_argument("--output-dir", required=True)
    return capture(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
