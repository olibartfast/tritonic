#!/usr/bin/env python3
"""Generate comparison.md for the GPU preprocessing variants (spec R-14, R-15, V-15).

Inputs, all inside one result directory:
  summary.json          run.py output (schema v4), required
  perf.json             perf.py output, optional
  gates/*.json          check reports, optional:
                          tensors_<VARIANT>.json   compare_tensors.py
                          detections.json          compare_detections.py
                          invalid_inputs.json      invalid_inputs.py
                          decode.json              compare_decode.py
                          soak.json                soak.py
  recommendation.md     the reviewed recommendation text, optional

The output depends only on these files and the repository tree, so rerunning
reproduces it byte for byte (V-15). The recommendation is written by a person
from the numbers and embedded verbatim, not generated.

    python3 benchmarks/yolo26-det/compare.py benchmarks/yolo26-det/results/<dir>
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
STAGES = ("preprocess", "infer", "postprocess", "total")

# Variant-specific sources, for the line counts in the cost-of-ownership table.
SOURCES = {
    "B1": ["deploy/object_detection/yolo26/ensemble/generate_pipeline.py"],
    "V12": ["backends/yolopreprocess/src", "backends/yolopreprocess/CMakeLists.txt",
            "backends/yolopreprocess/build_backend.sh", "src/cuda/letterbox.cu",
            "include/tritonic/cuda/letterbox_pixel.hpp",
            "include/tritonic/cuda/letterbox_geometry.hpp",
            "include/tritonic/cuda/letterbox.hpp"],
    "V13": ["backends/yolopreprocess/src", "backends/yolopreprocess/CMakeLists.txt",
            "backends/yolopreprocess/build_backend.sh",
            "deploy/object_detection/yolo26/ensemble/trt_pre"],
    "V2": ["src/cuda", "include/tritonic/cuda", "include/tritonic/core/device_preprocessor.hpp",
           "include/tritonic/triton/device_input.hpp"],
}
OWNERSHIP = {
    "B0": ("none", "none", "neuriplo-tasks release"),
    "B1": ("serialize DALI pipeline in the Triton image",
           "Triton DALI backend",
           "DALI version of each Triton release (re-serialize per bump)"),
    "V12": ("build backend .so in the TensorRT image",
            "Triton backend SDK, nvJPEG",
            "backend SDK tag + CUDA of each Triton release (rebuild per bump)"),
    "V13": ("build backend .so; build engine with in-graph letterbox",
            "Triton backend SDK, nvJPEG, TensorRT ONNX parser",
            "backend SDK tag + CUDA; engine per TensorRT version and per model"),
    "V2": ("build tritonic with TRITONIC_ENABLE_CUDA=ON",
           "CUDA toolkit on the client",
           "client CUDA runtime vs. driver; same host and GPU as Triton"),
}
SOURCE_SUFFIXES = {".cc", ".cpp", ".cu", ".hpp", ".h", ".py", ".sh", ".txt"}


def count_lines(paths) -> int:
    total = 0
    for relative in paths:
        path = REPO / relative
        files = sorted(path.rglob("*")) if path.is_dir() else [path]
        for file in files:
            if file.is_file() and file.suffix in SOURCE_SUFFIXES:
                total += sum(1 for line in file.read_text(errors="replace").splitlines()
                             if line.strip())
    return total


def load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def fmt(value, digits=2, suffix=""):
    return "—" if value is None else f"{value:.{digits}f}{suffix}"


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def gate_status(gates: Path, variants):
    detections = load(gates / "detections.json")
    invalid = load(gates / "invalid_inputs.json")
    soak = load(gates / "soak.json")
    decode = load(gates / "decode.json")

    def verdict(items):
        if not items:
            return "not run"
        return "pass" if all(item["passed"] for item in items) else "**FAIL**"

    rows = []
    for key in variants:
        base = key.split("-")[0]
        tensors = load(gates / f"tensors_{base}.json")
        rows.append([
            key,
            verdict(tensors["results"]) if tensors else "not run",
            verdict([r for r in (detections or {}).get("results", []) if r["candidate"] == key]),
            verdict([r for r in (invalid or {}).get("results", [])
                     if r["variant"] == base and r.get("gated", True)]),
            verdict([r for r in (soak or {}).get("results", []) if r["variant"] == key]),
            verdict((decode or {}).get("results", [])) if base == "V13" else "n/a",
        ])
    return table(["Variant", "Tensor parity", "Detection parity", "Invalid input", "Soak",
                  "Decode parity"], rows)


def latency_section(summary):
    parts = []
    for protocol, fixtures in summary["protocols"].items():
        rows = []
        for key in summary["variants"]:
            runs = [fixture[key] for fixture in fixtures.values() if "timings" in fixture[key]]
            if not runs:
                rows.append([key, *["failed"] * 5])
                continue
            medians = [statistics.mean(run["timings"][stage]["median_ms"] for run in runs)
                       for stage in STAGES]
            p95 = statistics.mean(run["timings"]["total"]["p95_ms"] for run in runs)
            rows.append([key, *(fmt(value) for value in medians), fmt(p95)])
        parts.append(f"### {protocol}\n\nMean over fixtures of per-fixture medians (ms).\n\n"
                     + table(["Variant", "Preprocess", "Infer", "Postprocess", "Total",
                              "Total p95"], rows))

        per_fixture = []
        for name, entries in fixtures.items():
            per_fixture.append([name, *(fmt(entries[key]["timings"]["total"]["median_ms"])
                                        if "timings" in entries[key] else "failed"
                                        for key in summary["variants"])])
        parts.append(f"Total median per fixture, {protocol} (ms):\n\n"
                     + table(["Fixture", *summary["variants"]], per_fixture))
    return "\n\n".join(parts)


def server_section(summary):
    protocol = next(iter(summary["protocols"]))
    fixtures = summary["protocols"][protocol]
    rows = []
    for key in summary["variants"]:
        servers = [fixture[key]["server"] for fixture in fixtures.values()
                   if "server" in fixture[key]]
        if not servers:
            continue
        for model in servers[0]:
            values = [s[model]["compute_infer_ms"] for s in servers
                      if s[model]["compute_infer_ms"] is not None]
            inputs = [s[model]["compute_input_ms"] for s in servers
                      if s[model]["compute_input_ms"] is not None]
            rows.append([key, model, fmt(statistics.mean(inputs) if inputs else None, 3),
                         fmt(statistics.mean(values) if values else None, 3)])
    return (f"Per-model server time per inference, {protocol}, mean over fixtures (ms).\n\n"
            + table(["Variant", "Model", "compute_input", "compute_infer"], rows))


def perf_section(perf):
    if not perf:
        return "_perf.json not present._"
    rows = []
    for key, levels in perf["variants"].items():
        if "skipped" in levels:
            rows.append([key, "—", "—", "—", "—", f"skipped: {levels['skipped']}"])
            continue
        for level, data in sorted(levels.items(), key=lambda item: int(item[0])):
            if "error" in data:
                rows.append([key, level, "—", "—", "—", "error"])
                continue
            rows.append([key, level, fmt(data["inferences_per_second"], 1),
                         fmt(data["p95_us"] / 1000), fmt(data["gpu_utilization_mean"], 2),
                         fmt(data["gpu_memory_peak_mib"], 0, " MiB")])
    return (f"perf_analyzer on {perf['fixture']}, gRPC.\n\n"
            + table(["Variant", "Concurrency", "Infer/s", "p95 ms", "GPU util", "Peak GPU mem"],
                    rows))


def bytes_section(summary):
    protocol = next(iter(summary["protocols"]))
    fixtures = summary["protocols"][protocol]
    rows = [[name, *(f"{entries[key]['request_bytes']:,}" for key in summary["variants"])]
            for name, entries in fixtures.items()]
    return table(["Fixture", *summary["variants"]], rows)


def ownership_section(variants):
    rows = []
    for key in variants:
        base = key.split("-")[0]
        build, deps, coupling = OWNERSHIP.get(base, ("—", "—", "—"))
        lines = count_lines(SOURCES[base]) if base in SOURCES else 0
        rows.append([key, lines, build, deps, coupling])
    return ("Source lines count non-blank lines of each variant's own files; the shared "
            "letterbox kernel is counted in every variant that uses it.\n\n"
            + table(["Variant", "Source lines", "Build steps", "Extra dependencies",
                     "Version coupling"], rows))


def environment_section(summary):
    env = summary["environment"]
    gpus = ", ".join(f"{g['name']} (driver {g['driver']})" for g in env["gpus"]) or "unknown"
    server = env["server_image"]
    return table(["Item", "Value"], [
        ["GPU", gpus],
        ["Client CPU", env["client_cpu"]],
        ["Server image", f"{server['image']} {', '.join(server.get('digest') or ['no digest'])}"],
        ["CUDA / TensorRT", f"{server.get('cuda_version')} / "
                            f"{server.get('trt_version') or server.get('tensorrt_version')}"],
        ["Images consistent (A-4)", env["images_consistent"]],
        ["Commit", f"{env['commit']} ({env['dirty_files']} dirty files)"],
        ["Iterations", f"{summary['warmup']} warmup + {summary['iterations']} measured"],
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()

    summary = load(args.result_dir / "summary.json")
    if not summary or summary.get("schema_version") != 4:
        sys.exit("summary.json (schema v4) not found")
    variants = list(summary["variants"])
    recommendation = load_text(args.result_dir / "recommendation.md")

    document = "\n\n".join([
        f"# YOLO26 detection: GPU preprocessing comparison ({summary['label']})",
        "Generated by `benchmarks/yolo26-det/compare.py` from `summary.json`, `perf.json` and "
        "`gates/`. Spec: `specs/2026-10-01-gpu-preprocess-alternatives/`.",
        "## Environment", environment_section(summary),
        "## Variants", table(["Key", "Description", "Request model"],
                             [[k, v["title"], f"`{v['model']}`"]
                              for k, v in summary["variants"].items()]),
        "## Correctness gates", gate_status(args.result_dir / "gates", variants),
        "## Client latency", latency_section(summary),
        "## Server breakdown", server_section(summary),
        "## Throughput", perf_section(load(args.result_dir / "perf.json")),
        "## Request bytes", bytes_section(summary),
        "## Cost of ownership", ownership_section(variants),
        "## Recommendation",
        recommendation or "_Pending: write `recommendation.md` in the result directory after "
                          "reviewing the numbers above, then regenerate._",
    ]) + "\n"
    (args.result_dir / "comparison.md").write_text(document)
    print(args.result_dir / "comparison.md")
    return 0


def load_text(path: Path):
    return path.read_text().strip() if path.exists() else None


if __name__ == "__main__":
    raise SystemExit(main())
