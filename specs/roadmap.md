# Roadmap

Status values: `idea` · `planned` · `in progress` · `done` · `blocked` · `deferred`

This file tracks work that uses spec-driven feature packets. Earlier plans are
kept where they are:
[feature roadmap](../docs/ROADMAP.md) (async, streaming and other Triton
features) and the [ensemble inference roadmap](../docs/ENSEMBLE_INFERENCE_ROADMAP.md)
(DALI GPU preprocessing, which passed its Option 1 gate on 2026-07-21).

## Phase 1 — GPU preprocessing alternatives to DALI (yolo26det)
- Status: done 2026-10-10. The A10G gate passed (evidence in the packet's
  `evidence/2026-10-09-brev-a10g/`). Recommendation: DALI stays the server-side
  default; V2 for clients on the GPU host; V12 is the DALI-free option; V13 retired.
- Outcome: three CUDA C++ preprocessing variants (custom Triton backend,
  TensorRT-graph preprocessing with an nvJPEG decode step, client-side CUDA),
  each parity-checked against DALI and benchmarked under one protocol, plus a
  comparison report that recommends a default.
- Proves: whether DALI can be replaced for yolo26det without loss of parity, and
  at what latency, throughput and maintenance cost.
- Spec: `specs/2026-10-01-gpu-preprocess-alternatives/`   Branch: `feature/gpu-preprocess-alternatives`
- Supersedes: Phase 2 of the ensemble inference roadmap (native C++/CUDA
  preprocessing), now extended to three variants and to yolo26det.

## Deferred / Revisit
- Python-backend GPU preprocessing — dropped by decision on 2026-10-01 (see
  tech-stack non-choices).
- Batching, video transport and PNG for the CUDA variants — revisit once Phase 1
  picks a default.
- Rolling the chosen variant out to yolo26seg / yolo11seg — after Phase 1.
- V12 large-image decode: at 1080p and 2160p the backend's nvJPEG decode trails DALI
  (8.0 vs 6.5 ms mean server preprocessing, 117 vs 162 infer/s at 1080p). Try the
  GPU-Huffman/hybrid nvJPEG backend chosen by image size, plus batched decode; if V12
  then matches DALI it can become the DALI-free default.
- V2 server throughput: perf_analyzer cannot drive client-side preprocessing; needs a
  dedicated concurrent client benchmark.
- FP16 YOLO26 engines under TensorRT 11 (strongly typed only, no `--fp16`): the ONNX FP16
  conversion leaves mixed-type `Concat` inputs in the end2end head (2026-10-09). The Phase 1
  gate runs FP32 for all variants; the existing README's "TensorRT 10.14.1 FP16 engine"
  instructions need a TensorRT 11 path.

_Revision: 2026-10-10 — Phase 1 done: GPU gate passed on an A10G; follow-ups added._
_Revision: 2026-10-02 — Phase 1 in progress: all tasks have code, and GPU exit checks are pending._
_Revision: 2026-10-01 — created; Phase 1 added from the GPU preprocessing interview._
