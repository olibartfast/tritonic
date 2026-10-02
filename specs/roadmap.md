# Roadmap

Status values: `idea` · `planned` · `in progress` · `done` · `blocked` · `deferred`

This file tracks work that uses spec-driven feature packets. Earlier plans are
kept where they are:
[feature roadmap](../docs/ROADMAP.md) (async, streaming and other Triton
features) and the [ensemble inference roadmap](../docs/ENSEMBLE_INFERENCE_ROADMAP.md)
(DALI GPU preprocessing, which passed its Option 1 gate on 2026-07-21).

## Phase 1 — GPU preprocessing alternatives to DALI (yolo26det)
- Status: in progress (code complete 2026-10-02; GPU validation pending, see the
  packet's plan.md implementation status)
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

_Revision: 2026-10-02 — Phase 1 in progress: all tasks have code, and GPU exit checks are pending._
_Revision: 2026-10-01 — created; Phase 1 added from the GPU preprocessing interview._
