# Mission — Tritonic

## Problem
Running vision and multimodal models behind Nvidia Triton or an OpenAI-compatible
chat endpoint takes per-model client glue: input encoding, pre/postprocessing,
rendering, and transport tuning (HTTP/gRPC, shared memory, ensembles).

## Audience
Engineers deploying and benchmarking CV models (detection, segmentation,
classification, pose, depth, optical flow, video classification) on Triton, and
users who want one CLI for VLM/LLM chat endpoints. It is not a training
framework, not a model zoo, and not a Triton server distribution.

## Product Promise
One C++20 client that runs a model on Triton or a chat endpoint with a single
command. Model-specific logic lives in `neuriplo-tasks`, and the client is
measured on real GPUs against its CPU and server-side alternatives.

## Success Criteria
- Every model type in the AGENTS.md tag table runs end to end with the
  documented command.
- Server-side GPU paths (ensembles) match the CPU reference path within
  recorded tolerances, with published latency evidence.
- Unit tests run without a live server (mock `ITriton` / `IChatBackend`).

## Product Principles
- Correctness parity before speed: a faster path that changes detections is a bug.
- Performance claims cite measured evidence with hardware and version identifiers.
- Defaults stay backward compatible, and new behavior is opt-in through flags.

## Open Questions
- None recorded.

_Revision: 2026-10-01 — initial constitution reconstructed from AGENTS.md, README.md
and docs/ENSEMBLE_INFERENCE_ROADMAP.md._
