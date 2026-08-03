# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Version numbers follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.4.0] - 2026-08-03

### Added
- YOLO11-seg GPU ensemble deployment: DALI GPU preprocess → TensorRT → DALI GPU
  postprocess, in polygon and packed-mask variants, with the CUDA DALI plugins,
  pipeline generators, model repository configs and setup script.
- YOLO26 detection GPU ensemble deployment for the NMS-free detection head,
  emitting the `NUM_DETECTIONS`/`BOXES`/`SCORES`/`CLASSES` envelope.
- DALI ensemble inference with GPU segmentation output paths: GPU polygon
  output, GPU packed-mask output, and video processing on both.
- `--model_type` support for the YOLO11-seg and YOLO26 ensembles with
  `--input_mode=encoded-image` and `--postprocess_mode=gpu`.
- Benchmark harnesses for YOLO11-seg and YOLO26-det (four-path CPU/GPU
  preprocess × postprocess timing), rebuilt on shared CPU-vs-GPU agreement
  checkers in `benchmarks/common`, with a generated dense fixture that
  exercises the detection cap.
- Tests pinning the empty-frame GPU segmentation contract and the GPU
  detection decode/validation paths.
- Per-model deployment READMEs, the YOLO11-seg ensemble README, and a
  documentation index (`docs/README.md`).

### Changed
- Pinned neuriplo-tasks to `v0.7.0`, replacing a commit-SHA pin that referenced
  an unreleased feature branch. Brings in polygon segmentation output.
- Deployment scripts take their container images, CUDA architecture, build
  parallelism and DALI cache volume from overridable environment variables
  (`TRITON_IMAGE`, `TENSORRT_IMAGE`, `CUDA_ARCH`, `BUILD_JOBS`, `DALI_VOLUME`)
  instead of repeated literals.
- Deployment naming is scale-agnostic: manifests record the model family plus
  the actual engine file and its sha256, while benchmark results carry the
  engine label so timings stay attributable.
- Restructured the README around a nested model index, split the Docker script
  and chat backend references into their own documents, and rewrote the CI
  workflows README to describe the pipeline that actually runs.

### Fixed
- YOLO11 GPU postprocess ranked and capped detections incorrectly: candidates
  are now sorted by score before the cap, NMS runs before the output limit is
  applied, and the channel stride is derived from the tensor shape instead of a
  hardcoded 8400-anchor assumption. GPU postprocessing now matches CPU
  postprocessing exactly on all fixtures.
- YOLO11 emitted a truncated `MASK_OFFSETS` tensor when a frame contained no
  detections, aborting any video whose first frame was empty.
- YOLO26 detection validated neither rank nor shape of the output tensor before
  reading it, over-copying from the device for engines emitting fewer rows; the
  normalized-coordinates heuristic is now an explicit DALI argument.
- Ensemble setup scripts are reproducible from a clean checkout: they install
  the `config.pbtxt` files, the YOLO26 detection model name is consistent
  across script, config and ensembles, and the generated manifest records the
  correct detection output shape.
- Processed video output is named after the serving model instead of a fixed
  `processed.avi` that successive runs silently overwrote.
- Benchmark metadata recorded the raw model type as the family, GPU-mode
  dispatch compared an unnormalized model type, and the encoded-image video
  path decoded the GPU envelope even under CPU postprocessing.

## [0.3.0] - 2026-07-15

### Added
- Automatic GitHub Release publication from curated changelog sections for
  version tags reachable from `master`.

### Changed
- Migrated the neuriplo-tasks boundary from OpenCV types to native
  `vision::Image`, `vision::Size`, and `vision::PixelType` values while
  retaining OpenCV inside tritonic through the optional adapter target.
- Pinned neuriplo-tasks to `v0.6.0` and linked
  `neuriplo-tasks::vision-opencv`.

## [0.2.0] - 2026-07-01

### Added
- Added `--inference_timeout=<ms>` for chat CURL calls, Triton infer requests, and model-load readiness waits.

### Changed
- Batched image inference now respects the configured `--batch_size`, capped by the model `max_batch_size`.

## [0.1.0] - 2026-06-25

### Added
- Versioning infrastructure: `VERSION` file (read by CMake), `CHANGELOG.md`,
  and a gitflow release workflow documented in `docs/Versioning.md`.
- Batched image inference via the neuriplo-tasks v0.5.0 Track B helpers
  (`batchPreprocess` / `batchPostprocess`). Independent-image tasks
  (classification, detection, instance segmentation, pose, depth, open-vocab)
  now run through a single batched Triton call when `ModelInfo.max_batch_size_`
  is greater than 1, falling back to the per-image loop otherwise. See
  `AGENTS.md` → "Batched image inference" for details.
- Pinned `neuriplo-tasks` to **v0.5.0**, which adds the `RfDetrPose` keypoint
  task and batch-ready postprocessors/preprocess strategies.

[Unreleased]: https://github.com/olibartfast/tritonic/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/olibartfast/tritonic/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/olibartfast/tritonic/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/olibartfast/tritonic/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/olibartfast/tritonic/releases/tag/v0.1.0
