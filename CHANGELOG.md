# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Version numbers follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

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

[Unreleased]: https://github.com/olibartfast/tritonic/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/olibartfast/tritonic/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/olibartfast/tritonic/releases/tag/v0.1.0
