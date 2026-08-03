# YOLO GPU Preprocessing Ensemble Contract

This directory will contain the Triton model repositories for the DALI reference
implementation and the later native C++/CUDA implementation. The two
preprocessors must implement the same external tensor contract.

## Versioned reference

- Tritonic dependency: `neuriplo-tasks` v0.6.0, pinned in the root
  `CMakeLists.txt`.
- Reference task: `--model_type=yolo` with a YOLO11s TensorRT model.
- Reference input shape: `[N, 3, 640, 640]`, `FP32`, `FORMAT_NCHW`.
- First milestone: one encoded JPEG per request, batch size 1.

The checked reference repository expects TensorRT input `images` and output
`output0` with shape `[N, 84, 8400]`. Other exports require their own
component and ensemble configs because export settings can change tensor names
and output layouts.

## Exact preprocessing semantics

The server-side implementation must reproduce `YoloPreprocessor` from
`neuriplo-tasks` v0.6.0:

1. Decode the image as 8-bit, three-channel BGR, matching `cv::imread` color
   semantics and honoring the JPEG orientation behavior selected for the golden
   fixtures.
2. Compute `scale = min(640 / source_width, 640 / source_height)`.
3. Compute scaled dimensions by truncation toward zero:
   `new_width = int(source_width * scale)` and
   `new_height = int(source_height * scale)`.
4. Resize to `new_width x new_height` with linear interpolation.
5. Allocate a zero-filled `640 x 640 x 3` `UINT8` canvas. Padding value is
   `[0, 0, 0]`, not the common Ultralytics value 114.
6. Place the resized image at
   `x_offset = (640 - new_width) / 2` and
   `y_offset = (640 - new_height) / 2`, using integer division. An odd remainder
   therefore falls on the right or bottom.
7. Swap BGR to RGB.
8. Convert to `FP32` and multiply by `1/255`. Do not apply ImageNet mean/std
   normalization.
9. Convert interleaved HWC to planar NCHW in R, G, B plane order.

The produced buffer contains `3 * 640 * 640` native-endian floats per sample.

## Ensemble boundary

The public ensemble input for the first milestone is:

| Name | Triton type | Shape | Meaning |
|---|---|---|---|
| `IMAGE` | `TYPE_UINT8` | `[-1]` | Original encoded JPEG bytes |

The DALI or CUDA preprocessing component produces the TensorRT model's declared
input tensor. The ensemble exposes the TensorRT model outputs unchanged until
the separately gated GPU-postprocessing phase.

## DALI reference component

Generate the pinned DALI 1.51.2 artifact with the locally matched Triton 25.12
image:

```bash
docker run --gpus all --rm \
  -v "$(pwd):/workspace" -w /workspace \
  nvcr.io/nvidia/tritonserver:25.12-py3 \
  python3 deploy/object_detection/yolo/ensemble/dali/generate_pipeline.py \
    --output deploy/object_detection/yolo/ensemble/dali/model_repository/yolo_dali_preprocess/1/model.dali
```

After generating the CPU baseline, run the GPU tensor-parity gate:

```bash
docker run --gpus all --rm \
  -v "$(pwd):/workspace:ro" \
  -v /tmp/tritonic-yolo-baseline:/baseline:ro \
  nvcr.io/nvidia/tritonserver:25.12-py3 \
  python3 /workspace/integration-tests/yolo-ensemble/compare_dali.py
```

The checked limits are normalized-pixel error bounds: max 0.15, mean 0.003,
and p99 0.016. Geometry mismatches such as round-versus-truncate letterbox sizes
exceed these limits and fail the test. Detection parity is checked separately
through the existing `neuriplo-tasks` postprocessor.

Tritonic invokes the ensemble with the ordinary Triton infer API. It must read
the original JPEG bytes without decode/re-encode. The inner TensorRT model
metadata, rather than the encoded ensemble input metadata, constructs the
`neuriplo-tasks` postprocessor. With frame display and writing disabled,
Tritonic reads dimensions from JPEG headers and does not decode on the CPU;
rendering requests still require a client-side decode.

## Reproducible Option 1 workflow

Start with a TensorRT FP32 plan whose bindings match the checked `yolo_trt`
config: `images [1,3,640,640] FP32` to `output0 [1,84,8400] FP32`.
Install it and serialize the DALI pipeline:

```bash
deploy/object_detection/yolo/ensemble/dali/setup_model_repository.sh \
  --engine /path/to/yolo11s_fp32.plan
```

Start Triton with strict component configs:

```bash
docker run --gpus all --rm \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v "$(pwd)/deploy/object_detection/yolo/ensemble/dali/model_repository:/models:ro" \
  nvcr.io/nvidia/tritonserver:25.12-py3 \
  tritonserver --model-repository=/models --disable-auto-complete-config
```

Run the encoded-JPEG ensemble over HTTP or gRPC:

```bash
./build/tritonic \
  --source=data/images/bus.jpg --model_type=yolo \
  --model=yolo_dali_ensemble --task_model=yolo_trt \
  --input_mode=encoded-image --labelsFile=labels/coco.txt \
  --protocol=http --port=8000 --write_frame=false

./build/tritonic \
  --source=data/images/bus.jpg --model_type=yolo \
  --model=yolo_dali_ensemble --task_model=yolo_trt \
  --input_mode=encoded-image --labelsFile=labels/coco.txt \
  --protocol=grpc --port=8001 --write_frame=false
```

Run all correctness gates:

```bash
cmake --build build --target tritonic tritonic_unit_tests yolo_preprocess_baseline
./build/tritonic_unit_tests
integration-tests/yolo-ensemble/run.py --output-dir /tmp/tritonic-yolo-baseline

docker run --gpus all --rm \
  -v "$(pwd):/workspace:ro" \
  -v /tmp/tritonic-yolo-baseline:/baseline:ro \
  nvcr.io/nvidia/tritonserver:25.12-py3 \
  python3 /workspace/integration-tests/yolo-ensemble/compare_dali.py

python3 integration-tests/yolo-ensemble/compare_inference.py
```

The inference gate compares detection count, class, confidence within 0.03, and
restored boxes within 12 pixels for all fixtures over both protocols. It also
sends invalid encoded bytes, requires controlled HTTP and gRPC errors, and
verifies that Triton remains ready.

### Local GPU acceptance evidence

The 2026-07-21 acceptance run used Triton 25.12, DALI 1.51.2, TensorRT 10.14.1,
and an RTX 3060 Laptop GPU with driver 580.159.03. The reference engine SHA-256
was `8c1f1ec3e1b5509fd1edac64582c07233d8bc91cd4d1cd97d80d3412d8fb2b48`.
All four fixtures matched over HTTP and gRPC. After one warmup request per
model and protocol, one four-image run measured:

| Protocol | Direct mean | Ensemble mean | Direct p95 | Ensemble p95 |
|---|---:|---:|---:|---:|
| HTTP | 112.75 ms | 69.5 ms | 118 ms | 77 ms |
| gRPC | 125.5 ms | 74 ms | 129 ms | 80 ms |

Triton metrics reached 14% GPU utilization during 500 sequential encoded
requests and reported 510 MiB GPU memory in use. These figures are
hardware-specific acceptance evidence, not a portable performance guarantee.

## Golden fixtures

The initial tracked inputs are existing repository images so no duplicate binary
fixtures are required:

| Fixture | Dimensions | SHA-256 | Coverage |
|---|---:|---|---|
| `data/images/bus.jpg` | 810x1080 | `33b198a1d2839bb9ac4c65d61f9e852196793cae9a0781360859425f6022b69c` | Portrait |
| `data/images/horses.jpg` | 773x512 | `c8f0a677a1356569e2ce71d2fa88c1030c0ae57ecf5e14170e02d9a86a20dcb4` | Non-integer resize |
| `data/images/person.jpg` | 640x424 | `cdcbab947e46110fc2b77784ac54ddbbab2640f1e44cb5e91fc8984a9793a7d1` | Wide, non-square |
| `data/images/mug.jpg` | 1280x960 | `8461bab2b8c2ea98bb381ce4d69d86252aed3be7952c921147a0737bca1cd50c` | Progressive JPEG |

Regenerate and verify the checked CPU baseline with:

```bash
cmake --build build --target yolo_preprocess_baseline
integration-tests/yolo-ensemble/run.py --output-dir /tmp/tritonic-yolo-baseline
```

The harness checks source hashes, regenerates the untracked FP32 tensors, and
compares their hashes with
`tests/data/ensemble/yolo/cpu_preprocess_sha256.json`.

Before DALI acceptance, the baseline harness must store or regenerate:

- the CPU-preprocessed tensor for each fixture;
- the TensorRT output tensors for the pinned engine;
- final detections at the pinned confidence and NMS thresholds;
- absolute and relative comparison tolerances;
- engine hash, Triton image, GPU, driver, and repository commit identifiers.

## Model-specific manifest

Every installed TensorRT engine must provide a checked manifest containing:

```text
engine_sha256:
triton_image:
input_name:
input_shape:
input_datatype:
input_format:
output_names:
output_shapes:
output_datatypes:
confidence_threshold:
nms_threshold:
```

An engine is not the reference engine until every field is populated from live
Triton metadata and the golden detection run passes over HTTP and gRPC.
