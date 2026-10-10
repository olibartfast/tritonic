# YOLO26 Detection DALI/TensorRT Ensemble

Server-side GPU pipeline: DALI JPEG decode + letterbox resize → TensorRT
inference → CUDA bbox decoding.  NMS remains client-side.

## Setup

Prepare the repository from a TensorRT 10.14.1 FP16 YOLO26 detection engine:

```bash
deploy/object_detection/yolo26/ensemble/setup_model_repository.sh \
  --engine /path/to/yolo26-det.engine
```

The setup builds the custom DALI CUDA postprocess operator, serializes
preprocessing and bbox-decoding pipelines, and stages the engine in
`model_repository/`.

Start Triton with the DALI plugin enabled:

```bash
docker run --rm --gpus all --shm-size=1g \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v "$PWD/deploy/object_detection/yolo26/ensemble/model_repository:/models:ro" \
  nvcr.io/nvidia/tritonserver:26.08-py3 \
  tritonserver --model-repository=/models --disable-auto-complete-config \
  --allow-client-shm=true --backend-config=dali,plugin_libs=/models/libyolo26_det_dali.so
```

`--allow-client-shm=true` is required since Triton 26.08 for clients that register CUDA or system shared memory (variant V2).

Models:
- `yolo26det_trt`: raw TensorRT model
- `yolo26det_gpu_pre_cpu_post`: DALI preprocess + raw TensorRT outputs (client postprocesses)
- `yolo26det_gpu_pre_gpu_post`: DALI preprocess + TensorRT + GPU bbox decode

## Output Contract

The GPU-post ensemble accepts one encoded JPEG in `IMAGE` and returns decoded
detection metadata:

| Tensor | Type | Shape | Meaning |
|---|---|---|---|
| `NUM_DETECTIONS` | INT32 | `[1]` | Valid detections |
| `BOXES` | INT32 | `[100,4]` | `(x, y, width, height)` in original image coords |
| `SCORES` | FP32 | `[100]` | Confidence scores |
| `CLASSES` | INT32 | `[100]` | Class identifiers |

Run the GPU-post ensemble:

```bash
./build/tritonic \
  --source=data/images/bus.jpg \
  --model_type=yolo26 \
  --model=yolo26det_gpu_pre_gpu_post \
  --task_model=yolo26det_trt \
  --labelsFile=labels/coco.txt \
  --input_mode=encoded-image \
  --postprocess_mode=gpu
```

## Ensemble Boundary

The public ensemble input:

| Name | Triton type | Shape | Meaning |
|---|---|---|---|
| `IMAGE` | `TYPE_UINT8` | `[-1]` | Encoded JPEG bytes |

The DALI preprocessing component produces the TensorRT model's declared input
tensor.  The ensemble exposes decoded bboxes directly; NMS is applied by the
client.

## GPU Preprocessing Semantics

Identical to the [YOLO detection ensemble contract](../../yolo/ensemble/README.md):
decode as RGB, letterbox to 640×640 with zero padding (matching
`neuriplo-tasks` v0.6.0 `YoloPreprocessor`), normalize by 1/255, no ImageNet
mean/std.

## Non-DALI GPU preprocessing variants

Three CUDA C++ alternatives to the DALI preprocessing step are being evaluated against
it ([spec](../../../../specs/2026-10-01-gpu-preprocess-alternatives/requirements.md)).
They use the same letterbox semantics as above and are opt-in: the default
repository contains only the DALI models.

| Variant | Request model | Preprocessing |
|---|---|---|
| V12 | `yolo26det_cudapre_cpu_post`, `yolo26det_cudapre_gpu_post` | Custom Triton backend `yolopreprocess`: nvJPEG decode + CUDA letterbox kernel |
| V13 | `yolo26det_trtpre_cpu_post`, `yolo26det_trtpre_gpu_post` | Same backend in decode-only mode, then letterbox inside the TensorRT engine `yolo26det_trt_pre` |
| V2 | `yolo26det_trt` | Client-side CUDA letterbox written into CUDA shared memory (`--preprocess_device=cuda`) |

Stage V12 and V13 next to the DALI models (the variant configs come from
`variants/`; V13 also needs the YOLO26 ONNX export of the same model):

```bash
deploy/object_detection/yolo26/ensemble/setup_model_repository.sh \
  --engine /path/to/yolo26-det.engine \
  --trt-pre-onnx /path/to/yolo26-det.onnx
```

`--with-cuda-backend` stages only V12. Triton loads the backend from
`libtriton_yolopreprocess.so` in each model directory, so the `docker run` command
above needs no extra mount.

Run V12 or V13 exactly like the DALI ensembles, changing only `--model`:

```bash
./build/tritonic --source=data/images/bus.jpg --model_type=yolo26 \
  --model=yolo26det_cudapre_cpu_post --task_model=yolo26det_trt \
  --labelsFile=labels/coco.txt --input_mode=encoded-image
```

V2 needs a client build with `-DTRITONIC_ENABLE_CUDA=ON` on the same host and GPU
as Triton. The Triton server must run with `--allow-client-shm=true` and share the host IPC namespace with the client (`docker run --ipc=host ...`):

```bash
./build-cuda/tritonic --source=data/images/bus.jpg --model_type=yolo26 \
  --model=yolo26det_trt --labelsFile=labels/coco.txt --protocol=grpc --port=8001 \
  --preprocess_device=cuda --shared_memory_type=cuda
```

Results and the recommendation are in the [A10G comparison report](../../../../specs/2026-10-01-gpu-preprocess-alternatives/evidence/2026-10-09-brev-a10g/comparison.md); the gate is reproducible with [run_gpu_host_gate.sh](../../../../integration-tests/gpu-preprocess/run_gpu_host_gate.sh) ([runbook](../../../../specs/2026-10-01-gpu-preprocess-alternatives/gpu-validation-runbook.md)).
