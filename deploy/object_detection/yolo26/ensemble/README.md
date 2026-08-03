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
  nvcr.io/nvidia/tritonserver:25.12-py3 \
  tritonserver --model-repository=/models --disable-auto-complete-config \
  --backend-config=dali,plugin_libs=/models/libyolo26_det_dali.so
```

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
  --model_type=yolo \
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
