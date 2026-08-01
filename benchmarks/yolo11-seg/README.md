# YOLO11m-seg GPU Ensemble Benchmarks

Four-path benchmark comparing DALI GPU pre/postprocessing against CPU baseline.

## Setup

1. Build tritonic:
```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j$(nproc)
```

2. Deploy the model repository:
```bash
./deploy/instance_segmentation/yolo11/ensemble/setup_model_repository.sh \
  --engine /path/to/yolo11m-seg.engine
```

3. Start Triton server:
```bash
docker run --rm --gpus all --shm-size=1g \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v "$PWD/deploy/instance_segmentation/yolo11/ensemble/model_repository:/models:ro" \
  nvcr.io/nvidia/tritonserver:25.12-py3 \
  tritonserver --model-repository=/models --disable-auto-complete-config \
  --backend-config=dali,plugin_libs=/models/libyolo11_seg_dali.so
```

## Run Benchmarks

```bash
python3 benchmarks/yolo11-seg/run.py
```

This runs 4 paths across 4 fixtures (bus, horses, person, mug):
- `cpu_pre_cpu_post` — standard CPU preprocessing + CPU postprocessing
- `gpu_pre_cpu_post` — DALI GPU preprocess + CPU postprocess
- `gpu_pre_gpu_mask_post` — DALI GPU preprocess + GPU packed-mask postprocess
- `gpu_pre_gpu_post` — DALI GPU preprocess + GPU polygon postprocess

## Validate Results

```bash
python3 benchmarks/yolo11-seg/check_results.py benchmarks/yolo11-seg/results/<run_dir>/
python3 benchmarks/yolo11-seg/check_mask_results.py benchmarks/yolo11-seg/results/<run_dir>/
```

## Architecture

```
IMAGE (JPEG bytes)
    │
    ▼
[DALI preprocess] ── GPU decode, resize, pad, normalize to [3,640,640] FP32
    │
    ▼
[TensorRT] ── YOLO11m-seg TRT engine, outputs:[116,8400] + [32,160,160]
    │
    ▼
[DALI postprocess] ── GPU: NMS → mask synthesis → contour tracing → convex hull
    │
    ▼
BOXES, SCORES, CLASSES, MASK_DATA / POLYGON_POINTS
```
