# YOLO26m-seg DALI/TensorRT ensemble

Prepare the repository from a TensorRT 10.14.1 FP16 `yolo26m-seg` engine:

```bash
deploy/instance_segmentation/yolo26/ensemble/setup_model_repository.sh \
  --engine /path/to/yolo26m-seg.engine
```

The setup builds the custom DALI CUDA postprocess operator against the DALI
libraries from Triton 25.12, serializes both DALI pipelines, and stages the
engine and plugin in `model_repository/`.

Start Triton with the DALI plugin enabled:

```bash
docker run --rm --gpus all --shm-size=1g \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v "$PWD/deploy/instance_segmentation/yolo26/ensemble/model_repository:/models:ro" \
  nvcr.io/nvidia/tritonserver:25.12-py3 \
  tritonserver --model-repository=/models --disable-auto-complete-config \
  --backend-config=dali,plugin_libs=/models/libyolo26_seg_dali.so
```

Models:

- `yolo26seg_trt`: raw TensorRT model.
- `yolo26seg_gpu_pre_cpu_post`: DALI preprocessing plus raw TensorRT outputs.
- `yolo26seg_gpu_pre_gpu_post`: DALI preprocessing, TensorRT, and DALI CUDA polygon postprocessing.

## Polygon output contract

The GPU-post ensemble accepts one encoded JPEG in `IMAGE` and returns detection
metadata plus compact polygon rings:

| Tensor | Type | Shape | Meaning |
|---|---|---|---|
| `NUM_DETECTIONS` | INT32 | `[1]` | Valid detections |
| `BOXES` | INT32 | `[100,4]` | `(x, y, width, height)` |
| `SCORES` | FP32 | `[100]` | Confidence scores |
| `CLASSES` | INT32 | `[100]` | Class identifiers |
| `INSTANCE_RING_OFFSETS` | INT64 | `[101]` | Detection-to-ring offsets |
| `RING_POINT_OFFSETS` | INT64 | `[-1]` | Ring-to-point offsets |
| `POLYGON_POINTS` | INT32 | `[-1,2]` | Image-space `(x, y)` points |

Exterior rings have positive signed area in image coordinates; holes have
negative signed area. The client assigns holes to the smallest containing
exterior. Temporary raster masks remain inside the CUDA operator and are not
part of the public model output.

Run the polygon ensemble with:

```bash
./build/tritonic \
  --source=data/images/bus.jpg \
  --model_type=yolo26seg \
  --model=yolo26seg_gpu_pre_gpu_post \
  --task_model=yolo26seg_trt \
  --labelsFile=labels/coco.txt \
  --input_mode=encoded-image \
  --postprocess_mode=gpu \
  --segmentation_output=polygon
```

`--postprocess_mode=gpu` requires `--segmentation_output=polygon`. CPU
postprocessing accepts `mask` (the default) or `polygon`. See the
[validated three-path benchmark](../../../../benchmarks/yolo26-seg/README.md).
