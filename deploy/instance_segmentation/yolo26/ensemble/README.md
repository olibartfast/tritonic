# YOLO26m-seg DALI/TensorRT ensemble

Prepare the repository from a TensorRT 10.14.1 FP16 `yolo26m-seg` engine:

```bash
deploy/instance_segmentation/yolo26/ensemble/setup_model_repository.sh \
  --engine /path/to/yolo26m-seg.engine
```

The setup builds the custom DALI CUDA postprocess operators against the DALI
libraries from Triton 25.12, serializes preprocessing plus mask and polygon
postprocessing pipelines, and stages the engine and plugin in `model_repository/`.

Start Triton with the DALI plugin enabled:

```bash
docker run --rm --gpus all --shm-size=1g \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v "$PWD/deploy/instance_segmentation/yolo26/ensemble/model_repository:/models:ro" \
  nvcr.io/nvidia/tritonserver:25.12-py3 \
  tritonserver --model-repository=/models --disable-auto-complete-config \
  --backend-config=dali,plugin_libs=/models/libyolo26_seg_dali.so
```

This command does not use `/home/oli/model_repository`. From the Tritonic
repository root, `$PWD/deploy/instance_segmentation/yolo26/ensemble/model_repository`
is the host directory bind-mounted read-only at `/models`; Triton then loads
`--model-repository=/models` inside the container.

Models:

- `yolo26seg_trt`: raw TensorRT model.
- `yolo26seg_gpu_pre_cpu_post`: DALI preprocessing plus raw TensorRT outputs.
- `yolo26seg_gpu_pre_gpu_mask_post`: DALI preprocessing, TensorRT, and mask-only CUDA postprocessing.
- `yolo26seg_gpu_pre_gpu_post`: DALI preprocessing, TensorRT, and polygon CUDA postprocessing.

The GPU mask and polygon contracts are separate models. Selecting the mask
ensemble does not allocate contour scratch or run boundary tracing and convex-hull kernels.

## Mask output contract

The mask-only ensemble accepts one encoded JPEG in `IMAGE` and returns detection
metadata plus packed bbox-local binary masks:

| Tensor | Type | Shape | Meaning |
|---|---|---|---|
| `NUM_DETECTIONS` | INT32 | `[1]` | Valid detections |
| `BOXES` | INT32 | `[100,4]` | `(x, y, width, height)` |
| `SCORES` | FP32 | `[100]` | Confidence scores |
| `CLASSES` | INT32 | `[100]` | Class identifiers |
| `MASK_OFFSETS` | INT64 | `[101]` | Detection-to-mask-byte offsets |
| `MASK_DATA` | UINT8 | `[-1]` | Packed bbox-local masks, values 0 or 255 |

For detection `i`, `MASK_DATA[MASK_OFFSETS[i]:MASK_OFFSETS[i+1]]` is a
row-major mask of shape `(BOXES[i].height, BOXES[i].width)`.

Run the mask ensemble with:

```bash
./build/tritonic \
  --source=data/images/bus.jpg \
  --model_type=yolo26seg \
  --model=yolo26seg_gpu_pre_gpu_mask_post \
  --task_model=yolo26seg_trt \
  --labelsFile=labels/coco.txt \
  --input_mode=encoded-image \
  --postprocess_mode=gpu
```

## Polygon output contract

The GPU-post ensemble accepts one encoded JPEG in `IMAGE` and returns detection
metadata plus compact convex-hull polygon rings:

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
exterior. Contour extraction and convex-hull reduction run inside the CUDA
operator. Temporary raster masks remain device-local and are not part of the
public model output.

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

Video sources use the same arguments. Each OpenCV-decoded frame is JPEG-encoded
in memory and sent through the full DALI-pre/TensorRT/DALI-post ensemble; the
rendered polygon video is written to `<video-directory>/output/processed.avi`.

Use the matching ensemble model for `--segmentation_output=mask` (the default)
or `polygon`; Tritonic rejects a mismatched tensor contract. CPU postprocessing
also accepts either representation. See the
[validated three-path benchmark](../../../../benchmarks/yolo26-seg/README.md).
