# YOLO11-seg DALI/TensorRT ensemble

Same architecture and tensor contracts as the
[YOLO26-seg ensemble](../../yolo26/ensemble/README.md) — DALI GPU preprocessing,
TensorRT inference, and custom CUDA postprocess operators producing either packed
masks or convex-hull polygons. The difference is the detection head: YOLO11 emits a
raw `[116, 8400]` anchor grid in spatial order, so the plugin must rank candidates
by score and run NMS, whereas YOLO26's end-to-end head emits 300 pre-sorted rows.

## Export and deploy

Any scale works — n/s/m/l/x share the same tensor contract, since `[116, 8400]`
and `[32, 160, 160]` depend on image size and class count, not on the scale
variant. Substitute the size letter throughout; `m` is used here as an example.

```bash
# 1. Export ONNX from Ultralytics
python3 -c "from ultralytics import YOLO; YOLO('yolo11m-seg.pt').export(format='onnx', imgsz=640, simplify=True)"

# 2. Build a TensorRT FP16 engine
docker run --rm --gpus all -v "$PWD:/workspace" -w /workspace \
  nvcr.io/nvidia/tensorrt:25.12-py3 \
  trtexec --onnx=yolo11m-seg.onnx --saveEngine=yolo11m-seg.engine --fp16

# 3. Stage the repository
deploy/instance_segmentation/yolo11/ensemble/setup_model_repository.sh \
  --engine "$PWD/yolo11m-seg.engine"
```

Step 3 builds the CUDA plugin against the DALI libraries from Triton 25.12,
serializes the preprocess, mask-postprocess and polygon-postprocess pipelines,
copies each model's `config.pbtxt`, and stages the engine and plugin. Pass
`--repository <path>` to stage somewhere other than the in-tree
`model_repository/`; the path must stay inside the repository root.

## Start Triton

```bash
docker run --rm --gpus all --shm-size=1g \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v "$PWD/deploy/instance_segmentation/yolo11/ensemble/model_repository:/models:ro" \
  nvcr.io/nvidia/tritonserver:25.12-py3 \
  tritonserver --model-repository=/models --disable-auto-complete-config \
  --backend-config=dali,plugin_libs=/models/libyolo11_seg_dali.so
```

`--backend-config=dali,plugin_libs=...` is required. Without it every DALI model
fails to load with `No schema found for operator "Yolo11SegPostprocess"`, because
the custom operators live in the plugin `.so` rather than in DALI itself.

## Models

| Model | Pipeline |
| --- | --- |
| `yolo11seg_trt` | Raw TensorRT engine |
| `yolo11seg_gpu_pre_cpu_post` | DALI preprocess → TensorRT → raw outputs (client postprocesses) |
| `yolo11seg_gpu_pre_gpu_mask_post` | DALI preprocess → TensorRT → CUDA mask postprocess |
| `yolo11seg_gpu_pre_gpu_post` | DALI preprocess → TensorRT → CUDA polygon postprocess |

Mask and polygon are separate models on purpose: selecting the mask ensemble does
not allocate contour scratch or run the boundary-tracing and convex-hull kernels.

The output tensor contracts are identical to YOLO26-seg and are documented in full
in the [YOLO26-seg ensemble README](../../yolo26/ensemble/README.md); the client
validates them in `include/tritonic/core/gpu_segmentation.hpp`.

## Running

```bash
# Polygon output
./build/tritonic --source=data/images/bus.jpg \
  --model_type=yolo11seg --model=yolo11seg_gpu_pre_gpu_post \
  --task_model=yolo11seg_trt --labelsFile=labels/coco.txt \
  --input_mode=encoded-image --postprocess_mode=gpu \
  --segmentation_output=polygon

# Mask output (default); substitute the mask ensemble
./build/tritonic --source=data/images/bus.jpg \
  --model_type=yolo11seg --model=yolo11seg_gpu_pre_gpu_mask_post \
  --task_model=yolo11seg_trt --labelsFile=labels/coco.txt \
  --input_mode=encoded-image --postprocess_mode=gpu
```

Video sources work with the same ensembles; Tritonic JPEG-encodes each decoded
frame in memory and writes the rendered result to
`<video-directory>/output/processed_<model-name>.avi`.

## Thresholds

Confidence, NMS and mask thresholds are baked into the serialized DALI pipelines
at setup time (defaults 0.5 / 0.4 / 0.5) and are **not** affected by the client's
`--confidence_threshold` or `--nms_threshold` flags, which apply only to CPU
postprocessing. To change them for the GPU path, edit the `generate_*_pipeline.py`
arguments, re-run the setup script, and restart Triton.

## Benchmarks

The four-path timing comparison and the CPU-versus-GPU parity gate are in
[benchmarks/yolo11-seg](../../../../benchmarks/yolo11-seg/README.md). The gate
includes a generated dense fixture, because the stock images yield only 10–50
anchors above the confidence gate — well under the 100-detection cap — and so
cannot distinguish a truncating postprocessor from a correct one.
