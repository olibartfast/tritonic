# YOLO26-seg Tritonic benchmark

The committed results were produced with `yolo26m-seg`; the ensemble itself is
scale-independent.

This benchmark measures the real `./build/tritonic` C++ application with the
official `yolo26m-seg` model exported to an FP16 TensorRT 10.14.1 engine. It
compares:

1. C++ CPU preprocessing, TensorRT inference, C++ CPU postprocessing.
2. DALI GPU preprocessing, TensorRT inference, C++ CPU postprocessing.
3. DALI GPU preprocessing, TensorRT inference, custom DALI CUDA postprocessing.

The GPU-post path uses only the Triton DALI backend. Its custom CUDA operator
creates masks as temporary device scratch, enumerates boundary edges in parallel,
traces ordered exterior and hole rings, computes their convex hulls on CUDA, and returns compact polygon tensors. Masks
are never exposed by the ensemble, and there is no Triton Python backend in this
path.

A separate `yolo26seg_gpu_pre_gpu_mask_post` ensemble returns packed bbox-local
`UINT8` masks. It writes the CUDA mask-synthesis kernel directly into the public
output tensor and does not run or allocate any contour, ring, or hull stage. Use
`--segmentation_output=mask` (the default) with that model.

## Validated result

Hardware: NVIDIA GeForce RTX 3060 Laptop GPU. Each path used 5 warmups and 30
measured iterations on `bus.jpg`, `horses.jpg`, `person.jpg`, and `mug.jpg`
(120 measured samples per path).

| Path | Pre median | Infer median | Post median | Total median | Total p95 |
|---|---:|---:|---:|---:|---:|
| CPU pre / CPU post | 30.08 ms | 27.17 ms | 44.08 ms | 100.43 ms | 134.79 ms |
| DALI pre / CPU post | 0.03 ms | 29.86 ms | 43.79 ms | 73.62 ms | 92.29 ms |
| DALI pre / DALI post | 0.02 ms | 28.68 ms | 0.02 ms | 28.76 ms | 51.08 ms |

DALI pre/post is 3.49x faster than CPU pre/post and 2.56x faster than DALI
pre/CPU post by aggregate median end-to-end latency.

Timing boundaries differ by placement: for DALI-pre paths, `infer` includes
server-side DALI preprocessing plus TensorRT. For DALI-pre/DALI-post it also
includes server-side DALI postprocessing; its reported `post` is only the C++
result-contract adapter. End-to-end `total` is the directly comparable metric.

The semantic checker passed all four fixtures. CPU versus DALI preprocessing
passed class-aware canonical matching, box IoU >= 0.95, polygon IoU >= 0.90,
ring winding, convexity, hole-containment, and polygon sanity checks. The observed
minimum CPU-versus-DALI polygon IoU was 0.983. DALI GPU postprocessing was
polygon-identical to CPU postprocessing for the same DALI-preprocessed tensors
(polygon IoU 1.0, box IoU 1.0, score delta <= 1e-6).

Raw timing JSON, final detections with polygon rings, and the generated summary are
saved at:

`benchmarks/yolo26-seg/results/2026-07-29_yolo26m-seg_polygon_rtx3060/`

Revalidate them with:

```bash
python3 benchmarks/yolo26-seg/check_results.py \
  benchmarks/yolo26-seg/results/2026-07-29_yolo26m-seg_polygon_rtx3060
```

## Validated mask-only result

The separate mask ensemble was measured on the same RTX 3060 Laptop GPU with
the same four fixtures, 5 warmups, and 30 measured iterations per fixture.

| Path | Pre median | Infer median | Post median | Total median | Total p95 |
|---|---:|---:|---:|---:|---:|
| CPU pre / CPU post | 30.58 ms | 27.38 ms | 35.96 ms | 106.79 ms | 116.86 ms |
| DALI pre / CPU post | 0.03 ms | 30.02 ms | 34.92 ms | 65.46 ms | 84.96 ms |
| DALI pre / DALI mask post | 0.03 ms | 24.95 ms | 2.34 ms | 27.51 ms | 69.65 ms |

The mask-only GPU path is 3.88x faster than CPU pre/post and 2.38x faster than
DALI pre/CPU post by aggregate median end-to-end latency. Its compact mask
result adapter necessarily copies the returned mask bytes into the C++ result;
the contour and hull stages are absent.

The mask checker passed class-aware canonical matching, box IoU >= 0.95, mask
IoU >= 0.90, non-empty mask validation, and artifact loading. GPU postprocessing
was pixel-identical to CPU postprocessing for the same DALI tensors (minimum mask
IoU 1.0); the minimum CPU-pre versus DALI-pre mask IoU was 0.974.

Results are saved under:

`benchmarks/yolo26-seg/results/2026-07-30_yolo26m-seg_mask_rtx3060/`

Revalidate them with:

```bash
python3 benchmarks/yolo26-seg/check_mask_results.py \
  benchmarks/yolo26-seg/results/2026-07-30_yolo26m-seg_mask_rtx3060
```
