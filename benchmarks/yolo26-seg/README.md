# YOLO26m-seg Tritonic benchmark

This benchmark measures the real `./build/tritonic` C++ application with the
official `yolo26m-seg` model exported to an FP16 TensorRT 10.14.1 engine. It
compares:

1. C++ CPU preprocessing, TensorRT inference, C++ CPU postprocessing.
2. DALI GPU preprocessing, TensorRT inference, C++ CPU postprocessing.
3. DALI GPU preprocessing, TensorRT inference, custom DALI CUDA postprocessing.

The GPU-post path uses only the Triton DALI backend. Its custom CUDA operator
creates masks as temporary device scratch, enumerates boundary edges in parallel,
traces ordered exterior and hole rings, and returns compact polygon tensors. Masks
are never exposed by the ensemble, and there is no Triton Python backend in this
path.

## Validated result

Hardware: NVIDIA GeForce RTX 3060 Laptop GPU. Each path used 5 warmups and 30
measured iterations on `bus.jpg`, `horses.jpg`, `person.jpg`, and `mug.jpg`
(120 measured samples per path).

| Path | Pre median | Infer median | Post median | Total median | Total p95 |
|---|---:|---:|---:|---:|---:|
| CPU pre / CPU post | 31.37 ms | 26.89 ms | 40.93 ms | 99.35 ms | 134.16 ms |
| DALI pre / CPU post | 0.03 ms | 29.97 ms | 40.12 ms | 69.29 ms | 90.54 ms |
| DALI pre / DALI post | 0.02 ms | 25.12 ms | 0.06 ms | 25.24 ms | 35.14 ms |

DALI pre/post is 3.94x faster than CPU pre/post and 2.75x faster than DALI
pre/CPU post by aggregate median end-to-end latency.

Timing boundaries differ by placement: for DALI-pre paths, `infer` includes
server-side DALI preprocessing plus TensorRT. For DALI-pre/DALI-post it also
includes server-side DALI postprocessing; its reported `post` is only the C++
result-contract adapter. End-to-end `total` is the directly comparable metric.

The semantic checker passed all four fixtures. CPU versus DALI preprocessing
passed class-aware canonical matching, box IoU >= 0.95, polygon IoU >= 0.90,
ring winding and hole-containment checks, and polygon sanity checks. The observed
minimum CPU-versus-DALI polygon IoU was 0.975. DALI GPU postprocessing was
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
