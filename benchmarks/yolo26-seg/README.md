# YOLO26m-seg Tritonic benchmark

This benchmark measures the real `./build/tritonic` C++ application with the
official `yolo26m-seg` model exported to an FP16 TensorRT 10.14.1 engine. It
compares:

1. C++ CPU preprocessing, TensorRT inference, C++ CPU postprocessing.
2. DALI GPU preprocessing, TensorRT inference, C++ CPU postprocessing.
3. DALI GPU preprocessing, TensorRT inference, custom DALI CUDA postprocessing.

The GPU-post MVP uses only the Triton DALI backend. Its custom DALI operator
performs confidence filtering, mask coefficient/prototype evaluation, crop,
bilinear resize, and thresholding. There is no Triton Python backend in this
path.

## Validated result

Hardware: NVIDIA GeForce RTX 3060 Laptop GPU. Each path used 5 warmups and 30
measured iterations on `bus.jpg`, `horses.jpg`, `person.jpg`, and `mug.jpg`
(120 measured samples per path).

| Path | Pre median | Infer median | Post median | Total median | Total p95 |
|---|---:|---:|---:|---:|---:|
| CPU pre / CPU post | 33.85 ms | 29.32 ms | 38.45 ms | 100.90 ms | 146.77 ms |
| DALI pre / CPU post | 0.03 ms | 30.58 ms | 36.84 ms | 67.87 ms | 93.16 ms |
| DALI pre / DALI post | 0.03 ms | 24.48 ms | 2.59 ms | 27.04 ms | 70.91 ms |

DALI pre/post is 3.73x faster than CPU pre/post and 2.51x faster than DALI
pre/CPU post by aggregate median end-to-end latency.

Timing boundaries differ by placement: for DALI-pre paths, `infer` includes
server-side DALI preprocessing plus TensorRT. For DALI-pre/DALI-post it also
includes server-side DALI postprocessing; its reported `post` is only the C++
result-contract adapter. End-to-end `total` is the directly comparable metric.

The semantic checker passed all four fixtures. CPU versus DALI preprocessing
passed class-aware canonical matching, box IoU >= 0.95, mask IoU >= 0.90, and
mask sanity checks. DALI GPU postprocessing was byte-for-byte mask-identical to
CPU postprocessing for the same DALI-preprocessed tensors (mask IoU 1.0, box IoU
1.0, score delta <= 1e-6).

Raw timing JSON, final detections, binary masks, and the generated summary are
saved at:

`benchmarks/yolo26-seg/results/2026-07-29_yolo26m-seg_rtx3060/`

Revalidate them with:

```bash
python3 benchmarks/yolo26-seg/check_results.py \
  benchmarks/yolo26-seg/results/2026-07-29_yolo26m-seg_rtx3060
```
