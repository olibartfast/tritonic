**Default: keep DALI (B1) for server-side preprocessing. Offer V2 as the opt-in choice for clients on the Triton host. Keep V12 as a DALI-free alternative and retire V13.**

| Use case | Choose | Evidence (A10G, Triton 26.08, gRPC) |
|---|---|---|
| Remote or CPU-only clients sending JPEG (the common case) | **B1, DALI** | Highest server throughput on bus_1080p: 162 inferences/s at concurrency 8, against 117 for V12 and V13. Best large-image latency: 15.8 ms at 2160p, against 28.2 ms for V12. Smallest code footprint: 55 source lines, against 815 for V12. |
| Client on the same host and GPU as Triton | **V2, client CUDA into CUDA shared memory** | 3.1 ms mean total, against 9.3 ms for B1 (3×) and 27.0 ms for B0. 0 request bytes, against 0.11–1.6 MB of JPEG. Tensor parity within D-5 (p99 0.0157, limit 0.016). Detection parity on all fixtures. Client memcheck clean. |
| Deployments that must not depend on DALI | **V12, custom C++ backend** | Tensors bit-exact with DALI (max 0.0 on bus and mug; 1 of 1.2 M elements differs by 1/255 on horses). About 1 ms faster than B1 up to about 1 MP (bus 8.9 vs 9.9 ms, horses 4.5 vs 5.4 ms). 11,000-request soak with 0 errors. Server memcheck clean. |
| — | not V13 | Same speed as V12 (11.24 vs 11.16 ms mean, 117 inferences/s for both). Its in-graph letterbox adds about 0.1 ms of TensorRT time (1.98 vs 1.87 ms), and it couples one engine to each model and each TensorRT version. It has no advantage over V12. |

**Why V12 is not the default.**
- The backend's nvJPEG decode loses at large sizes. Mean server preprocessing time is 8.0 ms for V12, against 6.5 ms for DALI. At concurrency 4–8, V12 delivers 117 inferences/s, 28% below DALI's 162.
- The most likely cause is the decoder strategy. DALI's mixed decoder moves Huffman decoding to the GPU for large images, while the backend uses nvJPEG's default backend. This has not been verified.
- Follow-up: select the GPU-Huffman (hybrid) backend by image size, and batch decodes. Then measure again. If V12 then matches B1 at 1080p and 2160p, it becomes the DALI-free default.

**V2 constraints.**
- V2 needs a client GPU, CUDA IPC on the same host, and a server started with `--allow-client-shm=true`.
- A CUDA-enabled build still runs on CPU-only clients with the default `--preprocess_device=cpu` (R-17, V-18).
- perf_analyzer cannot drive client-side preprocessing, so V2's server throughput is not measured.

**Caveats.**
- All engines are FP32. TensorRT 11 builds strongly typed networks only, and the FP16 conversion of the end-to-end export fails, so FP16 is a follow-up.
- One GPU model (A10G).
- perf_analyzer exited without a report for B0 at concurrency 8. Levels 1 and 4 are valid.
- Detection confidence tolerance is ±0.04, amended from ±0.03 because the B1 and B0 baselines differ by 0.032 on bus_1080p.
