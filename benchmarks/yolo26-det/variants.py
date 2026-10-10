"""Preprocessing variants compared by specs/2026-10-01-gpu-preprocess-alternatives.

One registry shared by the benchmark runner, the perf_analyzer wrapper, the
parity/robustness checks, and the comparison report, so a variant is defined in
exactly one place.

Each variant names the Triton model Tritonic sends requests to, the client flags
that select its transport, the server models whose per-model statistics make up
its server-side cost, and the bytes one request carries.
"""

from __future__ import annotations

from dataclasses import dataclass, field

TASK_MODEL = "yolo26det_trt"
PREPROCESSED_BYTES = 3 * 640 * 640 * 4


@dataclass(frozen=True)
class Variant:
    key: str
    title: str
    model: str
    input_mode: str
    postprocess_mode: str = "cpu"
    task_model: str | None = None
    server_models: tuple[str, ...] = ()
    extra_args: tuple[str, ...] = ()
    # "encoded": request carries the JPEG file; "preprocessed": FP32 tensor;
    # "cuda-shm": only a CUDA IPC handle travels, the tensor stays on the GPU.
    payload: str = "encoded"
    supports_perf_analyzer: bool = True
    tags: tuple[str, ...] = field(default_factory=tuple)

    def client_args(self) -> list[str]:
        args = [
            f"--model={self.model}",
            f"--input_mode={self.input_mode}",
            f"--postprocess_mode={self.postprocess_mode}",
        ]
        if self.task_model:
            args.append(f"--task_model={self.task_model}")
        return args + list(self.extra_args)

    def request_bytes(self, jpeg_bytes: int) -> int:
        if self.payload == "encoded":
            return jpeg_bytes
        if self.payload == "preprocessed":
            return PREPROCESSED_BYTES
        return 0


def _encoded(key, title, model, preprocess_models, postprocess):
    gpu_post = postprocess == "gpu"
    models = (*preprocess_models, *(("yolo26det_dali_postprocess",) if gpu_post else ()))
    return Variant(
        key=key,
        title=title,
        model=model,
        input_mode="encoded-image",
        postprocess_mode=postprocess,
        task_model=TASK_MODEL,
        server_models=models,
        tags=("gpu-post",) if gpu_post else (),
    )


VARIANTS: dict[str, Variant] = {
    variant.key: variant
    for variant in (
        Variant(
            key="B0",
            title="CPU preprocessing (client, neuriplo-tasks)",
            model=TASK_MODEL,
            input_mode="preprocessed",
            server_models=(TASK_MODEL,),
            payload="preprocessed",
        ),
        _encoded("B1", "DALI preprocessing", "yolo26det_gpu_pre_cpu_post",
                 ("yolo26det_dali_preprocess", TASK_MODEL), "cpu"),
        _encoded("B1-gpupost", "DALI preprocessing + GPU postprocess",
                 "yolo26det_gpu_pre_gpu_post", ("yolo26det_dali_preprocess", TASK_MODEL), "gpu"),
        _encoded("V12", "Custom C++ backend (nvJPEG + CUDA kernel)", "yolo26det_cudapre_cpu_post",
                 ("yolo26det_cuda_preprocess", TASK_MODEL), "cpu"),
        _encoded("V12-gpupost", "Custom C++ backend + GPU postprocess",
                 "yolo26det_cudapre_gpu_post", ("yolo26det_cuda_preprocess", TASK_MODEL), "gpu"),
        _encoded("V13", "nvJPEG decode + letterbox in TensorRT graph", "yolo26det_trtpre_cpu_post",
                 ("yolo26det_nvjpeg_decode", "yolo26det_trt_pre"), "cpu"),
        _encoded("V13-gpupost", "nvJPEG decode + TensorRT-graph letterbox + GPU postprocess",
                 "yolo26det_trtpre_gpu_post", ("yolo26det_nvjpeg_decode", "yolo26det_trt_pre"),
                 "gpu"),
        Variant(
            key="V2",
            title="Client-side CUDA kernel into CUDA shared memory",
            model=TASK_MODEL,
            input_mode="preprocessed",
            server_models=(TASK_MODEL,),
            extra_args=("--preprocess_device=cuda", "--shared_memory_type=cuda"),
            payload="cuda-shm",
            # The client-side kernel is the variant; perf_analyzer would bypass it.
            supports_perf_analyzer=False,
        ),
    )
}

DEFAULT_VARIANTS = ("B0", "B1", "V12", "V13", "V2")


def select(keys: str | None) -> list[Variant]:
    names = [key.strip() for key in keys.split(",")] if keys else list(DEFAULT_VARIANTS)
    unknown = [name for name in names if name not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variants {unknown}; known: {sorted(VARIANTS)}")
    return [VARIANTS[name] for name in names]
