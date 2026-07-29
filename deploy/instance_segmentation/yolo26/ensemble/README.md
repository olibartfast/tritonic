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
- `yolo26seg_gpu_pre_gpu_post`: DALI preprocessing, TensorRT, and DALI CUDA postprocessing.
