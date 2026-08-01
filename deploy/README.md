# Triton Deployment Tools

Triton Inference Server model repository setup, ensemble configurations, and
deployment scripts.  For model export instructions, see
[neuriplo-tasks export documentation](https://github.com/olibartfast/neuriplo-tasks/blob/master/export/README.md).

## Directory Structure

```
deploy/
├── classifier/
│   └── vit/                          # [ViT Classifier](classifier/vit/README.md)
│       ├── python_pipeline/           # HuggingFace pipeline backend
│       ├── python_standard/           # VideoMAE-style Python backend
│       └── docker/                    # Environment setup helper
├── instance_segmentation/
│   ├── rf-detr/                       # [RF-DETR seg](instance_segmentation/rf-detr/README.md)
│   │   ├── export_trt.sh             # ONNX → TensorRT via Docker
│   │   └── deploy_triton_model.sh    # Copy engine to Triton repo
│   └── yolo26/                        # [YOLO26 instance seg](instance_segmentation/yolo26/README.md)
│       └── ensemble/                  # [DALI/TensorRT ensemble](instance_segmentation/yolo26/ensemble/README.md)
│           ├── setup_model_repository.sh
│           ├── dali_plugin/           # Custom CUDA postprocess operators
│           └── model_repository/      # Ready-to-use Triton models
└── object_detection/
    ├── yolo/                          # [YOLOv5–v12, NAS](object_detection/yolo/README.md)
    │   └── ensemble/                  # [GPU preprocessing ensemble](object_detection/yolo/ensemble/README.md)
    │       └── dali/
    │           ├── generate_pipeline.py
    │           ├── setup_model_repository.sh
    │           └── model_repository/  # DALI ensemble + TensorRT slot
    └── yolo26/                        # [YOLO26 detection](object_detection/yolo26/README.md)
        └── ensemble/                  # [DALI/TensorRT ensemble](object_detection/yolo26/ensemble/README.md)
            ├── generate_pipeline.py
            ├── generate_postprocess_pipeline.py
            ├── setup_model_repository.sh
            ├── dali_plugin/           # CUDA bbox decode operator
            └── model_repository/      # Ready-to-use Triton models
```

## Triton-Specific Features

1. **Model Repository Setup** — scripts to organize engines in Triton's `1/model.plan` layout with `config.pbtxt`
2. **Ensemble Configurations** — multi-stage pipelines (DALI preprocess → TensorRT → DALI postprocess)
3. **TensorRT Deployment** — Docker-based ONNX→TensorRT conversion and engine staging
4. **CUDA Plugins** — custom DALI operators for GPU-accelerated postprocessing
