# Tritonic Deployment Tools

This directory contains Triton Inference Server-specific deployment utilities.

## ⚠️ Important: Export Tools Moved

**Generic model export scripts have been moved to the vision-core repository.**

For model export (ONNX, TorchScript, TensorRT conversion), please use:
- **Vision-Core Export Tools**: [https://github.com/olibartfast/vision-core](https://github.com/olibartfast/vision-core)

## What's in this directory

This directory now contains only Triton-specific deployment files:

```
deploy/
├── classifier/
│   └── vit/
│       ├── python_pipeline/     # Triton pipeline configs
│       └── python_standard/     # Triton model repository structures
├── instance_segmentation/
│   └── rf-detr/
│       ├── deploy_triton_model.sh  # Deploy TensorRT to Triton
│       └── export_trt.sh           # TensorRT conversion
└── pose_estimation/
    └── vitpose/                 # Triton ensemble configs
```

## Quick Migration Guide

### Before (old structure):
```bash
# Old way
python deploy/object_detection/yolo/export.py --model yolov8n.pt
```

### After (new structure):
```bash
# New way - use vision-core export tools
git clone https://github.com/olibartfast/vision-core.git
cd vision-core/export/detection/yolo
python export.py --model yolov8n.pt

# Then deploy to Triton if needed
cd ../../tritonic/deploy/instance_segmentation/rf-detr
./deploy_triton_model.sh model.engine
```

## Triton-Specific Features

The remaining files in this directory provide:

1. **Triton Model Repository Setup**: Scripts to organize models in Triton's expected directory structure
2. **Triton Configuration**: `config.pbtxt` templates and ensemble configurations  
3. **TensorRT Deployment**: Scripts to deploy TensorRT engines to Triton
4. **Pipeline Configurations**: Multi-stage inference pipelines for Triton

## Usage Workflow

1. **Export Model**: Use vision-core export tools to create ONNX/TensorRT models
2. **Deploy to Triton**: Use scripts in this directory to set up Triton model repository
3. **Configure**: Edit `config.pbtxt` files for your specific model requirements
4. **Run**: Start Triton server with your model repository

For detailed export instructions, see: [vision-core export documentation](https://github.com/olibartfast/vision-core/tree/main/export)