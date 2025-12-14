# YOLO Model Export

This directory contains scripts for exporting YOLO models (v5, v6, v7, v8, v9, v10, v11, v12, NAS) to ONNX and TensorRT formats for deployment with Triton Inference Server.

## Quick Start

### Ultralytics Models (v8, v9, v10, v11, v12)

These models can be exported directly using the `ultralytics` pip package:

```bash
# Install dependencies
pip install ultralytics onnx onnxsim

# Export YOLOv8
python export.py --model yolov8n.pt --format onnx

# Export YOLO11
python export.py --model yolo11s.pt --format onnx --imgsz 640

# Export with auto-download
python export.py --model yolov8n --download-weights --format onnx
```

### YOLOv5 (Original Repository)

**Note:** The `ultralytics` pip package is NOT compatible with original YOLOv5 models. You must clone the original repository:

```bash
# Clone the original YOLOv5 repository
./clone_repo.sh --version v5 --output-dir ./repositories

# Install requirements
cd repositories/yolov5 && pip install -r requirements.txt && cd ../..

# Export with repo path
python export.py --model yolov5s.pt --version v5 --repo-dir ./repositories/yolov5 --format onnx
```

### YOLOv6/v7 (Repository-based)

These versions require cloning their respective repositories:

```bash
# Clone repository
./clone_repo.sh --version v7 --output-dir ./repositories

# Export with repo path
python export.py --model yolov7.pt --version v7 --repo-dir ./repositories/yolov7 --format onnx
```

### YOLO-NAS

```bash
# Install dependencies
pip install super-gradients onnx onnxsim

# Export YOLO-NAS
python export.py --model yolo_nas_s --version nas --format onnx
```

## Scripts

| Script | Description |
|--------|-------------|
| `export.py` | Universal YOLO export script |
| `export.sh` | Shell wrapper for export.py |
| `clone_repo.sh` | Clone YOLO repositories (v5, v6, v7, v12) |
| `setup_env.sh` | Setup Python environment for export |

## Supported Models

### Ultralytics pip package (`pip install ultralytics`)

| Version | Models | Repo Required |
|---------|--------|---------------|
| YOLOv8 | n, s, m, l, x | No |
| YOLOv9 | t, s, m, c, e | No |
| YOLOv10 | n, s, m, l, x | No (end-to-end, no NMS) |
| YOLO11 | n, s, m, l, x | No |
| YOLOv12 | n, s, m, l, x | No |

### Repository-based (requires `--repo-dir`)

| Version | Repository | Notes |
|---------|------------|-------|
| YOLOv5 | ultralytics/yolov5 | **Different from ultralytics pip!** |
| YOLOv6 | meituan/YOLOv6 | Requires repo clone |
| YOLOv7 | WongKinYiu/yolov7 | Requires repo clone |

### Other

| Version | Package | Notes |
|---------|---------|-------|
| YOLO-NAS | super-gradients | `pip install super-gradients` |

## Export Options

```bash
python export.py --help

# Common options:
--model PATH          Model weights or name
--version VERSION     Force version (auto, v5-v12, nas)
--format FORMAT       onnx, tensorrt, both
--repo-dir PATH       Path to cloned repo (required for v5, v6, v7)
--imgsz SIZE          Input size (default: 640)
--batch-size N        Batch size (default: 1)
--dynamic             Enable dynamic batch size
--download-weights    Auto-download weights
```

## Examples

### Ultralytics Models (v8+)

```bash
# YOLOv8
python export.py --model yolov8n.pt --format onnx

# YOLO11 with custom size
python export.py --model yolo11s.pt --format onnx --imgsz 640

# With dynamic batch
python export.py --model yolov8n.pt --format onnx --dynamic
```

### Repository-based Models (v5, v6, v7)

```bash
# Step 1: Clone repository
./clone_repo.sh --version v5 --output-dir ./repositories

# Step 2: Export with repo path
python export.py --model yolov5s.pt --version v5 --repo-dir ./repositories/yolov5 --format onnx

# YOLOv7
./clone_repo.sh --version v7 --output-dir ./repositories
python export.py --model yolov7.pt --version v7 --repo-dir ./repositories/yolov7 --format onnx
```

**Note on YOLOv7 Export:**
- The default export creates standard YOLO format `[1, 25200, 85]` compatible with **ONNX Runtime**
- Model type to use: `yolo` (same as v5/v6/v8)
- For end-to-end export with TensorRT NMS plugin (requires **TensorRT backend**), manually add `--grid --end2end` flags to the YOLOv7 export.py command and use model type `yolov7e2e`

### TensorRT Export

```bash
# Export to TensorRT (FP16)
python export.py --model yolov8n.pt --format tensorrt

# TensorRT without FP16
python export.py --model yolov8n.pt --format tensorrt --no-fp16
```

## Environment Setup

```bash
# Create environment for Ultralytics models
./setup_env.sh --version v8 --env-name yolo-ultralytics

# Create environment for YOLOv5 (original)
./setup_env.sh --version v5 --env-name yolo-v5

# With TensorRT support
./setup_env.sh --version v8 --tensorrt
```

## Important Notes

### YOLOv5 Compatibility

⚠️ **The `ultralytics` pip package (designed for v8+) is NOT compatible with original YOLOv5 models.**

- Original YOLOv5 uses: `ultralytics/yolov5` repository
- Ultralytics v8+ uses: `pip install ultralytics`

They have different:
- Model architectures and weights format
- Export APIs
- Configuration systems

Always use the original repository for YOLOv5 export.

### Version Detection

The script auto-detects versions from model names:
- `yolov5*.pt` → v5 (requires repo)
- `yolov8*.pt` → v8 (ultralytics pip)
- `yolo11*.pt` → v11 (ultralytics pip)

Force a specific version with `--version v5`.

## Deployment to Triton

After export, copy the ONNX model to your Triton model repository:

```
model_repository/
└── yolov8n/
    ├── config.pbtxt  # Optional
    └── 1/
        └── model.onnx
```

Run inference with tritonic:

```bash
./tritonic \
    --source=/path/to/image.jpg \
    --model_type=yolo \
    --model=yolov8n \
    --labelsFile=/path/to/coco.txt \
    --protocol=grpc \
    --serverAddress=localhost \
    --port=8001
```

## Troubleshooting

### "ultralytics not compatible with YOLOv5"

Use the original repository:
```bash
./clone_repo.sh --version v5 --output-dir ./repositories
python export.py --model yolov5s.pt --version v5 --repo-dir ./repositories/yolov5
```

### "Repository required for export"

Clone the required repository first:
```bash
./clone_repo.sh --version v7 --output-dir ./repositories
```

### ONNX Export Fails

```bash
# Try without simplification
python export.py --model yolov8n.pt --format onnx --no-simplify
```
