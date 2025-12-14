#!/bin/bash

# YOLO Universal Export Launcher Script
# 
# This script provides a convenient interface for exporting YOLO models
# across all versions (v5, v6, v7, v8, v9, v10, v11, v12, NAS) to ONNX and TensorRT formats.
#
# Usage:
#   bash export.sh --model yolov8n.pt --format onnx
#   bash export.sh --model yolo11s.pt --format onnx --imgsz 640
#   bash export.sh --help

set -e  # Exit on any error

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPORT_SCRIPT="$SCRIPT_DIR/export.py"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Help function
show_help() {
    cat << EOF
YOLO Universal Export Launcher

DESCRIPTION:
    Launch YOLO model export for all versions (v5, v6, v7, v8, v9, v10, v11, v12, NAS)
    with support for ONNX and TensorRT formats.

USAGE:
    $0 [OPTIONS]

REQUIRED OPTIONS:
    -m, --model PATH        Path to model weights or model name

VERSION OPTIONS:
    -v, --version VERSION   Force specific version: auto, v5, v6, v7, v8, v9, v10, v11, v12, nas
                           (default: auto-detect from model name)

EXPORT OPTIONS:
    -f, --format FORMAT     Export format: onnx, tensorrt, both (default: onnx)
    -o, --output-dir PATH   Output directory (default: ./exported_models)

REPOSITORY OPTIONS:
    --repo-dir PATH         Path to cloned YOLO repository (required for v5, v6, v7)

MODEL OPTIONS:
    --imgsz SIZE            Input image size (default: 640)
    -b, --batch-size SIZE   Batch size (default: 1)

ONNX OPTIONS:
    --no-simplify           Skip ONNX model simplification
    --opset VERSION         ONNX opset version (default: 12)
    --dynamic               Enable dynamic batch size

TENSORRT OPTIONS:
    --no-fp16               Disable FP16 precision
    --workspace-size SIZE   TensorRT workspace size (default: 4g)

WEIGHT OPTIONS:
    --download-weights      Download model weights if not present
    --weights-dir PATH      Directory to download weights to (default: ./weights)

UTILITY OPTIONS:
    --model-info            Display model FLOPs and parameters
    --skip-venv-check       Skip virtual environment check

EXAMPLES:
    # Export YOLOv8 to ONNX (uses ultralytics pip)
    $0 --model yolov8n.pt --format onnx

    # Export YOLO11 with custom input size
    $0 --model yolo11s.pt --format onnx --imgsz 640

    # Export YOLOv5 (requires repo clone!)
    $0 --model yolov5s.pt --version v5 --repo-dir ./repositories/yolov5 --format onnx

    # Export YOLOv7 (requires repo clone!)
    $0 --model yolov7.pt --version v7 --repo-dir ./repositories/yolov7 --format onnx

    # Export YOLO-NAS
    $0 --model yolo_nas_s --version nas --format onnx

    # Export to TensorRT with FP16
    $0 --model yolov8n.pt --format tensorrt

    # Export with dynamic batch size
    $0 --model yolov8n.pt --format onnx --dynamic

SUPPORTED MODELS:
    Ultralytics pip (v8, v9, v10, v11, v12) - No repo required:
        yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
        yolov9t, yolov9s, yolov9m, yolov9c, yolov9e
        yolov10n, yolov10s, yolov10m, yolov10l, yolov10x
        yolo11n, yolo11s, yolo11m, yolo11l, yolo11x
        yolov12n, yolov12s, yolov12m, yolov12l, yolov12x

    YOLOv5 (ultralytics/yolov5 repo) - Requires --repo-dir:
        yolov5n, yolov5s, yolov5m, yolov5l, yolov5x

    YOLOv6 (meituan/YOLOv6 repo) - Requires --repo-dir:
        yolov6n, yolov6s, yolov6m, yolov6l

    YOLOv7 (WongKinYiu/yolov7 repo) - Requires --repo-dir:
        yolov7, yolov7-tiny, yolov7x, yolov7-w6, yolov7-e6, yolov7-d6, yolov7-e6e

    YOLO-NAS (super-gradients pip):
        yolo_nas_s, yolo_nas_m, yolo_nas_l

EOF
}

# Initialize variables
MODEL=""
VERSION="auto"
FORMAT="onnx"
OUTPUT_DIR="./exported_models"
REPO_DIR=""
IMGSZ=640
BATCH_SIZE=1
SIMPLIFY=true
OPSET=12
DYNAMIC=false
FP16=true
WORKSPACE_SIZE="4g"
DOWNLOAD_WEIGHTS=false
WEIGHTS_DIR="./weights"
MODEL_INFO=false
SKIP_VENV_CHECK=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -v|--version)
            VERSION="$2"
            shift 2
            ;;
        -f|--format)
            FORMAT="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --repo-dir)
            REPO_DIR="$2"
            shift 2
            ;;
        --imgsz|--img-size)
            IMGSZ="$2"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --no-simplify)
            SIMPLIFY=false
            shift
            ;;
        --opset)
            OPSET="$2"
            shift 2
            ;;
        --dynamic)
            DYNAMIC=true
            shift
            ;;
        --no-fp16)
            FP16=false
            shift
            ;;
        --workspace-size)
            WORKSPACE_SIZE="$2"
            shift 2
            ;;
        --download-weights)
            DOWNLOAD_WEIGHTS=true
            shift
            ;;
        --weights-dir)
            WEIGHTS_DIR="$2"
            shift 2
            ;;
        --model-info)
            MODEL_INFO=true
            shift
            ;;
        --skip-venv-check)
            SKIP_VENV_CHECK=true
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$MODEL" ]]; then
    log_error "Model path is required"
    show_help
    exit 1
fi

# Check if export script exists
if [[ ! -f "$EXPORT_SCRIPT" ]]; then
    log_error "Export script not found: $EXPORT_SCRIPT"
    exit 1
fi

# Build Python command
CMD="python $EXPORT_SCRIPT"
CMD+=" --model $MODEL"
CMD+=" --version $VERSION"
CMD+=" --format $FORMAT"
CMD+=" --output-dir $OUTPUT_DIR"
CMD+=" --imgsz $IMGSZ"
CMD+=" --batch-size $BATCH_SIZE"
CMD+=" --opset $OPSET"
CMD+=" --weights-dir $WEIGHTS_DIR"
CMD+=" --workspace-size $WORKSPACE_SIZE"

if [[ -n "$REPO_DIR" ]]; then
    CMD+=" --repo-dir $REPO_DIR"
fi

if [[ "$SIMPLIFY" == "false" ]]; then
    CMD+=" --no-simplify"
fi

if [[ "$DYNAMIC" == "true" ]]; then
    CMD+=" --dynamic"
fi

if [[ "$FP16" == "false" ]]; then
    CMD+=" --no-fp16"
fi

if [[ "$DOWNLOAD_WEIGHTS" == "true" ]]; then
    CMD+=" --download-weights"
fi

if [[ "$MODEL_INFO" == "true" ]]; then
    CMD+=" --model-info"
fi

if [[ "$SKIP_VENV_CHECK" == "true" ]]; then
    CMD+=" --skip-venv-check"
fi

# Print configuration
log_info "YOLO Export Configuration:"
echo "  Model:         $MODEL"
echo "  Version:       $VERSION"
echo "  Format:        $FORMAT"
echo "  Output Dir:    $OUTPUT_DIR"
if [[ -n "$REPO_DIR" ]]; then
echo "  Repo Dir:      $REPO_DIR"
fi
echo "  Image Size:    $IMGSZ"
echo "  Batch Size:    $BATCH_SIZE"
echo ""

# Run export
log_info "Running export..."
eval $CMD

log_success "Export completed!"
