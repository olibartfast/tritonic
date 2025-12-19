#!/bin/bash

# RT-DETR Universal Export Launcher Script
# 
# This script provides a convenient interface for exporting RT-DETR models
# across all versions (v1, v2, v3, v4) to ONNX and TensorRT formats.
#
# Usage:
# bash deploy/object_detection/rt-detr/export.sh --config \
#    3rdparty/repositories/pytorch/RT-DETRv4/configs/rtv4/rtv4_hgnetv2_s_coco.yml -r weights/rtv4_hgnetv2_s_model.pth \
#    --repo-dir \
#    3rdparty/repositories/pytorch/RT-DETRv4 --download-weights --weights-dir ./weights --format onnx --skip-venv-check
#
# bash deploy/object_detection/rt-detr/export.sh  --help
#

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
RT-DETR Universal Export Launcher

DESCRIPTION:
    Launch RT-DETR model export for all versions (v1, v2, v3, v4) with support for
    ONNX and TensorRT formats. Handles automatic repository cloning and dependency
    installation.

USAGE:
    $0 [OPTIONS]

REQUIRED OPTIONS:
    -c, --config PATH       Path to model configuration file
    -r, --checkpoint PATH   Path to model checkpoint file

EXPORT OPTIONS:
    --format FORMAT         Export format: onnx, tensorrt, both (default: onnx)
    --output-dir PATH       Output directory (default: ./exported_models)

REPOSITORY OPTIONS:
    --clone-repo           Clone appropriate RT-DETR repository if needed
    --install-deps         Install dependencies from requirements.txt
    --repo-dir PATH        Custom repository directory path
    --version VERSION      Force specific version: v1, v2, v3, v4, dfine, deim

ONNX OPTIONS:
    --no-check             Skip ONNX model validation
    --no-simplify          Skip ONNX model simplification

TENSORRT OPTIONS:
    --no-fp16              Disable FP16 precision
    --workspace-size SIZE  TensorRT workspace size (default: 4g)

ANALYSIS OPTIONS:
    --model-info           Display model FLOPs, MACs, and parameters
    --benchmark            Run performance benchmarks
    --coco-dir PATH        COCO dataset path for benchmarking

ENVIRONMENT OPTIONS:
    --skip-venv-check      Skip virtual environment check

MODEL WEIGHTS OPTIONS:
    --download-weights     Automatically download model weights based on config
    --weights-dir PATH     Directory to download weights to (default: ./weights)

EXAMPLES:
    # Basic ONNX export with auto-download of weights
    $0 -c configs/rtv4/rtv4_hgnetv2_s_coco.yml -r model.pth --download-weights --format onnx

    # Complete setup with repo cloning, dependencies, and weight download
    $0 -c configs/rtv4/rtv4_hgnetv2_s_coco.yml -r model.pth --clone-repo --install-deps --download-weights --format onnx

    # RT-DETRv3 (PaddlePaddle) export
    $0 -c configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml -r model.pdparams --format onnx --clone-repo

    # Both ONNX and TensorRT with benchmarking
    $0 -c configs/rtv2/rtv2_r50_coco.yml -r model.pth --format both --benchmark --coco-dir /data/COCO2017

    # D-FINE model export
    $0 -c configs/dfine/dfine_hgnetv2_l_coco.yml -r model.pth --format tensorrt --version dfine

    # Custom repository location
    $0 -c my_config.yml -r my_model.pth --repo-dir /path/to/RT-DETR --format onnx

SUPPORTED MODELS:
    - RT-DETR v1 (PyTorch)    - github.com/lyuwenyu/RT-DETR
    - RT-DETR v2 (PyTorch)    - github.com/lyuwenyu/RT-DETR  
    - RT-DETR v3 (PaddlePaddle) - github.com/clxia12/RT-DETRv3
    - RT-DETR v4 (PyTorch)    - github.com/RT-DETRs/RT-DETRv4
    - D-FINE (PyTorch)        - github.com/Peterande/D-FINE
    - DEIM (PyTorch)          - github.com/Intellindust-AI-Lab/DEIM

NOTE:
    RT-DETRv3 requires PaddlePaddle and paddle2onnx:
        pip install paddlepaddle-gpu paddle2onnx==1.0.5

EOF
}

# Check if Python script exists
check_dependencies() {
    if [[ ! -f "$EXPORT_SCRIPT" ]]; then
        log_error "Export script not found: $EXPORT_SCRIPT"
        exit 1
    fi

    # Check if python is available
    if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
        log_error "Python not found. Please install Python 3.7+"
        exit 1
    fi

    # Prefer python3 if available
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    else
        PYTHON_CMD="python"
    fi
}

# Parse command line arguments
parse_arguments() {
    # If no arguments provided, show help
    if [[ $# -eq 0 ]]; then
        show_help
        exit 0
    fi

    # Convert arguments for Python script
    PYTHON_ARGS=()
    DOWNLOAD_WEIGHTS=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -c|--config)
                PYTHON_ARGS+=("-c" "$2")
                CONFIG_FILE="$2"
                shift 2
                ;;
            -r|--checkpoint)
                PYTHON_ARGS+=("-r" "$2")
                CHECKPOINT_FILE="$2"
                shift 2
                ;;
            --format)
                PYTHON_ARGS+=("--format" "$2")
                shift 2
                ;;
            --output-dir)
                PYTHON_ARGS+=("--output-dir" "$2")
                shift 2
                ;;
            --clone-repo)
                PYTHON_ARGS+=("--clone-repo")
                shift
                ;;
            --install-deps)
                PYTHON_ARGS+=("--install-deps")
                shift
                ;;
            --repo-dir)
                PYTHON_ARGS+=("--repo-dir" "$2")
                shift 2
                ;;
            --version)
                PYTHON_ARGS+=("--version" "$2")
                shift 2
                ;;
            --no-check)
                PYTHON_ARGS+=("--no-check")
                shift
                ;;
            --no-simplify)
                PYTHON_ARGS+=("--no-simplify")
                shift
                ;;
            --no-fp16)
                PYTHON_ARGS+=("--no-fp16")
                shift
                ;;
            --workspace-size)
                PYTHON_ARGS+=("--workspace-size" "$2")
                shift 2
                ;;
            --model-info)
                PYTHON_ARGS+=("--model-info")
                shift
                ;;
            --benchmark)
                PYTHON_ARGS+=("--benchmark")
                shift
                ;;
            --coco-dir)
                PYTHON_ARGS+=("--coco-dir" "$2")
                shift 2
                ;;
            --skip-venv-check)
                PYTHON_ARGS+=("--skip-venv-check")
                shift
                ;;
            --download-weights)
                PYTHON_ARGS+=("--download-weights")
                DOWNLOAD_WEIGHTS=true
                shift
                ;;
            --weights-dir)
                PYTHON_ARGS+=("--weights-dir" "$2")
                shift 2
                ;;
            *)
                log_error "Unknown argument: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
}

# Validate required arguments
validate_arguments() {
    if [[ -z "${CONFIG_FILE:-}" ]]; then
        log_error "Config file is required. Use -c/--config to specify."
        exit 1
    fi

    if [[ -z "${CHECKPOINT_FILE:-}" ]]; then
        log_error "Checkpoint file is required. Use -r/--checkpoint to specify."
        exit 1
    fi

    # Check if files exist
    if [[ ! -f "$CONFIG_FILE" ]]; then
        log_error "Config file not found: $CONFIG_FILE"
        exit 1
    fi

    # Only check if checkpoint exists if not downloading weights
    if [[ "$DOWNLOAD_WEIGHTS" != "true" && ! -f "$CHECKPOINT_FILE" ]]; then
        log_error "Checkpoint file not found: $CHECKPOINT_FILE"
        log_info "Hint: Use --download-weights to automatically download model weights"
        exit 1
    fi
}

# Main execution
main() {
    log_info "Starting RT-DETR Universal Export Launcher"
    
    # Parse command line arguments
    parse_arguments "$@"
    
    # Validate dependencies and arguments
    check_dependencies
    validate_arguments
    
    log_info "Configuration: $CONFIG_FILE"
    log_info "Checkpoint: $CHECKPOINT_FILE"
    log_info "Python command: $PYTHON_CMD"
    
    # Execute Python export script
    log_info "Launching export process..."
    
    if $PYTHON_CMD "$EXPORT_SCRIPT" "${PYTHON_ARGS[@]}"; then
        log_success "Export completed successfully!"
    else
        exit_code=$?
        log_error "Export failed with exit code: $exit_code"
        exit $exit_code
    fi
}

# Execute main function with all arguments
main "$@"