#!/bin/bash

# YOLO Virtual Environment Setup Script
# 
# This script creates optimized virtual environments for YOLO export pipelines.
# Different YOLO versions may have different dependency requirements.
#
# Usage:
#   ./setup_env.sh --version v8 --env-name yolo-export
#   ./setup_env.sh --help

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
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

log_step() {
    echo -e "${PURPLE}[STEP]${NC} $1"
}

# Default values
VERSION="v8"
ENV_NAME=""
PYTHON_VERSION="3.11"
CUDA_VERSION="11.8"
USE_CONDA=true
FORCE_CREATE=false
INSTALL_TENSORRT=false

# Help function
show_help() {
    cat << EOF
YOLO Virtual Environment Setup Script

DESCRIPTION:
    Create optimized Python environments for YOLO model export.
    Supports conda and venv environments.

USAGE:
    $0 [OPTIONS]

OPTIONS:
    -v, --version VERSION   YOLO version: v5, v6, v7, v8, v9, v10, v11, v12, nas, all
                           (default: v8)
    -n, --env-name NAME     Environment name (default: yolo-<version>)
    --python VERSION        Python version (default: 3.11)
    --cuda VERSION          CUDA version for PyTorch (default: 11.8)
    --use-venv              Use Python venv instead of conda
    -f, --force             Force create even if environment exists
    --tensorrt              Install TensorRT dependencies

EXAMPLES:
    # Create environment for Ultralytics (v8-v11)
    $0 --version v8 --env-name yolo-ultralytics

    # Create environment for YOLOv7
    $0 --version v7 --env-name yolo-v7

    # Create environment for YOLO-NAS
    $0 --version nas --env-name yolo-nas

    # Create environment with TensorRT support
    $0 --version v8 --tensorrt

DEPENDENCY GROUPS:
    v5, v6, v7:     PyTorch, OpenCV, specific repo requirements
    v8, v9, v10, v11, v12: ultralytics package (includes PyTorch)
    nas:            super-gradients package

EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -v|--version)
            VERSION="$2"
            shift 2
            ;;
        -n|--env-name)
            ENV_NAME="$2"
            shift 2
            ;;
        --python)
            PYTHON_VERSION="$2"
            shift 2
            ;;
        --cuda)
            CUDA_VERSION="$2"
            shift 2
            ;;
        --use-venv)
            USE_CONDA=false
            shift
            ;;
        -f|--force)
            FORCE_CREATE=true
            shift
            ;;
        --tensorrt)
            INSTALL_TENSORRT=true
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Set default environment name
if [[ -z "$ENV_NAME" ]]; then
    ENV_NAME="yolo-$VERSION"
fi

log_info "Setting up YOLO environment"
log_info "Version: $VERSION"
log_info "Environment: $ENV_NAME"
log_info "Python: $PYTHON_VERSION"

# Check for conda
if [[ "$USE_CONDA" == "true" ]]; then
    if ! command -v conda &> /dev/null; then
        log_warning "Conda not found, falling back to venv"
        USE_CONDA=false
    fi
fi

# Create environment
if [[ "$USE_CONDA" == "true" ]]; then
    log_step "Creating conda environment..."
    
    # Check if environment exists
    if conda env list | grep -q "^$ENV_NAME "; then
        if [[ "$FORCE_CREATE" == "true" ]]; then
            log_warning "Removing existing environment: $ENV_NAME"
            conda env remove -n "$ENV_NAME" -y
        else
            log_info "Environment already exists: $ENV_NAME"
            log_info "Use --force to recreate"
            log_info "Activate with: conda activate $ENV_NAME"
            exit 0
        fi
    fi
    
    # Create environment
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
    
    # Activate environment
    eval "$(conda shell.bash hook)"
    conda activate "$ENV_NAME"
    
else
    log_step "Creating venv environment..."
    
    VENV_DIR="./envs/$ENV_NAME"
    
    if [[ -d "$VENV_DIR" ]]; then
        if [[ "$FORCE_CREATE" == "true" ]]; then
            log_warning "Removing existing environment: $VENV_DIR"
            rm -rf "$VENV_DIR"
        else
            log_info "Environment already exists: $VENV_DIR"
            log_info "Use --force to recreate"
            log_info "Activate with: source $VENV_DIR/bin/activate"
            exit 0
        fi
    fi
    
    python${PYTHON_VERSION} -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
fi

# Upgrade pip
log_step "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install PyTorch
log_step "Installing PyTorch..."
case $CUDA_VERSION in
    11.8)
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
        ;;
    12.1)
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
        ;;
    cpu)
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
        ;;
    *)
        pip install torch torchvision
        ;;
esac

# Install version-specific dependencies
log_step "Installing YOLO dependencies for version $VERSION..."

case $VERSION in
    v5)
        log_info "Installing YOLOv5 dependencies..."
        pip install ultralytics  # Ultralytics supports v5 export
        pip install onnx onnxsim onnxruntime
        ;;
    v6)
        log_info "Installing YOLOv6 dependencies..."
        pip install onnx onnxsim onnxruntime
        pip install opencv-python scipy
        log_warning "For full YOLOv6 support, clone repo and install requirements.txt"
        ;;
    v7)
        log_info "Installing YOLOv7 dependencies..."
        pip install onnx onnxsim onnxruntime
        pip install opencv-python scipy matplotlib
        log_warning "For full YOLOv7 support, clone repo and install requirements.txt"
        ;;
    v8|v9|v10|v11|v12)
        log_info "Installing Ultralytics (supports v8/v9/v10/v11/v12)..."
        pip install ultralytics
        pip install onnx onnxsim onnxruntime
        ;;
    nas)
        log_info "Installing YOLO-NAS (super-gradients)..."
        pip install super-gradients
        pip install onnx onnxsim onnxruntime
        ;;
    all)
        log_info "Installing all YOLO dependencies..."
        pip install ultralytics
        pip install super-gradients
        pip install onnx onnxsim onnxruntime
        ;;
    *)
        log_error "Unknown version: $VERSION"
        exit 1
        ;;
esac

# Install TensorRT if requested
if [[ "$INSTALL_TENSORRT" == "true" ]]; then
    log_step "Installing TensorRT dependencies..."
    pip install tensorrt
    log_warning "Make sure TensorRT is properly installed on your system"
    log_info "See: https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html"
fi

# Install common utilities
log_step "Installing common utilities..."
pip install opencv-python-headless pillow

# Print activation instructions
echo ""
log_success "Environment setup complete!"
echo ""

if [[ "$USE_CONDA" == "true" ]]; then
    log_info "Activate environment with:"
    echo "    conda activate $ENV_NAME"
else
    log_info "Activate environment with:"
    echo "    source $VENV_DIR/bin/activate"
fi

echo ""
log_info "Test installation with:"
case $VERSION in
    v8|v9|v10|v11|v12|v5)
        echo "    python -c 'from ultralytics import YOLO; print(\"Ultralytics OK\")'"
        ;;
    nas)
        echo "    python -c 'from super_gradients.training import models; print(\"SuperGradients OK\")'"
        ;;
esac

echo ""
log_info "Export example:"
case $VERSION in
    v8|v9|v10|v11|v12|v5)
        echo "    python export.py --model yolov8n.pt --format onnx"
        ;;
    nas)
        echo "    python export.py --model yolo_nas_s --version nas --format onnx"
        ;;
esac
