#!/bin/bash -ex

# Usage: 
#   ./raft_exporter.sh [--use-venv] [--cpu-only] [--format=FORMAT]
#   USE_VENV=true CPU_ONLY=true EXPORT_FORMAT=onnx ./raft_exporter.sh
# Default: Use Docker with CUDA, traced format
# With --use-venv: Use virtual environment
# With --cpu-only: Force CPU execution
# With --format: Export format (all, traced, scripted, onnx)

# Initialize variables (can be overridden by environment variables)
USE_VENV=${USE_VENV:-false}
CPU_ONLY=${CPU_ONLY:-false}
EXPORT_FORMAT=${EXPORT_FORMAT:-traced}

# Parse command line arguments (override environment variables)
for arg in "$@"; do
    case $arg in
        --use-venv)
            USE_VENV=true
            ;;
        --cpu-only)
            CPU_ONLY=true
            ;;
        --format=*)
            EXPORT_FORMAT="${arg#*=}"
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: $0 [--use-venv] [--cpu-only] [--format=FORMAT]"
            echo "Or set environment variables: USE_VENV=true CPU_ONLY=true EXPORT_FORMAT=onnx $0"
            echo "Available formats: all, traced, scripted, onnx"
            exit 1
            ;;
    esac
done

mkdir -p exports

if [[ "$USE_VENV" == "true" ]]; then
    echo "Running RAFT export with virtual environment..."
    
    # Check if we're in a virtual environment
    if [[ -z "${VIRTUAL_ENV}" && -z "${CONDA_DEFAULT_ENV}" ]]; then
        echo "Warning: Not in a virtual environment. Consider activating one first."
        echo "Example: conda activate your_env or source venv/bin/activate"
    fi
    
    # Install dependencies if not present
    echo "Installing required dependencies..."
    pip install onnx-graphsurgeon
    
    # Get script directory and build Python command
    SCRIPT_DIR="$(dirname "$0")"
    PYTHON_CMD="python $SCRIPT_DIR/raft_exporter.py --model-type large --output-dir exports --format $EXPORT_FORMAT"
    
    if [[ "$CPU_ONLY" == "true" ]]; then
        PYTHON_CMD="$PYTHON_CMD --cpu-only"
    else
        PYTHON_CMD="$PYTHON_CMD --device cuda"
    fi
    
    # Run the export script directly
    $PYTHON_CMD
    
else
    echo "Running RAFT export with Docker..."
    
    # Build Docker command
    if [[ "$CPU_ONLY" == "true" ]]; then
        DOCKER_CMD="docker run --rm -it"
        DEVICE_ARG="--cpu-only"
    else
        DOCKER_CMD="docker run --rm -it --gpus=all"
        DEVICE_ARG="--device cuda"
    fi
    
    # Run RAFT model export script with Docker
    $DOCKER_CMD \
      -v $(pwd)/exports:/exports \
      -v $(pwd)/raft_exporter.py:/workspace/raft_exporter.py \
      -u $(id -u):$(id -g) \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
      -w /workspace \
      nvcr.io/nvidia/pytorch:24.12-py3 /bin/bash -cx \
      "pip install onnx-graphsurgeon && python raft_exporter.py --model-type large --output-dir /exports $DEVICE_ARG --format $EXPORT_FORMAT"
fi

echo "RAFT model ready."