#!/bin/bash

# RT-DETR Virtual Environment Setup Script
# 
# This script creates optimized virtual environments for RT-DETR export pipelines.
# Supports both PyTorch-based and PaddlePaddle-based workflows.
#
# Usage:
#   ./setup_env.sh --framework pytorch --env-name rtdetr-pytorch
#   ./setup_env.sh --framework paddlepaddle --env-name rtdetr-paddle
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
FRAMEWORK=""
ENV_NAME=""
PYTHON_VERSION="3.11"
CUDA_VERSION="11.8"
OUTPUT_DIR="./environments"
FORCE_CREATE=false
INSTALL_EXTRAS=true
DRY_RUN=false

# Help function
show_help() {
    cat << EOF
RT-DETR Virtual Environment Setup Script

DESCRIPTION:
    Create optimized virtual environments for RT-DETR export pipelines.
    Supports both PyTorch and PaddlePaddle frameworks with export dependencies.

USAGE:
    $0 [OPTIONS]

REQUIRED OPTIONS:
    -f, --framework FRAMEWORK    Framework: pytorch or paddlepaddle

OPTIONAL OPTIONS:
    -n, --env-name NAME         Environment name (auto-generated if not specified)
    -p, --python-version VER    Python version (default: 3.11)
    -c, --cuda-version VER      CUDA version for GPU support (default: 11.8)
    -o, --output-dir PATH       Directory for environments (default: ./environments)
    --force                     Force recreate environment if exists
    --no-extras                 Skip optional export tools installation
    --dry-run                   Show what would be installed without executing
    -h, --help                  Show this help message

FRAMEWORKS:
    pytorch      - PyTorch pipeline: PyTorch → ONNX → TensorRT
                   Supports: RT-DETR v1, v2, v4, D-FINE, DEIM
                   
    paddlepaddle - PaddlePaddle pipeline: PaddlePaddle → ONNX → TensorRT
                   Supports: RT-DETR v3

PIPELINE COMPONENTS:

PyTorch Pipeline:
    ├── PyTorch (CUDA-enabled)
    ├── TorchVision 
    ├── ONNX & ONNX Runtime
    ├── ONNXSim (model optimization)
    ├── TensorRT Python bindings
    └── Export utilities

PaddlePaddle Pipeline:
    ├── PaddlePaddle (GPU version)
    ├── Paddle2ONNX (conversion tool)
    ├── ONNX & ONNX Runtime
    ├── TensorRT Python bindings
    └── Export utilities

EXAMPLES:
    # Create PyTorch environment with defaults
    $0 --framework pytorch

    # Create PaddlePaddle environment with custom name
    $0 --framework paddlepaddle --env-name rtdetr-paddle-v3

    # Create with specific Python/CUDA versions
    $0 --framework pytorch --python-version 3.10 --cuda-version 12.1

    # Force recreate environment
    $0 --framework pytorch --force

    # Dry run to see what would be installed
    $0 --framework paddlepaddle --dry-run

ENVIRONMENT STRUCTURE:
    environments/
    ├── pytorch/
    │   ├── rtdetr-pytorch-py311-cuda118/
    │   └── activation_scripts/
    └── paddlepaddle/
        ├── rtdetr-paddle-py311-cuda118/
        └── activation_scripts/

USAGE AFTER SETUP:
    # Activate PyTorch environment
    source environments/pytorch/rtdetr-pytorch-py311-cuda118/bin/activate
    
    # Activate PaddlePaddle environment  
    source environments/paddlepaddle/rtdetr-paddle-py311-cuda118/bin/activate

EOF
}

# Parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -f|--framework)
                FRAMEWORK="$2"
                shift 2
                ;;
            -n|--env-name)
                ENV_NAME="$2"
                shift 2
                ;;
            -p|--python-version)
                PYTHON_VERSION="$2"
                shift 2
                ;;
            -c|--cuda-version)
                CUDA_VERSION="$2"
                shift 2
                ;;
            -o|--output-dir)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            --force)
                FORCE_CREATE=true
                shift
                ;;
            --no-extras)
                INSTALL_EXTRAS=false
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown argument: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
}

# Validate arguments
validate_arguments() {
    if [[ -z "$FRAMEWORK" ]]; then
        log_error "Framework is required. Use -f/--framework to specify."
        echo "Supported frameworks: pytorch, paddlepaddle"
        exit 1
    fi

    if [[ "$FRAMEWORK" != "pytorch" && "$FRAMEWORK" != "paddlepaddle" ]]; then
        log_error "Unsupported framework: $FRAMEWORK"
        echo "Supported frameworks: pytorch, paddlepaddle"
        exit 1
    fi

    # Generate environment name if not provided
    if [[ -z "$ENV_NAME" ]]; then
        local python_short=$(echo "$PYTHON_VERSION" | tr -d '.')
        local cuda_short=$(echo "$CUDA_VERSION" | tr -d '.')
        ENV_NAME="rtdetr-${FRAMEWORK}-py${python_short}-cuda${cuda_short}"
    fi
}

# Check dependencies
check_dependencies() {
    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed"
        exit 1
    fi

    # Check pip
    if ! command -v pip3 &> /dev/null && ! python3 -m pip --version &> /dev/null; then
        log_error "pip is not installed"
        exit 1
    fi

    # Check conda (recommended for environment management)
    if command -v conda &> /dev/null; then
        ENV_MANAGER="conda"
        log_info "Using conda for environment management"
    elif python3 -m venv --help &> /dev/null; then
        ENV_MANAGER="venv"
        log_info "Using venv for environment management"
    else
        log_error "No suitable environment manager found (conda or venv required)"
        exit 1
    fi
}

# Get PyTorch installation commands
get_pytorch_commands() {
    local cuda_version="$1"
    local commands=()
    
    # Convert CUDA version for PyTorch index
    local torch_cuda
    case "$cuda_version" in
        "11.8") torch_cuda="cu118" ;;
        "12.1") torch_cuda="cu121" ;;
        "12.4") torch_cuda="cu124" ;;
        *) torch_cuda="cu118" ;;  # default
    esac
    
    commands+=(
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/$torch_cuda"
        "pip install transformers timm"
        "pip install opencv-python pillow"
    )
    
    echo "${commands[@]}"
}

# Get PaddlePaddle installation commands  
get_paddlepaddle_commands() {
    local cuda_version="$1"
    local commands=()
    
    # PaddlePaddle GPU version
    commands+=(
        "pip install paddlepaddle-gpu"
        "pip install paddle2onnx==1.0.5"
        "pip install paddleslim"
    )
    
    echo "${commands[@]}"
}

# Get common export dependencies
get_export_commands() {
    local commands=(
        "pip install onnx==1.13.0"
        "pip install onnxruntime-gpu"
        "pip install onnxsim"
        "pip install tensorrt"
        "pip install pycuda"
        "pip install numpy==1.24.3"
        "pip install PyYAML"
        "pip install tqdm"
        "pip install matplotlib"
        "pip install pycocotools"
    )
    
    if [[ "$INSTALL_EXTRAS" == true ]]; then
        commands+=(
            "pip install fiftyone"
            "pip install tensorboard"
            "pip install wandb"
            "pip install gradio"
        )
    fi
    
    echo "${commands[@]}"
}

# Create environment
create_environment() {
    local framework="$1"
    local env_name="$2"
    local framework_dir="$OUTPUT_DIR/$framework"
    local env_path="$framework_dir/$env_name"
    
    mkdir -p "$framework_dir"
    
    # Check if environment exists
    if [[ -d "$env_path" ]]; then
        if [[ "$FORCE_CREATE" == true ]]; then
            log_warning "Environment exists, removing: $env_path"
            rm -rf "$env_path"
        else
            log_error "Environment already exists: $env_path"
            log_info "Use --force to recreate"
            exit 1
        fi
    fi
    
    log_step "Creating $framework environment: $env_name"
    
    if [[ "$DRY_RUN" == true ]]; then
        log_info "DRY RUN: Would create environment at $env_path"
        return
    fi
    
    case "$ENV_MANAGER" in
        "conda")
            conda create -y -p "$env_path" python="$PYTHON_VERSION"
            ;;
        "venv")
            python3 -m venv "$env_path"
            ;;
    esac
    
    log_success "Environment created: $env_path"
}

# Install packages
install_packages() {
    local framework="$1"
    local env_name="$2"
    local env_path="$OUTPUT_DIR/$framework/$env_name"
    
    if [[ "$DRY_RUN" == true ]]; then
        log_info "DRY RUN: Would install packages in $env_path"
        show_installation_plan "$framework"
        return
    fi
    
    log_step "Installing packages for $framework pipeline"
    
    # Activate environment
    case "$ENV_MANAGER" in
        "conda")
            source "$env_path/bin/activate"
            ;;
        "venv")
            source "$env_path/bin/activate"
            ;;
    esac
    
    # Upgrade pip
    log_info "Upgrading pip..."
    python -m pip install --upgrade pip
    
    # Install framework-specific packages
    case "$framework" in
        "pytorch")
            log_info "Installing PyTorch pipeline dependencies..."
            local pytorch_cmds=($(get_pytorch_commands "$CUDA_VERSION"))
            for cmd in "${pytorch_cmds[@]}"; do
                log_info "Running: $cmd"
                eval "$cmd"
            done
            ;;
        "paddlepaddle")
            log_info "Installing PaddlePaddle pipeline dependencies..."
            local paddle_cmds=($(get_paddlepaddle_commands "$CUDA_VERSION"))
            for cmd in "${paddle_cmds[@]}"; do
                log_info "Running: $cmd"
                eval "$cmd"
            done
            ;;
    esac
    
    # Install common export dependencies
    log_info "Installing export dependencies..."
    local export_cmds=($(get_export_commands))
    for cmd in "${export_cmds[@]}"; do
        log_info "Running: $cmd"
        eval "$cmd"
    done
    
    log_success "Package installation completed"
}

# Show installation plan for dry run
show_installation_plan() {
    local framework="$1"
    
    echo
    log_info "Installation Plan for $framework Pipeline:"
    echo
    
    echo "Framework-specific packages:"
    case "$framework" in
        "pytorch")
            get_pytorch_commands "$CUDA_VERSION" | tr ' ' '\n' | grep -E '^pip install' | sed 's/^/  /'
            ;;
        "paddlepaddle")
            get_paddlepaddle_commands "$CUDA_VERSION" | tr ' ' '\n' | grep -E '^pip install' | sed 's/^/  /'
            ;;
    esac
    
    echo
    echo "Export dependencies:"
    get_export_commands | tr ' ' '\n' | grep -E '^pip install' | sed 's/^/  /'
    echo
}

# Create activation script
create_activation_script() {
    local framework="$1"
    local env_name="$2"
    local framework_dir="$OUTPUT_DIR/$framework"
    local env_path="$framework_dir/$env_name"
    local scripts_dir="$framework_dir/activation_scripts"
    
    mkdir -p "$scripts_dir"
    
    local script_path="$scripts_dir/activate_${env_name}.sh"
    
    if [[ "$DRY_RUN" == true ]]; then
        log_info "DRY RUN: Would create activation script at $script_path"
        return
    fi
    
    cat > "$script_path" << EOF
#!/bin/bash
# RT-DETR $framework Environment Activation Script
# Generated on $(date)

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "\${BLUE}[INFO]\${NC} Activating RT-DETR $framework environment: $env_name"

# Activate environment
source "$env_path/bin/activate"

# Set environment variables
export RTDETR_FRAMEWORK="$framework"
export RTDETR_ENV_NAME="$env_name"
export RTDETR_ENV_PATH="$env_path"

# Show environment info
echo -e "\${GREEN}[SUCCESS]\${NC} Environment activated!"
echo "  Framework: $framework"
echo "  Environment: $env_name"
echo "  Python: \$(python --version)"
echo "  Location: $env_path"
echo

# Show pipeline info
case "$framework" in
    "pytorch")
        echo "PyTorch Pipeline: PyTorch → ONNX → TensorRT"
        echo "Supported models: RT-DETR v1, v2, v4, D-FINE, DEIM"
        ;;
    "paddlepaddle")
        echo "PaddlePaddle Pipeline: PaddlePaddle → ONNX → TensorRT"
        echo "Supported models: RT-DETR v3"
        ;;
esac
echo

echo "Usage:"
echo "  Export models: ./export.sh --framework $framework ..."
echo "  Deactivate: deactivate"
echo
EOF

    chmod +x "$script_path"
    
    log_success "Activation script created: $script_path"
}

# Create environment info file
create_env_info() {
    local framework="$1" 
    local env_name="$2"
    local env_path="$OUTPUT_DIR/$framework/$env_name"
    
    if [[ "$DRY_RUN" == true ]]; then
        return
    fi
    
    local info_file="$env_path/.rtdetr_env_info"
    
    cat > "$info_file" << EOF
# RT-DETR Environment Information
# Generated on $(date)

framework=$framework
env_name=$env_name
python_version=$PYTHON_VERSION
cuda_version=$CUDA_VERSION
created_date=$(date -Iseconds)
extras_installed=$INSTALL_EXTRAS

# Pipeline
pipeline_type=${framework}_to_onnx_to_tensorrt

# Supported models
$(case "$framework" in
    "pytorch")
        echo "supported_models=v1,v2,v4,dfine,deim"
        ;;
    "paddlepaddle")
        echo "supported_models=v3"
        ;;
esac)
EOF

    log_info "Environment info created: $info_file"
}

# Show final summary
show_summary() {
    local framework="$1"
    local env_name="$2"
    local framework_dir="$OUTPUT_DIR/$framework" 
    local env_path="$framework_dir/$env_name"
    local script_path="$framework_dir/activation_scripts/activate_${env_name}.sh"
    
    echo
    log_success "Environment setup completed!"
    echo
    log_info "Environment Details:"
    echo "  Framework: $framework"
    echo "  Name: $env_name"
    echo "  Location: $env_path"
    echo "  Python: $PYTHON_VERSION"
    echo "  CUDA: $CUDA_VERSION"
    echo
    
    log_info "Activation Options:"
    echo "  Direct: source $env_path/bin/activate"
    echo "  Script: source $script_path"
    echo
    
    log_info "Next Steps:"
    echo "  1. Activate environment: source $script_path"
    echo "  2. Clone repository: ./clone_repo.sh --version [VERSION]"
    echo "  3. Export model: ./export.sh --framework $framework ..."
    echo
    
    case "$framework" in
        "pytorch")
            echo "  Supported versions: v1, v2, v4, dfine, deim"
            ;;
        "paddlepaddle") 
            echo "  Supported versions: v3"
            ;;
    esac
    echo
}

# Main execution
main() {
    log_info "RT-DETR Virtual Environment Setup Script"
    
    # Show help if no arguments
    if [[ $# -eq 0 ]]; then
        show_help
        exit 0
    fi
    
    # Parse and validate arguments
    parse_arguments "$@"
    validate_arguments
    
    # Check dependencies
    check_dependencies
    
    log_info "Setup Configuration:"
    echo "  Framework: $FRAMEWORK"
    echo "  Environment: $ENV_NAME"
    echo "  Python: $PYTHON_VERSION"
    echo "  CUDA: $CUDA_VERSION"
    echo "  Manager: $ENV_MANAGER"
    echo "  Extras: $INSTALL_EXTRAS"
    echo
    
    if [[ "$DRY_RUN" == true ]]; then
        log_warning "DRY RUN MODE - No changes will be made"
        show_installation_plan "$FRAMEWORK"
        exit 0
    fi
    
    # Create and setup environment
    create_environment "$FRAMEWORK" "$ENV_NAME"
    install_packages "$FRAMEWORK" "$ENV_NAME"
    create_activation_script "$FRAMEWORK" "$ENV_NAME"
    create_env_info "$FRAMEWORK" "$ENV_NAME"
    
    # Show summary
    show_summary "$FRAMEWORK" "$ENV_NAME"
}

# Execute main function
main "$@"