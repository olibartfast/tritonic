#!/bin/bash

# YOLO Repository Cloning Script
# 
# This script clones the appropriate YOLO repository based on version
# and sets up the directory structure for export pipelines.
#
# Usage:
#   ./clone_repo.sh --version v7 --output-dir ./repositories
#   ./clone_repo.sh --help

set -e  # Exit on any error

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

# Default values
VERSION=""
OUTPUT_DIR="./repositories"
FORCE_CLONE=false
SHALLOW_CLONE=true
BRANCH=""

# Repository configurations
declare -A REPO_URLS=(
    ["v5"]="https://github.com/ultralytics/yolov5.git"
    ["v6"]="https://github.com/meituan/YOLOv6.git"
    ["v7"]="https://github.com/WongKinYiu/yolov7.git"
    ["v12"]="https://github.com/sunsmarterjie/yolov12.git"
)

declare -A REPO_DIRS=(
    ["v5"]="yolov5"
    ["v6"]="YOLOv6"
    ["v7"]="yolov7"
    ["v12"]="yolov12"
)

declare -A REPO_BRANCHES=(
    ["v5"]="master"
    ["v6"]="main"
    ["v7"]="main"
    ["v12"]="main"
)

# Help function
show_help() {
    cat << EOF
YOLO Repository Cloning Script

DESCRIPTION:
    Clone YOLO repositories for model export. Most YOLO versions (v8, v9, v10, v11)
    use the ultralytics pip package and don't require repository cloning.

USAGE:
    $0 [OPTIONS]

REQUIRED OPTIONS:
    -v, --version VERSION   YOLO version to clone: v5, v6, v7, v12
                           Note: v8, v9, v10, v11, nas use pip packages

OPTIONS:
    -o, --output-dir PATH   Output directory (default: ./repositories)
    -f, --force             Force clone even if directory exists
    --no-shallow            Clone full repository history
    -b, --branch BRANCH     Specific branch to clone

EXAMPLES:
    # Clone YOLOv7 repository
    $0 --version v7 --output-dir ./3rdparty/repositories

    # Clone YOLOv6 with force overwrite
    $0 --version v6 --force

    # Clone specific branch
    $0 --version v5 --branch v7.0

NOTES:
    - YOLOv8, v9, v10, v11 use 'pip install ultralytics' (no repo needed)
    - YOLO-NAS uses 'pip install super-gradients' (no repo needed)
    - YOLOv5, v6, v7, v12 may require repository cloning for export

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
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -f|--force)
            FORCE_CLONE=true
            shift
            ;;
        --no-shallow)
            SHALLOW_CLONE=false
            shift
            ;;
        -b|--branch)
            BRANCH="$2"
            shift 2
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate version
if [[ -z "$VERSION" ]]; then
    log_error "Version is required"
    show_help
    exit 1
fi

# Check if version requires cloning
case $VERSION in
    v8|v9|v10|v11)
        log_info "YOLOv8/v9/v10/v11 use the ultralytics pip package"
        log_info "Install with: pip install ultralytics"
        log_info "No repository cloning needed"
        exit 0
        ;;
    nas)
        log_info "YOLO-NAS uses the super-gradients pip package"
        log_info "Install with: pip install super-gradients"
        log_info "No repository cloning needed"
        exit 0
        ;;
    v5|v6|v7|v12)
        # Continue with cloning
        ;;
    *)
        log_error "Unknown version: $VERSION"
        log_info "Supported versions: v5, v6, v7, v8, v9, v10, v11, v12, nas"
        exit 1
        ;;
esac

# Get repository info
REPO_URL="${REPO_URLS[$VERSION]}"
REPO_DIR="${REPO_DIRS[$VERSION]}"
DEFAULT_BRANCH="${REPO_BRANCHES[$VERSION]}"

if [[ -z "$BRANCH" ]]; then
    BRANCH="$DEFAULT_BRANCH"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

TARGET_DIR="$OUTPUT_DIR/$REPO_DIR"

# Check if already exists
if [[ -d "$TARGET_DIR" ]]; then
    if [[ "$FORCE_CLONE" == "true" ]]; then
        log_warning "Removing existing directory: $TARGET_DIR"
        rm -rf "$TARGET_DIR"
    else
        log_info "Repository already exists: $TARGET_DIR"
        log_info "Use --force to overwrite"
        exit 0
    fi
fi

# Clone repository
log_info "Cloning YOLO$VERSION repository..."
log_info "URL: $REPO_URL"
log_info "Branch: $BRANCH"
log_info "Target: $TARGET_DIR"

CLONE_CMD="git clone"
if [[ "$SHALLOW_CLONE" == "true" ]]; then
    CLONE_CMD+=" --depth 1"
fi
CLONE_CMD+=" --branch $BRANCH"
CLONE_CMD+=" $REPO_URL"
CLONE_CMD+=" $TARGET_DIR"

eval $CLONE_CMD

log_success "Repository cloned successfully!"
log_info "Location: $TARGET_DIR"

# Print next steps
echo ""
log_info "Next steps - Option A (use our wrapper script):"
case $VERSION in
    v5)
        echo "  python export.py --model yolov5s.pt --version v5 --repo-dir $TARGET_DIR --format onnx"
        ;;
    v6)
        echo "  python export.py --model yolov6s.pt --version v6 --repo-dir $TARGET_DIR --format onnx"
        ;;
    v7)
        echo "  python export.py --model yolov7.pt --version v7 --repo-dir $TARGET_DIR --format onnx"
        ;;
    v12)
        echo "  python export.py --model yolov12s.pt --format onnx  # Uses ultralytics pip"
        ;;
esac

echo ""
log_info "Next steps - Option B (use repo's export script directly):"
case $VERSION in
    v5)
        echo "  cd $TARGET_DIR"
        echo "  pip install -r requirements.txt"
        echo "  python export.py --weights yolov5s.pt --include onnx"
        ;;
    v6)
        echo "  cd $TARGET_DIR"
        echo "  pip install -r requirements.txt"
        echo "  python deploy/ONNX/export_onnx.py --weights yolov6s.pt"
        ;;
    v7)
        echo "  cd $TARGET_DIR"
        echo "  pip install -r requirements.txt"
        echo "  python export.py --weights yolov7.pt --grid --end2end --simplify"
        ;;
    v12)
        echo "  # YOLOv12 uses ultralytics pip package:"
        echo "  pip install ultralytics"
        echo "  python -c \"from ultralytics import YOLO; YOLO('yolov12s.pt').export(format='onnx')\""
        ;;
esac

