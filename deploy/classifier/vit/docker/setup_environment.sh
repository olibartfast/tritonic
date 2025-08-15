#!/bin/bash

# Setup environment for ViT model deployment

set -e

echo "🚀 Setting up ViT deployment environment..."

# Function to print colored output
print_status() {
    echo -e "\033[1;32m$1\033[0m"
}

print_error() {
    echo -e "\033[1;31m$1\033[0m"
}

print_warning() {
    echo -e "\033[1;33m$1\033[0m"
}

# Check if CUDA is available
check_cuda() {
    if command -v nvidia-smi &> /dev/null; then
        print_status "✅ NVIDIA GPU detected:"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    else
        print_warning "⚠️  No NVIDIA GPU detected. Will use CPU-only deployment."
    fi
}

# Install Python dependencies
install_dependencies() {
    print_status "📦 Installing Python dependencies..."
    
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    else
        pip install transformers torch torchvision onnx onnxruntime-gpu pillow opencv-python numpy tritonclient[all]
    fi
    
    print_status "✅ Dependencies installed successfully"
}

# Create model repository structure
setup_model_repository() {
    print_status "📁 Setting up model repository structure..."
    
    mkdir -p model_repository/{vit_pipeline,vit_standard,vit_onnx}/{1,labels}
    
    # Create labels directory with common label files
    if [ ! -f "model_repository/labels/imagenet_classes.txt" ]; then
        print_status "📥 Downloading ImageNet labels..."
        curl -s https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt > model_repository/labels/imagenet_classes.txt
    fi
    
    print_status "✅ Model repository structure created"
}

# Test Docker setup
test_docker_setup() {
    print_status "🐳 Testing Docker setup..."
    
    if command -v docker &> /dev/null; then
        print_status "✅ Docker is installed"
        
        if docker info | grep -q "nvidia"; then
            print_status "✅ NVIDIA Docker runtime detected"
        else
            print_warning "⚠️  NVIDIA Docker runtime not detected. Install nvidia-container-toolkit for GPU support."
        fi
    else
        print_error "❌ Docker is not installed. Please install Docker first."
        return 1
    fi
    
    if command -v docker-compose &> /dev/null; then
        print_status "✅ Docker Compose is available"
    else
        print_warning "⚠️  Docker Compose not found. Consider installing it for easier multi-container management."
    fi
}

# Create example deployment script
create_example_scripts() {
    print_status "📝 Creating example scripts..."
    
    # Quick deploy script
    cat > quick_deploy.sh << 'EOF'
#!/bin/bash
# Quick deployment script for ViT models

MODEL=${1:-"google/vit-base-patch16-224"}
METHOD=${2:-"all"}

echo "🚀 Quick deploying ViT model: $MODEL"
echo "📋 Methods: $METHOD"

case $METHOD in
    "pipeline")
        python python_pipeline/deploy.py --model $MODEL
        ;;
    "standard")
        python python_standard/deploy.py --model $MODEL
        ;;
    "onnx")
        python onnx/deploy.py --model $MODEL
        ;;
    "all")
        echo "🔄 Deploying all methods..."
        python python_pipeline/deploy.py --model $MODEL
        python python_standard/deploy.py --model $MODEL
        python onnx/deploy.py --model $MODEL
        ;;
    *)
        echo "❌ Unknown method: $METHOD"
        echo "Available methods: pipeline, standard, onnx, all"
        exit 1
        ;;
esac

echo "✅ Deployment completed!"
EOF
    chmod +x quick_deploy.sh
    
    # Test script
    cat > test_all.sh << 'EOF'
#!/bin/bash
# Test all ViT deployments

echo "🧪 Testing all ViT deployments..."

# Test pipeline method
if [ -d "model_repository/vit_pipeline" ]; then
    echo "Testing pipeline method..."
    cd python_pipeline && python deploy.py --test && cd ..
fi

# Test standard method
if [ -d "model_repository/vit_standard" ]; then
    echo "Testing standard method..."
    cd python_standard && python deploy.py --test && cd ..
fi

# Test ONNX method
if [ -d "model_repository/vit_onnx" ]; then
    echo "Testing ONNX method..."
    cd onnx && python deploy.py --test && cd ..
fi

echo "✅ All tests completed!"
EOF
    chmod +x test_all.sh
    
    print_status "✅ Example scripts created"
}

# Main setup function
main() {
    print_status "🎯 Starting ViT deployment environment setup..."
    
    # Run checks and setup
    check_cuda
    install_dependencies
    setup_model_repository
    test_docker_setup
    create_example_scripts
    
    print_status "🎉 Environment setup completed successfully!"
    print_status ""
    print_status "📋 Next steps:"
    print_status "1. Deploy a model: ./quick_deploy.sh google/vit-base-patch16-224 all"
    print_status "2. Start Triton server: docker-compose --profile all up -d"
    print_status "3. Test deployment: ./test_all.sh"
    print_status ""
    print_status "📚 For more options, check individual deployment scripts in:"
    print_status "   - python_pipeline/deploy.py"
    print_status "   - python_standard/deploy.py"
    print_status "   - onnx/deploy.py"
}

# Run main function
main "$@"
