#!/usr/bin/env python3

# ViT Model Deployment Script - Method 2: Python Standard
# This script creates a standard Python backend deployment following VideoMAE pattern

import os
import argparse
import json
import time
from pathlib import Path
from transformers import AutoImageProcessor, ViTForImageClassification

def create_standard_deployment(model_name, output_dir, max_batch_size=8, instance_kind="KIND_GPU"):
    """Create standard Python backend deployment"""
    
    print(f"🚀 Creating standard deployment for {model_name}")
    
    # Create directory structure
    model_dir = Path(output_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    version_dir = model_dir / "1"
    version_dir.mkdir(exist_ok=True)
    
    # Download and save model files locally (like VideoMAE pattern)
    print(f"📥 Downloading model files to {version_dir}")
    try:
        processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
        model = ViTForImageClassification.from_pretrained(model_name)
        
        # Save model and processor to local directory
        processor.save_pretrained(version_dir)
        model.save_pretrained(version_dir)
        
        print(f"✅ Successfully downloaded and saved model files")
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        raise
    
    input_size = processor.size.get("height", 224)
    num_classes = model.config.num_labels
    
    # Create config.pbtxt
    config_content = f'''name: "{model_dir.name}"
backend: "python"
max_batch_size: {max_batch_size}

input [
  {{
    name: "pixel_values"
    data_type: TYPE_FP32
    dims: [ 3, {input_size}, {input_size} ]
  }}
]

output [
  {{
    name: "logits"
    data_type: TYPE_FP32
    dims: [ {num_classes} ]
  }}
]

instance_group [
  {{
    count: 1
    kind: {instance_kind}
  }}
]

dynamic_batching {{
  max_queue_delay_microseconds: 100
}}
'''
    
    with open(model_dir / "config.pbtxt", 'w') as f:
        f.write(config_content)
    
    # Create model.py following VideoMAE pattern
    model_py_content = f'''import json
import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from transformers import AutoImageProcessor, ViTForImageClassification
import time
import os

class TritonPythonModel:
    """ViT model for Triton Inference Server."""

    def initialize(self, args):
        """Called once when the model is loaded."""
        self.model_config = json.loads(args["model_config"])

        # Get model path - use the directory where model.py is located
        model_path = os.path.dirname(os.path.abspath(__file__))
        print(f"Loading model from: {{model_path}}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoImageProcessor.from_pretrained(model_path, use_fast=True)
        self.model = ViTForImageClassification.from_pretrained(model_path).to(self.device)
        self.model.eval()

        # Cache model info
        self.input_size = self.processor.size.get("height", 224)
        self.num_classes = self.model.config.num_labels
        
        print(f"✅ Loaded ViT model on {{self.device}}")
        print(f"📊 Model specifications:")
        print(f"   - Input size: {{self.input_size}}x{{self.input_size}}")
        print(f"   - Number of classes: {{self.num_classes}}")

        # Get output configuration
        output_config = pb_utils.get_output_config_by_name(self.model_config, "logits")
        self.output_dtype = pb_utils.triton_string_to_numpy(output_config["data_type"])

    def execute(self, requests):
        """Handle inference requests."""
        responses = []

        for request in requests:
            try:
                # Get input tensor (pixel_values) from the request
                input_tensor = pb_utils.get_input_tensor_by_name(request, "pixel_values")
                input_np = input_tensor.as_numpy()  # Shape: [batch_size, 3, 224, 224]

                # Validate input shape
                if input_np.shape[1:] != (3, self.input_size, self.input_size):
                    raise ValueError(f"Expected input shape [batch, 3, {{self.input_size}}, {{self.input_size}}], got {{input_np.shape}}")

                # Convert NumPy input to PyTorch tensor and move to device
                pixel_values = torch.from_numpy(input_np).to(self.device)

                # Run inference
                start_time = time.time()
                with torch.no_grad():
                    outputs = self.model(pixel_values=pixel_values)
                    logits = outputs.logits  # Shape: [batch_size, num_classes]
                inference_time = time.time() - start_time
                print(f"Inference time: {{inference_time:.4f}} seconds")

                # Convert logits to NumPy for Triton response
                logits_np = logits.cpu().numpy().astype(self.output_dtype)

                # Log prediction info for debugging
                batch_size = logits_np.shape[0]
                for b in range(batch_size):
                    top_logit_idx = np.argmax(logits_np[b])
                    top_logit_val = logits_np[b][top_logit_idx]
                    print(f"📊 Batch {{b}}: Top prediction class {{top_logit_idx}} with logit {{top_logit_val:.4f}}")

                # Create output tensor
                output_tensor = pb_utils.Tensor("logits", logits_np)
                response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
                responses.append(response)

            except Exception as e:
                print(f"❌ Error during inference: {{e}}")
                # Return error response
                error_response = pb_utils.InferenceResponse(
                    output_tensors=[],
                    error=pb_utils.TritonError(f"ViT inference failed: {{str(e)}}")
                )
                responses.append(error_response)

        return responses

    def finalize(self):
        """Clean up when unloading the model."""
        print("🧹 Cleaning up ViT model...")
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'processor'):
            del self.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
'''
    
    with open(version_dir / "model.py", 'w') as f:
        f.write(model_py_content)
    
    # Create metadata
    metadata = {
        "deployment_type": "python_standard",
        "model_name": model_name,
        "input_size": [3, input_size, input_size],
        "num_classes": num_classes,
        "client_preprocessing": {
            "note": "All preprocessing handled by TritonIC client",
            "steps": [
                "BGR to RGB conversion",
                f"Resize to {input_size}x{input_size}",
                "Normalize to [0,1] range",
                "Apply ImageNet normalization",
                "Convert to NCHW format",
                "Convert to float32"
            ]
        },
        "deployment_pattern": "VideoMAE-style local model loading",
        "advantages": [
            "Clean separation of preprocessing and inference",
            "No duplicate preprocessing overhead",
            "Consistent with TritonIC client architecture",
            "Local model files for faster loading",
            "Simple server-side inference only"
        ],
        "disadvantages": [
            "Requires TritonIC client for preprocessing",
            "Python dependency on server",
            "Model files stored locally (disk space)"
        ],
        "usage": {
            "command": f"./tritonic --source=image.jpg --model_type=vit-classifier --model={model_dir.name} --labelsFile=labels.txt --protocol=http --serverAddress=localhost --port=8000",
            "expected_input": f"Preprocessed tensor [batch, 3, {input_size}, {input_size}] - fully processed by client",
            "expected_output": f"Raw logits tensor [batch, {num_classes}] - client applies softmax"
        }
    }
    
    with open(model_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Create README explaining the deployment
    readme_content = f'''# ViT Standard Deployment - {model_name}

## Overview
This is a VideoMAE-style deployment optimized for the TritonIC client.

## Key Features
- **No Server-Side Preprocessing**: All preprocessing is handled by the TritonIC client
- **Local Model Files**: Model and processor are stored locally for fast loading
- **Clean Architecture**: Server only handles inference, client handles preprocessing

## Expected Input
- **Format**: Float32 tensor
- **Shape**: [batch_size, 3, {input_size}, {input_size}]
- **Preprocessing**: Already done by TritonIC client
  - BGR to RGB conversion
  - Resized to {input_size}x{input_size}
  - Normalized to [0,1] range
  - ImageNet normalization applied
  - NCHW format

## Expected Output
- **Format**: Float32 tensor
- **Shape**: [batch_size, {num_classes}]
- **Content**: Raw logits (TritonIC client applies softmax)

## Usage
```bash
./tritonic \\
    --source=image.jpg \\
    --model_type=vit-classifier \\
    --model={model_dir.name} \\
    --labelsFile=labels.txt \\
    --protocol=http \\
    --serverAddress=localhost \\
    --port=8000
```

## Model Files
- `model.py`: Triton Python backend model
- `config.pbtxt`: Triton model configuration
- `pytorch_model.bin`: Model weights
- `config.json`: Model configuration
- `preprocessor_config.json`: Preprocessing configuration (for reference)

## Performance
- Fast loading (local model files)
- No preprocessing overhead on server
- Optimized for production deployment
'''
    
    readme_file = model_dir / "README.md"
    with open(readme_file, 'w') as f:
        f.write(readme_content)
    
    # Create test script
    test_script = f'''#!/bin/bash
# Test script for {model_dir.name} - Server-only inference deployment

echo "🧪 Testing {model_dir.name} deployment..."
echo "ℹ️  This model expects preprocessed data from TritonIC client"
echo "📁 Model files are stored locally for fast loading"
echo "🚫 No server-side preprocessing - client handles everything"

# Test with TritonIC client (the only supported method)
echo "📱 Testing with TritonIC client..."
if command -v ./tritonic &> /dev/null; then
    echo "✅ TritonIC client found - running inference test..."
    ./tritonic \\
        --source=../../../data/images/cat.jpeg \\
        --model_type=vit-classifier \\
        --model={model_dir.name} \\
        --labelsFile=../../../labels/imagenet.txt \\
        --protocol=http \\
        --serverAddress=localhost \\
        --port=8000
else
    echo "❌ TritonIC client not found!"
    echo "💡 This deployment requires the TritonIC client for preprocessing"
    echo "� Install and build TritonIC client first"
    exit 1
fi

echo "\\n📋 Deployment Summary:"
echo "   - Pattern: VideoMAE-style inference-only server"
echo "   - Model files: Stored locally in ./1/ directory"
echo "   - Preprocessing: 100% handled by TritonIC client"
echo "   - Server role: Pure inference (no preprocessing)"
echo "   - Performance: Optimized for production"

echo "\\n✅ Test completed for {model_dir.name}"
echo "💡 Always use TritonIC client - no other preprocessing supported"
'''
    
    test_file = model_dir / "test.sh"
    with open(test_file, 'w') as f:
        f.write(test_script)
    test_file.chmod(0o755)
    
    print(f"✅ Standard deployment created successfully!")
    print(f"📁 Output directory: {output_dir}")
    print(f"🐍 Model file: {version_dir / 'model.py'}")
    print(f"⚙️  Config: {model_dir / 'config.pbtxt'}")
    print(f"📄 Metadata: {model_dir / 'metadata.json'}")
    print(f"� README: {readme_file}")
    print(f"🧪 Test script: {test_file}")
    print(f"📦 Model files: Saved locally in {version_dir}")
    print(f"🚫 No server-side preprocessing - client handles everything")
    print(f"💡 Use TritonIC client for complete preprocessing and inference pipeline")
    
    return str(model_dir)

def main():
    parser = argparse.ArgumentParser(description="Deploy ViT model using Python Standard backend")
    parser.add_argument("--model", default="google/vit-base-patch16-224", 
                      help="HuggingFace model name")
    parser.add_argument("--output", default="./model_repository/vit_standard", 
                      help="Output directory for Triton model")
    parser.add_argument("--batch_size", type=int, default=8,
                      help="Maximum batch size")
    parser.add_argument("--kind", default="KIND_GPU", 
                      choices=["KIND_GPU", "KIND_CPU"], 
                      help="Instance kind (KIND_GPU or KIND_CPU)")
    parser.add_argument("--test", action="store_true",
                      help="Run test after deployment")
    
    args = parser.parse_args()
    
    # Create deployment
    model_path = create_standard_deployment(args.model, args.output, args.batch_size, args.kind)
    
    if args.test:
        print(f"\\n🧪 Running test...")
        test_script = Path(model_path) / "test.sh"
        if test_script.exists():
            os.system(f"chmod +x {test_script} && {test_script}")

if __name__ == "__main__":
    main()
