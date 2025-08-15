#!/bin/bash

# ViT Model Deployment Script - Method 1: Python Pipeline
# This script creates a Triton deployment using HuggingFace pipeline

import os
import argparse
import json
from pathlib import Path

def create_pipeline_deployment(model_name, output_dir, max_batch_size=8, instance_kind="KIND_GPU"):
    """Create Triton deployment using HuggingFace pipeline"""
    
    print(f"🚀 Creating pipeline deployment for {model_name}")
    
    # Create directory structure
    model_dir = Path(output_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    version_dir = model_dir / "1"
    version_dir.mkdir(exist_ok=True)
    
    # Create config.pbtxt
    config_content = f'''name: "{model_dir.name}"
backend: "python"
max_batch_size: {max_batch_size}

input [
  {{
    name: "pixel_values"
    data_type: TYPE_FP32
    dims: [ 3, 224, 224 ]
  }}
]

output [
  {{
    name: "logits"
    data_type: TYPE_FP32
    dims: [ 1000 ]
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
    
    # Create model.py
    model_py_content = f'''import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import pipeline
import torch
import json
from PIL import Image

class TritonPythonModel:
    def initialize(self, args):
        """Initialize the model"""
        self.model_config = json.loads(args["model_config"])
        self.model_name = "{model_name}"
        
        # Create pipeline with explicit device
        try:
            # Check for CUDA availability using torch
            import torch
            device = 0 if torch.cuda.is_available() and torch.cuda.device_count() > 0 else -1
            self.classifier = pipeline(
                "image-classification",
                model=self.model_name,
                device=device,
                torch_dtype="auto" if device >= 0 else None
            )
            print(f"✅ Loaded pipeline model: {{self.model_name}} on device {{device}}")
        except Exception as e:
            print(f"❌ Failed to load pipeline model: {{e}}")
            raise
        
        # Get model info
        self.processor = getattr(self.classifier, 'feature_extractor', None)
        if self.processor is None:
            self.processor = getattr(self.classifier, 'image_processor', None)
        
        # Cache model info
        self.input_size = self.processor.size.get("height", 224) if self.processor else 224
        self.num_classes = self.classifier.model.config.num_labels
        
        print(f"📊 Model specifications:")
        print(f"   - Input size: {{self.input_size}}x{{self.input_size}}")
        print(f"   - Number of classes: {{self.num_classes}}")
        
        # Get output configuration
        output_config = pb_utils.get_output_config_by_name(self.model_config, "logits")
        self.output_dtype = pb_utils.triton_string_to_numpy(output_config["data_type"])

    def execute(self, requests):
        """Execute inference"""
        responses = []
        
        for request in requests:
            try:
                # Get input tensor (pixel_values) from the request
                input_tensor = pb_utils.get_input_tensor_by_name(request, "pixel_values")
                input_np = input_tensor.as_numpy()  # Shape: [batch_size, 3, 224, 224]

                # Validate input shape
                if input_np.shape[1:] != (3, 224, 224):
                    raise ValueError(f"Expected input shape [batch, 3, 224, 224], got {{input_np.shape}}")

                # Convert NumPy input to PyTorch tensor and move to device
                pixel_values = torch.from_numpy(input_np)
                if hasattr(self.classifier, 'device') and self.classifier.device.type == 'cuda':
                    pixel_values = pixel_values.cuda()

                # Run inference using the pipeline's model directly
                with torch.no_grad():
                    outputs = self.classifier.model(pixel_values=pixel_values)
                    logits = outputs.logits  # Shape: [batch_size, num_classes]

                # Convert logits to NumPy for Triton response
                logits_np = logits.cpu().numpy().astype(np.float32)

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
        """Clean up"""
        print("🧹 Cleaning up ViT Pipeline model...")
        if hasattr(self, 'classifier'):
            del self.classifier
'''
    
    with open(version_dir / "model.py", 'w') as f:
        f.write(model_py_content)
    
    # Create metadata
    metadata = {
        "deployment_type": "python_pipeline",
        "model_name": model_name,
        "description": "HuggingFace pipeline deployment with consistent TritonIC client interface",
        "dependencies": [
            "transformers",
            "torch", 
            "numpy"
        ],
        "client_preprocessing": {
            "note": "All preprocessing handled by TritonIC client (same as standard method)",
            "steps": [
                "BGR to RGB conversion",
                "Resize to 224x224",
                "Normalize to [0,1] range",
                "Apply ImageNet normalization",
                "Convert to NCHW format",
                "Convert to float32"
            ]
        },
        "advantages": [
            "Consistent with TritonIC client interface",
            "Same preprocessing as standard/ONNX methods",
            "No preprocessing duplication",
            "Uses HuggingFace pipeline internally"
        ],
        "disadvantages": [
            "Requires TritonIC client for preprocessing",
            "Python dependency on server",
            "Slightly more overhead than standard method"
        ],
        "usage": {
            "command": f"./tritonic --source=image.jpg --model_type=vit-classifier --model={model_dir.name} --labelsFile=labels.txt --protocol=http --serverAddress=localhost --port=8000",
            "expected_input": f"Preprocessed tensor [batch, 3, 224, 224] - fully processed by client",
            "expected_output": f"Raw logits tensor [batch, 1000] - client applies softmax"
        }
    }
    
    with open(model_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Create example test script
    test_script = f'''#!/bin/bash
# Test script for {model_dir.name} - Consistent with TritonIC client interface

echo "🧪 Testing {model_dir.name} deployment..."
echo "ℹ️  This model expects preprocessed data from TritonIC client"
echo "🔄 Uses HuggingFace pipeline internally but consistent interface"

# Test with TritonIC client (the recommended method)
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
    echo "🔧 Install and build TritonIC client first"
    exit 1
fi

echo "\\n📋 Deployment Summary:"
echo "   - Pattern: HuggingFace pipeline with TritonIC client interface"
echo "   - Input: Preprocessed float32 tensors from client"
echo "   - Output: Raw logits (client applies softmax)"
echo "   - Preprocessing: 100% handled by TritonIC client"

echo "\\n✅ Test completed for {model_dir.name}"
echo "💡 Interface is consistent with vit_standard and vit_onnx"
'''
    
    test_file = model_dir / "test.sh"
    with open(test_file, 'w') as f:
        f.write(test_script)
    test_file.chmod(0o755)
    
    print(f"✅ Pipeline deployment created successfully!")
    print(f"📁 Output directory: {output_dir}")
    print(f"🐍 Model file: {version_dir / 'model.py'}")
    print(f"⚙️  Config: {model_dir / 'config.pbtxt'}")
    print(f"📄 Metadata: {model_dir / 'metadata.json'}")
    print(f"🧪 Test script: {test_file}")
    
    return str(model_dir)

def main():
    parser = argparse.ArgumentParser(description="Deploy ViT model using Python Pipeline backend")
    parser.add_argument("--model", default="google/vit-base-patch16-224", 
                      help="HuggingFace model name")
    parser.add_argument("--output", default="./model_repository/vit_pipeline", 
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
    model_path = create_pipeline_deployment(args.model, args.output, args.batch_size, args.kind)
    
    if args.test:
        print(f"\\n🧪 Running test...")
        test_script = Path(model_path) / "test.sh"
        if test_script.exists():
            os.system(f"chmod +x {test_script} && {test_script}")

if __name__ == "__main__":
    main()
