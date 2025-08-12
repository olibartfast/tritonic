#!/usr/bin/env python3

# ViT Model Deployment Script - Method 2: Python Standard
# This script creates a standard Python backend deployment with more control

import os
import argparse
import json
from pathlib import Path
from transformers import ViTImageProcessor, ViTForImageClassification

def create_standard_deployment(model_name, output_dir, max_batch_size=8):
    """Create standard Python backend deployment"""
    
    print(f"🚀 Creating standard deployment for {model_name}")
    
    # Load model to get info
    try:
        processor = ViTImageProcessor.from_pretrained(model_name)
        model = ViTForImageClassification.from_pretrained(model_name)
        print(f"✅ Successfully loaded model info")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        raise
    
    # Create directory structure
    model_dir = Path(output_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    version_dir = model_dir / "1"
    version_dir.mkdir(exist_ok=True)
    
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
    kind: KIND_GPU
  }}
]

parameters {{
  key: "EXECUTION_ENV_PATH"
  value: {{string_value: "/opt/tritonserver/python_envs/vit_env"}}
}}

dynamic_batching {{
  max_queue_delay_microseconds: 100
}}
'''
    
    with open(model_dir / "config.pbtxt", 'w') as f:
        f.write(config_content)
    
    # Create model.py
    model_py_content = f'''import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import ViTImageProcessor, ViTForImageClassification
import torch
import json

class TritonPythonModel:
    def initialize(self, args):
        """Initialize the model"""
        self.model_name = "{model_name}"
        
        try:
            # Load model and processor
            self.processor = ViTImageProcessor.from_pretrained(self.model_name)
            self.model = ViTForImageClassification.from_pretrained(self.model_name)
            self.model.eval()
            
            # Setup device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            
            # Cache model info
            self.input_size = self.processor.size.get("height", 224)
            self.num_classes = self.model.config.num_labels
            self.image_mean = getattr(self.processor, 'image_mean', [0.485, 0.456, 0.406])
            self.image_std = getattr(self.processor, 'image_std', [0.229, 0.224, 0.225])
            
            print(f"✅ Loaded model: {{self.model_name}} on {{self.device}}")
            print(f"📊 Model specifications:")
            print(f"   - Input size: {{self.input_size}}x{{self.input_size}}")
            print(f"   - Number of classes: {{self.num_classes}}")
            print(f"   - Normalization mean: {{self.image_mean}}")
            print(f"   - Normalization std: {{self.image_std}}")
            
        except Exception as e:
            print(f"❌ Failed to initialize model: {{e}}")
            raise

    def execute(self, requests):
        """Execute inference"""
        responses = []
        
        for request in requests:
            try:
                # Get input tensor
                pixel_values_tensor = pb_utils.get_input_tensor_by_name(request, "pixel_values")
                pixel_values = pixel_values_tensor.as_numpy()
                
                # Validate input shape
                expected_shape = [-1, 3, self.input_size, self.input_size]
                if pixel_values.shape[1:] != (3, self.input_size, self.input_size):
                    raise ValueError(f"Expected input shape [batch, 3, {{self.input_size}}, {{self.input_size}}], got {{pixel_values.shape}}")
                
                # Convert to torch tensor
                pixel_values_torch = torch.from_numpy(pixel_values).to(self.device)
                
                # Validate tensor range (should be normalized)
                if pixel_values_torch.min() < -3 or pixel_values_torch.max() > 3:
                    print(f"⚠️  Warning: Input values outside expected normalized range [{{pixel_values_torch.min():.3f}}, {{pixel_values_torch.max():.3f}}]")
                
                # Run inference
                with torch.no_grad():
                    outputs = self.model(pixel_values=pixel_values_torch)
                    logits = outputs.logits
                
                # Convert back to numpy
                logits_np = logits.cpu().numpy().astype(np.float32)
                
                # Create output tensor
                output_tensor = pb_utils.Tensor("logits", logits_np)
                response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
                responses.append(response)
                
            except Exception as e:
                print(f"❌ Error during inference: {{e}}")
                # Return error response
                error_response = pb_utils.InferenceResponse(
                    output_tensors=[],
                    error=pb_utils.TritonError(f"Standard inference failed: {{str(e)}}")
                )
                responses.append(error_response)
        
        return responses

    def finalize(self):
        """Clean up"""
        print("🧹 Cleaning up ViT Standard model...")
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
        "normalization": {
            "mean": getattr(processor, 'image_mean', [0.485, 0.456, 0.406]),
            "std": getattr(processor, 'image_std', [0.229, 0.224, 0.225])
        },
        "advantages": [
            "More control over preprocessing",
            "Better performance than pipeline",
            "Easy debugging",
            "Custom preprocessing possible"
        ],
        "disadvantages": [
            "Requires manual preprocessing",
            "More complex setup",
            "Python dependency"
        ],
        "usage": {
            "command": f"./tritonic --source=image.jpg --model_type=vit-classifier --model={model_dir.name} --labelsFile=labels.txt --protocol=http --serverAddress=localhost --port=8000",
            "expected_input": f"Normalized tensor [batch, 3, {input_size}, {input_size}]",
            "expected_output": f"Logits tensor [batch, {num_classes}]"
        }
    }
    
    with open(model_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Create preprocessing helper
    preprocess_script = f'''#!/usr/bin/env python3
"""
Preprocessing helper for {model_dir.name}
Usage: python preprocess.py input_image.jpg output_tensor.npy
"""

import numpy as np
import cv2
import sys
from pathlib import Path

def preprocess_image(image_path, output_path=None):
    """Preprocess image for ViT model"""
    
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image: {{image_path}}")
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize to model input size
    img = cv2.resize(img, ({input_size}, {input_size}))
    
    # Convert to float and normalize to [0, 1]
    img = img.astype(np.float32) / 255.0
    
    # Apply ImageNet normalization
    mean = np.array({processor.image_mean})
    std = np.array({processor.image_std})
    img = (img - mean) / std
    
    # Convert to CHW format and add batch dimension
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, axis=0)   # Add batch dimension
    
    if output_path:
        np.save(output_path, img)
        print(f"✅ Preprocessed tensor saved to {{output_path}}")
    
    return img

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python preprocess.py input_image.jpg [output_tensor.npy]")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else input_path.with_suffix('.npy')
    
    try:
        tensor = preprocess_image(input_path, output_path)
        print(f"📊 Tensor shape: {{tensor.shape}}")
        print(f"📊 Tensor range: [{{tensor.min():.3f}}, {{tensor.max():.3f}}]")
    except Exception as e:
        print(f"❌ Error: {{e}}")
        sys.exit(1)
'''
    
    preprocess_file = model_dir / "preprocess.py"
    with open(preprocess_file, 'w') as f:
        f.write(preprocess_script)
    preprocess_file.chmod(0o755)
    
    # Create test script
    test_script = f'''#!/bin/bash
# Test script for {model_dir.name}

echo "🧪 Testing {model_dir.name} deployment..."

# Test preprocessing
python preprocess.py ../../../data/images/cat.jpeg test_input.npy

# Test with curl (HTTP)
echo "Testing inference..."
curl -X POST localhost:8000/v2/models/{model_dir.name}/infer \\
    -H "Content-Type: application/json" \\
    -d '{{
        "inputs": [{{
            "name": "pixel_values",
            "shape": [1, 3, {input_size}, {input_size}],
            "datatype": "FP32",
            "data": []
        }}]
    }}'

echo "\\n✅ Test completed for {model_dir.name}"
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
    print(f"🔧 Preprocessing helper: {preprocess_file}")
    print(f"🧪 Test script: {test_file}")
    
    return str(model_dir)

def main():
    parser = argparse.ArgumentParser(description="Deploy ViT model using Python Standard backend")
    parser.add_argument("--model", default="google/vit-base-patch16-224", 
                      help="HuggingFace model name")
    parser.add_argument("--output", default="./model_repository/vit_standard", 
                      help="Output directory for Triton model")
    parser.add_argument("--batch_size", type=int, default=8,
                      help="Maximum batch size")
    parser.add_argument("--test", action="store_true",
                      help="Run test after deployment")
    
    args = parser.parse_args()
    
    # Create deployment
    model_path = create_standard_deployment(args.model, args.output, args.batch_size)
    
    if args.test:
        print(f"\\n🧪 Running test...")
        test_script = Path(model_path) / "test.sh"
        if test_script.exists():
            os.system(f"chmod +x {test_script} && {test_script}")

if __name__ == "__main__":
    main()
