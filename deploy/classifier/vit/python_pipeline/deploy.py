#!/bin/bash

# ViT Model Deployment Script - Method 1: Python Pipeline
# This script creates a Triton deployment using HuggingFace pipeline

import os
import argparse
import json
from pathlib import Path

def create_pipeline_deployment(model_name, output_dir, max_batch_size=8):
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
    name: "image"
    data_type: TYPE_UINT8
    dims: [ -1, -1, 3 ]  # Variable size RGB image
  }}
]

output [
  {{
    name: "predictions"
    data_type: TYPE_FP32
    dims: [ -1, 2 ]  # [score, class_id] pairs
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
'''
    
    with open(model_dir / "config.pbtxt", 'w') as f:
        f.write(config_content)
    
    # Create model.py
    model_py_content = f'''import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import pipeline
import cv2
import json
from PIL import Image

class TritonPythonModel:
    def initialize(self, args):
        """Initialize the model"""
        self.model_name = "{model_name}"
        
        # Create pipeline with explicit device
        try:
            device = 0 if pb_utils.get_cuda_device_count() > 0 else -1
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
        
        print(f"📊 Model info:")
        print(f"   - Input size: {{getattr(self.processor, 'size', 'auto')}}")
        print(f"   - Normalization: mean={{getattr(self.processor, 'image_mean', 'auto')}}")

    def execute(self, requests):
        """Execute inference"""
        responses = []
        
        for request in requests:
            try:
                # Get input image
                image_tensor = pb_utils.get_input_tensor_by_name(request, "image")
                image_data = image_tensor.as_numpy()
                
                # Convert numpy array to PIL Image
                if len(image_data.shape) == 4:  # Batch of images
                    batch_results = []
                    for img_array in image_data:
                        # Convert BGR to RGB
                        if img_array.shape[-1] == 3:
                            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                        
                        # Convert to PIL Image
                        pil_image = Image.fromarray(img_array.astype('uint8'))
                        
                        # Run pipeline
                        results = self.classifier(pil_image, top_k=1)
                        top_result = results[0]
                        
                        # Extract class ID (try to parse as int, fallback to 0)
                        try:
                            class_id = int(top_result['label'].split(' ')[0]) if ' ' in top_result['label'] else int(top_result['label'])
                        except (ValueError, AttributeError):
                            class_id = 0
                        
                        batch_results.append([top_result['score'], float(class_id)])
                    
                    output_array = np.array(batch_results, dtype=np.float32)
                else:  # Single image
                    # Convert BGR to RGB
                    if image_data.shape[-1] == 3:
                        image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
                    
                    # Convert to PIL Image
                    pil_image = Image.fromarray(image_data.astype('uint8'))
                    
                    # Run pipeline
                    results = self.classifier(pil_image, top_k=1)
                    top_result = results[0]
                    
                    # Extract class ID
                    try:
                        class_id = int(top_result['label'].split(' ')[0]) if ' ' in top_result['label'] else int(top_result['label'])
                    except (ValueError, AttributeError):
                        class_id = 0
                    
                    output_array = np.array([[top_result['score'], float(class_id)]], dtype=np.float32)
                
                # Create output tensor
                output_tensor = pb_utils.Tensor("predictions", output_array)
                response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
                responses.append(response)
                
            except Exception as e:
                print(f"❌ Error during inference: {{e}}")
                # Return error response
                error_response = pb_utils.InferenceResponse(
                    output_tensors=[],
                    error=pb_utils.TritonError(f"Pipeline inference failed: {{str(e)}}")
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
        "description": "HuggingFace pipeline deployment with automatic preprocessing",
        "advantages": [
            "Simplest setup",
            "Automatic preprocessing",
            "No ONNX export needed",
            "Easy debugging"
        ],
        "disadvantages": [
            "Slower inference",
            "Higher memory usage",
            "Less control over preprocessing"
        ],
        "usage": {
            "command": f"./tritonic --source=image.jpg --model_type=vit-classifier --model={model_dir.name} --protocol=http --serverAddress=localhost --port=8000",
            "expected_input": "Raw RGB image (any size)",
            "expected_output": "Top prediction [score, class_id]"
        }
    }
    
    with open(model_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Create example test script
    test_script = f'''#!/bin/bash
# Test script for {model_dir.name}

echo "🧪 Testing {model_dir.name} deployment..."

# Test with curl (HTTP)
curl -X POST localhost:8000/v2/models/{model_dir.name}/infer \\
    -H "Content-Type: application/json" \\
    -d '{{
        "inputs": [{{
            "name": "image",
            "shape": [1, 224, 224, 3],
            "datatype": "UINT8",
            "data": []
        }}]
    }}'

echo "\\n✅ Test completed for {model_dir.name}"
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
    parser.add_argument("--test", action="store_true",
                      help="Run test after deployment")
    
    args = parser.parse_args()
    
    # Create deployment
    model_path = create_pipeline_deployment(args.model, args.output, args.batch_size)
    
    if args.test:
        print(f"\\n🧪 Running test...")
        test_script = Path(model_path) / "test.sh"
        if test_script.exists():
            os.system(f"chmod +x {test_script} && {test_script}")

if __name__ == "__main__":
    main()
