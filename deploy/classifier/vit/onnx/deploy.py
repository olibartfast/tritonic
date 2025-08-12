#!/usr/bin/env python3

# ViT Model Deployment Script - ONNX Backend Method
# This script exports ViT to ONNX and creates high-performance deployment

import torch
from transformers import ViTImageProcessor, ViTForImageClassification
import argparse
import os
import json
from pathlib import Path
import onnx
import onnxruntime as ort

def create_onnx_deployment(model_name, output_dir, input_size=(224, 224), opset_version=17, optimize=True):
    """Export ViT model to ONNX and create Triton deployment"""
    
    print(f"🚀 Creating ONNX deployment for {model_name}")
    
    # Load model and processor
    try:
        processor = ViTImageProcessor.from_pretrained(model_name)
        model = ViTForImageClassification.from_pretrained(model_name)
        model.eval()
        print(f"✅ Successfully loaded model")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        raise
    
    # Create directory structure
    model_dir = Path(output_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    version_dir = model_dir / "1"
    version_dir.mkdir(exist_ok=True)
    
    # Create dummy input
    batch_size = 1
    channels = 3
    height, width = input_size
    dummy_input = torch.randn(batch_size, channels, height, width)
    
    print(f"📤 Exporting to ONNX...")
    print(f"   - Input shape: {dummy_input.shape}")
    print(f"   - Opset version: {opset_version}")
    
    # Export to ONNX
    onnx_path = version_dir / "model.onnx"
    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['pixel_values'],
            output_names=['logits'],
            dynamic_axes={
                'pixel_values': {0: 'batch_size'},
                'logits': {0: 'batch_size'}
            },
            verbose=False
        )
        print(f"✅ ONNX export successful")
    except Exception as e:
        print(f"❌ ONNX export failed: {e}")
        raise
    
    # Verify ONNX model
    try:
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        print(f"✅ ONNX model verification passed")
    except Exception as e:
        print(f"⚠️  ONNX model verification warning: {e}")
    
    # Test ONNX inference
    try:
        ort_session = ort.InferenceSession(str(onnx_path))
        test_input = dummy_input.numpy()
        ort_outputs = ort_session.run(None, {'pixel_values': test_input})
        print(f"✅ ONNX inference test passed - Output shape: {ort_outputs[0].shape}")
    except Exception as e:
        print(f"⚠️  ONNX inference test warning: {e}")
    
    num_classes = model.config.num_labels
    
    # Create config.pbtxt with TensorRT optimization
    config_content = f'''name: "{model_dir.name}"
backend: "onnxruntime"
max_batch_size: 8

input [
  {{
    name: "pixel_values"
    data_type: TYPE_FP32
    dims: [ {channels}, {height}, {width} ]
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
'''

    if optimize:
        config_content += f'''
optimization {{
  execution_accelerators {{
    gpu_execution_accelerator : [ {{
      name : "tensorrt"
      parameters {{ key: "precision_mode" value: "FP16" }}
      parameters {{ key: "max_workspace_size_bytes" value: "2147483648" }}
      parameters {{ key: "trt_max_partition_iterations" value: "1000" }}
      parameters {{ key: "trt_min_subgraph_size" value: "1" }}
      parameters {{ key: "trt_fp16_enable" value: "true" }}
    }} ]
  }}
}}

dynamic_batching {{
  max_queue_delay_microseconds: 100
  preserve_ordering: false
}}
'''
    
    with open(model_dir / "config.pbtxt", 'w') as f:
        f.write(config_content)
    
    # Create model optimization script
    optimize_script = f'''#!/usr/bin/env python3
"""
ONNX Model Optimization Script for {model_dir.name}
"""

import onnx
from onnxruntime.tools import optimizer
import argparse

def optimize_model(input_path, output_path):
    """Optimize ONNX model for better performance"""
    
    print(f"🔧 Optimizing ONNX model...")
    
    # Load model
    model = onnx.load(input_path)
    
    # Apply optimizations
    optimized_model = optimizer.optimize_model(
        input_path,
        model_type='bert',  # Use BERT optimizer for transformer models
        num_heads=12,       # Adjust based on your model
        hidden_size=768,    # Adjust based on your model
        optimization_level=optimizer.OptimizationLevel.ORT_ENABLE_ALL
    )
    
    # Save optimized model
    onnx.save(optimized_model, output_path)
    print(f"✅ Optimized model saved to {{output_path}}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="model.onnx")
    parser.add_argument("--output", default="model_optimized.onnx")
    args = parser.parse_args()
    
    optimize_model(args.input, args.output)
'''
    
    optimize_file = model_dir / "optimize.py"
    with open(optimize_file, 'w') as f:
        f.write(optimize_script)
    optimize_file.chmod(0o755)
    
    # Create metadata
    metadata = {
        "deployment_type": "onnx",
        "model_name": model_name,
        "input_size": [channels, height, width],
        "num_classes": num_classes,
        "opset_version": opset_version,
        "optimization": "TensorRT FP16" if optimize else "None",
        "normalization": {
            "mean": getattr(processor, 'image_mean', [0.485, 0.456, 0.406]),
            "std": getattr(processor, 'image_std', [0.229, 0.224, 0.225])
        },
        "advantages": [
            "Best performance",
            "TensorRT optimization",
            "Low memory usage",
            "Production ready"
        ],
        "disadvantages": [
            "Export complexity",
            "Limited model support",
            "Harder debugging"
        ],
        "usage": {
            "command": f"./tritonic --source=image.jpg --model_type=vit-classifier --model={model_dir.name} --labelsFile=labels.txt --protocol=http --serverAddress=localhost --port=8000",
            "expected_input": f"Normalized tensor [batch, 3, {height}, {width}]",
            "expected_output": f"Logits tensor [batch, {num_classes}]"
        }
    }
    
    with open(model_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Create benchmark script
    benchmark_script = f'''#!/usr/bin/env python3
"""
Benchmark script for {model_dir.name}
"""

import numpy as np
import time
import onnxruntime as ort
import argparse

def benchmark_model(model_path, batch_sizes=[1, 2, 4, 8], num_runs=100):
    """Benchmark ONNX model performance"""
    
    print(f"🚀 Benchmarking {{model_path}}...")
    
    # Create session
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(model_path, providers=providers)
    
    print(f"📊 Using providers: {{session.get_providers()}}")
    
    for batch_size in batch_sizes:
        print(f"\\n📈 Batch size: {{batch_size}}")
        
        # Create dummy input
        dummy_input = np.random.randn(batch_size, {channels}, {height}, {width}).astype(np.float32)
        
        # Warmup
        for _ in range(10):
            _ = session.run(None, {{'pixel_values': dummy_input}})
        
        # Benchmark
        start_time = time.time()
        for _ in range(num_runs):
            outputs = session.run(None, {{'pixel_values': dummy_input}})
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / num_runs
        throughput = batch_size / avg_time
        
        print(f"   Average latency: {{avg_time*1000:.2f}}ms")
        print(f"   Throughput: {{throughput:.2f}} images/sec")
        print(f"   Output shape: {{outputs[0].shape}}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="1/model.onnx")
    parser.add_argument("--runs", type=int, default=100)
    args = parser.parse_args()
    
    benchmark_model(args.model, num_runs=args.runs)
'''
    
    benchmark_file = model_dir / "benchmark.py"
    with open(benchmark_file, 'w') as f:
        f.write(benchmark_script)
    benchmark_file.chmod(0o755)
    
    # Create test script
    test_script = f'''#!/bin/bash
# Test script for {model_dir.name}

echo "🧪 Testing {model_dir.name} deployment..."

# Test ONNX model directly
echo "Testing ONNX inference..."
python benchmark.py --model 1/model.onnx --runs 10

# Test with curl (HTTP)
echo "\\nTesting Triton inference..."
curl -X POST localhost:8000/v2/models/{model_dir.name}/infer \\
    -H "Content-Type: application/json" \\
    -d '{{
        "inputs": [{{
            "name": "pixel_values",
            "shape": [1, 3, {height}, {width}],
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
    
    print(f"✅ ONNX deployment created successfully!")
    print(f"📁 Output directory: {output_dir}")
    print(f"🔧 ONNX model: {onnx_path}")
    print(f"⚙️  Config: {model_dir / 'config.pbtxt'}")
    print(f"📄 Metadata: {model_dir / 'metadata.json'}")
    print(f"🔧 Optimization script: {optimize_file}")
    print(f"📊 Benchmark script: {benchmark_file}")
    print(f"🧪 Test script: {test_file}")
    
    if optimize:
        print(f"🚀 Optimizations enabled: TensorRT FP16, Dynamic Batching")
    
    return str(model_dir)

def main():
    parser = argparse.ArgumentParser(description="Deploy ViT model using ONNX backend")
    parser.add_argument("--model", default="google/vit-base-patch16-224", 
                      help="HuggingFace model name")
    parser.add_argument("--output", default="./model_repository/vit_onnx", 
                      help="Output directory for Triton model")
    parser.add_argument("--input_size", nargs=2, type=int, default=[224, 224],
                      help="Input image size (height width)")
    parser.add_argument("--opset", type=int, default=17,
                      help="ONNX opset version")
    parser.add_argument("--no-optimize", action="store_true",
                      help="Disable TensorRT optimization")
    parser.add_argument("--test", action="store_true",
                      help="Run test after deployment")
    
    args = parser.parse_args()
    
    # Create deployment
    model_path = create_onnx_deployment(
        args.model, 
        args.output, 
        tuple(args.input_size), 
        args.opset,
        not args.no_optimize
    )
    
    if args.test:
        print(f"\\n🧪 Running test...")
        test_script = Path(model_path) / "test.sh"
        if test_script.exists():
            os.system(f"chmod +x {test_script} && {test_script}")

if __name__ == "__main__":
    main()
