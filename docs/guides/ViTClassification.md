# Vision Transformer (ViT) Classification with TritonIC

This guide explains how to deploy and use Hugging Face Vision Transformer models with TritonIC using three different deployment strategies.

## Overview

**⚠️ Export tools have moved to vision-core for reusability across inference engines.**

ViT (Vision Transformer) models apply transformer architectures to image classification tasks by treating images as sequences of patches. TritonIC supports ViT models through three deployment methods:

1. **Python Backend with Pipeline** - Simplest setup with automatic preprocessing
2. **Python Backend Standard** - More control with custom preprocessing  
3. **ONNX Backend** - Best performance with GPU optimization

For model export, use:
```bash
# Clone vision-core if not already available  
git clone https://github.com/olibartfast/vision-core.git
cd vision-core/export/classification/vit/onnx
python export.py
```

See [vision-core export documentation](https://github.com/olibartfast/vision-core/tree/main/export) for detailed usage.

## Quick Comparison

| Method | Pros | Cons | Best For |
|--------|------|------|----------|
| **Python Pipeline** | ✅ Simplest setup<br>✅ Automatic preprocessing<br>✅ No ONNX export needed | ❌ Slower inference<br>❌ Less control | Prototyping, Quick demos |
| **Python Standard** | ✅ More control<br>✅ Custom preprocessing<br>✅ Easy debugging | ❌ Moderate performance<br>❌ Python dependency | Development, Custom models |
| **ONNX Backend** | ✅ Best performance<br>✅ TensorRT optimization<br>✅ Production ready | ❌ Export complexity<br>❌ Limited model support | Production, High throughput |

## Server Deployment

#### Method 1: Python Pipeline
```bash
cd deploy/vit/python_pipeline
python deploy.py --model google/vit-base-patch16-224 --output ./model_repository/vit_pipeline
```

#### Method 2: Python Standard  
```bash
cd deploy/vit/python_standard
python deploy.py --model google/vit-base-patch16-224 --output ./model_repository/vit_standard
```

#### Method 3: ONNX Backend
```bash
cd deploy/vit/onnx
python deploy.py --model google/vit-base-patch16-224 --output ./model_repository/vit_onnx
```

## Client Usage

### Basic Classification

Once your ViT model is deployed on Triton Server, use TritonIC client:

```bash
# Pipeline method (port 8000)
./tritonic \
    --source=data/images/cat.jpg \
    --model_type=vit-classifier \
    --model=vit_pipeline \
    --labelsFile=data/labels/imagenet_labels.txt \
    --protocol=http \
    --serverAddress=localhost \
    --port=8000

# Standard method (port 8010)  
./tritonic \
    --source=data/images/cat.jpg \
    --model_type=vit-classifier \
    --model=vit_standard \
    --labelsFile=data/labels/imagenet_labels.txt \
    --protocol=http \
    --serverAddress=localhost \
    --port=8010

# ONNX method (port 8020)
./tritonic \
    --source=data/images/cat.jpg \
    --model_type=vit-classifier \
    --model=vit_onnx \
    --labelsFile=data/labels/imagenet_labels.txt \
    --protocol=http \
    --serverAddress=localhost \
    --port=8020
```

### Advanced Usage

#### Batch Processing
```bash
# Process multiple images
./tritonic \
    --source=data/images/ \
    --model_type=vit-classifier \
    --model=vit_onnx \
    --labelsFile=data/labels/imagenet_labels.txt \
    --protocol=grpc \
    --serverAddress=localhost \
    --port=8021
```

#### Custom Input Sizes
```bash
# For high-resolution ViT models (384x384)
./tritonic \
    --source=data/images/high_res.jpg \
    --model_type=vit-classifier \
    --model=vit_onnx_384 \
    --input_sizes="3,384,384" \
    --labelsFile=data/labels/imagenet_labels.txt \
    --protocol=http \
    --serverAddress=localhost \
    --port=8020
```

## Supported Models

### Popular ViT Models

- `google/vit-base-patch16-224` (88M parameters) - Standard resolution
- `google/vit-base-patch16-384` (88M parameters) - High resolution  
- `google/vit-large-patch16-224` (307M parameters) - Large model
- `google/vit-huge-patch14-224-in21k` (632M parameters) - Huge model
- `microsoft/beit-base-patch16-224` (87M parameters) - BEiT variant
- `microsoft/swin-base-patch4-window7-224` (88M parameters) - Swin Transformer

### Custom Models

To use custom fine-tuned ViT models:

1. Ensure your model follows the standard ViT architecture
2. Update the deployment script with your model name
3. Provide appropriate label files for your classes
4. Test with known samples first

Example:
```bash
python deploy.py --model your_username/custom-vit-model --output ./custom_vit
```

## Preprocessing Details

ViT models in TritonIC use standard ImageNet preprocessing:

- **Input Size**: 224×224 (or 384×384 for high-res models)
- **Normalization**: ImageNet statistics
  - Mean: [0.485, 0.456, 0.406] 
  - Std: [0.229, 0.224, 0.225]
- **Format**: RGB (automatically converted from BGR)
- **Range**: [0, 1] before normalization

The client automatically handles:
- BGR to RGB conversion
- Image resizing
- Normalization
- Format conversion (NCHW/NHWC)

## Output Format

ViT classifiers return top-k predictions with:
- `class_id`: Integer class identifier
- `class_confidence`: Softmax probability (0.0 to 1.0)

Example output:
```
Top 1: class 285 (Egyptian cat) with confidence 0.8234
Top 2: class 281 (tabby cat) with confidence 0.1123  
Top 3: class 282 (tiger cat) with confidence 0.0456
```

## Troubleshooting

### Common Issues

1. **Server Not Ready**
   ```bash
   # Check server health
   curl http://localhost:8000/v2/health/ready
   
   # Check server logs
   docker logs triton-vit-onnx
   ```

2. **ONNX Export Fails**
   - Ensure transformers library is up to date: `pip install transformers>=4.36.0`
   - Some models may require specific PyTorch versions

3. **Wrong Predictions**
   - Verify the labels file matches the model's training dataset
   - Check if model expects different normalization parameters

4. **Performance Issues**
   - Use ONNX backend for best performance
   - Enable TensorRT optimization
   - Consider batch processing for multiple images

### Debug Mode

```bash
# Enable verbose logging
./tritonic \
    --source=data/images/cat.jpg \
    --model_type=vit-classifier \
    --model=vit_onnx \
    --labelsFile=data/labels/imagenet_labels.txt \
    --protocol=http \
    --serverAddress=localhost \
    --port=8020 \
    --verbose
```

## Integration Examples

### Python Integration

```python
import tritonclient.http as httpclient
import numpy as np
from PIL import Image

# Initialize client
client = httpclient.InferenceServerClient(url="localhost:8000")

# Preprocess image (handled by TritonIC client normally)
image = Image.open("cat.jpg")
# ... preprocessing code ...

# Run inference
result = client.infer("vit_onnx", inputs, outputs)
```

### C++ Integration

The ViT classifier is fully integrated into the TritonIC C++ client:

```cpp
#include "ViTClassifier.hpp"

// Create ViT classifier instance
auto classifier = std::make_unique<ViTClassifier>(model_info);

// Process image
std::vector<cv::Mat> images = {cv::imread("cat.jpg")};
auto preprocessed = classifier->preprocess(images);

// Get results  
auto results = classifier->postprocess(frame_size, infer_results, infer_shapes);
```

## Best Practices

1. **Model Selection**
   - Use `vit-base-patch16-224` for balanced speed/accuracy
   - Use `vit-large-patch16-224` for maximum accuracy
   - Use high-resolution models  for fine-grained classification

2. **Deployment Strategy**
   - Use ONNX backend for production environments
   - Use Python pipeline for rapid prototyping
   - Use Python standard for custom preprocessing needs

3. **Performance Tuning**
   - Enable dynamic batching for high-throughput scenarios
   - Use appropriate batch sizes based on GPU memory
   - Monitor inference latency and throughput

4. **Error Handling**
   - Always check server health before inference
   - Validate input image formats and sizes
   - Handle network timeouts gracefully

## References

- [Hugging Face ViT Models](https://huggingface.co/models?pipeline_tag=image-classification&library=transformers)
- [Vision Transformer Paper](https://arxiv.org/abs/2010.11929)
- [Triton Inference Server Documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/)
- [TritonIC GitHub Repository](https://github.com/olibartfast/tritonic)
