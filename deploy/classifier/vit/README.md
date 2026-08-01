# ViT Classifier — Triton Deployment

Two Python-backend deployment strategies for Vision Transformer classification
models on Triton Inference Server.

## Setup

```bash
./docker/setup_environment.sh
```

Installs Python dependencies, creates the model repository directory skeleton,
and generates convenience scripts.

## Method 1: Python Pipeline (`python_pipeline/`)

Uses HuggingFace's `pipeline()` API internally.  Simpler to set up but the
pipeline abstraction adds a small overhead.

```bash
python3 python_pipeline/deploy.py \
    --model google/vit-base-patch16-224 \
    --output ./model_repository/vit_pipeline \
    --batch_size 8 --kind KIND_GPU
```

## Method 2: Python Standard (`python_standard/`)

Follows the VideoMAE pattern: model weights and config are downloaded and saved
locally in the version directory.  Leaner runtime, no pipeline overhead.

```bash
python3 python_standard/deploy.py \
    --model google/vit-base-patch16-224 \
    --output ./model_repository/vit_standard \
    --batch_size 8 --kind KIND_GPU
```

Both methods produce the same Triton contract: `pixel_values [N,3,224,224] FP32`
in, `logits [N,1000] FP32` out.  Client-side preprocessing (resize, BGR→RGB,
ImageNet normalization) is handled by tritonic's `vit-classifier` task.

## Export

For ONNX/TensorRT export see
[neuriplo-tasks export documentation](https://github.com/olibartfast/neuriplo-tasks/blob/master/export/README.md).
