# RF-DETR Instance Segmentation — Triton Deployment

Two-step workflow to convert a simplified ONNX model to TensorRT and deploy it
to a Triton model repository.

For model export (PyTorch → ONNX) see
[neuriplo-tasks export documentation](https://github.com/olibartfast/neuriplo-tasks/blob/master/export/README.md).

## Step 1: ONNX → TensorRT

Converts `output/inference_model.sim.onnx` to an FP16 TensorRT engine via the
NVIDIA TensorRT Docker container.

```bash
./export_trt.sh
```

The engine is written to `exports/model.engine`.

## Step 2: Deploy to Triton

Copies the TensorRT engine into a Triton model repository with the standard
`1/model.plan` layout and a minimal `config.pbtxt`.

```bash
./deploy_triton_model.sh exports/model.engine rf_detr /path/to/triton/model/repo
```

Default arguments:
- engine: `exports/model.engine`
- model name: `rf_detr`
- repo: `./triton_model_repo`
