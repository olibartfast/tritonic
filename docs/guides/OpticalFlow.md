# Optical Flow Models Export Guide

## RAFT

**⚠️ Export tools have moved to vision-core for reusability across inference engines.**

## Export Options
To export, use the raft_exporter from vision-core:

```bash
# Clone vision-core if not already available
git clone https://github.com/olibartfast/vision-core.git
cd vision-core/export/optical_flow/raft

python raft_exporter.py [options]
```

See [vision-core export documentation](https://github.com/olibartfast/vision-core/tree/main/export) for detailed usage.  

The export script supports several configuration options:

| Option | Values | Description |
|--------|---------|-------------|
| `--model-type` | `small`, `large` | RAFT model variant to export |
| `--format` | `traced`, `scripted`, `onnx` | Export format |
| `--device` | `cuda`, `cpu` | Device to use for export |
| `--output-dir` | path | Directory for exported models |
| `--batch-size` | integer | Batch size for sample inputs |
| `--height` | integer | Height of sample inputs |
| `--width` | integer | Width of sample inputs |

## Export Formats
### TorchScript Traced
- Filename format: `raft_{size}_traced_torchscript.pt`

### TorchScript Scripted
- Filename format: `raft_{size}_scripted_torchscript.pt`

### ONNX
- Filename format: `raft_{size}.onnx`

## Example Commands

Export large model in traced format:
```bash
docker run --rm -it --gpus=all \
  -v $(pwd)/exports:/exports \
  -v $(pwd)/raft_exporter.py:/workspace/raft_exporter.py \
  -u $(id -u):$(id -g) \
  -w /workspace \
  nvcr.io/nvidia/pytorch:24.12-py3 /bin/bash -cx \
  "python raft_exporter.py --model-type large --output-dir /exports --device cuda --format traced"
```

## GPU Troubleshooting
Common issues and solutions:
1. CUDA Out of Memory
   - Reduce batch size
   - Use smaller input dimensions
   - Try the small model variant

2. Export Failures
   - Check CUDA/GPU availability
   - Verify torchvision installation
   - Ensure correct container version
