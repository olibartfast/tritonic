#!/bin/bash -ex

PYTORCH_VERSION=25.11
mkdir -p exports

# Run RAFT model export script
docker run --rm -it --gpus=all \
  -v $(pwd)/exports:/exports \
  -v $(pwd)/raft_exporter.py:/workspace/raft_exporter.py \
  -u $(id -u):$(id -g) \
--ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -w /workspace \
  nvcr.io/nvidia/pytorch:${PYTORCH_VERSION}-py3 /bin/bash -cx \
  "pip install onnx-graphsurgeon && python raft_exporter.py --model-type large --output-dir /exports --device cuda --format traced"

echo "RAFT model ready."
