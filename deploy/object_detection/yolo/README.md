# YOLO Object Detection — Triton Deployment

Covers YOLOv5, v8, v9, v10, v11, v12, and YOLO-NAS.  For model export to
ONNX/TensorRT, see
[neuriplo-tasks export documentation](https://github.com/olibartfast/neuriplo-tasks/blob/master/export/README.md).

## Triton Model Repository Layout

After export, place the ONNX or TensorRT engine in a Triton model repository:

```
model_repository/
└── yolov8n/
    ├── config.pbtxt
    └── 1/
        └── model.onnx     # or model.plan for TensorRT
```

Run inference with tritonic:

```bash
./build/tritonic \
    --source=data/images/bus.jpg --model_type=yolo \
    --model=yolov8n --labelsFile=labels/coco.txt \
    --protocol=grpc --serverAddress=localhost --port=8001
```

## GPU Preprocessing Ensemble

Server-side DALI/CUDA preprocessing eliminates CPU decode and resize overhead.
See the [YOLO GPU preprocessing ensemble contract](ensemble/README.md) for the
tensor contract, DALI reference implementation, and correctness gates.
