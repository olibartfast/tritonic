### YOLOv8/YOLO11/YOLOv12
* Install  [following Ultralytics official documentation (pip ultralytics package version >= 8.3.0)](]https://docs.ultralytics.com/tasks/segment/) and export the model in different formats, you can use the following commands:

#### OnnxRuntime/Torchscript/Openvino

```
yolo export model=yolov8/yolo11/yolo12 n/s/m/x-seg.pt format=onnx/torchscript/openvino
```

## YoloV5 
#### OnnxRuntime/Torchscript
* Run from [yolov5 repo](https://github.com/ultralytics/yolov5) export script:  ```python export.py  --weights <yolov5seg_version>.pt  --include [onnx,torchscript]```

## RF-DETR
### ONNX Export for Instance Segmentation

For instance segmentation, use the `RFDETRSegPreview` model class or the provided export script.

#### Using Python Script

```bash
python deploy/instance_segmentation/rf-detr/export_segmentation.py --simplify --input_size 432
```

**Available Options:**
- `--output_dir`: Path to save exported model (default: current directory)
- `--opset_version`: ONNX opset version (default: 17)
- `--simplify`: Simplify ONNX model using onnxsim
- `--batch_size`: Batch size for export (default: 1)
- `--input_size`: Input image size (default: 640)

#### Using Python API

```python
from rfdetr import RFDETRSegPreview

model = RFDETRSegPreview(pretrain_weights=<CHECKPOINT_PATH>)

model.export(
    opset_version=17,
    simplify=True,
    batch_size=1
)
```

**Model Outputs:**
- `dets`: Bounding boxes `[batch, num_queries, 4]` in cxcywh format (normalized)
- `labels`: Class logits `[batch, num_queries, num_classes]`
- `masks`: Segmentation masks `[batch, num_queries, mask_h, mask_w]` (e.g., 108x108)

This command saves the ONNX segmentation model to the `output` directory.

---

### TensorRT Export

For GPU deployment, you can convert the ONNX model to TensorRT format for optimized performance.

### Detection or Segmentation Models

```bash
trtexec --onnx=/path/to/model.onnx \
        --saveEngine=/path/to/model.engine \
        --memPoolSize=workspace:4096 \
        --fp16 \
        --useCudaGraph \
        --useSpinWait \
        --warmUp=500 \
        --avgRuns=1000 \
        --duration=10
```

### Using TensorRT Docker Container

```bash
export NGC_TAG_VERSION=24.12

docker run --rm -it --gpus=all \
    -v $(pwd)/exports:/exports \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v $(pwd)/model.onnx:/workspace/model.onnx \
    -w /workspace \
    nvcr.io/nvidia/tensorrt:${NGC_TAG_VERSION}-py3 \
    /bin/bash -cx "trtexec --onnx=model.onnx \
                            --saveEngine=/exports/model.engine \
                            --memPoolSize=workspace:4096 \
                            --fp16 \
                            --useCudaGraph \
                            --useSpinWait \
                            --warmUp=500 \
                            --avgRuns=1000 \
                            --duration=10"
```

 