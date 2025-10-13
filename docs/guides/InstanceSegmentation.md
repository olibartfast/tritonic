### YOLOv8/YOLO11/YOLOv12
* Install  [following Ultralytics official documentation (pip ultralytics package version >= 8.3.0)](]https://docs.ultralytics.com/tasks/segment/) and export the model in different formats, you can use the following commands:

#### OnnxRuntime/Torchscript/Openvino

```
yolo export model=yolov8/yolo11/yolo12 n/s/m/x-seg.pt format=onnx/torchscript/openvino
```

## YoloV5 
#### OnnxRuntime/Torchscript
* Run from [yolov5 repo](https://github.com/ultralytics/yolov5) export script:  ```python export.py  --weights <yolov5seg_version>.pt  --include [onnx,torchscript]```

## RF-DETR Segmentation
RF-DETR is a real-time DETR-based detector that also supports instance segmentation. The segmentation variant outputs bounding boxes, class labels, and segmentation masks.

### Model Export
```bash
# Clone the RF-DETR repository
git clone https://github.com/Peterande/RF-DETR.git
cd RF-DETR

# Export RF-DETR segmentation model to ONNX
python deploy/onnx_export.py --model rfdetr_seg --weights <path_to_checkpoint> --output rfdetr_seg.onnx
```

### Model Outputs
The RF-DETR segmentation model produces three outputs:
- **dets**: Bounding boxes in [x_center, y_center, width, height] format (normalized coordinates)
- **labels**: Class logits for each detection
- **masks**: Segmentation masks in [num_queries, mask_h, mask_w] format

### Usage Example
```bash
./build/tritonic \
    --source data/bus.jpg \
    --serverAddress localhost \
    --labelsFile labels/coco.txt \
    --model rfdetr_seg \
    --model_type rfdetrseg
```