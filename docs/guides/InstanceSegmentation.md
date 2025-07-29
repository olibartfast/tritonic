### YOLOv8/YOLO11/YOLOv12
* Install  [following Ultralytics official documentation (pip ultralytics package version >= 8.3.0)](]https://docs.ultralytics.com/tasks/segment/) and export the model in different formats, you can use the following commands:

#### OnnxRuntime/Torchscript/Openvino

```
yolo export model=yolov8/yolo11/yolo12 n/s/m/x-seg.pt format=onnx/torchscript/openvino
```

## YoloV5 
#### OnnxRuntime/Torchscript
* Run from [yolov5 repo](https://github.com/ultralytics/yolov5) export script:  ```python export.py  --weights <yolov5seg_version>.pt  --include [onnx,torchscript]```