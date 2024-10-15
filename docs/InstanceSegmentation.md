### YOLOv8/YOLO11
* Install  [following Ultralytics official documentation (pip ultralytics package version >= 8.3.0)](]https://docs.ultralytics.com/tasks/segment/) and export the model in different formats, you can use the following commands:

#### OnnxRuntime/Torchscript/Openvino

```
yolo export model=yolov8/yolo11 n/s/m/x-seg.pt format=onnx/torchscript/openvino
```