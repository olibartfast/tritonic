## Instructions

### Export the model to deploy inside model repository referenced by Triton Server

## YOLO11
Install YOLO11 [following Ultralytics official documentation (pip ultralytics package version >= > 8.3.0)](https://docs.ultralytics.com/quickstart/) and export the model in different formats, you can use the following commands:

#### Torchscript

To export the model in the TorchScript format:

```
yolo export model=best.pt(the best corrisponding to your trained yolo11n/s/m/x) format=torchscript
```

#### OnnxRuntime

To export the model in the ONNXRuntime format:

```
yolo export model=best.pt format=onnx
```

#### TensorRT

To export the model in the TensorRT format:

```
yolo export model=best.pt format=engine
```

Please note that when using TensorRT, ensure that the version installed under Ultralytics python environment matches the C++ version you plan to use for inference. Another way to export the model is to use `trtexec` with the following command:

```
trtexec --onnx=best.onnx --saveEngine=best.engine
```

## YOLOv10
### OnnxRuntime 
* From [yolov10 repo](https://github.com/THU-MIG/yolov10) or [ultralytics package](https://pypi.org/project/ultralytics/):
```
yolo export format=onnx model=yolov10model.pt

```
#### Torchscript
* Same way as above:
```
yolo export format=torchscript model=yolov10model.pt

```

#### Tensorrt
```
trtexec --onnx=yolov10model.onnx --saveEngine=yolov10model.engine --fp16
```


## YOLOv9
from [yolov9 repo](https://github.com/WongKinYiu/yolov9):
#### OnnxRuntime
```
 python export.py --weights yolov9-c/e-converted.pt --include onnx
```
#### Torchscript
```
 python export.py --weights yolov9-c/e-converted.pt --include torchscript
```
#### TensorRT
```
 trtexec --onnx=yolov9-c/e-converted.onnx --saveEngine=yolov9-c/e.engine --fp16
```

## YoloV8
* same as YOLO11, check ultralytics documentation



## YoloV5 
#### OnnxRuntime
* Run from [yolov5 repo](https://github.com/ultralytics/yolov5/issues/251) export script:  ```python export.py  --weights <yolov5_version>.pt  --include onnx```

#### Libtorch
* from yolov5 repo: ```python export.py  --weights <yolov5_version>.pt  --include torchscript```

## YoloV6
#### OnnxRuntime
Weights to export in ONNX format or download from [yolov6 repo](https://github.com/meituan/YOLOv6/tree/main/deploy/ONNX). Posteprocessing code is identical to yolov5-v7.


## YoloV7
#### OnnxRuntime and/or Libtorch
* Run from [yolov7 repo](https://github.com/WongKinYiu/yolov7#export): ```python export.py --weights <yolov7_version>.pt --grid  --simplify --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640 --max-wh 640``` (Don't use end-to-end parameter)


## Yolo-Nas export 
#### OnnxRuntime
* Weights can be export in ONNX format like in [YoloNAS Quickstart](https://github.com/Deci-AI/super-gradients/blob/master/documentation/source/YoloNASQuickstart.md#export-to-onnx).  
I export the model specifying input and output layers name, for example here below in the case of yolo_nas_s version:
```
from super_gradients.training import models

net = models.get("yolo_nas_s", pretrained_weights="coco")
models.convert_to_onnx(model=net, input_shape=(3,640,640), out_path="yolo_nas_s.onnx", torch_onnx_export_kwargs={"input_names": ['input'], "output_names": ['output0', 'output1']})
```


