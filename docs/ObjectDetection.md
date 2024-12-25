## Instructions

### Export the model to deploy inside model repository referenced by Triton Server

## YOLOv8/YOLO11
Install  [following Ultralytics official documentation (pip ultralytics package version >= 8.3.0)](https://docs.ultralytics.com/quickstart/) and export the model in different formats, you can use the following commands:

#### OnnxRuntime/Torchscript

To export the model in the TorchScript format:

```
yolo export model=best.pt(the best corrisponding to your trained or yolov8/yolo11n/s/m/x ) format=onnx/torchscript
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
### OnnxRuntime/Torchscript
* From [yolov10 repo](https://github.com/THU-MIG/yolov10) or [ultralytics package](https://pypi.org/project/ultralytics/):
```
yolo export format=onnx/torchscript model=yolov10model.pt

```

#### Tensorrt
```
trtexec --onnx=yolov10model.onnx --saveEngine=yolov10model.engine --fp16
```


## YOLOv9
from [yolov9 repo](https://github.com/WongKinYiu/yolov9):
#### OnnxRuntime/Torchscript
```
 python export.py --weights yolov9-c/e-converted.pt --include onnx/torchscript
```

#### TensorRT
```
 trtexec --onnx=yolov9-c/e-converted.onnx --saveEngine=yolov9-c/e.engine --fp16
```



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


# RT-DETR Export Instructions

From the [lyuwenyu RT-DETR repository](https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetr_pytorch):

## OnnxRuntime
```bash
cd RT-DETR/rtdetr_pytorch
python tools/export_onnx.py -c configs/rtdetr/rtdetr_r18vd_6x_coco.yml -r path/to/checkpoint --check
```
Note: You can use other versions instead of `rtdetr_r18vd_6x_coco.yml`.

## TensorRT
```bash
trtexec --onnx=<model>.onnx --saveEngine=rtdetr_r18vd_dec3_6x_coco_from_paddle.engine --minShapes=images:1x3x640x640,orig_target_sizes:1x2 --optShapes=images:1x3x640x640,orig_target_sizes:1x2 --maxShapes=images:1x3x640x640,orig_target_sizes:1x2
```
Note: This assumes you exported the ONNX model in the previous step.


# RT-DETR (Ultralytics) Export Instructions

Always use the [Ultralytics pip package](https://docs.ultralytics.com/quickstart/) to export the model.

## OnnxRuntime
Export the model to ONNX using the following command:

```bash
yolo export model=best.pt format=onnx
```
Note: In this case, `best.pt` is a trained RTDETR-L or RTDETR-X model.

## Libtorch
Similar to the ONNX case, change format to torchscript:

```bash
yolo export model=best.pt format=torchscript 
```

## TensorRT
Same as explained for YOLOv8:

```bash
trtexec --onnx=yourmodel.onnx --saveEngine=yourmodel.engine
```

Or:

```bash
yolo export model=yourmodel.pt format=engine
```

For more information, visit: https://docs.ultralytics.com/models/rtdetr/


# **D-FINE Export Instructions**  

## **Exporting ONNX Models with ONNXRuntime**  
To export D-FINE models to ONNX format, follow the steps below:  

### **Repository**  
[Peterande D-FINE Repository](https://github.com/Peterande/D-FINE)  

### **Steps:**  
1. Navigate to the D-FINE repository directory:  
   ```bash
   cd D-FINE
   ```  

2. Define the model size you want to export (`n`, `s`, `m`, `l`, or `x`). For example:  
   ```bash
   export model=l
   ```  

3. Run the export script:  
   ```bash
   python tools/deployment/export_onnx.py --check -c configs/dfine/dfine_hgnetv2_${model}_coco.yml -r model.pth
   ```  

### **Notes:**  
- Ensure the batch size hardcoded in the `export_onnx.py` script is appropriate for your system's available RAM. If not, modify the batch size in the script to avoid out-of-memory errors.  
- Verify that `model.pth` corresponds to the correct pre-trained model checkpoint for the configuration file you're using.  
- The `--check` flag ensures that the exported ONNX model is validated after the export process.  

### **Example:**  
To export the large model (`l`) with the corresponding configuration:  
```bash
cd D-FINE
export model=l
python tools/deployment/export_onnx.py --check -c configs/dfine/dfine_hgnetv2_l_coco.yml -r model.pth
```  
