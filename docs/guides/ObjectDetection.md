# Object Detection Model Export Guide for Triton Server Deployment

## YOLOv8/YOLO11/YOLOv12
Install using [Ultralytics official documentation (pip ultralytics package version >= 8.3.0)](https://docs.ultralytics.com/quickstart/)

### OnnxRuntime/TorchScript
```bash
yolo export model=best.pt format=onnx   # for ONNX format
# OR
yolo export model=best.pt format=torchscript   # for TorchScript format
```

### TensorRT
```bash
yolo export model=best.pt format=engine
```
**Note**: Ensure TensorRT version in your Python environment matches the C++ version for inference. Alternatively:
```bash
trtexec --onnx=best.onnx --saveEngine=best.engine
```

## YOLOv10
From [yolov10 repo](https://github.com/THU-MIG/yolov10) or [ultralytics package](https://pypi.org/project/ultralytics/):

### OnnxRuntime/TorchScript
```bash
yolo export format=onnx model=yolov10model.pt   # for ONNX format
# OR
yolo export format=torchscript model=yolov10model.pt   # for TorchScript format
```

### TensorRT
```bash
trtexec --onnx=yolov10model.onnx --saveEngine=yolov10model.engine --fp16
```

## YOLOv9
From [yolov9 repo](https://github.com/WongKinYiu/yolov9):

### OnnxRuntime/TorchScript
```bash
python export.py --weights yolov9-c/e-converted.pt --include onnx   # for ONNX format
# OR
python export.py --weights yolov9-c/e-converted.pt --include torchscript   # for TorchScript format
```

### TensorRT
```bash
trtexec --onnx=yolov9-c/e-converted.onnx --saveEngine=yolov9-c/e.engine --fp16
```

## YOLOv5
### OnnxRuntime
From [yolov5 repo](https://github.com/ultralytics/yolov5/issues/251):
```bash
python export.py --weights <yolov5_version>.pt --include onnx
```

### Libtorch
```bash
python export.py --weights <yolov5_version>.pt --include torchscript
```

## YOLOv6
### OnnxRuntime
Export weights to ONNX format or download from [yolov6 repo](https://github.com/meituan/YOLOv6/tree/main/deploy/ONNX). Postprocessing code is identical to YOLOv5-v7.

## YOLOv7
### OnnxRuntime/Libtorch
From [yolov7 repo](https://github.com/WongKinYiu/yolov7#export):
```bash
python export.py --weights <yolov7_version>.pt --grid --simplify --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640 --max-wh 640
```
**Note**: Don't use the end-to-end parameter.

## YOLO-NAS
### OnnxRuntime
Follow [YoloNAS Quickstart](https://github.com/Deci-AI/super-gradients/blob/master/documentation/source/YoloNASQuickstart.md#export-to-onnx). Example for yolo_nas_s:

```python
from super_gradients.training import models

net = models.get("yolo_nas_s", pretrained_weights="coco")
models.convert_to_onnx(model=net, input_shape=(3,640,640), out_path="yolo_nas_s.onnx", 
                      torch_onnx_export_kwargs={"input_names": ['input'], 
                                               "output_names": ['output0', 'output1']})
```

## RT-DETR/RT-DETRv2
From [lyuwenyu RT-DETR repository](https://github.com/lyuwenyu/RT-DETR/):

### OnnxRuntime
```bash
export RTDETR_VERSION=rtdetr  # or rtdetrv2
export MODEL_VERSION=rtdetr_r18vd_6x_coco  # or select other model from RT-DETR/RT-DETRV2 model zoo
cd RT-DETR/${RTDETR_VERSION}_pytorch
python tools/export_onnx.py -c configs/${RTDETR_VERSION}/${MODEL_VERSION}.yml -r path/to/checkpoint --check
```

### TensorRT
```bash
trtexec --onnx=<model>.onnx --saveEngine=<model>.engine --minShapes=images:1x3x640x640,orig_target_sizes:1x2 --optShapes=images:1x3x640x640,orig_target_sizes:1x2 --maxShapes=images:1x3x640x640,orig_target_sizes:1x2
```

## RT-DETR (Ultralytics)
Use the [Ultralytics pip package](https://docs.ultralytics.com/quickstart/):

### OnnxRuntime
```bash
yolo export model=best.pt format=onnx
```
**Note**: `best.pt` should be a trained RTDETR-L or RTDETR-X model.

### Libtorch
```bash
yolo export model=best.pt format=torchscript
```

### TensorRT
```bash
trtexec --onnx=yourmodel.onnx --saveEngine=yourmodel.engine
```
OR
```bash
yolo export model=yourmodel.pt format=engine
```

For more information: https://docs.ultralytics.com/models/rtdetr/

## D-FINE
From [Peterande D-FINE Repository](https://github.com/Peterande/D-FINE):

### OnnxRuntime
```bash
cd D-FINE
export model=l  # Choose from n, s, m, l, or x
python tools/deployment/export_onnx.py --check -c configs/dfine/dfine_hgnetv2_${model}_coco.yml -r model.pth
```

**Notes**:
- Ensure the batch size in `export_onnx.py` is appropriate for your system's RAM
- Verify `model.pth` corresponds to the correct pre-trained model for your config
- The `--check` flag validates the exported ONNX model

## DEIM
From [ShihuaHuang95 DEIM Repository](https://github.com/ShihuaHuang95/DEIM):

### OnnxRuntime
```bash
cd DEIM
python tools/deployment/export_onnx.py --check -c configs/deim_dfine/deim_hgnetv2_s_coco.yml -r deim_dfine_hgnetv2_s_coco_120e.pth
```
**Notes**:
- Same considerations as D-FINE regarding batch size and model verification

## DEIMv2
From [Intellidust AI Lab DEIMv2 Repository](https://github.com/Intellindust-AI-Lab/DEIMv2):

### OnnxRuntime
```bash
cd DEIMv2
download vitt_distill.pt to ckpts folder
export MODEL=deimv2_dinov3_s_coco, or check from modelzoo for others
python tools/deployment/export_onnx.py --check -c configs/deimv2/$MODEL.yml -r /path/to/$MODEL.pth
```
**Notes**:
- Same considerations as D-FINE regarding batch size and model verification



### TensorRT for D-FINE DEIM and DEIMv2
* Same as for lyuwenyu RT-DETR models


## RF-DETR

Follow the procedure described in the [RF-DETR documentation](https://github.com/roboflow/rf-detr?tab=readme-ov-file#onnx-export).

### ONNX Export for ONNX Runtime

#### Option 1 — Using the Python Script

Run the export script with your desired parameters:

```bash
python deploy/object_detection/rf-detr/export.py <input_params>  # use --help for details
```

#### Option 2 — Using Python API

You can also export the model directly from Python:

```python
from rfdetr import RFDETRBase

model = RFDETRBase(pretrain_weights=<CHECKPOINT_PATH>)
model.export()
```

The ONNX model will be saved automatically in the `output` directory.


### TensorRT Export
```bash
trtexec --onnx=/path/to/model.onnx --saveEngine=/path/to/model.engine --memPoolSize=workspace:4096 --fp16 --useCudaGraph --useSpinWait --warmUp=500 --avgRuns=1000 --duration=10
```

* using a tensorrt docker container:
```bash
export NGC_TAG_VERSION=24.12
docker run --rm -it --gpus=all -v $(pwd)/exports:/exports --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v $(pwd)/model.onnx:/workspace/model.onnx -w /workspace nvcr.io/nvidia/tensorrt$NGC_TAG_VERSION:-py3 /bin/bash -cx "trtexec --onnx=model.onnx --saveEngine=/exports/model.engine --memPoolSize=workspace:4096 --fp16 --useCudaGraph --useSpinWait --warmUp=500 --avgRuns=1000 --duration=10"
```
 