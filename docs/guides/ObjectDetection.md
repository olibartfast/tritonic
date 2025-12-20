# Object Detection Model Export Guide for Triton Server Deployment

## Quick Start with Universal YOLO Exporter

**⚠️ Export tools have moved to vision-core for reusability across inference engines.**

Use the universal YOLO exporter supporting **all YOLO versions** (v5-v12, NAS):

```bash
# Clone vision-core if not already available
git clone https://github.com/olibartfast/vision-core.git
cd vision-core/export/detection/yolo

# For ultralytics-based models (v8, v9, v10, v11, v12)
python3 export.py --model yolov8n.pt --version v8 --format onnx --download-weights

# For repository-based models (v5, v6, v7)
./clone_repo.sh --version v5 --output-dir ./repositories
python3 export.py --model yolov5s.pt --version v5 --repo-dir ./repositories/yolov5 --format onnx
```

See [vision-core export documentation](https://github.com/olibartfast/vision-core/tree/main/export) for detailed usage.

---
### Manual Export (Alternative)
## YOLOv8/YOLO11/YOLOv12
Install using [Ultralytics official documentation (pip ultralytics package version >= 8.3.0)](https://docs.ultralytics.com/quickstart/)

### OnnxRuntime/TorchScript
```bash
yolo export model=best.pt format=onnx   # for ONNX format
# OR
yolo export model=best.pt format=torchscript   # for TorchScript format
```

**Output Format**: `[1, 84, 8400]` where 84 = 4 bbox coords + 80 class scores (no objectness score)
**Model Type**: `yolo`

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

**Output Format**: `[1, 300, 6]` where 6 = [x1, y1, x2, y2, confidence, class_id] (end-to-end with NMS)
**Model Type**: `yolov10`

### TensorRT
```bash
trtexec --onnx=yolov10model.onnx --saveEngine=yolov10model.engine --fp16
```

## YOLOv9
From [yolov9 repo](https://github.com/WongKinYiu/yolov9) or ultralytics package:

### OnnxRuntime/TorchScript
```bash
# From ultralytics (recommended)
yolo export model=yolov9c.pt format=onnx

# OR from original repo
python export.py --weights yolov9-c/e-converted.pt --include onnx   # for ONNX format
# OR
python export.py --weights yolov9-c/e-converted.pt --include torchscript   # for TorchScript format
```

**Output Format**: `[1, 84, 8400]` where 84 = 4 bbox coords + 80 class scores (no objectness score)
**Model Type**: `yolo`

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

**Output Format**: `[1, 25200, 85]` where 85 = 4 bbox coords + 1 objectness + 80 class scores
**Model Type**: `yolo`
**Note**: YOLOv5/v6/v7 have objectness score at index 4, which is handled automatically by vision-core

### Libtorch
```bash
python export.py --weights <yolov5_version>.pt --include torchscript
```

## YOLOv6
### OnnxRuntime
Export weights to ONNX format or download from [yolov6 repo](https://github.com/meituan/YOLOv6/tree/main/deploy/ONNX).

**Output Format**: `[1, 25200, 85]` where 85 = 4 bbox coords + 1 objectness + 80 class scores
**Model Type**: `yolo`
**Note**: Postprocessing code is identical to YOLOv5/v7

## YOLOv7
### OnnxRuntime/Libtorch
From [yolov7 repo](https://github.com/WongKinYiu/yolov7#export):
```bash
python export.py --weights <yolov7_version>.pt --grid --simplify --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640 --max-wh 640
```

**Output Format**: `[1, 25200, 85]` where 85 = 4 bbox coords + 1 objectness + 80 class scores
**Model Type**: `yolo`

**Important Notes**:
- Use `--grid` to combine multi-scale outputs into single tensor
- Use `--simplify` to optimize ONNX graph
- **Do NOT use** `--end2end` flag (adds TensorRT NMS plugin, requires TensorRT backend)
- For TensorRT end-to-end format, add `--end2end` and use model type `yolov7e2e`

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

## RT-DETR Family
From [lyuwenyu RT-DETR repository](https://github.com/lyuwenyu/RT-DETR/):

### Universal Export Launcher Script (Recommended for All Versions)

**⚠️ Export tools have moved to vision-core for reusability across inference engines.**

Use the universal export launcher that supports **all RT-DETR versions** (v1, v2, v3, v4) as well as **D-FINE** and **DEIM** models:

```bash
# Clone vision-core if not already available
git clone https://github.com/olibartfast/vision-core.git
cd vision-core/export/detection/rtdetr

bash export.sh \
    --config <path-to-config.yml> \
    --resume <path-to-checkpoint.pth> \
    --repo-dir <path-to-repo> \
    --download-weights \
    --weights-dir ./weights \
    --format onnx
```

**Options:**
- `--config` or `-c`: Path to model configuration file
- `--resume` or `-r`: Path to checkpoint file (will be downloaded if `--download-weights` is used)
- `--repo-dir`: Path to RT-DETR repository (auto-detected based on config)
- `--download-weights`: Automatically download pretrained weights
- `--weights-dir`: Directory to store downloaded weights (default: ./weights)
- `--format`: Export format: `onnx`, `tensorrt`, or `both` (default: onnx)
- `--version`: Force specific version: `v1`, `v2`, `v3`, `v4`, `dfine`, `deim`
- `--clone-repo`: Clone appropriate repository if needed
- `--install-deps`: Install dependencies from requirements.txt
- `--skip-venv-check`: Skip virtual environment validation
- `--no-check`: Skip ONNX model validation
- `--no-simplify`: Skip ONNX model simplification
- `--model-info`: Display model FLOPs, MACs, and parameters
- `--benchmark`: Run performance benchmarks

**Examples:**

RT-DETRv4:
```bash
bash export.sh \
    --config 3rdparty/repositories/pytorch/RT-DETRv4/configs/rtv4/rtv4_hgnetv2_s_coco.yml \
    --resume weights/rtv4_hgnetv2_s_model.pth \
    --repo-dir 3rdparty/repositories/pytorch/RT-DETRv4 \
    --download-weights --format onnx
```

RT-DETRv2:
```bash
bash export.sh \
    --config 3rdparty/repositories/pytorch/RT-DETR/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml \
    --resume weights/rtdetrv2_r50vd_6x_coco.pth \
    --repo-dir 3rdparty/repositories/pytorch/RT-DETR/rtdetrv2_pytorch \
    --format onnx
```

**Notes:**
- The script automatically detects RT-DETR version from the config file
- **⚠️ Batch Size Warning**: The export script patches the hardcoded batch size 32 → 1 at runtime to prevent OOM errors. If you experience memory issues, ensure this patching works correctly or manually modify `export_onnx.py`
- For RT-DETRv3 (PaddlePaddle-based), the script handles the different export pipeline automatically

### Manual Export (Alternative)

#### RT-DETR/RT-DETRv2 (PyTorch)
```bash
export RTDETR_VERSION=rtdetr  # or rtdetrv2
export MODEL_VERSION=rtdetr_r18vd_6x_coco  # or select from model zoo
cd RT-DETR/${RTDETR_VERSION}_pytorch
python tools/export_onnx.py -c configs/${RTDETR_VERSION}/${MODEL_VERSION}.yml -r path/to/checkpoint --check
```

#### RT-DETRv4 (PyTorch)
```bash
cd 3rdparty/repositories/pytorch/RT-DETRv4
python tools/deployment/export_onnx.py \
    -c configs/rtv4/rtv4_hgnetv2_s_coco.yml \
    -r /path/to/checkpoint.pth \
    --check --simplify
```

**Available RT-DETRv4 Models:**
- `rtv4_hgnetv2_s_coco` - Small model
- `rtv4_hgnetv2_m_coco` - Medium model  
- `rtv4_hgnetv2_l_coco` - Large model
- `rtv4_hgnetv2_x_coco` - Extra large model

### TensorRT (All RT-DETR Variants)
```bash
trtexec --onnx=<model>.onnx --saveEngine=<model>.engine \
    --minShapes=images:1x3x640x640,orig_target_sizes:1x2 \
    --optShapes=images:1x3x640x640,orig_target_sizes:1x2 \
    --maxShapes=images:1x3x640x640,orig_target_sizes:1x2
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

**⚠️ Export tools have moved to vision-core.**

Run the export script with your desired parameters:

```bash
# Clone vision-core if not already available
git clone https://github.com/olibartfast/vision-core.git
cd vision-core/export/detection/rfdetr

python export.py <input_params>  # use --help for details
```

#### Option 2 — Using Python API

You can also export the model directly from Python:

```python
from rfdetr import RFDETRBase  # or RFDETRBase/Nano/Small/Medium/Large

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
 
