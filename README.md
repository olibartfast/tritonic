# C++ Triton Client for Computer Vision Models

This C++ application enables machine learning tasks, such as object detection and classification, using the Nvidia Triton Server. Triton manages multiple framework backends for streamlined model deployment.

## Table of Contents
- [Supported Models](#supported-models)
- [Dependencies](#dependencies)
- [Building](#building)
- [Tasks](#tasks)
- [Model Deployment](#model-deployment)
- [Running Inference](#running-inference)
- [Docker Support](#docker-support)
- [Demo](#demo)
- [References](#references)
- [Feedback](#feedback)

## Supported Models

### Object Detection
- YOLOv5
- YOLOv6
- YOLOv7
- YOLOv8
- YOLOv9
- YOLOv10
- YOLO11
- YOLO-NAS
- RT-DETR
- D-FINE

### Instance Segmentation
- YOLOv5
- YOLOv8
- YOLO11

### Classification
- Torchvision API-based models
- Tensorflow-Keras API (saved_model export)

## Dependencies

Ensure you have the following components installed:

1. **Nvidia Triton Inference Server**:
```bash
docker pull nvcr.io/nvidia/tritonserver:24.09-py3
```

2. **Triton client libraries**: See [Building Triton Client Libraries](triton-client-build.md)
   - Tested on Release r24.09

3. **System Libraries**:
```bash
# Install RapidJSON
apt install rapidjson-dev

# Install libcurl
apt install libcurl4-openssl-dev
```

4. **OpenCV 4**: Tested version: 4.7.0

## Building

1. Set the environment variable `TritonClientBuild_DIR` or update `CMakeLists.txt` with the path to your installed Triton client libraries.

2. Build the application:
```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
```

Optional build flags:
- `-DSHOW_FRAME`: Enable to display processed frames after inference
- `-DWRITE_FRAME`: Enable to write processed frames to disk

## Tasks

### Export Instructions
- [Object Detection](docs/ObjectDetection.md)
- [Classification](docs/Classification.md)
- [Instance Segmentation](docs/InstanceSegmentation.md)

*Other tasks like Pose Estimation, Optical Flow, LLM are in TODO list.*

## Model Deployment

### Model Repository Structure
```
model_repository/
├── model_name/
│   ├── config.pbtxt        # Optional unless using OpenVino/Ensemble
│   └── 1/
│       └── model.binary
```

### Starting Triton Server
```bash
docker run --gpus=1 --rm \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v /full/path/to/model_repository:/models \
  nvcr.io/nvidia/tritonserver:24.09-py3 tritonserver \
  --model-repository=/models
```

*Omit `--gpus` flag if using CPU version*

## Running Inference

### Command-Line Usage
```bash
./computer-vision-triton-cpp-client \
    --source=/path/to/source.format \
    --task_type=<task_type> \
    --model_type=<model_type> \
    --model=<model_name_folder_on_triton> \
    --labelsFile=/path/to/labels/coco.names \
    --protocol=<http or grpc> \
    --serverAddress=<triton-ip> \
    --port=<8000 for http, 8001 for grpc>
```

For dynamic input sizes:
```bash
    --input_sizes="c w h"
```

### Model Type Parameters
| Model | Parameter |
|-------|-----------|
| YOLOv5 | yolov5 |
| YOLOv6 | yolov6 |
| YOLOv7 | yolov7 |
| YOLOv8 | yolov8 |
| YOLOv9 | yolov9 |
| YOLOv10 | yolov10 |
| YOLO11 | yolo11 |
| RT-DETR | rtdetr |
| D-FINE | dfine |
| Torchvision Classifier | torchvision-classifier |
| Tensorflow Classifier | tensorflow-classifier |
| YOLOv5/v8/11 Segmentation | yoloseg |

## Docker Support

### Build Image
```bash
docker build --rm -t computer-vision-triton-cpp-client .
```

### Run Container
```bash
docker run --rm \
  -v /path/to/host/data:/app/data \
  computer-vision-triton-cpp-client \
  --network host \
  --source=<path_to_source_on_container> \
  --task_type=<task_type> \
  --model_type=<model_type> \
  --model=<model_name_folder_on_triton> \
  --labelsFile=<path_to_labels_on_container> \
  --protocol=<http or grpc> \
  --serverAddress=<triton-ip> \
  --port=<8000 for http, 8001 for grpc>
```

## Demo

Real-time inference test (GPU RTX 3060):
- YOLOv7-tiny (ONNX): [Demo Video](https://youtu.be/lke5TcbP2a0)
- YOLO11s (ONNX): [Demo Video](https://youtu.be/whP-FF__4IM)

## References
- [Triton Client Example](https://github.com/triton-inference-server/client/blob/r21.08/src/c%2B%2B/examples/image_client.cc)
- [Triton User Guide](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/getting_started/quickstart.html)
- [Triton Tutorials](https://github.com/triton-inference-server/tutorials)
- [ONNX Models](https://onnx.ai/models/)
- [Torchvision Models](https://pytorch.org/vision/stable/models.html)
- [Tensorflow Model Garden](https://github.com/tensorflow/models/tree/master/official)

## Feedback
For bug reports, suggestions, or questions, please open an [issue](https://github.com/olibartfast/computer-vision-triton-cpp-client/issues).