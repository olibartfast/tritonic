![Alt text](data/tritonic.jpeg)

# TritonIC - C++ Triton Inference Client for Computer Vision Models

[![CI](https://github.com/olibartfast/tritonic/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/olibartfast/tritonic/actions/workflows/ci.yml)

This C++ application enables machine learning tasks (e.g. object detection, classification, optical flow ...) using the Nvidia Triton Server. Triton manages multiple framework backends for streamlined model deployment.

> 🚧 Status: Under Development — expect frequent updates.

## Table of Contents
- [Project Structure](#project-structure)
- [Tested Models](#tested-models)
- [Build Client Libraries](#build-client-libraries)
- [Dependencies](#dependencies)
- [Build and Compile](#build-and-compile)
- [Tasks](#tasks)
- [Notes](#notes)
- [Deploying Models](#deploying-models)
- [Running Inference](#running-inference)
- [Docker Support](#docker-support)
- [Demo](#demo)
- [References](#references)
- [Feedback](#feedback)

## Project Structure

```
tritonic/
├── src/                          # Source code (main app, triton client, tasks, utils)
├── include/                      # Header files
├── deploy/                       # Model export scripts (per task type)
├── scripts/                      # Docker, setup, and utility scripts
├── config/                       # Configuration files
├── docs/                         # Documentation and guides
├── labels/                       # Label files (COCO, ImageNet, etc.)
├── data/                         # Data files (images, videos, outputs)
└── tests/                        # Unit and integration tests
```

**CMake Fetched Dependencies:**
- [vision-core](https://github.com/olibartfast/vision-core) - Model pre/post processing
- [fmt](https://github.com/fmtlib/fmt) - Formatting library

## Tested Models

## Object Detection

- [YOLOv5](https://github.com/ultralytics/yolov5)
- [YOLOv6](https://github.com/meituan/YOLOv6)
- [YOLOv7](https://github.com/WongKinYiu/yolov7)
- [YOLOv8](https://github.com/ultralytics/ultralytics)
- [YOLOv9](https://github.com/WongKinYiu/yolov9)
- [YOLOv10](https://github.com/THU-MIG/yolov10)
- [YOLO11](https://github.com/ultralytics/ultralytics)
- [YOLOv12](https://github.com/sunsmarterjie/yolov12)
- [YOLO-NAS](https://github.com/Deci-AI/super-gradients)
- [RT-DETR](https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetr_pytorch)
- [RT-DETRv2](https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch)
- [RT-DETRv4](https://github.com/RT-DETRs/RT-DETRv4)
- [D-FINE](https://github.com/Peterande/D-FINE)
- [DEIM](https://github.com/ShihuaHuang95/DEIM)
- [DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2)
- [RF-DETR](https://github.com/roboflow/rf-detr)

## Instance Segmentation

- [YOLOv5](https://github.com/ultralytics/yolov5)
- [YOLOv8](https://github.com/ultralytics/ultralytics)
- [YOLO11](https://github.com/ultralytics/ultralytics)
- [YOLOv12](https://github.com/sunsmarterjie/yolov12)
- [RF-DETR](https://github.com/roboflow/rf-detr)

## Classification

- [Torchvision Models](https://pytorch.org/vision/stable/models.html)
- [TensorFlow-Keras Models](https://www.tensorflow.org/api_docs/python/tf/keras/applications)
- [Hugging Face Vision Transformers (ViT)](https://huggingface.co/docs/transformers/model_doc/vit)

## Optical Flow

- [RAFT](https://pytorch.org/vision/stable/models/raft.html)


## Build Client Libraries

To build the client libraries, refer to the official [Triton Inference Server client libraries](https://github.com/triton-inference-server/client/tree/r25.06).

### Alternative: Extract Client Libraries from Docker

For convenience, you can extract the pre-built Triton client libraries from the official NVIDIA Triton Server SDK image using [Docker](docs/guides/Docker_setup.md):

```bash
# Run the extraction script
./scripts/docker/extract_triton_libs.sh
```

This script will:
1. Create a temporary Docker container from the `nvcr.io/nvidia/tritonserver:25.06-py3-sdk` image
2. Extract the Triton client libraries from `/workspace/install`
3. Copy additional Triton server headers and libraries if available
4. Save everything to `./triton_client_libs/` directory

After extraction, set the environment variable:
```bash
export TritonClientBuild_DIR=$(pwd)/triton_client_libs/install
```

The extracted directory structure will contain:
- `install/` - Triton client build artifacts
- `triton_server_include/` - Triton server headers  
- `triton_server_lib/` - Triton server libraries
- `workspace/` - Additional workspace files

## Dependencies

Ensure the following dependencies are installed:

1. **Nvidia Triton Inference Server**:
```bash
docker pull nvcr.io/nvidia/tritonserver:25.06-py3
```

2. **Triton client libraries**: Tested on Release r25.06
3. **Protobuf and gRPC++**: Versions compatible with Triton
4. **RapidJSON**:
```bash
apt install rapidjson-dev
```

5. **libcurl**:
```bash
apt install libcurl4-openssl-dev
```

6. **OpenCV 4**: Tested version: 4.7.0
```bash
apt install libopencv-dev
```

## Development Setup

### Pre-commit Hooks (Recommended)

To maintain code quality and consistency, install pre-commit hooks:

```bash
# Run the setup script
./scripts/setup/pre_commit_setup.sh

# Or install manually
pip install pre-commit
pre-commit install
```

### Build and Compile

1. Set the environment variable `TritonClientBuild_DIR` or update the `CMakeLists.txt` with the path to your installed Triton client libraries.

2. Create a build directory:
```bash
mkdir build
```

3. Navigate to the build directory:
```bash
cd build
```

4. Run CMake to configure the build:
```bash
cmake -DCMAKE_BUILD_TYPE=Release ..
```

Optional flags:
- `-DSHOW_FRAME`: Enable to display processed frames after inference
- `-DWRITE_FRAME`: Enable to write processed frames to disk

5. Build the application:
```bash
cmake --build .
```

## Tasks

### Export Instructions
- [Object Detection](docs/guides/ObjectDetection.md)
- [Classification](docs/guides/Classification.md)
- [ViT Classification](docs/guides/ViTClassification.md)
- [Instance Segmentation](docs/guides/InstanceSegmentation.md)
- [Optical Flow](docs/guides/OpticalFlow.md)


*Other tasks are in TODO list.*

## Model Pre/Post Processing

All model-specific preprocessing and postprocessing logic is handled by the [vision-core](https://github.com/olibartfast/vision-core) library, which is automatically fetched as a CMake dependency via `FetchContent`. This modular approach keeps the Triton client code clean and allows the vision processing logic to be reused across different inference backends.

## Notes

Ensure the model export versions match those supported by your Triton release. Check Triton releases [here](https://github.com/triton-inference-server/server/releases).

## Deploying Models

To deploy models, set up a model repository following the [Triton Model Repository schema](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_repository.md). The `config.pbtxt` file is optional unless you're using the OpenVino backend, implementing an Ensemble pipeline, or passing custom inference parameters.

### Model Repository Structure
```
<model_repository>/
    <model_name>/
        config.pbtxt
        <model_version>/
            <model_binary>
```

### Starting Triton Server

Use the provided script for easy setup:
```bash
# Start Triton server with GPU support
./scripts/docker/docker_triton_run.sh /path/to/model_repository 25.06 gpu

# Start with CPU only
./scripts/docker/docker_triton_run.sh /path/to/model_repository 25.06 cpu
```

Or manually with Docker:
```bash
docker run --gpus=1 --rm \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v /full/path/to/model_repository:/models \
  nvcr.io/nvidia/tritonserver:<xx.yy>-py3 tritonserver \
  --model-repository=/models
```

*Omit the `--gpus` flag if using the CPU version.*

## Running Inference

### Command-Line Inference on Video or Image
```bash
./tritonic \
    --source=/path/to/source.format \
    --model_type=<model_type> \
    --model=<model_name_folder_on_triton> \
    --labelsFile=/path/to/labels/coco.names \
    --protocol=<http or grpc> \
    --serverAddress=<triton-ip> \
    --port=<8000 for http, 8001 for grpc> \
```

For dynamic input sizes:
```bash
    --input_sizes="c,h,w"
```

### Shared Memory Support

Tritonic supports shared memory to improve inference performance by reducing data copying between the client and Triton server. Two types of shared memory are available:

#### System (POSIX) Shared Memory
Uses CPU-based shared memory for efficient data transfer:
```bash
./tritonic \
    --source=/path/to/source.format \
    --model=<model_name> \
    --shared_memory_type=system \
    ...
```

#### CUDA Shared Memory
Uses GPU memory directly for zero-copy inference (requires GPU support):
```bash
./tritonic \
    --source=/path/to/source.format \
    --model=<model_name> \
    --shared_memory_type=cuda \
    --cuda_device_id=0 \
    ...
```

**Configuration Options:**
- `--shared_memory_type` or `-smt`: Shared memory type (`none`, `system`, or `cuda`). Default: `none`
- `--cuda_device_id` or `-cdi`: CUDA device ID when using CUDA shared memory. Default: `0`

### Quick Start with Docker Scripts

Use the provided Docker scripts for quick testing:

```bash
# Run object detection
./scripts/docker/run_client.sh

# Run with debug mode
./scripts/docker/run_debug.sh

# Run optical flow
./scripts/docker/run_optical_flow.sh

# Run unit tests
./scripts/docker/run_tests.sh
```

#### Debugging Tips
Check [`.vscode/launch.json`](.vscode/launch.json) for additional configuration examples

#### Placeholder Descriptions
- **`/path/to/source.format`**: Path to the input video or image file, for optical flow you must pass two images as comma separated list
- **`<model_type>`**: Model type (e.g., `yolov5`, `yolov8`, `yolo11`, `yoloseg`, `torchvision-classifier`, `tensorflow-classifier`, `vit-classifier`, check below [Model Type Parameters](#model-type-tag-parameters))
- **`<model_name_folder_on_triton>`**: Name of the model folder on the Triton server
- **`/path/to/labels/coco.names`**: Path to the label file (e.g., COCO labels)
- **`<http or grpc>`**: Communication protocol (`http` or `grpc`)
- **`<triton-ip>`**: IP address of your Triton server
- **`<8000 for http, 8001 for grpc>`**: Port number
- **`<batch or b >`**: Batch size, currently only 1 is supported
- **`<input_sizes or -is>`**: Input sizes input for dynamic axes. Semi-colon separated list format: CHW;CHW;... (e.g., '3,224,224' for single input or '3,224,224;3,224,224' for two inputs, '3,640,640;2' for rtdetr/dfine models)


To view all available parameters, run:
```bash
./tritonic --help
```

#### Model Type Tag Parameters
| Model                  | Model Type Parameter   | Notes |
|------------------------|------------------------|-------|
| YOLOv5 / v6 / v7 / v8 / v9 / v11 / v12 | `yolo` | Any `yolo*` variant works. Standard format |
| YOLOv7 End-to-End      | `yolov7e2e`            | Only for YOLOv7 exported with `--grid --end2end` flags (requires TensorRT backend) |
| YOLOv10                | `yolov10`              | Specific output format |
| YOLO-NAS               | `yolonas`              | Specific output format |
| RT-DETR / RT-DETRv2 / RT-DETRv4 / D-FINE / DEIM / DEIMv2 | `rtdetr` | All RT-DETR style models share the same postprocessor |
| RT-DETR Ultralytics    | `rtdetrul`             |       |
| RF-DETR (Detection & Segmentation) | `rfdetr`  |       |
| YOLOv5/v8/v11/v12 Segmentation | `yoloseg`       |       |
| Torchvision Classifier | `torchvision-classifier` |     |
| Tensorflow Classifier  | `tensorflow-classifier` |      |
| ViT Classifier         | `vit-classifier`       |       |
| RAFT Optical Flow      | `raft`                 |       |


## Docker Support  
For detailed instructions on installing Docker and the NVIDIA Container Toolkit, refer to the [Docker Setup Document](docs/guides/Docker_setup.md).  

### Build

```bash
docker build --rm -t tritonic .
```

### Run Container
```bash
docker run --rm \
  -v /path/to/host/data:/app/data \
  tritonic \
  --network host \
  --source=<path_to_source_on_container> \
  --model_type=<model_type> \
  --model=<model_name_folder_on_triton> \
  --labelsFile=<path_to_labels_on_container> \
  --protocol=<http or grpc> \
  --serverAddress=<triton-ip> \
  --port=<8000 for http, 8001 for grpc>
```

## Demo

Real-time inference test (GPU RTX 3060):
- YOLOv7-tiny exported to ONNX: [Demo Video](https://youtu.be/lke5TcbP2a0)
- YOLO11s exported to onnx: [Demo Video](https://youtu.be/whP-FF__4IM)
- RAFT Optical Flow Large(exported to traced torchscript): [Demo Video](https://www.youtube.com/watch?v=UvKCjYI_9aQ)

## References
- [Triton Inference Server Client Example](https://github.com/triton-inference-server/client/blob/r21.08/src/c%2B%2B/examples/image_client.cc)
- [Triton User Guide](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/getting_started/quickstart.html)
- [Triton Tutorials](https://github.com/triton-inference-server/tutorials)
- [ONNX Models](https://onnx.ai/models/)
- [Torchvision Models](https://pytorch.org/vision/stable/models.html)
- [Tensorflow Model Garden](https://github.com/tensorflow/models/tree/master/official)

## Feedback
Any feedback is greatly appreciated. If you have any suggestions, bug reports, or questions, don't hesitate to open an [issue](https://github.com/olibartfast/tritonic/issues). Contributions, corrections, and suggestions are welcome to keep this repository relevant and useful.
