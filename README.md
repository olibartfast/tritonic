![Alt text](data/tritonic.jpeg)

# TritonIC - C++ Triton Inference Client for Computer Vision Models

This C++ application enables machine learning tasks (e.g. object detection, classification, optical flow ...) using the Nvidia Triton Server. Triton manages multiple framework backends for streamlined model deployment.

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
├── src/                          # Source code
│   ├── main/                     # Main application
│   │   └── client.cpp           # Main entry point
│   ├── triton/                   # Triton client code
│   ├── tasks/                    # Task implementations
│   └── utils/                    # Utility classes
├── include/                      # Header files
├── deploy/                       # Model deployment scripts
│   ├── classifier/               # Classification models
│   │   ├── tensorflow/           # TensorFlow deployments
│   │   ├── torchvision/          # Torchvision deployments
│   │   └── vit/                  # Vision Transformer deployments
│   ├── instance_segmentation/    # Instance segmentation models
│   │   └── rf-detr/              # RF-DETR segmentation deployments
│   │       ├── export.py         # ONNX export script
│   │       ├── export_trt.sh     # TensorRT engine export
│   │       └── deploy_triton_model.sh # Deploy to Triton repository
│   ├── object_detection/         # Object detection models
│   │   └── rf-detr/              # RF-DETR detection deployments
│   ├── optical_flow/             # Optical flow models
│   └── video_classification/     # Video classification models (TODO)
├── scripts/                      # All scripts
│   ├── docker/                   # Docker-related scripts
│   │   ├── docker_triton_run.sh  # Run Triton server
│   │   ├── extract_triton_libs.sh# Extract client libraries
│   │   ├── run_client.sh         # Run client application
│   │   ├── run_debug.sh          # Run with debug mode
│   │   ├── run_optical_flow.sh   # Run optical flow
│   │   └── run_tests.sh          # Run unit tests
│   ├── setup/                    # Setup scripts
│   └── tools/                    # Utility scripts
├── config/                       # Configuration files
│   └── environments/             # Environment configs
├── docs/                         # Documentation
│   └── guides/                   # User guides
├── labels/                       # Label files
│   ├── coco.txt                  # COCO class labels
│   └── imagenet.txt              # ImageNet class labels
├── data/                         # Data files
│   ├── images/                   # Test images
│   ├── videos/                   # Test videos
│   └── models/                   # Model files
└── tests/                        # Test files
    ├── mocks/                    # Mock objects
    ├── unit/                     # Unit tests
    └── integration/              # Integration tests
```

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

For convenience, you can extract pre-built Triton client libraries from the official NVIDIA Triton Server SDK Docker image:

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

### Quick Deploy: RF-DETR Segmentation

For RF-DETR segmentation models, use the automated deployment scripts:

```bash
# 1. Export ONNX model
cd deploy/instance_segmentation/rf-detr
python export.py --output_dir output --simplify

# 2. Convert to TensorRT engine (compatible with Triton 25.06)
bash export_trt.sh

# 3. Deploy to Triton model repository
bash deploy_triton_model.sh exports/model.engine rf-detr-seg-preview_trt /path/to/model_repository
```

The deployment script will:
- Create proper Triton model repository structure
- Copy the TensorRT engine as `model.plan`
- Generate appropriate `config.pbtxt` with input/output specifications
- Handle absolute path requirements for Docker deployments

*Other tasks are in TODO list.*

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
| Model                  | Model Type Parameter   |
|------------------------|------------------------|
| YOLOv5                 | yolov5                 |
| YOLOv6                 | yolov6                 |
| YOLOv7                 | yolov7                 |
| YOLOv8                 | yolov8                 |
| YOLOv9                 | yolov9                 |
| YOLOv10                | yolov10                |
| YOLO11                 | yolo11                 |
| YOLOv12                | yolov12                |
| RT-DETR                | rtdetr                 |
| RT-DETRV2              | rtdetrv2               |
| RT-DETR Ultralytics    | rtdetrul               |
| RF-DETR                | rfdetr                 |
| RF-DETR Segmentation   | rfdetr                 |
| D-FINE                 | dfine                  |
| DEIM                   | deim                   |
| DEIMv2                 | deim                   |
| Torchvision Classifier | torchvision-classifier |
| Tensorflow Classifier  | tensorflow-classifier  |
| ViT Classifier         | vit-classifier         |
| YOLOv5 Segmentation    | yoloseg                |
| YOLOv8 Segmentation    | yoloseg                |
| YOLO11 Segmentation    | yoloseg                |
| YOLO12 Segmentation    | yoloseg                |
| RF-DETR Segmentation   | rfdetr                 |
| RAFT Optical Flow      | raft                   |


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
