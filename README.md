# C++ Triton Client for Computer Vision Models

This C++ application enables machine learning tasks, such as object detection and classification, using the Nvidia Triton Server. Triton manages multiple framework backends for streamlined model deployment.

### Supported Models

- **Object Detection**: YOLOv5, YOLOv6, YOLOv7, YOLOv8, YOLOv9, YOLOv10, YOLO11, YOLO-NAS
- **Instance Segmentation**: YOLOv5, YOLOv8, YOLO11
- **Classification**: Torchvision API-based models, Tensorflow-Keras API(saved_model export)

---

## Build Client Libraries

To build the client libraries, refer to the official [Triton Inference Server client libraries](https://github.com/triton-inference-server/client/tree/r24.09).

---

## Dependencies

Ensure the following dependencies are installed:

- **Nvidia Triton Inference Server** (from NGC):  
  ```bash
  docker pull nvcr.io/nvidia/tritonserver:24.09-py3
  ```
- **Triton client libraries**: Tested on Release r24.09
- **Protobuf and gRPC++**: Versions compatible with Triton
- **RapidJSON**:  
  ```bash
  apt install rapidjson-dev
  ```
- **libcurl**:  
  ```bash
  apt install libcurl4-openssl-dev
  ```
- **OpenCV 4**: Tested version: 4.7.0

---

## Build and Compile

To build and compile the application, follow these steps:

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
   - `-DSHOW_FRAME`: Enable to display processed frames after inference.
   - `-DWRITE_FRAME`: Enable to write processed frames to disk.

5. Build the application:
   ```bash
   cmake --build .
   ```

---

## Machine Learning Tasks

### Export Instructions

- [Object Detection](docs/ObjectDetection.md)
- [Classification](docs/Classification.md)
- [Instance Segmentation](docs/InstanceSegmentation.md)
  
  _Other tasks like Pose Estimation, Optical Flow, LLM are in TODO list._

---

## Notes

Ensure the model export versions match those supported by your Triton release. Check Triton releases [here](https://github.com/triton-inference-server/server/releases).

---

## Deploying Models to Triton

To deploy models, set up a model repository following the [Triton Model Repository schema](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_repository.md). The `config.pbtxt` file is optional unless you're using the OpenVino backend, implementing an Ensemble pipeline, or passing custom inference parameters.

### Example Model Repository Structure

```
<model_repository>/
    <model_name>/
        config.pbtxt
        <model_version>/
            <model_binary>
```

To start Triton Server, run:

```bash
#!/bin/bash
docker run --gpus=1 --rm \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v /full/path/to/model_repository:/models \
  nvcr.io/nvidia/tritonserver:<xx.yy>-py3 tritonserver \
  --model-repository=/models
```

_Omit the `--gpus` flag if using the CPU version._

For more examples, check the [Triton Inference Server tutorials](https://github.com/triton-inference-server/tutorials).

---

## Running Inference

### Command-Line Inference on Video or Image

Use the following command to perform inference on a video or image:

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

_If the model has dynamic input sizes, use:_

```bash
    --input_sizes="c w h" 
```

#### Placeholder Descriptions:

- **`/path/to/source.format`**: Path to the input video or image file.
- **`<task_type>`**: Type of computer vision task (`detection`, `classification`, or `instance_segmentation`).
- **`<model_type>`**: Model type (e.g., `yolov5`, `yolov8`, `yolo11`, `yoloseg`, `torchvision-classifier`, `tensorflow-classifier`).
- **`<model_name_folder_on_triton>`**: Name of the model folder on the Triton server.
- **`/path/to/labels/coco.names`**: Path to the label file (e.g., COCO labels).
- **`<http or grpc>`**: Communication protocol (`http` or `grpc`).
- **`<triton-ip>`**: IP address of your Triton server.
- **`<8000 for http, 8001 for grpc>`**: Port number.

To view all available parameters, run:

```bash
./computer-vision-triton-cpp-client --help
```

---

## Running with Docker

### Build the Docker Image

```bash
docker build --rm -t computer-vision-triton-cpp-client .
```

### Run the Docker Container

```bash
docker run --rm \
  -v /path/to/host/data:/app/data \
  computer-vision-triton-cpp-client \
  --network host \
  --source=/app/data/source.format \
  --task_type=<task_type> \
  --model_type=<model_type> \
  --model=<model_name_folder_on_triton> \
  --labelsFile=/app/coco.names \
  --protocol=<http or grpc> \
  --serverAddress=<triton-ip> \
  --port=<8000 for http, 8001 for grpc>
```

- **`-v /path/to/host/data:/app/data`**: Maps host data to `/app/data` in the container for input/output.

### View Output

Processed output is saved to the mapped directory on the host.

---

## Demo

Watch a real-time inference test using YOLOv7-tiny exported to ONNX: [YOLOv7-tiny Inference Test](https://youtu.be/lke5TcbP2a0).

---

## References

- [Triton Inference Server Client Example](https://github.com/triton-inference-server/client/blob/r21.08/src/c%2B%2B/examples/image_client.cc)
- [Triton User Guide](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/getting_started/quickstart.html)
- [Triton Tutorials](https://github.com/triton-inference-server/tutorials)
- [ONNX Models](https://onnx.ai/models/)
- [Torchvision Models](https://pytorch.org/vision/stable/models.html)
- [Tensorflow Model Garden](https://github.com/tensorflow/models/tree/master/official)

---

## Feedback
- Any feedback is greatly appreciated, if you have any suggestions, bug reports or questions don't hesitate to open an [issue](https://github.com/olibartfast/computer-vision-triton-cpp-client/issues).
