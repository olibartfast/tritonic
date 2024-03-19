# C++ Triton Client for Computer Vision Models

This C++ application allows you to perform computer vision tasks such as object detection or classification using Nvidia Triton Server to manage multiple framework backends. It currently supports object detection models like YOLOv5, YOLOv6, YOLOv7, YOLOv8, YOLOv9, and YOLO-NAS, inference for classification models from the Torchvision API.

## Build Client Libraries

To build the client libraries, please refer to the Triton Inference Server client libraries located [here](https://github.com/triton-inference-server/client/tree/r23.08).

## Dependencies

Ensure that you have the following dependencies installed:

- Nvidia Triton Inference Server container pulled from NGC (`docker pull nvcr.io/nvidia/tritonserver:23.08-py3`).
- Triton client libraries (Tested Release 23.08).
- Protobuf and gRPC++ (versions compatible with Triton Server).
- RapidJSON (`apt install rapidjson-dev`).
- libcurl (`apt install libcurl4-openssl-dev`).
- OpenCV 4 (Tested version: 4.7.0).

## Build and Compile

Follow these steps to build and compile the application:

1. Set the environment variable `TritonClientBuild_DIR` (path/to/client/build/install), or link to the folder where you have installed Triton client libraries, or Triton client libraries directly and edit `CMakeLists` accordingly.

2. Create a build directory: `mkdir build`.

3. Navigate to the build directory: `cd build`.

4. Run CMake to configure the build:

   ```shell
   cmake -DCMAKE_BUILD_TYPE=Release ..
   ```

   Optional flags:

   - `-DSHOW_FRAME`: Enable to show processed frames after inference.
   - `-DWRITE_FRAME`: Enable to write processed frames after inference.

5. Build the application: `cmake --build .`

# Computer Vision Tasks

- [Object Detection](docs/ObjectDetection.md)
- [Classification](docs/Classification.md) 
- TODO Instance Segmentation, PoseEstimation... 

## Notes

- Ensure that the versions of libraries used for exporting models match the versions supported in the Triton release you are using. Check [Triton Server releases here](https://github.com/triton-inference-server/server/releases).

## Deploy to Triton

To deploy the models to Triton, set up a model repository folder following the [Triton Model Repository schema](https://github.com/triton-inference-server/server/blob/main/docs/model_repository.md). Usually, the `config.pbtxt` file is optional unless you use the OpenVino backend.

Example repository structure:

```
<model_repository>
    -> 
        <model_name>
            -> 
                [config.pbtx]
                <model_version>
                    ->
                        <model_binary>
```

Then, run the Triton server:

```shell
#!/bin/bash
$ docker run --gpus=1 --rm \
-p8000:8000 -p8001:8001 -p8002:8002 \
-v/full/path/to/docs/examples/model_repository:/models \
nvcr.io/nvidia/tritonserver:<xx.yy>-py3 tritonserver \
--model-repository=/models
```

If you plan to run on CPU, omit the `--gpus` parameter.

For more information and examples, refer to the [Triton Inference Server tutorials](https://github.com/triton-inference-server/tutorials).

## How to Run

### Performing Inference on a Video or Image Source

By using the `--source` parameter with the path to either a video or an image, you can perform computer vision tasks on your chosen input type. Follow these instructions:

```shell
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
Add input sizes if your model has dynamic axis
```
    --input_sizes="c w h" 
```

Replace the following placeholders:

- `/path/to/source.format`: The path to your video or image file.
- `<task_type>`: Choose the computer vision task type (e.g., `detection` or `classification`).
- `<model_type>`: Specify the model type (e.g., one of detectors: `yolov5`, `yolov6`, `yolov7`, `yolov8`, `yolov9`, `yolonas`  or classification models: `torchvision-classifier`).
- `<model_name_folder_on_triton>`: The name of the model folder in the Triton server where your chosen model is deployed.
- `/path/to/labels/coco.names`: The path to the file containing label names (e.g., COCO labels).
- `<http or grpc>`: Choose either `http` or `grpc` as the protocol based on your Triton server setup.
- `<triton-ip>`: The IP address of your Triton server.
- `<8000 for http, 8001 for grpc>`: The port number, which is usually `8000` for HTTP or `8001` for gRPC, depending on your Triton server configuration.

Use `./computer-vision-triton-cpp-client --help` to view all available parameters.

## How to Run with Docker

### Build the Docker Image

```bash
docker build --rm -t computer-vision-triton-cpp-client .
```

This command will create a Docker image based on the provided Dockerfile.

### Run the Docker Container

Replace the placeholders with your desired options and paths:

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
  --port<8000 for http, 8001 for grpc>
```

- `-v /path/to/host/data:/app/data`: Map a host directory to `/app/data` inside the container, allowing you to access input and output data.
- Adjust the rest of the parameters to match your specific setup.

### View Output

The program will process the specified video or image based on your options. You can find the processed output in the `/path/to/host/data` directory on your host machine.

## Demo Video: Realtime Inference Test

Watch a test of YOLOv7-tiny exported to ONNX [here](https://youtu.be/lke5TcbP2a0).

## References

- [Triton Inference Server Client Example](https://github.com/triton-inference-server/client/blob/r21.08/src/c%2B%2B/examples/image_client.cc)
- [YOLOv5 ONNX Runtime](https://github.com/itsnine/yolov5-onnxruntime)
- [TensorRTX](https://github.com/wang-xinyu/tensorrt)
- [Triton user guide](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_repository.html)



## Feedback
- Any feedback is greatly appreciated, if you have any suggestions, bug reports or questions don't hesitate to open an [issue](https://github.com/olibartfast/computer-vision-triton-cpp-client/issues).
