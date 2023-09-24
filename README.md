# C++ Triton Client for Object Detection Models

This C++ application allows you to perform object detection inference using Nvidia Triton Server to manage multiple framework backends. It currently supports object detection models such as YoloV5, YoloV6, YoloV7, YoloV8, and Yolo-Nas.

## Build Client Libraries

To build the client libraries, refer to the Triton Inference Server client libraries located [here](https://github.com/triton-inference-server/client/tree/r23.08).

## Dependencies

Ensure you have the following dependencies installed:

- Nvidia Triton Inference Server container pulled from NGC (docker pull nvcr.io/nvidia/tritonserver:23.08-py3)
- Triton client libraries (Tested Release 23.08)
- Protobuf, Grpc++ (versions compatible with Triton Server)
- RapidJSON (apt install rapidjson-dev)
- libcurl (apt install libcurl4-openssl-dev)
- OpenCV 4 (Tested version: 4.7.0)

## Build and Compile

Follow these steps to build and compile the application:

1. Set environment variable `TritonClientBuild_DIR` (path/to/client/build/install), or however find a way to link the folder where you have installed triton client libraries, or triton client libraries directly, edit CMakeLists accordingly.
2. Create a build directory: `mkdir build`
3. Navigate to the build directory: `cd build`
4. Run CMake to configure the build:

   ```shell
   cmake -DCMAKE_BUILD_TYPE=Release ..
   ```
   
   Optional flags:
   - `-DSHOW_FRAME`: Enable to show processed frames after inference.
   - `-DWRITE_FRAME`: Enable to write processed frames after inference.
   
5. Build the application: `cmake --build .`

## YoloV5 Export

To export YoloV5 models in ONNX format, you can run the export script from the [yolov5 repository](https://github.com/ultralytics/yolov5/issues/251).

## YoloV6 Export

Export YoloV6 models in ONNX format from the [YoloV6 repository](https://github.com/meituan/YOLOv6/tree/main/deploy/ONNX). Post-processing code for YoloV6 is identical to YoloV5.

## YoloV7 Export

To export YoloV7 models, run the export script from the [YoloV7 repository](https://github.com/WongKinYiu/yolov7#export). Make sure to specify necessary export parameters, such as input and output sizes.

## YoloV8 Export

YoloV8 models can be exported in ONNX format similarly to YoloV5. You can refer to the [Ultralytics YoloV8-CPP-Inference example](https://github.com/ultralytics/ultralytics/tree/main/examples/YOLOv8-CPP-Inference).

## Yolo-Nas Export

To export Yolo-Nas models in ONNX format, follow the instructions in the [YoloNAS Quickstart](https://github.com/Deci-AI/super-gradients/blob/master/documentation/source/YoloNASQuickstart.md#export-to-onnx).

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
### Performing Inference on a Video or image source
By using the `--source` parameter with the path to either a video or an image, you can perform object detection on your chosen input type. Follow these instructions:

```shell
./object-detection-triton-cpp-client \
    --source=/path/to/source.format \
    --model_type=<yolo_version> \
    --model=<model_name_folder_on_triton> \
    --labelsFile=/path/to/labels/coco.names \
    --protocol=<http or grpc> \
    --serverAddress=<triton-ip> \
    --port=<8000 for http, 8001 for grpc>
```

Replace the following placeholders:
- `/path/to/source.format`: The path to your video or image file.
- `<yolo_version>`: The YOLO version you want to use (e.g., yolov5, yolov6, yolov7, yolov8).
- `<model_name_folder_on_triton>`: The name of the model folder in the Triton server where your chosen YOLO model is deployed.
- `/path/to/labels/coco.names`: The path to the file containing label names (e.g., COCO labels).
- `<http or grpc>`: Choose either `http` or `grpc` as the protocol based on your Triton server setup.
- `<triton-ip>`: The IP address of your Triton server.
- `<8000 for http, 8001 for grpc>`: The port number, which is usually `8000` for HTTP or `8001` for gRPC, depending on your Triton server configuration.

Use `./object-detection-triton-cpp-client --help` to view all available parameters.


## How to Run with Docker

### Build the Docker Image

```bash
docker build --rm -t object-detection-triton-cpp-client .
```

This command will create a Docker image based on the provided Dockerfile.

### Run the Docker Container

Replace the placeholders with your desired options and paths:

```bash
docker run --rm \
  -v /path/to/host/data:/app/data \
  object-detection-triton-cpp-client \
   --network host \
  --source=/app/data/source.format \
  --model_type=<yolo_version> \
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

Watch a test of YoloV7-tiny exported to ONNX [here](https://youtu.be/lke5TcbP2a0).

## References

- [Triton Inference Server Client Example](https://github.com/triton-inference-server/client/blob/r21.08/src/c%2B%2B/examples/image_client.cc)
- [YoloV5 ONNX Runtime](https://github.com/itsnine/yolov5-onnxruntime)
- [TensorRTX](https://github.com/wang-xinyu/tensorrtx)

## TO DO

- Manage models with dynamic axis (i.e., input layer -1, -1, -1, -1)
- NMS (Non-Maximum Suppression) on GPU

- The project tree is currently organised according to the structure below, in the long term I plan to add other computer vision tasks (and rename the repo accordingly).
```
└── src
    ├── tasks
    │   ├── classification
    │   ├── instance_segmentation
    │   └── object_detection
    │       ├── include
    │       │   └── DetectorInterface.hpp
    │       └── src
    │           ├── Yolo.hpp
    │           └── YoloNas.hpp
    └── Triton.hpp
```
