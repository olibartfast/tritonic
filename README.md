## C++ Triton client to infer YoloV5 and YoloV7 models 

## Build client libraries
https://github.com/triton-inference-server/client/tree/r22.08


## Dependencies
* Nvidia Triton Inference Server container pulled from NGC (docker pull nvcr.io/nvidia/tritonserver:22.07-py3)
* Triton client libraries (Tested Release 22.08)
* Protobuf, Grpc++(versions according to the ones used within Triton server project. I used libraries built inside Triton Client third party folder)
* rapidjson (on ubuntu sudo apt install rapidjson-dev)
* Opencv4(Tested 4.5.4)



## Build and compile
* Set environment variables TritonClientThirdParty_DIR(path/to/client/build/third-party) and TritonClientBuild_DIR(path/to/client/build/install)
* mkdir build 
* cd build 
* cmake -DCMAKE_BUILD_TYPE=Release .. 
* make

## YoloV7 export
* Run from [yolov7 repo](https://github.com/WongKinYiu/yolov7#export) export script(e.g. to onnx) : ```python export.py --weights <yolov7_version>.pt --grid  --simplify --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640 --max-wh 640``` (Don't use end-to-end parameter)

## YoloV5 export
* Run from [yolov5 repo](https://github.com/ultralytics/yolov5/issues/251) export script(always onnx case example):  ```python export.py  --weights <yolov5_version>.pt  --include onnx```

### Notes
When you export your model to tensorrt your version MUST match the one supported by your Triton version inside its container

### Deploy to Triton
Set a model repository folder following this [schema](https://github.com/triton-inference-server/server/blob/main/docs/model_repository.md):
```
<model_repository> 
    -> 
        <model_name> 
            -> 
                <model_version>
                    ->
                         <model_binary>
```

then [run](https://github.com/triton-inference-server/server/blob/main/docs/quickstart.md) the server
```
#!/bin/bash
$ docker run --gpus=1 --rm \
-p8000:8000 -p8001:8001 -p8002:8002 \
-v/full/path/to/docs/examples/model_repository:/models \
nvcr.io/nvidia/tritonserver:<xx.yy>-py3 tritonserver \
--model-repository=/models
```
If you plan to run on CPU omit --gpus parameter

## How to run
* ./yolo-triton-cpp-client  --video=/path/to/video/videoname.format  --model=model_name_folder_on_triton --labelsFile=/path/to/labels/coco.names
* ./yolo-triton-cpp-client  --help for all available parameters

### Realtime inference test on video


### References
* https://github.com/triton-inference-server/client/blob/r21.08/src/c%2B%2B/examples/image_client.cc
* https://github.com/itsnine/yolov5-onnxruntime
* https://github.com/wang-xinyu/tensorrtx
