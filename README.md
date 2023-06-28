## C++ Triton client to infer Yolo series models 
* Currently YoloV5/V6/V7/V8, Yolo-Nas, only object detection. Instance segmentation in to-do list/in progress.
## Build client libraries
https://github.com/triton-inference-server/client/tree/r22.08


## Dependencies
* Nvidia Triton Inference Server container pulled from NGC (docker pull nvcr.io/nvidia/tritonserver:22.07-py3)
* Triton client libraries (Tested Release 22.08)
* Protobuf, Grpc++(versions according to the ones used within Triton server project. I used libraries built inside Triton Client third party folder)
* rapidjson (on ubuntu sudo apt install rapidjson-dev)
* libcurl
* Opencv4(Tested 4.5.4)



## Build and compile
* Set environment variables TritonClientThirdParty_DIR(path/to/client/build/third-party) and TritonClientBuild_DIR(path/to/client/build/install)
* mkdir build 
* cd build 
* cmake -DCMAKE_BUILD_TYPE=Release (or optional -DSHOW_FRAME -DWRITE_FRAME if you want show or write processed frame after inference) .. 
* make

## YoloV5 export
* Run from [yolov5 repo](https://github.com/ultralytics/yolov5/issues/251) export script(always onnx case example):  ```python export.py  --weights <yolov5_version>.pt  --include onnx```

## YoloV6 export
Weights to export in ONNX format or download from [yolov6 repo](https://github.com/meituan/YOLOv6/tree/main/deploy/ONNX). Posteprocessing code is identical to yolov5-v7.

## YoloV7 export
* Run from [yolov7 repo](https://github.com/WongKinYiu/yolov7#export) export script(e.g. to onnx) : ```python export.py --weights <yolov7_version>.pt --grid  --simplify --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640 --max-wh 640``` (Don't use end-to-end parameter)

## YoloV8 export 
* Weights to export in ONNX format, [same way as yolov5](https://github.com/ultralytics/ultralytics/tree/main/examples/YOLOv8-CPP-Inference).

## Yolo-Nas export 
* Weights can be export in ONNX format like in [YoloNAS Quickstart](https://github.com/Deci-AI/super-gradients/blob/master/documentation/source/YoloNASQuickstart.md#export-to-onnx).  
I export the model specifying input and output layers name, for example here below in the case of yolo_nas_s version:
```
from super_gradients.training import models

net = models.get("yolo_nas_s", pretrained_weights="coco")
models.convert_to_onnx(model=net, input_shape=(3,640,640), out_path="yolo_nas_s.onnx", torch_onnx_export_kwargs={"input_names": ['input'], "output_names": ['output0', 'output1']})
```


## Notes
*  If you trying to export the model to TensorRT or OpenVino, your installed version of previous libraries MUST match the ones supported in Triton release you are using, check [server releases here](https://github.com/triton-inference-server/server/releases) 

### Deploy to Triton
Set a model repository folder following this [schema](https://github.com/triton-inference-server/server/blob/main/docs/model_repository.md), usually config.pbtxt file is optional except in case of use of OpenVino backend:
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
```
 ./yolo-triton-cpp-client  --video=/path/to/video/videoname.format  --model_type=<yolo_version> --model=<model_name_folder_on_triton> --labelsFile=/path/to/labels/coco.names --serverAddress=<address:port>
```
```
 ./yolo-triton-cpp-client  --help for all available parameters
```

### Realtime inference test on video
Testing yolov7-tiny exported to onnx  
https://youtu.be/lke5TcbP2a0


### References
* https://github.com/triton-inference-server/client/blob/r21.08/src/c%2B%2B/examples/image_client.cc
* https://github.com/itsnine/yolov5-onnxruntime
* https://github.com/wang-xinyu/tensorrtx


### TO DO
* Manage models with dynamic axis(i.e. input layer -1, -1, -1, -1)
* Nms on gpu
* Instance segmentation
