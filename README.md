## C++ Triton client to infer YoloV5 and YoloV7 models 

## Work in progress

## Build client libraries
https://github.com/triton-inference-server/client/tree/r22.08


## Dependencies
* Nvidia Triton Inference Server container pulled from NGC (docker pull nvcr.io/nvidia/tritonserver:22.07-py3)
* Triton client libraries (Tested Release 22.08)
* Protobuf, Grpc++(versions according to the ones used within Triton server project. I used libraries built inside Triton Client third party folder)
* rapidjson (on ubuntu sudo apt install rapidjson-dev)
* Opencv4(Tested 4.6.0)



## Build and compile
* Set environment variables TritonClientThirdParty_DIR(path/to/client/build/third-party) and TritonClientBuild_DIR(path/to/client/build/install)
* mkdir build 
* cd build 
* cmake -DCMAKE_BUILD_TYPE=Release .. 
* make

## YoloV7 Onnx export
* Run from yolov7 repo export script i.e.: python export.py --weights <yolov7_version>.pt --grid  --simplify --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640 --max-wh 640 (Don't use end-to-end parameter)

### YoloV7 TensorRT export
* To Do


## YoloV5 Onnx export
* Run from yolov5 repo export script:  python export.py  --weights <yolov5_version>.pt  --include onnx

### YoloV5 TensorRT export
* To Do

### Notes
When you export your model to tensorrt your version MUST match the one supported by your Triton version


## How to run
* ./yolo-triton-cpp-client  --video=/path/to/video/videoname.format
* ./yolo-triton-cpp-client  --help for all available parameters

### Realtime inference test on video


### References
https://github.com/triton-inference-server/client/blob/r21.08/src/c%2B%2B/examples/image_client.cc
https://github.com/itsnine/yolov5-onnxruntime
https://github.com/wang-xinyu/tensorrtx
