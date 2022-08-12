## C++ Triton client to infer YoloV4/V5 and YoloV7 models 

## Work in progress

## Build client libraries
https://github.com/triton-inference-server/client/tree/r22.06


## Dependencies
* Nvidia Triton Inference Server container pulled from NGC(Tested Release 22.06)
* Triton client libraries
* Protobuf, Grpc++, Rapidjson(versions according to the ones used within Triton server project. I used libraries built inside Triton Client third party folder)
* Cuda(Tested 11.7)
* Opencv4(Tested 4.6.0)

## Build and compile
* Set environment variables TritonClientThirdParty_DIR(path/to/client/build/third-party) and TritonClientBuild_DIR(path/to/client/build/install)
* mkdir build 
* cd build 
* cmake -DCMAKE_BUILD_TYPE=Release .. 
* make

## YoloV7
* go to repo
* export to onnx the model python export.py --weights <yolo_version>.pt --grid  --simplify --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640 --max-wh 640

### YoloV7 TensorRT export
* To Do


## YoloV5
* To Do

## YoloV4
* To Do

### Notes
When you export your model to tensorrt your version MUST match the one supported by your Triton version


## How to run
* ./yolo-triton-cpp-client  --video=/path/to/video/videoname.format
* ./yolo-triton-cpp-client  --help for all available parameters

### Realtime inference test on video

