## C++ Triton YoloV5 and YoloV7 client 


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

## How to run
* ./yolo-triton-cpp-client  --video=/path/to/video/videoname.format
* ./yolo-triton-cpp-client  --help for all available parameters

### Realtime inference test on video
* Inference test ran from VS Code: https://youtu.be/IUdbplJlspg
* other video inference test: https://youtu.be/VsENXGMNlhA