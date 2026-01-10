# same level of Dockerfile
# docker build --rm -t tritonic .
docker run --rm \
--network host \
--user root \
-v ${PWD}/data:/app/data tritonic:latest \
  --source='/app/data/images/raft/frame_001.png,/app/data/images/raft/frame_002.png' \
  --model_type=raft \
  --model=raft_large_trt  \
  --serverAddress=localhost \
  --input_sizes='3,520,960;3,520,960'


# supposing export with command:
# export NGC_TAG_VERSION=25.12

# export MODEL_ONNX_DIR=$HOME/model_repository/raft_large_onnx/1
# export MODEL_TRT_DIR=$HOME/model_repository/raft_large_trt/1
# export MIN_SHAPES="input1:1x3x256x256,input2:1x3x256x256"
# export OPT_SHAPES="input1:1x3x520x960,input2:1x3x520x960"
# export MAX_SHAPES="input1:1x3x1080x1920,input2:1x3x1080x1920"

# docker run --rm -it --gpus=all \
#     -v $MODEL_ONNX_DIR:/workspace/input \
#     -v $MODEL_TRT_DIR:/workspace/output \
#     --ipc=host \
#     --ulimit memlock=-1 \
#     --ulimit stack=67108864 \
#     -w /workspace \
#     nvcr.io/nvidia/tensorrt:${NGC_TAG_VERSION}-py3 \
#     /bin/bash -c "trtexec --onnx=/workspace/input/model.onnx \
#         --saveEngine=/workspace/output/model.plan \
#         --memPoolSize=workspace:4096 \
#         --fp16 \
#         --useCudaGraph \
#         --useSpinWait \
#         --warmUp=500 \
#         --avgRuns=1000 \
#         --duration=10 \
#         --minShapes=$MIN_SHAPES \
#         --optShapes=$OPT_SHAPES \
#         --maxShapes=$MAX_SHAPES"


# Supposing using the following model configuration for raft_large_trt:
# name: "raft_large_trt"
# platform: "tensorrt_plan"
# max_batch_size: 1

# input [
#   {
#     name: "input1"
#     data_type: TYPE_FP32
#     dims: [ 3, -1, -1 ]
#   },
#   {
#     name: "input2"
#     data_type: TYPE_FP32
#     dims: [ 3, -1, -1 ]
#   }
# ]

# output [
#   {
#     name: "flow_prediction"
#     data_type: TYPE_FP32
#     dims: [ 2, -1, -1 ]
#   }
# ]

# dynamic_batching {
#   max_queue_delay_microseconds: 100
# }

# instance_group [
#   {
#     count: 1
#     kind: KIND_GPU
#     gpus: [ 0 ]
#   }
# ]

# optimization {
#   cuda {
#     graphs: true
#     graph_spec {
#       batch_size: 1
#       input {
#         key: "input1"
#         value {
#           dim: [ 3, 520, 960 ]
#         }
#       }
#       input {
#         key: "input2"
#         value {
#           dim: [ 3, 520, 960 ]
#         }
#       }
#     }
#   }
# }

