# same level of Dockerfile
# docker build --rm -t computer-vision-triton-cpp-client .
docker run --rm \
--network host \
-v ${PWD}/data:/app/data computer-vision-triton-cpp-client \
  --source=/app/data/person.jpg \
  --model_type=yoloseg \
  --model=yolo11s-seg_onnx  \
  --labelsFile=/app/labels/coco.txt \
  --serverAddress=localhost
