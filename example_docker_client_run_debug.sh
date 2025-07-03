# same level of Dockerfile
# docker build --rm -t computer-vision-triton-cpp-client .
docker run --rm \
--network host \
-v ${PWD}/data:/app/data computer-vision-triton-cpp-client \
  --source=/app/data/person.jpg \
  --model_type=yolov8 \
  --model=yolov8m_onnx  \
  --labelsFile=/app/labels/coco.txt \
  --serverAddress=localhost \
  --input_sizes="3,640,640" \
  --log_level=debug
