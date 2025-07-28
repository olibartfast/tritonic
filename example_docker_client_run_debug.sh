# same level of Dockerfile
# docker build --rm -t computer-vision-triton-cpp-client:latest .
docker run --rm \
--network host \
--user root \
-v ${PWD}/data:/app/data computer-vision-triton-cpp-client:latest \
  --source=/app/data/person.jpg \
  --model_type=yolov8 \
  --model=yolov8m_onnx  \
  --labelsFile=/app/labels/coco.txt \
  --serverAddress=192.168.1.212 \
  --input_sizes="3,640,640" \
  --log_level=debug
