# same level of Dockerfile
# docker build --rm -t tritonic:latest .
docker run --rm \
--network host \
--user root \
-v ${PWD}/data:/app/data tritonic:latest \
  --source=/app/data/images/person.jpg \
  --model_type=yolov8 \
  --model=yolov8m_onnx  \
  --labelsFile=/app/labels/coco.txt \
  --serverAddress=localhost \
  --input_sizes="3,640,640" \
  --log_level=debug
