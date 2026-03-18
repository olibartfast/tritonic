# build from repository root
# docker build --rm -t tritonic -f docker/Dockerfile .
docker run --rm \
--network host \
--user root \
-v ${PWD}/data:/app/data tritonic:latest \
  --source=/app/data/images/person.jpg \
  --model_type=yoloseg \
  --model=yolo11s-seg_onnx  \
  --labelsFile=/app/labels/coco.txt \
  --serverAddress=localhost
