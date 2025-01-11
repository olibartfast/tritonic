# same level of Dockerfile
# docker build --rm -t computer-vision-triton-cpp-client .
# video from https://www.pexels.com/it-it/video/montagne-natura-volando-spiaggia-5614887/
docker run --rm \
--network host \
-v ${PWD}/data:/app/data computer-vision-triton-cpp-client \
  --source=/app/data/5614887-hd_1280_720_25fps.mp4 \
  --model_type=raft \
  --model=raft_large_torchscript  \
  --serverAddress=localhost \
  --input_sizes='3,520,960;3,520,960'
