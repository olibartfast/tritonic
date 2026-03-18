#!/bin/bash
# docker build --rm -t tritonic:dev -f docker/Dockerfile .
echo "Running unit tests in Docker container..."
docker run --rm --entrypoint /bin/sh tritonic:dev -c "cd /app/build && ./tests/run_tests" 
