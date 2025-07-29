#!/bin/bash
# docker build -f Dockerfile.dev --rm -t tritonic:dev .
echo "Running unit tests in Docker container..."
docker run --rm --entrypoint /bin/sh tritonic:dev -c "cd /app/build && ./tests/run_tests" 
