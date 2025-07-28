#!/bin/bash
echo "Running unit tests in Docker container..."
docker run --platform linux/amd64 --rm --entrypoint /bin/sh tritonic:dev -c "cd /app/build && ./tests/run_tests" 
