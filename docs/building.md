# Building TritonIC

This page covers dependencies, client library setup, and build instructions. See the [project README](../README.md) for quick-start instructions.

## Dependencies

Ensure the following dependencies are installed:

1. **Nvidia Triton Inference Server**:
```bash
docker pull nvcr.io/nvidia/tritonserver:26.08-py3
```

2. **Triton client libraries**: Release r26.08 (pinned as `TRITON_REPO_TAG` in [versions.env](../versions.env))
3. **Protobuf and gRPC++**: Versions compatible with Triton
4. **RapidJSON**:
```bash
apt install rapidjson-dev
```

5. **libcurl**:
```bash
apt install libcurl4-openssl-dev
```

6. **OpenCV 4**: Tested version: 4.7.0
```bash
apt install libopencv-dev
```

## Build Client Libraries

To build the client libraries, refer to the official [Triton Inference Server client libraries](https://github.com/triton-inference-server/client/tree/r26.08).

### Alternative: Extract Client Libraries from Docker

For convenience, you can extract the pre-built Triton client libraries from the official NVIDIA Triton Server SDK image using [Docker](guides/Docker_setup.md):

```bash
# Run the extraction script
./docker/scripts/extract_triton_libs.sh
```

This script will:
1. Create a temporary Docker container from the `nvcr.io/nvidia/tritonserver:26.08-py3-sdk` image
2. Extract the Triton client libraries from `/workspace/install`
3. Copy additional Triton server headers and libraries if available
4. Save everything to `./triton_client_libs/` directory

After extraction, set the environment variable:
```bash
export TritonClientBuild_DIR=$(pwd)/triton_client_libs/install
```

The extracted directory structure will contain:
- `install/` - Triton client build artifacts
- `triton_server_include/` - Triton server headers  
- `triton_server_lib/` - Triton server libraries
- `workspace/` - Additional workspace files

## Development Setup

### Pre-commit Hooks

To maintain code quality and consistency, install git hooks for automatic formatting on commit:

```bash
# Run the setup script (RECOMMENDED)
./scripts/setup/install_git_hooks.sh

# Or manually format all files before committing
./scripts/format_all.sh

# Or use the legacy pre-commit setup (requires Python)
./scripts/setup/pre_commit_setup.sh
```

> **Important:** Install the git hook to avoid CI format check failures. The hook automatically formats C++ files before each commit.

### Version Pins

Dependency and container versions are centralized in `versions.env`, which is read by `cmake/versions.cmake` (for CMake builds) and `scripts/versions.sh` (for shell scripts). After editing `versions.env`, run `scripts/check_version_sync.sh` to verify consistency across all configuration files:

```bash
./scripts/check_version_sync.sh
```

### Build and Compile

1. Set the environment variable `TritonClientBuild_DIR` or update the `CMakeLists.txt` with the path to your installed Triton client libraries.

2. Create a build directory:
```bash
mkdir build
```

3. Navigate to the build directory:
```bash
cd build
```

4. Run CMake to configure the build:
```bash
cmake -DCMAKE_BUILD_TYPE=Release ..
```

Optional CMake flags:

| Option | Default | Effect |
|---|---|---|
| `-DWITH_SHOW_FRAME=ON/OFF` | OFF | Display frames in a window during inference |
| `-DWITH_WRITE_FRAME=ON/OFF` | ON | Write output frames/video to disk |
| `-DWITH_CHAT_BACKEND=ON/OFF` | ON | OpenAI-compatible chat backend |
| `-DTRITONIC_ENABLE_CUDA=ON/OFF` | OFF | Client-side CUDA preprocessing (`--preprocess_device=cuda`); the binary still runs on CPU-only hosts |
| `-DBUILD_TESTING=ON/OFF` | OFF | Unit and integration tests |
| `-DTRITONIC_STRICT_WARNINGS=ON/OFF` | OFF | Strict compiler warnings for tritonic targets |

5. Build the application:
```bash
cmake --build . -j$(nproc)
```

## Tests

To build and run unit and integration tests:

```bash
cmake -S . -B build -DBUILD_TESTING=ON && cmake --build build -j && ctest --test-dir build --output-on-failure
```
