# Local CI Checks

Run these commands from the repository root to mirror the CI pipeline locally.

## Prerequisites

- Install `clang-format`, `clang-tidy`, `cppcheck`, `cmake`, and `ctest`
- Install build dependencies: OpenCV, Protobuf, RapidJSON, and `libcurl`
- Ensure Triton client libraries exist at `triton_client_libs/install`

Export the Triton client path before build-based checks:

```bash
export TritonClientBuild_DIR="$(pwd)/triton_client_libs/install"
```

If the Triton client libraries are missing:

```bash
./docker/scripts/extract_triton_libs.sh
```

## Format Check

```bash
mapfile -t files < <(find include src tests -type f \( -name "*.hpp" -o -name "*.cpp" \) | sort)
clang-format --dry-run --Werror "${files[@]}"
```

## Clang-Tidy

```bash
cmake -S . -B build-tidy -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
mapfile -t files < <(find src -type f -name "*.cpp" | sort)
clang-tidy -p build-tidy --header-filter='^(include|src)/' "${files[@]}"
```

## Cppcheck

```bash
cppcheck \
  --enable=warning,style,performance,portability \
  --error-exitcode=1 \
  --inline-suppr \
  --std=c++20 \
  --language=c++ \
  --suppress=missingIncludeSystem \
  include src tests
```

## Build & Test

```bash
cmake -S . -B build-ci -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build-ci -j"$(nproc)"
ctest --test-dir build-ci --output-on-failure
```

## Build with Strict Warnings

```bash
cmake -S . -B build-strict \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON \
  -DTRITONIC_STRICT_WARNINGS=ON
cmake --build build-strict -j"$(nproc)"
```

`TRITONIC_STRICT_WARNINGS` enables `-Wall -Wextra -Wpedantic -Werror` on tritonic targets and tests.
