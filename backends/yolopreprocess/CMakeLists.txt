# Triton C++ backend "yolopreprocess": nvJPEG decode + the shared tritonic CUDA letterbox
# (specs/2026-10-01-gpu-preprocess-alternatives, variants V12 and V13).
# Built in the TensorRT image of the pinned Triton release by build_backend.sh.
cmake_minimum_required(VERSION 3.24)
project(tritonyolopreprocessbackend LANGUAGES C CXX CUDA)

# Load centralized version pins from the parent tritonic build.
include("${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/versions.cmake")

set(TRITON_MIN_CXX_STANDARD 17 CACHE STRING "Minimum C++ standard for this backend.")
set(TRITON_REPO_ORGANIZATION "https://github.com/triton-inference-server" CACHE STRING
    "Git repository to pull from")
# Pinned to the reference server release (specs/tech-stack.md).
set(TRITON_COMMON_REPO_TAG "" CACHE STRING
    "triton-inference-server/common tag override; empty uses TRITON_REPO_TAG")
set(TRITON_CORE_REPO_TAG "" CACHE STRING
    "triton-inference-server/core tag override; empty uses TRITON_REPO_TAG")
set(TRITON_BACKEND_REPO_TAG "" CACHE STRING
    "triton-inference-server/backend tag override; empty uses TRITON_REPO_TAG")
if(NOT TRITON_COMMON_REPO_TAG)
  set(TRITON_COMMON_REPO_TAG "${TRITON_REPO_TAG}")
endif()
if(NOT TRITON_CORE_REPO_TAG)
  set(TRITON_CORE_REPO_TAG "${TRITON_REPO_TAG}")
endif()
if(NOT TRITON_BACKEND_REPO_TAG)
  set(TRITON_BACKEND_REPO_TAG "${TRITON_REPO_TAG}")
endif()
set(CUDA_ARCH "" CACHE STRING
    "CUDA architectures override; empty uses CUDA_ARCHITECTURES from versions.env")
if(CUDA_ARCH)
  set(_triton_backend_cuda_arch "${CUDA_ARCH}")
else()
  set(_triton_backend_cuda_arch "${CUDA_ARCHITECTURES}")
endif()
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES ${_triton_backend_cuda_arch})
endif()
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# The backend utilities must be built with GPU support so BackendModelInstance owns a
# CUDA stream per GPU instance.
set(TRITON_ENABLE_GPU ON CACHE BOOL "" FORCE)
set(TRITON_ENABLE_STATS ON CACHE BOOL "" FORCE)

find_package(CUDAToolkit REQUIRED)

include(FetchContent)
FetchContent_Declare(repo-common
  GIT_REPOSITORY ${TRITON_REPO_ORGANIZATION}/common.git
  GIT_TAG ${TRITON_COMMON_REPO_TAG}
  GIT_SHALLOW ON)
FetchContent_Declare(repo-core
  GIT_REPOSITORY ${TRITON_REPO_ORGANIZATION}/core.git
  GIT_TAG ${TRITON_CORE_REPO_TAG}
  GIT_SHALLOW ON)
FetchContent_Declare(repo-backend
  GIT_REPOSITORY ${TRITON_REPO_ORGANIZATION}/backend.git
  GIT_TAG ${TRITON_BACKEND_REPO_TAG}
  GIT_SHALLOW ON)
FetchContent_MakeAvailable(repo-common repo-core repo-backend)

set(TRITONIC_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
configure_file(src/libtriton_yolopreprocess.ldscript libtriton_yolopreprocess.ldscript COPYONLY)

add_library(triton-yolopreprocess-backend SHARED
  src/yolopreprocess.cc
  src/gray_to_rgb.cu
  ${TRITONIC_ROOT}/src/cuda/letterbox.cu
)
target_include_directories(triton-yolopreprocess-backend PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${TRITONIC_ROOT}/include
)
target_compile_features(triton-yolopreprocess-backend PRIVATE cxx_std_${TRITON_MIN_CXX_STANDARD})
target_compile_definitions(triton-yolopreprocess-backend PRIVATE TRITON_ENABLE_GPU TRITON_ENABLE_STATS)
target_compile_options(triton-yolopreprocess-backend PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wno-unused-parameter -Wno-type-limits -Werror>)
target_link_libraries(triton-yolopreprocess-backend PRIVATE
  triton-core-serverapi
  triton-core-backendapi
  triton-core-serverstub
  triton-backend-utils
  CUDA::cudart
  CUDA::nvjpeg
)
set_target_properties(triton-yolopreprocess-backend PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  OUTPUT_NAME triton_yolopreprocess
  LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libtriton_yolopreprocess.ldscript
  LINK_FLAGS "-Wl,--version-script libtriton_yolopreprocess.ldscript"
)

include(GNUInstallDirs)
install(TARGETS triton-yolopreprocess-backend
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/yolopreprocess)
