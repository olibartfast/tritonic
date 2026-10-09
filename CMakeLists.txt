cmake_minimum_required(VERSION 3.20)

# Read version from VERSION file (single source of truth, read by CMake).
# A `-dev` suffix (e.g. `0.1.0-dev`) marks unreleased work on `develop`.
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" PROJECT_VERSION_RAW)
string(STRIP "${PROJECT_VERSION_RAW}" PROJECT_VERSION_RAW)
string(REGEX MATCH "^[0-9]+\\.[0-9]+\\.[0-9]+" PROJECT_SEMVER "${PROJECT_VERSION_RAW}")

project(tritonic VERSION ${PROJECT_SEMVER})
message(STATUS "${PROJECT_NAME} ${PROJECT_VERSION}")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Load centralized version pins for FetchContent and CMake caches.
include(cmake/versions.cmake)

set(TRITON_CLIENT_BUILD_DIR "$ENV{TritonClientBuild_DIR}" CACHE PATH "Path to Triton client build directory")

# Set default if not provided
if(NOT TRITON_CLIENT_BUILD_DIR OR TRITON_CLIENT_BUILD_DIR STREQUAL "")
    set(TRITON_CLIENT_BUILD_DIR "${CMAKE_SOURCE_DIR}/triton_client_libs/install" CACHE PATH "Path to Triton client build directory" FORCE)
    message(STATUS "TritonClientBuild_DIR not set, using default: ${TRITON_CLIENT_BUILD_DIR}")
endif()

message(STATUS "Using Triton Client Build Dir: ${TRITON_CLIENT_BUILD_DIR}")

# Fetch neuriplo-tasks repository for tasks
include(FetchContent)

# Auto-detect offline mode by testing network connectivity
if(NOT DEFINED FETCHCONTENT_FULLY_DISCONNECTED)
  execute_process(
    COMMAND ping -c 1 -W 1 github.com
    RESULT_VARIABLE NO_CONNECTION
    OUTPUT_QUIET
    ERROR_QUIET
  )
  if(NO_CONNECTION)
    message(STATUS "No network connection detected - enabling offline mode")
    set(FETCHCONTENT_FULLY_DISCONNECTED ON CACHE BOOL "Operate in offline mode" FORCE)
  else()
    set(FETCHCONTENT_FULLY_DISCONNECTED OFF CACHE BOOL "Operate in offline mode")
  endif()
endif()

# Set GIT_CONFIG to allow credential helper or use token from environment
if(DEFINED ENV{GITHUB_TOKEN})
  set(NEURIPLO_TASKS_REPO "https://$ENV{GITHUB_TOKEN}@github.com/olibartfast/neuriplo-tasks.git")
else()
  set(NEURIPLO_TASKS_REPO "https://github.com/olibartfast/neuriplo-tasks.git")
endif()

set(NEURIPLO_TASKS_WITH_OPENCV ON CACHE BOOL "" FORCE)

set(TRITONIC_NEURIPLO_TASKS_SOURCE_DIR "" CACHE PATH
    "Optional local neuriplo-tasks checkout used instead of the pinned release")
if(TRITONIC_NEURIPLO_TASKS_SOURCE_DIR)
  message(STATUS "Using local neuriplo-tasks: ${TRITONIC_NEURIPLO_TASKS_SOURCE_DIR}")
  FetchContent_Declare(
    neuriplo-tasks
    SOURCE_DIR "${TRITONIC_NEURIPLO_TASKS_SOURCE_DIR}"
  )
else()
  FetchContent_Declare(
    neuriplo-tasks
    GIT_REPOSITORY ${NEURIPLO_TASKS_REPO}
    GIT_TAG ${NEURIPLO_TASKS_VERSION}
  )
endif()
FetchContent_MakeAvailable(neuriplo-tasks)


option(WITH_SHOW_FRAME "Option description" OFF)
option(WITH_WRITE_FRAME "Option description" ON)
option(WITH_CHAT_BACKEND "Build OpenAI-compatible chat backend (requires CURL, already a dep)" ON)
option(TRITONIC_STRICT_WARNINGS "Enable strict compiler warnings for tritonic targets." OFF)
option(TRITONIC_ENABLE_CUDA
    "Build CUDA support: CUDA shared memory and client-side GPU preprocessing (needs the CUDA toolkit)"
    OFF)

function(tritonic_apply_strict_warnings target_name)
  if(NOT TRITONIC_STRICT_WARNINGS)
    return()
  endif()

  if(MSVC)
    target_compile_options(${target_name} PRIVATE /W4 /WX)
  else()
    target_compile_options(${target_name} PRIVATE -Wall -Wextra -Wpedantic -Werror)
  endif()
endfunction()

if(WITH_SHOW_FRAME)
  add_definitions(-DSHOW_FRAME)
  message(STATUS "Show frame enabled")
endif()
if(WITH_WRITE_FRAME)
  add_definitions(-DWRITE_FRAME)
  message(STATUS "Write frame enabled")
endif()

set(TritonClient_DIR "${TRITON_CLIENT_BUILD_DIR}/lib/cmake/TritonClient")
set(TritonCommon_DIR "${TRITON_CLIENT_BUILD_DIR}/lib/cmake/TritonCommon")


find_package(OpenCV REQUIRED)
find_package(TritonCommon REQUIRED)
find_package(TritonClient REQUIRED)
find_package(CURL REQUIRED)
find_package(Protobuf REQUIRED)
find_package(RapidJSON REQUIRED)

# CUDA is opt-in: with TRITONIC_ENABLE_CUDA=OFF the build has no CUDA dependency.
if(TRITONIC_ENABLE_CUDA)
    set(CUDA_ARCH "" CACHE STRING
        "CUDA architectures override; empty uses CUDA_ARCHITECTURES from versions.env")
    if(CUDA_ARCH)
        set(_tritonic_cuda_arch "${CUDA_ARCH}")
    else()
        set(_tritonic_cuda_arch "${CUDA_ARCHITECTURES}")
    endif()
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES ${_tritonic_cuda_arch})
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    add_library(tritonic_cuda STATIC
        ${PROJECT_SOURCE_DIR}/src/cuda/letterbox.cu
        ${PROJECT_SOURCE_DIR}/src/cuda/letterbox_preprocessor.cpp
    )
    target_include_directories(tritonic_cuda PUBLIC
        ${PROJECT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
    )
    target_compile_features(tritonic_cuda PUBLIC cxx_std_20)
    # TRITON_ENABLE_GPU makes the Triton client's ipc.h use the real cudaIpcMemHandle_t,
    # matching the GPU-enabled client libraries' RegisterCudaSharedMemory signature.
    target_compile_definitions(tritonic_cuda PUBLIC TRITONIC_ENABLE_CUDA TRITON_ENABLE_GPU)
    target_link_libraries(tritonic_cuda PUBLIC CUDA::cudart ${OpenCV_LIBS})
    set_target_properties(tritonic_cuda PROPERTIES POSITION_INDEPENDENT_CODE ON)
    message(STATUS "CUDA support enabled (architectures: ${CMAKE_CUDA_ARCHITECTURES})")
endif()

# Set the source files (only tritonic-specific sources)
set(SOURCES
    ${PROJECT_SOURCE_DIR}/src/main/client.cpp
    ${PROJECT_SOURCE_DIR}/src/main/App.cpp
    # Add any additional tritonic-specific source files here
)

# Chat backend (OpenAI-compatible) — CURL is already a required dep
set(CHAT_SOURCES)
if(WITH_CHAT_BACKEND)
    list(APPEND CHAT_SOURCES
        ${PROJECT_SOURCE_DIR}/src/chat/ChatBackend.cpp
        ${PROJECT_SOURCE_DIR}/src/chat/ChatSession.cpp
    )
    add_definitions(-DWITH_CHAT_BACKEND)
    message(STATUS "Chat backend enabled")
endif()

# Set the source files
set(TRITON_SOURCES 
    ${PROJECT_SOURCE_DIR}/src/triton/Triton.cpp
)

set(INFRA_SOURCES
    ${PROJECT_SOURCE_DIR}/src/main/Logger.cpp
    ${PROJECT_SOURCE_DIR}/src/main/ConfigManager.cpp
)


# Add an executable target
add_executable(${PROJECT_NAME}
    ${SOURCES}
    ${TRITON_SOURCES}
    ${INFRA_SOURCES}
    ${CHAT_SOURCES}
)
tritonic_apply_strict_warnings(${PROJECT_NAME})

# Set include directories for the target
target_include_directories(${PROJECT_NAME}
    PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/src/utils
    ${PROJECT_SOURCE_DIR}/src/triton
    ${PROJECT_SOURCE_DIR}/src/chat
    # neuriplo-tasks includes are provided automatically through target_link_libraries
)

# Third-party headers from the Triton SDK (grpc, protobuf, abseil) are not held to
# TRITONIC_STRICT_WARNINGS: the 26.08 abseil headers fail -Wpedantic -Werror.
target_include_directories(${PROJECT_NAME} SYSTEM PRIVATE ${TRITON_CLIENT_BUILD_DIR}/include)

# Set link directories for the target
target_link_directories(${PROJECT_NAME}
    PRIVATE
    ${TRITON_CLIENT_BUILD_DIR}/lib
    # Add any additional link directories here
)

# Link libraries for the target
target_link_libraries(${PROJECT_NAME}
    PRIVATE
    neuriplo-tasks::neuriplo-tasks
    neuriplo-tasks::vision-opencv
    grpcclient
    httpclient
    ${OpenCV_LIBS}
    CURL::libcurl
    ${PROTOBUF_LIBRARIES}
)
if(TRITONIC_ENABLE_CUDA)
    target_link_libraries(${PROJECT_NAME} PRIVATE tritonic_cuda)
endif()

# Unoptimized builds against the 26.08+ SDK instantiate abseil CHECK helpers from the
# bundled protobuf headers; their definitions live in this static library, which the
# shared client libraries do not export. Release builds inline them away.
set(TRITONIC_ABSL_CHECK_LIBS)
foreach(absl_lib log_internal_check_op leak_check strings strings_internal int128
        raw_logging_internal throw_delegate base spinlock_wait log_severity)
    set(absl_path "${TRITON_CLIENT_BUILD_DIR}/lib/libabsl_${absl_lib}.a")
    if(EXISTS "${absl_path}")
        list(APPEND TRITONIC_ABSL_CHECK_LIBS "${absl_path}")
    endif()
endforeach()
add_library(tritonic_sdk_absl INTERFACE)
if(TRITONIC_ABSL_CHECK_LIBS)
    target_link_libraries(tritonic_sdk_absl INTERFACE
        -Wl,--start-group ${TRITONIC_ABSL_CHECK_LIBS} -Wl,--end-group)
endif()
target_link_libraries(${PROJECT_NAME} PRIVATE tritonic_sdk_absl)

# Add option for testing
option(BUILD_TESTING "Build the testing tree." OFF)

# Enable testing if requested
if(BUILD_TESTING)
    enable_testing()
    
    # Fetch GoogleTest (reuse FetchContent from above if already included)
    if(NOT TARGET gtest)
        include(FetchContent)
        FetchContent_Declare(
          googletest
          GIT_REPOSITORY https://github.com/google/googletest.git
          GIT_TAG ${GTEST_VERSION}
        )
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)
    endif()
    
    # Collect test files
    file(GLOB UNIT_TEST_SOURCES "tests/unit/*.cpp")
    file(GLOB INTEGRATION_TEST_SOURCES "tests/integration/*.cpp")
    
    # Main test runner
    set(TEST_MAIN tests/test_main.cpp)
    
    # Create unit test executable if unit tests exist
    if(UNIT_TEST_SOURCES)
        add_executable(tritonic_unit_tests
            ${TEST_MAIN}
            ${UNIT_TEST_SOURCES}
            src/main/App.cpp
            src/triton/Triton.cpp
            ${INFRA_SOURCES}
            ${CHAT_SOURCES}
        )
        tritonic_apply_strict_warnings(tritonic_unit_tests)

        # The Triton SDK install tree ships its own gtest/gmock headers. Included through
        # -I they shadow the googletest these tests link (googletest's own directories are
        # -isystem, searched after every -I, and GCC drops an -I that repeats an -isystem
        # path), and gMock expectations then crash on the header/library mismatch. So the
        # tests take the client headers as -isystem too, linked after gtest/gmock so that
        # googletest's directories come first. They still precede /usr/include.
        add_library(tritonic_test_triton_client_headers INTERFACE)
        target_include_directories(tritonic_test_triton_client_headers SYSTEM INTERFACE
            ${TRITON_CLIENT_BUILD_DIR}/include)

        target_include_directories(tritonic_unit_tests PRIVATE
            ${PROJECT_SOURCE_DIR}/include
            ${PROJECT_SOURCE_DIR}/src
            ${PROJECT_SOURCE_DIR}/src/triton
            ${PROJECT_SOURCE_DIR}/src/chat
            ${PROJECT_SOURCE_DIR}/src/common
            ${PROJECT_SOURCE_DIR}/tests
            ${PROJECT_SOURCE_DIR}/tests/mocks
            ${OpenCV_INCLUDE_DIRS}
        )

        target_link_directories(tritonic_unit_tests PRIVATE
            ${TRITON_CLIENT_BUILD_DIR}/lib
        )
        
        target_link_libraries(tritonic_unit_tests PRIVATE
            gtest gtest_main gmock
            tritonic_test_triton_client_headers
            neuriplo-tasks::neuriplo-tasks
            neuriplo-tasks::vision-opencv
            grpcclient
            httpclient
            ${OpenCV_LIBS}
            CURL::libcurl
            ${PROTOBUF_LIBRARIES}
            tritonic_sdk_absl
        )
        if(TRITONIC_ENABLE_CUDA)
            target_link_libraries(tritonic_unit_tests PRIVATE tritonic_cuda)
        endif()

        include(GoogleTest)
        gtest_discover_tests(tritonic_unit_tests
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
            PROPERTIES TIMEOUT 60
        )
    endif()

    add_executable(yolo_preprocess_baseline
        tests/tools/yolo_preprocess_baseline.cpp
    )
    tritonic_apply_strict_warnings(yolo_preprocess_baseline)
    target_link_libraries(yolo_preprocess_baseline PRIVATE
        neuriplo-tasks::neuriplo-tasks
        neuriplo-tasks::vision-opencv
        ${OpenCV_LIBS}
    )

    if(TRITONIC_ENABLE_CUDA)
        add_executable(cuda_preprocess_dump tests/tools/cuda_preprocess_dump.cpp)
        tritonic_apply_strict_warnings(cuda_preprocess_dump)
        target_include_directories(cuda_preprocess_dump PRIVATE ${OpenCV_INCLUDE_DIRS})
        target_link_libraries(cuda_preprocess_dump PRIVATE tritonic_cuda ${OpenCV_LIBS})
    endif()

    # Create integration test executable if integration tests exist
    if(INTEGRATION_TEST_SOURCES)
        add_executable(tritonic_integration_tests
            ${TEST_MAIN}
            ${INTEGRATION_TEST_SOURCES}
            ${INFRA_SOURCES}
            ${CHAT_SOURCES}
        )
        tritonic_apply_strict_warnings(tritonic_integration_tests)

        target_include_directories(tritonic_integration_tests PRIVATE
            ${PROJECT_SOURCE_DIR}/include
            ${PROJECT_SOURCE_DIR}/src
            ${PROJECT_SOURCE_DIR}/src/chat
            ${PROJECT_SOURCE_DIR}/src/common
            ${PROJECT_SOURCE_DIR}/tests
            ${PROJECT_SOURCE_DIR}/tests/mocks
            ${OpenCV_INCLUDE_DIRS}
        )

        target_link_libraries(tritonic_integration_tests PRIVATE
            gtest gtest_main gmock
            ${OpenCV_LIBS}
            CURL::libcurl
        )
        
        add_test(NAME IntegrationTests COMMAND tritonic_integration_tests)
        set_tests_properties(IntegrationTests PROPERTIES
            TIMEOUT 120
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        )
    endif()
endif()
