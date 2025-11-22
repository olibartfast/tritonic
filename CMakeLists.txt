cmake_minimum_required(VERSION 3.20)

project(tritonic)
message(STATUS ${PROJECT_NAME})
message(STATUS $ENV{TritonClientBuild_DIR})
set(CMAKE_CXX_STANDARD 20)

# Fetch vision-core repository for tasks
include(FetchContent)

# Set GIT_CONFIG to allow credential helper or use token from environment
if(DEFINED ENV{GITHUB_TOKEN})
  set(VISION_CORE_REPO "https://$ENV{GITHUB_TOKEN}@github.com/olibartfast/vision-core.git")
else()
  set(VISION_CORE_REPO "https://github.com/olibartfast/vision-core.git")
endif()

FetchContent_Declare(
  vision-core
  GIT_REPOSITORY ${VISION_CORE_REPO}
  GIT_TAG main  # or specify a branch/tag/commit
  SOURCE_SUBDIR ""  # Don't try to build it as a CMake project
)
FetchContent_MakeAvailable(vision-core)

# Set the tasks directory to the fetched content
set(TASKS_DIR ${vision-core_SOURCE_DIR}/src/tasks)

option(WITH_SHOW_FRAME "Option description" OFF)
option(WITH_WRITE_FRAME "Option description" ON)

if(WITH_SHOW_FRAME)
  add_definitions(-DSHOW_FRAME)
  message(STATUS "Show frame enabled")
endif()
if(WITH_WRITE_FRAME)
  add_definitions(-DWRITE_FRAME)
  message(STATUS "Write frame enabled")
endif()

set(TritonClient_DIR "$ENV{TritonClientBuild_DIR}/lib/cmake/TritonClient")
set(TritonCommon_DIR "$ENV{TritonClientBuild_DIR}/lib/cmake/TritonCommon")


find_package(OpenCV REQUIRED)
find_package(TritonCommon REQUIRED)
find_package(TritonClient REQUIRED)
find_package(CURL REQUIRED)
find_package(Protobuf REQUIRED)
find_package(RapidJSON REQUIRED)

# Set the source files
set(SOURCES
    ${PROJECT_SOURCE_DIR}/src/main/client.cpp
    ${TASKS_DIR}/task_factory.cpp
    # Add any additional source files here
)

set(DETECTOR_SOURCES ${TASKS_DIR}  
  object_detection/src/YOLO.cpp 
  object_detection/src/YoloNas.cpp
  object_detection/src/YOLOv10.cpp 
  object_detection/src/RTDetr.cpp
  object_detection/src/RFDetr.cpp
  object_detection/src/RTDetrUltralytics.cpp
  )

set(CLASSIFIER_SOURCES ${TASKS_DIR}  
  classification/src/TensorflowClassifier.cpp
  classification/src/TorchvisionClassifier.cpp
  classification/src/ViTClassifier.cpp
  )

set(INSTANCE_SEGMENTATION_SOURCES ${TASKS_DIR}
   instance_segmentation/src/YOLOSeg.cpp
   instance_segmentation/src/RFDetrSeg.cpp
  )

set(OPTICAL_FLOW_SOURCES ${TASKS_DIR}
  optical_flow/src/RAFT.cpp
  )  

# Set the source files
set(TRITON_SOURCES ${PROJECT_SOURCE_DIR}/src/triton/Triton.cpp)

set(UTILS_SOURCES ${PROJECT_SOURCE_DIR}/src/utils/utils.cpp)

# Add configuration and logging sources
set(CONFIG_SOURCES ${PROJECT_SOURCE_DIR}/src/utils/ConfigManager.cpp)
set(LOGGER_SOURCES ${PROJECT_SOURCE_DIR}/src/utils/Logger.cpp)
set(FILESYSTEM_SOURCES ${PROJECT_SOURCE_DIR}/src/utils/FileSystem.cpp)

# Add an executable target
add_executable(${PROJECT_NAME} ${SOURCES}
    ${UTILS_SOURCES} 
    ${CONFIG_SOURCES}
    ${LOGGER_SOURCES}
    ${DETECTOR_SOURCES} 
    ${CLASSIFIER_SOURCES} 
    ${INSTANCE_SEGMENTATION_SOURCES} 
    ${OPTICAL_FLOW_SOURCES}
    ${TRITON_SOURCES}
    )

# Set include directories for the target
target_include_directories(${PROJECT_NAME}
    PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    $ENV{TritonClientBuild_DIR}/include
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/src/utils
    ${TASKS_DIR}
    ${TASKS_DIR}/object_detection/include
    ${TASKS_DIR}/object_detection/src
    ${TASKS_DIR}/classification/include
    ${TASKS_DIR}/classification/src  
    ${TASKS_DIR}/instance_segmentation/include
    ${TASKS_DIR}/instance_segmentation/src  
    ${TASKS_DIR}/optical_flow/include
    ${TASKS_DIR}/optical_flow/src
    ${PROJECT_SOURCE_DIR}/src/triton
    # Add any additional include directories here
)

# Set link directories for the target
target_link_directories(${PROJECT_NAME}
    PRIVATE
    $ENV{TritonClientBuild_DIR}/lib
    # Add any additional link directories here
)

# Link libraries for the target
target_link_libraries(${PROJECT_NAME}
    PRIVATE
    grpcclient
    httpclient
    ${OpenCV_LIBS}
    CURL::libcurl
    ${PROTOBUF_LIBRARIES}
)

# Add option for testing
option(BUILD_TESTING "Build the testing tree." OFF)

# Enable testing if requested
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
