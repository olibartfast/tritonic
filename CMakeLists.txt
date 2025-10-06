cmake_minimum_required(VERSION 3.20)

project(tritonic)
message(STATUS ${PROJECT_NAME})
message(STATUS $ENV{TritonClientBuild_DIR})
set(CMAKE_CXX_STANDARD 20)

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
    ${PROJECT_SOURCE_DIR}/src/tasks/task_factory.cpp
    # Add any additional source files here
)

set(DETECTOR_SOURCES ${PROJECT_SOURCE_DIR}  
  src/tasks/object_detection/src/YOLO.cpp 
  src/tasks/object_detection/src/YoloNas.cpp
  src/tasks/object_detection/src/YOLOv10.cpp 
  src/tasks/object_detection/src/RTDetr.cpp
  src/tasks/object_detection/src/RFDetr.cpp
  src/tasks/object_detection/src/RTDetrUltralytics.cpp
  )

set(CLASSIFIER_SOURCES ${PROJECT_SOURCE_DIR}  
  src/tasks/classification/src/TensorflowClassifier.cpp
  src/tasks/classification/src/TorchvisionClassifier.cpp
  src/tasks/classification/src/ViTClassifier.cpp
  )

set(INSTANCE_SEGMENTATION_SOURCES ${PROJECT_SOURCE_DIR}
   src/tasks/instance_segmentation/src/YOLOSeg.cpp
   src/tasks/instance_segmentation/src/RFDetrSeg.cpp
  )

set(OPTICAL_FLOW_SOURCES ${PROJECT_SOURCE_DIR}
  src/tasks/optical_flow/src/RAFT.cpp
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
    ${PROJECT_SOURCE_DIR}/src/tasks
    ${PROJECT_SOURCE_DIR}/src/tasks/object_detection/include
    ${PROJECT_SOURCE_DIR}/src/tasks/object_detection/src
    ${PROJECT_SOURCE_DIR}/src/tasks/classification/include
    ${PROJECT_SOURCE_DIR}/src/tasks/classification/src  
    ${PROJECT_SOURCE_DIR}/src/tasks/instance_segmentation/include
    ${PROJECT_SOURCE_DIR}/src/tasks/instance_segmentation/src  
    ${PROJECT_SOURCE_DIR}/src/tasks/optical_flow/include
    ${PROJECT_SOURCE_DIR}/src/tasks/optical_flow/src
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
