cmake_minimum_required(VERSION 3.0)

project(computer-vision-triton-cpp-client)
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
find_package(RapidJSON REQUIRED)

# Set the source files
set(SOURCES
    ${PROJECT_SOURCE_DIR}/client.cpp
    # Add any additional source files here
)

set(DETECTOR_SOURCES ${PROJECT_SOURCE_DIR} src/tasks/object_detection/src/Yolo.cpp src/tasks/object_detection/src/YoloNas.cpp)

# Set the source files
set(TRITON_SOURCES ${PROJECT_SOURCE_DIR}/src/triton/Triton.cpp)

# Add an executable target
add_executable(${PROJECT_NAME} ${SOURCES} ${DETECTOR_SOURCES} ${TRITON_SOURCES})

# Set include directories for the target
target_include_directories(${PROJECT_NAME}
    PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    $ENV{TritonClientBuild_DIR}/include
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/src/tasks/object_detection/include
    ${PROJECT_SOURCE_DIR}/src/tasks/object_detection/src
    ${PROJECT_SOURCE_DIR}/src/tasks/classification/include
    ${PROJECT_SOURCE_DIR}/src/tasks/classification/src    
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
)
