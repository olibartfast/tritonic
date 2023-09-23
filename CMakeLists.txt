cmake_minimum_required(VERSION 3.0)

project(object-detection-triton-cpp-client)
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
set(TritonCore_DIR "$ENV{TritonClientBuild_DIR}/lib/cmake/TritonCore")

find_package(OpenCV REQUIRED)
find_package(TritonCommon REQUIRED)
find_package(TritonClient REQUIRED)
find_package(CURL REQUIRED)
find_package(RapidJSON REQUIRED)

add_executable(${PROJECT_NAME} ${PROJECT_SOURCE_DIR}/client.cpp)
target_include_directories(
    ${PROJECT_NAME} 
    PRIVATE ${OpenCV_INCLUDE_DIRS} $ENV{TritonClientBuild_DIR}/include
  )
target_link_directories(${PROJECT_NAME} PRIVATE $ENV{TritonClientBuild_DIR}/lib)
target_link_libraries(${PROJECT_NAME} 
PRIVATE
grpcclient
httpclient
${OpenCV_LIBS}
CURL::libcurl
)
