cmake_minimum_required(VERSION 3.0)

project(yolo-triton-cpp-client)
message(STATUS ${PROJECT_NAME}) 
message(STATUS $ENV{TritonClientThirdParty_DIR})
message(STATUS $ENV{TritonClientBuild_DIR})
set(CMAKE_CXX_STANDARD 17)


option(WITH_SHOW_FRAME "Option description" OFF)
option(WITH_WRITE_FRAME "Option description" OFF)

if(WITH_SHOW_FRAME)
  add_definitions(-DSHOW_FRAME)
  message(STATUS "Show frame enabled") 
endif()
if(WITH_WRITE_FRAME)
  add_definitions(-DWRITE_FRAME)
  message(STATUS "Write frame enabled") 
endif()


set(CURL_DIR "$ENV{TritonClientThirdParty_DIR}/curl/lib/cmake/CURL")
set(Protobuf_DIR "$ENV{TritonClientThirdParty_DIR}/protobuf/lib/cmake/protobuf")
set(gRPC_DIR "$ENV{TritonClientThirdParty_DIR}/grpc/lib/cmake/grpc")
set(c-ares_DIR "$ENV{TritonClientThirdParty_DIR}/c-ares/lib/cmake/c-ares")
set(absl_DIR "$ENV{TritonClientThirdParty_DIR}/absl/lib/cmake/absl")
set(TritonClient_DIR "$ENV{TritonClientBuild_DIR}/lib/cmake/TritonClient")
set(TritonCommon_DIR "$ENV{TritonClientBuild_DIR}/lib/cmake/TritonCommon")


find_package(OpenCV REQUIRED)
find_package(TritonCommon REQUIRED)
find_package(TritonClient REQUIRED)


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
)
