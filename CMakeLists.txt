cmake_minimum_required(VERSION 3.20)

project(tritonic)
message(STATUS ${PROJECT_NAME})
message(STATUS $ENV{TritonClientBuild_DIR})
set(CMAKE_CXX_STANDARD 20)

# Fetch vision-core repository for tasks
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
  set(VISION_CORE_REPO "https://$ENV{GITHUB_TOKEN}@github.com/olibartfast/vision-core.git")
else()
  set(VISION_CORE_REPO "https://github.com/olibartfast/vision-core.git")
endif()

FetchContent_Declare(
  vision-core
  GIT_REPOSITORY ${VISION_CORE_REPO}
  GIT_TAG master
)
FetchContent_MakeAvailable(vision-core)

# Set vision-infra repository URL with token support
if(DEFINED ENV{GITHUB_TOKEN})
  set(VISION_INFRA_REPO "https://$ENV{GITHUB_TOKEN}@github.com/olibartfast/vision-infra.git")
else()
  set(VISION_INFRA_REPO "https://github.com/olibartfast/vision-infra.git")
endif()

FetchContent_Declare(
  vision-infra
  GIT_REPOSITORY ${VISION_INFRA_REPO}
  GIT_TAG master
)

# Disable warnings as errors for vision-infra
set(ENABLE_WARNINGS_SAVED ${ENABLE_WARNINGS})
set(ENABLE_WARNINGS OFF CACHE BOOL "Disable warnings for vision-infra" FORCE)
FetchContent_MakeAvailable(vision-infra)
if(DEFINED ENABLE_WARNINGS_SAVED)
  set(ENABLE_WARNINGS ${ENABLE_WARNINGS_SAVED} CACHE BOOL "Restore warnings setting" FORCE)
endif()


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

# Set the source files (only tritonic-specific sources)
set(SOURCES
    ${PROJECT_SOURCE_DIR}/src/main/client.cpp
    ${PROJECT_SOURCE_DIR}/src/main/App.cpp
    # Add any additional tritonic-specific source files here
)  

# Set the source files
set(TRITON_SOURCES ${PROJECT_SOURCE_DIR}/src/triton/Triton.cpp)

set(UTILS_SOURCES ${PROJECT_SOURCE_DIR}/src/utils/utils.cpp)


# Add an executable target
add_executable(${PROJECT_NAME}
    ${SOURCES}
    ${UTILS_SOURCES}
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
    ${PROJECT_SOURCE_DIR}/src/triton
    # vision-core and vision-infra includes are provided automatically through target_link_libraries
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
    vision-core::vision-core
    vision-infra::vision-infra
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
