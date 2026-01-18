cmake_minimum_required(VERSION 3.20)

project(tritonic)
message(STATUS ${PROJECT_NAME})
set(CMAKE_CXX_STANDARD 20)

set(TRITON_CLIENT_BUILD_DIR "$ENV{TritonClientBuild_DIR}" CACHE PATH "Path to Triton client build directory")

# Set default if not provided
if(NOT TRITON_CLIENT_BUILD_DIR OR TRITON_CLIENT_BUILD_DIR STREQUAL "")
    set(TRITON_CLIENT_BUILD_DIR "${CMAKE_SOURCE_DIR}/triton_client_libs/install" CACHE PATH "Path to Triton client build directory" FORCE)
    message(STATUS "TritonClientBuild_DIR not set, using default: ${TRITON_CLIENT_BUILD_DIR}")
endif()

message(STATUS "Using Triton Client Build Dir: ${TRITON_CLIENT_BUILD_DIR}")

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

set(TritonClient_DIR "${TRITON_CLIENT_BUILD_DIR}/lib/cmake/TritonClient")
set(TritonCommon_DIR "${TRITON_CLIENT_BUILD_DIR}/lib/cmake/TritonCommon")


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
set(TRITON_SOURCES 
    ${PROJECT_SOURCE_DIR}/src/triton/Triton.cpp
    ${PROJECT_SOURCE_DIR}/src/triton/TritonClientWrapper.cpp
)

# Utils removed - using vision-infra utilities instead


# Add an executable target
add_executable(${PROJECT_NAME}
    ${SOURCES}
    ${TRITON_SOURCES}
)

# Set include directories for the target
target_include_directories(${PROJECT_NAME}
    PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${TRITON_CLIENT_BUILD_DIR}/include
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/src/utils
    ${PROJECT_SOURCE_DIR}/src/triton
    # vision-core and vision-infra includes are provided automatically through target_link_libraries
)

# Set link directories for the target
target_link_directories(${PROJECT_NAME}
    PRIVATE
    ${TRITON_CLIENT_BUILD_DIR}/lib
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
    enable_testing()
    
    # Fetch GoogleTest (reuse FetchContent from above if already included)
    if(NOT TARGET gtest)
        include(FetchContent)
        FetchContent_Declare(
          googletest
          GIT_REPOSITORY https://github.com/google/googletest.git
          GIT_TAG v1.14.0
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
            src/triton/Triton.cpp
            src/triton/TritonClientWrapper.cpp
        )
        
        target_include_directories(tritonic_unit_tests PRIVATE
            ${PROJECT_SOURCE_DIR}/include
            ${PROJECT_SOURCE_DIR}/src
            ${PROJECT_SOURCE_DIR}/src/triton
            ${PROJECT_SOURCE_DIR}/tests/mocks
            ${TRITON_CLIENT_BUILD_DIR}/include
            ${OpenCV_INCLUDE_DIRS}
        )

        target_link_directories(tritonic_unit_tests PRIVATE
            ${TRITON_CLIENT_BUILD_DIR}/lib
        )
        
        target_link_libraries(tritonic_unit_tests PRIVATE
            gtest gtest_main gmock
            vision-infra::vision-infra
            grpcclient
            httpclient
            ${OpenCV_LIBS}
            CURL::libcurl
            ${PROTOBUF_LIBRARIES}
        )
        
        add_test(NAME UnitTests COMMAND tritonic_unit_tests)
        set_tests_properties(UnitTests PROPERTIES
            TIMEOUT 60
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        )
    endif()
    
    # Create integration test executable if integration tests exist  
    if(INTEGRATION_TEST_SOURCES)
        add_executable(tritonic_integration_tests
            ${TEST_MAIN}
            ${INTEGRATION_TEST_SOURCES}
            # Integration tests can include project sources for full testing
            src/triton/Triton.cpp
            src/main/App.cpp
        )
        
        target_include_directories(tritonic_integration_tests PRIVATE
            ${PROJECT_SOURCE_DIR}/include
            ${PROJECT_SOURCE_DIR}/src
            ${PROJECT_SOURCE_DIR}/src/triton
            ${PROJECT_SOURCE_DIR}/tests/mocks
            ${OpenCV_INCLUDE_DIRS}
        )
        
        target_link_libraries(tritonic_integration_tests PRIVATE
            gtest gtest_main gmock
            vision-core::vision-core
            vision-infra::vision-infra
            grpcclient httpclient
            ${OpenCV_LIBS}
            CURL::libcurl
            ${PROTOBUF_LIBRARIES}
        )
        
        add_test(NAME IntegrationTests COMMAND tritonic_integration_tests)
        set_tests_properties(IntegrationTests PROPERTIES
            TIMEOUT 120
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        )
    endif()
endif()
