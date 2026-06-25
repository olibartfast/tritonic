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

FetchContent_Declare(
  neuriplo-tasks
  GIT_REPOSITORY ${NEURIPLO_TASKS_REPO}
  GIT_TAG v0.5.0
)
FetchContent_MakeAvailable(neuriplo-tasks)


option(WITH_SHOW_FRAME "Option description" OFF)
option(WITH_WRITE_FRAME "Option description" ON)
option(WITH_CHAT_BACKEND "Build OpenAI-compatible chat backend (requires CURL, already a dep)" ON)
option(TRITONIC_STRICT_WARNINGS "Enable strict compiler warnings for tritonic targets." OFF)

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
    ${TRITON_CLIENT_BUILD_DIR}/include
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/src/utils
    ${PROJECT_SOURCE_DIR}/src/triton
    ${PROJECT_SOURCE_DIR}/src/chat
    # neuriplo-tasks includes are provided automatically through target_link_libraries
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
    neuriplo-tasks::neuriplo-tasks
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
            ${INFRA_SOURCES}
            ${CHAT_SOURCES}
        )
        tritonic_apply_strict_warnings(tritonic_unit_tests)

        target_include_directories(tritonic_unit_tests PRIVATE
            ${PROJECT_SOURCE_DIR}/include
            ${PROJECT_SOURCE_DIR}/src
            ${PROJECT_SOURCE_DIR}/src/triton
            ${PROJECT_SOURCE_DIR}/src/chat
            ${PROJECT_SOURCE_DIR}/src/common
            ${PROJECT_SOURCE_DIR}/tests
            ${PROJECT_SOURCE_DIR}/tests/mocks
            ${TRITON_CLIENT_BUILD_DIR}/include
            ${OpenCV_INCLUDE_DIRS}
        )

        target_link_directories(tritonic_unit_tests PRIVATE
            ${TRITON_CLIENT_BUILD_DIR}/lib
        )
        
        target_link_libraries(tritonic_unit_tests PRIVATE
            gtest gtest_main gmock
            grpcclient
            httpclient
            ${OpenCV_LIBS}
            CURL::libcurl
            ${PROTOBUF_LIBRARIES}
        )
        
        include(GoogleTest)
        gtest_discover_tests(tritonic_unit_tests
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
            PROPERTIES TIMEOUT 60
        )
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
