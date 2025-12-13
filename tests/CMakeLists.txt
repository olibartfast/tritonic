# Test configuration
cmake_minimum_required(VERSION 3.20)

# Enable testing
enable_testing()

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.14.0
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Test source files
set(TEST_SOURCES
    test_main.cpp
    test_config_basic.cpp
    test_config_manager_basic.cpp
    test_logger.cpp
    test_triton.cpp
    test_classification_preprocess.cpp
    test_app.cpp
    test_rfdetr_repro.cpp
    test_optical_flow.cpp
    test_shared_memory.cpp
)

# Source files that need to be tested (without main)
set(TESTABLE_SOURCES
    ${PROJECT_SOURCE_DIR}/src/utils/ConfigManager.cpp
    ${PROJECT_SOURCE_DIR}/src/utils/Logger.cpp
    ${PROJECT_SOURCE_DIR}/src/utils/utils.cpp
    ${PROJECT_SOURCE_DIR}/src/utils/FileSystem.cpp
    ${PROJECT_SOURCE_DIR}/src/main/App.cpp
    ${PROJECT_SOURCE_DIR}/src/triton/Triton.cpp
)

# Create the test executable
add_executable(run_tests ${TEST_SOURCES} ${TESTABLE_SOURCES})

# IMPORTANT: The Triton client libs include bundled gmock/gtest headers that conflict
# with the FetchContent'd googletest. Use SYSTEM to make TritonClient a system include
# which is searched AFTER normal includes. Place googletest BEFORE all other includes.
target_include_directories(run_tests BEFORE
    PRIVATE
    ${googletest_SOURCE_DIR}/googletest/include
    ${googletest_SOURCE_DIR}/googlemock/include
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
    ${PROJECT_SOURCE_DIR}/tests/mocks
    ${OpenCV_INCLUDE_DIRS}
)

# Add TritonClient as SYSTEM include so it's searched AFTER googletest includes
target_include_directories(run_tests SYSTEM
    PRIVATE
    $ENV{TritonClientBuild_DIR}/include
)

# Link libraries
target_link_libraries(run_tests
    PRIVATE
    gtest
    gtest_main
    gmock
    grpcclient
    httpclient
    ${OpenCV_LIBS}
    CURL::libcurl
    ${PROTOBUF_LIBRARIES}
    vision-core::vision-core
    fmt::fmt
)

# Set the same compile definitions as the main project
target_compile_definitions(run_tests PRIVATE 
    $<$<BOOL:${WITH_SHOW_FRAME}>:SHOW_FRAME>
    $<$<BOOL:${WITH_WRITE_FRAME}>:WRITE_FRAME>
)

# Set link directories
target_link_directories(run_tests
    PRIVATE
    $ENV{TritonClientBuild_DIR}/lib
)

# Add test
add_test(NAME UnitTests COMMAND run_tests)

# Set test properties
set_tests_properties(UnitTests PROPERTIES
    TIMEOUT 300
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)
