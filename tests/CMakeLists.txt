# Test configuration
cmake_minimum_required(VERSION 3.20)

# Enable testing
enable_testing()

# Find required packages for testing
find_package(GTest REQUIRED)

# Include Google Mock (it's part of GTest)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Include test directories
include_directories(
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
    $ENV{TritonClientBuild_DIR}/include
)

# Test source files
set(TEST_SOURCES
    test_main.cpp
    test_config_basic.cpp
    test_config_manager_basic.cpp
    test_logger.cpp
    test_triton.cpp
    test_shared_memory.cpp
)

# Source files that need to be tested (without main)
set(TESTABLE_SOURCES
    ${PROJECT_SOURCE_DIR}/src/utils/ConfigManager.cpp
    ${PROJECT_SOURCE_DIR}/src/utils/Logger.cpp
    ${PROJECT_SOURCE_DIR}/src/utils/utils.cpp
    ${PROJECT_SOURCE_DIR}/src/utils/FileSystem.cpp
    ${PROJECT_SOURCE_DIR}/src/triton/Triton.cpp
)

# Create the test executable
add_executable(run_tests ${TEST_SOURCES} ${TESTABLE_SOURCES})

# Link libraries
target_link_libraries(run_tests
    PRIVATE
    GTest::gtest
    GTest::gtest_main
    grpcclient
    httpclient
    ${OpenCV_LIBS}
    CURL::libcurl
    ${PROTOBUF_LIBRARIES}
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
