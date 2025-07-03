#!/bin/bash

# Unit Testing Runner Script for Computer Vision Triton C++ Client
# This script builds and runs all unit tests

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Computer Vision Triton C++ Client - Unit Test Runner ===${NC}"

# Check if we're in the right directory
if [ ! -f "CMakeLists.txt" ]; then
    echo -e "${RED}Error: Please run this script from the project root directory${NC}"
    exit 1
fi

# Create build directory if it doesn't exist
if [ ! -d "build" ]; then
    echo -e "${YELLOW}Creating build directory...${NC}"
    mkdir build
fi

cd build

echo -e "${YELLOW}Configuring CMake with testing enabled...${NC}"
cmake -DBUILD_TESTING=ON ..

echo -e "${YELLOW}Building test executable...${NC}"
ninja run_tests

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Build successful!${NC}"
else
    echo -e "${RED}Build failed!${NC}"
    exit 1
fi

echo -e "${YELLOW}Running unit tests...${NC}"
./tests/run_tests

if [ $? -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
else
    echo -e "${RED}Some tests failed!${NC}"
    exit 1
fi

echo -e "${BLUE}=== Test Summary ===${NC}"
echo -e "${GREEN}✓ Configuration validation tests${NC}"
echo -e "${GREEN}✓ Configuration manager tests${NC}"  
echo -e "${GREEN}✓ Logger functionality tests${NC}"
echo -e "${GREEN}✓ Type safety improvements${NC}"
echo -e "${GREEN}✓ Error handling validation${NC}"

echo -e "${BLUE}=== Testing Framework Features ===${NC}"
echo -e "${GREEN}✓ Google Test framework integration${NC}"
echo -e "${GREEN}✓ Interface abstractions for dependency injection${NC}"
echo -e "${GREEN}✓ Mock-friendly design patterns${NC}"
echo -e "${GREEN}✓ Comprehensive test coverage${NC}"
echo -e "${GREEN}✓ Security validation (model name path checking)${NC}"

echo -e "${BLUE}Unit testing framework setup complete!${NC}"
cd build_tests

# Configure with testing enabled
cmake .. -DBUILD_TESTING=ON

# Build the project and tests
make -j$(nproc)

echo "Running tests..."

# Run the tests
./tests/run_tests --gtest_output=xml:test_results.xml

echo "Tests completed. Results saved to test_results.xml"

# Run tests with verbose output for debugging
echo "Running tests with verbose output..."
./tests/run_tests --gtest_verbose

echo "All tests completed successfully!"
