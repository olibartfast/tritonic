#include <gtest/gtest.h>
#include "common.hpp"
#include "triton/Triton.hpp"
#include "Config.hpp"
#include <memory>
#include <vector>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <chrono>

class SharedMemoryTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_shm_name_ = "/test_shared_memory_" + std::to_string(getpid());
    }

    void TearDown() override {
        shm_unlink(test_shm_name_.c_str());
    }

    std::string test_shm_name_;
};

TEST_F(SharedMemoryTest, SharedMemoryRegionDefaultConstruction) {
    SharedMemoryRegion region;
    
    EXPECT_TRUE(region.name.empty());
    EXPECT_TRUE(region.key.empty());
    EXPECT_EQ(region.ptr, nullptr);
    EXPECT_EQ(region.size, 0);
    EXPECT_EQ(region.fd, -1);
    EXPECT_EQ(region.type, SharedMemoryType::SYSTEM_SHARED_MEMORY);
    EXPECT_EQ(region.device_id, 0);
}

TEST_F(SharedMemoryTest, SharedMemoryRegionCleanupSystemMemory) {
    auto region = std::make_unique<SharedMemoryRegion>();
    region->type = SharedMemoryType::SYSTEM_SHARED_MEMORY;
    region->key = test_shm_name_;
    region->size = 1024;
    
    region->fd = shm_open(region->key.c_str(), O_CREAT | O_RDWR, 0666);
    ASSERT_NE(region->fd, -1);
    
    ASSERT_EQ(ftruncate(region->fd, region->size), 0);
    
    region->ptr = mmap(nullptr, region->size, PROT_READ | PROT_WRITE, MAP_SHARED, region->fd, 0);
    ASSERT_NE(region->ptr, MAP_FAILED);
    
    region->cleanup();
    
    EXPECT_EQ(region->ptr, nullptr);
    EXPECT_EQ(region->fd, -1);
}

TEST_F(SharedMemoryTest, SharedMemoryRegionCleanupCudaMemory) {
    SharedMemoryRegion region;
    region.type = SharedMemoryType::CUDA_SHARED_MEMORY;
    region.ptr = reinterpret_cast<void*>(0x1234);  // Mock pointer
    region.fd = 5;  // Mock fd
    
    region.cleanup();
    
    EXPECT_EQ(region.ptr, nullptr);
    EXPECT_EQ(region.fd, -1);
}

class TritonSharedMemoryTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_model_info_.input_names = {"input"};
        test_model_info_.input_shapes = {{1, 3, 224, 224}};
        test_model_info_.input_datatypes = {"FP32"};
        test_model_info_.input_formats = {"FORMAT_NCHW"};
        test_model_info_.output_names = {"output"};
        test_model_info_.max_batch_size_ = 1;
    }

    TritonModelInfo test_model_info_;
};

TEST_F(TritonSharedMemoryTest, TritonConstructorWithSystemSharedMemory) {
    EXPECT_NO_THROW({
        Triton triton("localhost", ProtocolType::HTTP, "test_model", "1", false, 
                     SharedMemoryType::SYSTEM_SHARED_MEMORY, 0);
    });
}

TEST_F(TritonSharedMemoryTest, TritonConstructorWithCudaSharedMemory) {
    EXPECT_NO_THROW({
        Triton triton("localhost", ProtocolType::HTTP, "test_model", "1", false, 
                     SharedMemoryType::CUDA_SHARED_MEMORY, 0);
    });
}

class ConfigSharedMemoryTest : public ::testing::Test {
protected:
    void SetUp() override {
        config_ = std::make_unique<Config>();
        config_->server_address = "localhost";
        config_->port = 8000;
        config_->model_name = "test_model";
        config_->model_type = "yolov8";
        config_->source = "test.jpg";
    }

    std::unique_ptr<Config> config_;
};

TEST_F(ConfigSharedMemoryTest, ValidConfigWithSharedMemoryNone) {
    config_->shared_memory_type = "none";
    
    EXPECT_TRUE(config_->isValid());
    EXPECT_TRUE(config_->getValidationErrors().empty());
}

TEST_F(ConfigSharedMemoryTest, ValidConfigWithSharedMemorySystem) {
    config_->shared_memory_type = "system";
    
    EXPECT_TRUE(config_->isValid());
    EXPECT_TRUE(config_->getValidationErrors().empty());
}

TEST_F(ConfigSharedMemoryTest, ValidConfigWithSharedMemoryCuda) {
    config_->shared_memory_type = "cuda";
    config_->cuda_device_id = 0;
    
    EXPECT_TRUE(config_->isValid());
    EXPECT_TRUE(config_->getValidationErrors().empty());
}

TEST_F(ConfigSharedMemoryTest, ValidCudaDeviceId) {
    config_->shared_memory_type = "cuda";
    config_->cuda_device_id = 1;
    
    EXPECT_TRUE(config_->isValid());
    EXPECT_TRUE(config_->getValidationErrors().empty());
}

TEST_F(ConfigSharedMemoryTest, DefaultSharedMemoryConfiguration) {
    Config default_config;
    
    EXPECT_EQ(default_config.shared_memory_type, "none");
    EXPECT_EQ(default_config.cuda_device_id, 0);
}

class TritonSharedMemoryIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_model_info_.input_names = {"input"};
        test_model_info_.input_shapes = {{1, 3, 224, 224}};
        test_model_info_.input_datatypes = {"FP32"};
        test_model_info_.input_formats = {"FORMAT_NCHW"};
        test_model_info_.output_names = {"output"};
        test_model_info_.max_batch_size_ = 1;
    }

    TritonModelInfo test_model_info_;
};

class TritonTestWrapper : public Triton {
public:
    TritonTestWrapper(SharedMemoryType type = SharedMemoryType::SYSTEM_SHARED_MEMORY) 
        : Triton("localhost", ProtocolType::HTTP, "test_model", "1", false, type, 0) {}
    
    size_t testCalculateTensorSize(const std::vector<int64_t>& shape, const std::string& datatype) {
        return calculateTensorSize(shape, datatype);
    }

private:
    using Triton::calculateTensorSize;
};

TEST_F(TritonSharedMemoryIntegrationTest, CalculateTensorSizeFloat32) {
    TritonTestWrapper triton;
    
    std::vector<int64_t> shape = {1, 3, 224, 224};
    std::string datatype = "FP32";
    
    size_t expected_size = 1 * 3 * 224 * 224 * sizeof(float);
    size_t actual_size = triton.testCalculateTensorSize(shape, datatype);
    
    EXPECT_EQ(actual_size, expected_size);
}

TEST_F(TritonSharedMemoryIntegrationTest, CalculateTensorSizeInt32) {
    TritonTestWrapper triton;
    
    std::vector<int64_t> shape = {1, 1000};
    std::string datatype = "INT32";
    
    size_t expected_size = 1 * 1000 * sizeof(int32_t);
    size_t actual_size = triton.testCalculateTensorSize(shape, datatype);
    
    EXPECT_EQ(actual_size, expected_size);
}

TEST_F(TritonSharedMemoryIntegrationTest, CalculateTensorSizeInt64) {
    TritonTestWrapper triton;
    
    std::vector<int64_t> shape = {1, 100};
    std::string datatype = "INT64";
    
    size_t expected_size = 1 * 100 * sizeof(int64_t);
    size_t actual_size = triton.testCalculateTensorSize(shape, datatype);
    
    EXPECT_EQ(actual_size, expected_size);
}

TEST_F(TritonSharedMemoryIntegrationTest, CalculateTensorSizeUint8) {
    TritonTestWrapper triton;
    
    std::vector<int64_t> shape = {1, 3, 256, 256};
    std::string datatype = "UINT8";
    
    size_t expected_size = 1 * 3 * 256 * 256 * sizeof(uint8_t);
    size_t actual_size = triton.testCalculateTensorSize(shape, datatype);
    
    EXPECT_EQ(actual_size, expected_size);
}

class SharedMemoryTypeTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup test environment
    }
};

TEST_F(SharedMemoryTypeTest, SharedMemoryTypeSystemValue) {
    EXPECT_EQ(static_cast<int>(SharedMemoryType::SYSTEM_SHARED_MEMORY), 0);
}

TEST_F(SharedMemoryTypeTest, SharedMemoryTypeCudaValue) {
    EXPECT_EQ(static_cast<int>(SharedMemoryType::CUDA_SHARED_MEMORY), 1);
}

class TritonConstructorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup for constructor tests
    }
};

TEST_F(TritonConstructorTest, ConstructorWithSystemSharedMemory) {
    EXPECT_NO_THROW({
        Triton triton("localhost", ProtocolType::HTTP, "test_model", "1", false, 
                     SharedMemoryType::SYSTEM_SHARED_MEMORY, 0);
    });
}

TEST_F(TritonConstructorTest, ConstructorWithCudaSharedMemory) {
    EXPECT_NO_THROW({
        Triton triton("localhost", ProtocolType::HTTP, "test_model", "1", false, 
                     SharedMemoryType::CUDA_SHARED_MEMORY, 0);
    });
}

TEST_F(TritonConstructorTest, ConstructorWithDefaultSharedMemory) {
    EXPECT_NO_THROW({
        Triton triton("localhost", ProtocolType::HTTP, "test_model");
    });
}

class SharedMemoryErrorHandlingTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_model_info_.input_names = {"input"};
        test_model_info_.input_shapes = {{1, 3, 224, 224}};
        test_model_info_.input_datatypes = {"FP32"};
        test_model_info_.input_formats = {"FORMAT_NCHW"};
        test_model_info_.output_names = {"output"};
        test_model_info_.max_batch_size_ = 1;
    }

    TritonModelInfo test_model_info_;
};

TEST_F(SharedMemoryErrorHandlingTest, InferWithSharedMemoryWithoutRegistration) {
    Triton triton("localhost", ProtocolType::HTTP, "test_model");
    
    std::vector<std::vector<uint8_t>> input_data = {
        std::vector<uint8_t>(1 * 3 * 224 * 224 * sizeof(float), 0)
    };
    
    EXPECT_THROW({
        triton.inferWithSharedMemory(input_data);
    }, std::runtime_error);
}

TEST_F(SharedMemoryErrorHandlingTest, SharedMemoryTypeValidation) {
    // Test that both shared memory types can be constructed
    EXPECT_NO_THROW({
        SharedMemoryRegion sys_region;
        sys_region.type = SharedMemoryType::SYSTEM_SHARED_MEMORY;
        
        SharedMemoryRegion cuda_region; 
        cuda_region.type = SharedMemoryType::CUDA_SHARED_MEMORY;
    });
}

class SharedMemoryPerformanceTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup for performance tests
    }
    
    void TearDown() override {
        // Cleanup after performance tests
    }
};

TEST_F(SharedMemoryPerformanceTest, SharedMemoryCreationPerformance) {
    const size_t num_iterations = 100;
    const size_t memory_size = 1024 * 1024;  // 1MB
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < num_iterations; ++i) {
        SharedMemoryRegion region;
        region.type = SharedMemoryType::SYSTEM_SHARED_MEMORY;
        region.key = "/test_perf_" + std::to_string(i);
        region.size = memory_size;
        region.name = "perf_test_" + std::to_string(i);
        
        region.fd = shm_open(region.key.c_str(), O_CREAT | O_RDWR, 0666);
        if (region.fd != -1) {
            ftruncate(region.fd, region.size);
            region.ptr = mmap(nullptr, region.size, PROT_READ | PROT_WRITE, MAP_SHARED, region.fd, 0);
            region.cleanup();
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Performance should complete within reasonable time (adjust as needed)
    EXPECT_LT(duration.count(), 5000);  // 5 seconds max for 100 iterations
}

class SharedMemoryEdgeCasesTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup for edge case tests
    }
};

TEST_F(SharedMemoryEdgeCasesTest, ZeroSizeSharedMemory) {
    SharedMemoryRegion region;
    region.type = SharedMemoryType::SYSTEM_SHARED_MEMORY;
    region.key = "/test_zero_size";
    region.size = 0;
    
    // Should handle zero size gracefully
    region.cleanup();
    EXPECT_EQ(region.ptr, nullptr);
    EXPECT_EQ(region.fd, -1);
}

TEST_F(SharedMemoryEdgeCasesTest, LargeSharedMemorySize) {
    const size_t large_size = 1024 * 1024 * 1024;  // 1GB
    
    // Test that large sizes are handled properly (may fail due to system limits)
    EXPECT_NO_THROW({
        SharedMemoryRegion region;
        region.type = SharedMemoryType::SYSTEM_SHARED_MEMORY;
        region.key = "/test_large";
        region.size = large_size;
        region.name = "large_test";
        
        // This may fail on systems with limited shared memory, which is expected
        region.fd = shm_open(region.key.c_str(), O_CREAT | O_RDWR, 0666);
        if (region.fd != -1) {
            // Only continue if we can create the shared memory object
            if (ftruncate(region.fd, region.size) == 0) {
                region.ptr = mmap(nullptr, region.size, PROT_READ | PROT_WRITE, MAP_SHARED, region.fd, 0);
            }
            region.cleanup();
        }
    });
}