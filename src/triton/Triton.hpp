#pragma once
#include "TritonModelInfo.hpp"
#include "ITriton.hpp"
#include <vision-infra/vision-infra.hpp>
#include <opencv2/opencv.hpp>
#include <curl/curl.h>
#include <rapidjson/document.h>
#include <variant>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

// Include heavy Triton dependencies only where needed
#include "grpc_client.h"
#include "http_client.h"
namespace tc = triton::client;

#ifdef TRITONIC_ENABLE_CUDA
#include <cuda_runtime.h>
#include <cuda.h>
#endif


enum class SharedMemoryType {
    SYSTEM_SHARED_MEMORY,  // Host CPU shared memory (POSIX)
    CUDA_SHARED_MEMORY     // GPU shared memory (CUDA)
};

struct SharedMemoryRegion {
    std::string name;
    std::string key;
    void* ptr;
    size_t size;
    int fd;
    SharedMemoryType type;
    int device_id;  // For CUDA shared memory
    
    SharedMemoryRegion() : ptr(nullptr), size(0), fd(-1), type(SharedMemoryType::SYSTEM_SHARED_MEMORY), device_id(0) {}
    ~SharedMemoryRegion() {
        cleanup();
    }
    
    void cleanup() {
        if (type == SharedMemoryType::SYSTEM_SHARED_MEMORY) {
            if (ptr && ptr != MAP_FAILED) {
                munmap(ptr, size);
            }
            if (fd != -1) {
                close(fd);
                shm_unlink(key.c_str());
            }
        } else if (type == SharedMemoryType::CUDA_SHARED_MEMORY) {
            // CUDA shared memory cleanup will be handled by CUDA runtime
            // when the context is destroyed
        }
        ptr = nullptr;
        fd = -1;
    }
};

 
union TritonClient
{
    TritonClient()
    {
        new (&httpClient) std::unique_ptr<tc::InferenceServerHttpClient>{};
    }
    ~TritonClient() {}

    std::unique_ptr<tc::InferenceServerHttpClient> httpClient;
    std::unique_ptr<tc::InferenceServerGrpcClient> grpcClient;
};

enum ProtocolType { HTTP = 0, GRPC = 1 };

class Triton : public ITriton {
private:
    TritonClient triton_client_;
    std::string url_; 
    bool verbose_; 
    ProtocolType protocol_;
    std::string model_name_;
    TritonModelInfo model_info_;
    std::string model_version_ ="";
    
    // Shared memory regions
    std::vector<std::unique_ptr<SharedMemoryRegion>> input_shm_regions_;
    std::vector<std::unique_ptr<SharedMemoryRegion>> output_shm_regions_;
    SharedMemoryType shared_memory_type_;
    int cuda_device_id_;

    void updateInputTypes();

public:
    Triton(const std::string& url, ProtocolType protocol, std::string modelName, std::string modelVersion ="", bool verbose = false, SharedMemoryType shm_type = SharedMemoryType::SYSTEM_SHARED_MEMORY, int cuda_device = 0) : 
        url_{url}, 
        verbose_{verbose}, 
        protocol_{protocol},
        model_name_{modelName},
        model_version_{modelVersion},
        shared_memory_type_{shm_type},
        cuda_device_id_{cuda_device}
    {
        createTritonClient();
    }

    ~Triton() {
        unregisterSharedMemory();
    }

    void printModelInfo(const TritonModelInfo& model_info) const override {
        auto logger = std::dynamic_pointer_cast<vision_infra::core::Logger>(
            vision_infra::core::LoggerManager::GetLogger("triton"));
        logger->Info("Model Information:");
        logger->Info("  Inputs:");
        for (size_t i = 0; i < model_info.input_names.size(); ++i) {
            logger->Info("    " + model_info.input_names[i] + ":");
            logger->Info("      Format: " + model_info.input_formats[i]);
            std::string shape_str = "      Shape: [";
            for (size_t j = 0; j < model_info.input_shapes[i].size(); ++j) {
                shape_str += std::to_string(model_info.input_shapes[i][j]);
                if (j < model_info.input_shapes[i].size() - 1) shape_str += ", ";
            }
            shape_str += "]";
            logger->Info(shape_str);
            logger->Info("      Type: " + getOpenCVTypeString(model_info.input_types[i]));
        }
        
        logger->Info("  Outputs:");
        for (const auto& output_name : model_info.output_names) {
            logger->Info("    " + output_name);
        }
        
        logger->Info("  Max Batch Size: " + std::to_string(model_info.max_batch_size_));
        logger->Info("  Batch Size: " + std::to_string(model_info.batch_size_));
    }

    std::string getOpenCVTypeString(int type) const {
        switch(type) {
            case CV_8U:  return "CV_8U";
            case CV_8S:  return "CV_8S";
            case CV_16U: return "CV_16U";
            case CV_16S: return "CV_16S";
            case CV_32S: return "CV_32S";
            case CV_32F: return "CV_32F";
            case CV_64F: return "CV_64F";
            default:     return "Unknown";
        }
    }    

    TritonModelInfo parseModelHttp(const std::string& modelName, const std::string& url, const std::vector<std::vector<int64_t>>& input_sizes = {});
    TritonModelInfo parseModelGrpc(const inference::ModelMetadataResponse& model_metadata, const inference::ModelConfigResponse& model_config);
    TritonModelInfo getModelInfo(const std::string& modelName, const std::string& url, const std::vector<std::vector<int64_t>>& input_sizes = {}) override;

    void setInputShapes(const std::vector<std::vector<int64_t>>& shapes) override;
    void setInputShape(const std::vector<int64_t>& shape) override;

    void createTritonClient() override;
    std::vector<Tensor> infer(const std::vector<std::vector<uint8_t>>& input_data) override;
    std::vector<Tensor> getInferResults(
        tc::InferResult* result,
        const size_t batch_size,
        const std::vector<std::string>& output_names);
    
    // Shared memory methods
    std::unique_ptr<SharedMemoryRegion> createSharedMemoryRegion(const std::string& name, size_t size);
    std::unique_ptr<SharedMemoryRegion> createSystemSharedMemoryRegion(const std::string& name, size_t size);
    std::unique_ptr<SharedMemoryRegion> createCudaSharedMemoryRegion(const std::string& name, size_t size);
    void registerInputSharedMemory() override;
    void registerOutputSharedMemory() override;
    void unregisterSharedMemory() override;
    std::vector<Tensor> inferWithSharedMemory(const std::vector<std::vector<uint8_t>>& input_data) override;
    size_t calculateTensorSize(const std::vector<int64_t>& shape, const std::string& datatype);
    void setSharedMemoryType(SharedMemoryType type, int cuda_device = 0);
};