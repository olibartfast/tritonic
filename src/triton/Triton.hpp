#pragma once
#include "common.hpp"
#include <curl/curl.h>
#include <rapidjson/document.h>
#include <variant>

using TensorElement = std::variant<float, int32_t, int64_t>;

struct TritonModelInfo {
    std::vector<std::string> output_names;
    std::vector<std::string> input_names;
    std::vector<std::vector<int64_t>> input_shapes;
    std::vector<std::string> input_formats;
    std::vector<std::string> input_datatypes;
    std::vector<int> input_types;

    int type1_{CV_32FC1};
    int type3_{CV_32FC3};
    int max_batch_size_;
    int batch_size_{1};
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

class Triton {
private:
    TritonClient triton_client_;
    const std::string& url_; 
    bool verbose_; 
    ProtocolType protocol_;
    std::string model_name_;
    TritonModelInfo model_info_;
    std::string model_version_ ="";

    void updateInputTypes();

public:
    Triton(const std::string& url, ProtocolType protocol, std::string modelName, std::string modelVersion ="", bool verbose = false) : 
        url_{url}, 
        verbose_{verbose}, 
        protocol_{protocol},
        model_name_{modelName},
        model_version_{modelVersion}
    {
        createTritonClient();
    }

    void printModelInfo(const TritonModelInfo& model_info) const {
        std::cout << "Model Information:" << std::endl;
        std::cout << "  Inputs:" << std::endl;
        for (size_t i = 0; i < model_info.input_names.size(); ++i) {
            std::cout << "    " << model_info.input_names[i] << ":" << std::endl;
            std::cout << "      Format: " << model_info.input_formats[i] << std::endl;
            std::cout << "      Shape: [";
            for (size_t j = 0; j < model_info.input_shapes[i].size(); ++j) {
                std::cout << model_info.input_shapes[i][j];
                if (j < model_info.input_shapes[i].size() - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
            std::cout << "      Type: " << getOpenCVTypeString(model_info.input_types[i]) << std::endl;
        }
        
        std::cout << "  Outputs:" << std::endl;
        for (const auto& output_name : model_info.output_names) {
            std::cout << "    " << output_name << std::endl;
        }
        
        std::cout << "  Max Batch Size: " << model_info.max_batch_size_ << std::endl;
        std::cout << "  Batch Size: " << model_info.batch_size_ << std::endl;
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
    TritonModelInfo getModelInfo(const std::string& modelName, const std::string& url, const std::vector<std::vector<int64_t>>& input_sizes = {});

    void setInputShapes(const std::vector<std::vector<int64_t>>& shapes);
    void setInputShape(const std::vector<int64_t>& shape);

    void createTritonClient();
    std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>> infer(const std::vector<std::vector<uint8_t>>& input_data);
    std::vector<const tc::InferRequestedOutput*> createInferRequestedOutput(const std::vector<std::string>& output_names_);
    std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>> getInferResults(
        tc::InferResult* result,
        const size_t batch_size,
        const std::vector<std::string>& output_names);
};