#pragma once
#include "common.hpp"
#include "TritonModelInfo.hpp"
#include "ITriton.hpp"
#include "Logger.hpp"
#include <curl/curl.h>
#include <rapidjson/document.h>
#include <variant>

using TensorElement = std::variant<float, int32_t, int64_t, uint8_t>;

 
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

    void printModelInfo(const TritonModelInfo& model_info) const override {
        logger.info("Model Information:");
        logger.info("  Inputs:");
        for (size_t i = 0; i < model_info.input_names.size(); ++i) {
            logger.infof("    {}:", model_info.input_names[i]);
            logger.infof("      Format: {}", model_info.input_formats[i]);
            std::string shape_str = "      Shape: [";
            for (size_t j = 0; j < model_info.input_shapes[i].size(); ++j) {
                shape_str += std::to_string(model_info.input_shapes[i][j]);
                if (j < model_info.input_shapes[i].size() - 1) shape_str += ", ";
            }
            shape_str += "]";
            logger.info(shape_str);
            logger.infof("      Type: {}", getOpenCVTypeString(model_info.input_types[i]));
        }
        
        logger.info("  Outputs:");
        for (const auto& output_name : model_info.output_names) {
            logger.infof("    {}", output_name);
        }
        
        logger.infof("  Max Batch Size: {}", model_info.max_batch_size_);
        logger.infof("  Batch Size: {}", model_info.batch_size_);
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
    std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>> infer(const std::vector<std::vector<uint8_t>>& input_data) override;
    std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>> getInferResults(
        tc::InferResult* result,
        const size_t batch_size,
        const std::vector<std::string>& output_names);
};