#pragma once

#include "ITritonClient.hpp"
#include "Triton.hpp"

/**
 * Adapter class that wraps the existing Triton class to implement ITritonClient interface
 */
class TritonClientAdapter : public ITritonClient {
private:
    std::unique_ptr<Triton> triton_client_;

public:
    TritonClientAdapter(std::unique_ptr<Triton> triton_client)
        : triton_client_(std::move(triton_client)) {}

    TritonModelInfo getModelInfo(const std::string& modelName, 
                               const std::string& url, 
                               const std::vector<std::vector<int64_t>>& input_sizes = {}) override {
        return triton_client_->getModelInfo(modelName, url, input_sizes);
    }
    
    std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>> 
    infer(const std::vector<std::vector<uint8_t>>& input_data) override {
        return triton_client_->infer(input_data);
    }
    
    void setInputShapes(const std::vector<std::vector<int64_t>>& shapes) override {
        triton_client_->setInputShapes(shapes);
    }
    
    void printModelInfo(const TritonModelInfo& model_info) const override {
        triton_client_->printModelInfo(model_info);
    }
};

/**
 * Real factory implementation for creating Triton clients
 */
class TritonClientFactory : public ITritonClientFactory {
public:
    std::unique_ptr<ITritonClient> createClient(
        const std::string& url,
        int protocol,
        const std::string& modelName,
        const std::string& modelVersion = "",
        bool verbose = false) override {
        
        auto triton = std::make_unique<Triton>(
            url, 
            static_cast<ProtocolType>(protocol), 
            modelName, 
            modelVersion, 
            verbose
        );
        
        return std::make_unique<TritonClientAdapter>(std::move(triton));
    }
};
