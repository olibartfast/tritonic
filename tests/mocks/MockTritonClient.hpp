#pragma once

#include <gmock/gmock.h>
#include "ITritonClient.hpp"

/**
 * Mock implementation of ITritonClient for unit testing
 */
class MockTritonClient : public ITritonClient {
public:
    MOCK_METHOD(TritonModelInfo, getModelInfo, 
                (const std::string& modelName, const std::string& url, 
                 const std::vector<std::vector<int64_t>>& input_sizes), (override));
    
    MOCK_METHOD((std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>>), 
                infer, (const std::vector<std::vector<uint8_t>>& input_data), (override));
    
    MOCK_METHOD(void, setInputShapes, (const std::vector<std::vector<int64_t>>& shapes), (override));
    
    MOCK_METHOD(void, printModelInfo, (const TritonModelInfo& model_info), (const, override));
};

/**
 * Mock factory for creating mock Triton clients
 */
class MockTritonClientFactory : public ITritonClientFactory {
public:
    MOCK_METHOD(std::unique_ptr<ITritonClient>, createClient,
                (const std::string& url, int protocol, const std::string& modelName,
                 const std::string& modelVersion, bool verbose), (override));
    
    // Helper method to create a mock client and set expectations
    std::unique_ptr<MockTritonClient> createMockClient() {
        return std::make_unique<MockTritonClient>();
    }
};
