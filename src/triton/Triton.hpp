#pragma once
#include "common.hpp"
#include "Yolo.hpp"
#include <curl/curl.h>
#include <rapidjson/document.h>


namespace Triton{

   struct TritonModelInfo {
        std::string output_name_;
        std::vector<std::string> output_names_;
        std::string input_name_;
        std::string input_datatype_;
        // The shape of the input
        int input_c_;
        int input_h_;
        int input_w_;
        // The format of the input
        std::string input_format_;
        int type1_{CV_32FC1};
        int type3_{CV_32FC3};
        int max_batch_size_;
        int batch_size_{1};
        std::vector<int64_t> shape_;

    };  

    enum ProtocolType { HTTP = 0, GRPC = 1 };
 
    // Callback function to handle the response data
    size_t WriteCallback(char* ptr, size_t size, size_t nmemb, std::string& data);

    TritonModelInfo parseModelHttp(const std::string& modelName, const std::string& url);


    TritonModelInfo setModel(const std::string& modelName, const std::string& url);

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

    // Function to create Triton client based on the protocol
    void createTritonClient(Triton::TritonClient& tritonClient, const std::string& url, bool verbose, Triton::ProtocolType protocol);

    std::vector<const tc::InferRequestedOutput*> createInferRequestedOutput(const std::vector<std::string>& output_names_);
    // Function to create Triton infer options
    tc::InferOptions createInferOptions(const std::string& modelName, const std::string& modelVersion);
    
    std::tuple<std::vector<std::vector<float>> , std::vector<std::vector<int64_t>>> getInferResults(
        tc::InferResult* result,
        const size_t batch_size,
        const std::vector<std::string>& output_names, const bool batching);
   

}


