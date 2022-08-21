#pragma once
#include "common.hpp"
#include "Yolo.hpp"

namespace Triton{

    enum ProtocolType { HTTP = 0, GRPC = 1 };


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
        int type1_;
        int type3_;
        int max_batch_size_;

        std::vector<int64_t> shape_;

    };


    TritonModelInfo setModel(const int batch_size){
        TritonModelInfo yoloModelInfo;
        yoloModelInfo.output_names_ = std::vector<std::string>{"output"};
        yoloModelInfo.input_name_ = "images";
        yoloModelInfo.input_datatype_ = std::string("FP32");
        // The shape of the input
        yoloModelInfo.input_c_ = Yolo::INPUT_C;
        yoloModelInfo.input_w_ = Yolo::INPUT_W;
        yoloModelInfo.input_h_ = Yolo::INPUT_H;
        // The format of the input
        yoloModelInfo.input_format_ = "FORMAT_NCHW";
        yoloModelInfo.type1_ = CV_32FC1;
        yoloModelInfo.type3_ = CV_32FC3;
        yoloModelInfo.max_batch_size_ = 32;
        yoloModelInfo.shape_.push_back(batch_size);
        yoloModelInfo.shape_.push_back(yoloModelInfo.input_c_);
        yoloModelInfo.shape_.push_back(yoloModelInfo.input_h_);
        yoloModelInfo.shape_.push_back(yoloModelInfo.input_w_);
        return yoloModelInfo;
    }

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


    auto
    getInferResults(
        tc::InferResult* result,
        const size_t batch_size,
        const std::vector<std::string>& output_names, const bool batching)
    {
        if (!result->RequestStatus().IsOk())
        {
            std::cerr << "inference  failed with error: " << result->RequestStatus()
                << std::endl;
            exit(1);
        }

        std::vector<float> infer_results;
        std::vector<int64_t> infer_shape;


        float* outputData;
        size_t outputByteSize;
        for (auto outputName : output_names)
        {
            if (outputName == "output")
            { 
                result->RawData(
                    outputName, (const uint8_t**)&outputData, &outputByteSize);

                tc::Error err = result->Shape(outputName, &infer_shape);
                infer_results = std::vector<float>(outputByteSize / sizeof(float));
                std::memcpy(infer_results.data(), outputData, outputByteSize);
                if (!err.IsOk())
                {
                    std::cerr << "unable to get data for " << outputName << std::endl;
                    exit(1);
                }
            }

        }

        return make_tuple(infer_results, infer_shape);
    }


}
