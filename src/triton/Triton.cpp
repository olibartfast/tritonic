#include "Triton.hpp"

namespace Triton 
{
    size_t WriteCallback(char* ptr, size_t size, size_t nmemb, std::string& data) {
        size_t totalSize = size * nmemb;
        data.append(ptr, totalSize);
        return totalSize;
    }

    TritonModelInfo parseModelHttp(const std::string& modelName, const std::string& url) {
        TritonModelInfo info;

        CURL* curl = curl_easy_init();
        if (!curl) {
            std::cerr << "Failed to initialize libcurl." << std::endl;
            std::exit(1);
        }

        const auto modelConfigUrl = "http://" + url + ":8000/v2/models/" + modelName + "/config";

        // Set the URL and callback function
        curl_easy_setopt(curl, CURLOPT_URL, modelConfigUrl.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);

        // Response data will be stored in this string
        std::string responseData;

        // Set the pointer to the response data string
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &responseData);

        // Perform the request
        CURLcode res = curl_easy_perform(curl);
        if (res != CURLE_OK) {
            std::cerr << "Failed to perform request: " << curl_easy_strerror(res) << std::endl;
            std::exit(1);
        }
        if (responseData.find("Request for unknown model") != std::string::npos) {
            std::cerr << "Unknown model: " << modelName << std::endl;
            std::exit(1);
        }

        // Response was successful, responseData contains the model configuration
        // Parse the JSON response
        rapidjson::Document responseJson;
        responseJson.Parse(responseData.c_str());

        // Fill the TritonModelInfo parameters from the parsed JSON
        info.input_name_ = responseJson["input"][0]["name"].GetString();

        // After retrieving the input dimensions from the model configuration
        const auto& inputDims = responseJson["input"][0]["dims"].GetArray();

        info.input_format_ = responseJson["input"][0]["format"].GetString();

        // Fix the input format if it is "FORMAT_NONE"
        // https://github.com/triton-inference-server/server/issues/1240
        if (info.input_format_ == "FORMAT_NONE") {
            info.input_format_ = "FORMAT_NCHW"; // or hardcode the string you know
        }        
        else if (info.input_format_ == "FORMAT_NCHW")
        {
            if (inputDims.Size() == 4) {
                // Batch size is included, and it's the first dimension    
                info.input_c_ = responseJson["input"][0]["dims"][1].GetInt();
                info.input_h_ = responseJson["input"][0]["dims"][2].GetInt();
                info.input_w_ = responseJson["input"][0]["dims"][3].GetInt();

            } else if (inputDims.Size() == 3) {
                // Batch size is not included, and you can treat it as 1
                info.input_c_ = responseJson["input"][0]["dims"][0].GetInt();
                info.input_h_ = responseJson["input"][0]["dims"][1].GetInt();
                info.input_w_ = responseJson["input"][0]["dims"][2].GetInt();
                info.shape_.push_back(info.batch_size_);
            } else {
            // Handle unexpected dimension size
            // You might want to throw an exception or handle this case based on your requirements
                std::cerr << "Can't manage this input size" << std::endl;
                std::exit(1);
            }


        }
        else if (info.input_format_ == "FORMAT_NHWC")
        {
            if (inputDims.Size() == 4) {
                // Batch size is included, and it's the first dimension    
                info.input_h_ = responseJson["input"][0]["dims"][1].GetInt();
                info.input_w_ = responseJson["input"][0]["dims"][2].GetInt();
                info.input_c_ = responseJson["input"][0]["dims"][3].GetInt();

            } else if (inputDims.Size() == 3) {
                // Batch size is not included, and you can treat it as 1
                info.input_h_ = responseJson["input"][0]["dims"][0].GetInt();
                info.input_w_ = responseJson["input"][0]["dims"][1].GetInt();
                info.input_c_ = responseJson["input"][0]["dims"][2].GetInt();
                info.shape_.push_back(info.batch_size_);
            } else {
            // Handle unexpected dimension size
            // You might want to throw an exception or handle this case based on your requirements
                std::cerr << "Can't manage this input size" << std::endl;
                std::exit(1);
            }            
        }
 

        for (const auto& dim : inputDims) 
        {
            info.shape_.push_back(dim.GetInt64());
        }

        info.max_batch_size_ = responseJson["max_batch_size"].GetInt();

        for (const auto& output : responseJson["output"].GetArray()) {
            info.output_names_.push_back(output["name"].GetString());
        }

        info.input_datatype_ = responseJson["input"][0]["data_type"].GetString();

        // After retrieving the input data type from the model configuration
        // Remove the "TYPE_" prefix from the input data type
        info.input_datatype_.erase(0, 5);

        // Other parameter assignments can be added based on the JSON structure

        // Cleanup
        curl_easy_cleanup(curl);

        return info;
    }


    TritonModelInfo setModel(const std::string& modelName, const std::string& url) {
        TritonModelInfo info = parseModelHttp(modelName, url);

        // Print the retrieved model information
        std::cout << "Retrieved model information: " << std::endl;
        std::cout << "Input Name: " << info.input_name_ << std::endl;
        std::cout << "Input Data Type: " << info.input_datatype_ << std::endl;
        std::cout << "Input Channels: " << info.input_c_ << std::endl;
        std::cout << "Input Height: " << info.input_h_ << std::endl;
        std::cout << "Input Width: " << info.input_w_ << std::endl;
        std::cout << "Input Format: " << info.input_format_ << std::endl;
        std::cout << "Max Batch Size: " << info.max_batch_size_ << std::endl;

        std::cout << "Output Names: ";
        for (const auto& outputName : info.output_names_) {
            std::cout << outputName << " ";
        }
        std::cout << std::endl;

        return info;
    }

    // Function to create Triton client based on the protocol
    void createTritonClient(Triton::TritonClient& tritonClient, const std::string& url, bool verbose, Triton::ProtocolType protocol)
    {
        tc::Error err;
        if (protocol == Triton::ProtocolType::HTTP) {
            err = tc::InferenceServerHttpClient::Create(
                &tritonClient.httpClient, url, verbose);
        }
        else {
            err = tc::InferenceServerGrpcClient::Create(
                &tritonClient.grpcClient, url, verbose);
        }
        if (!err.IsOk()) {
            std::cerr << "error: unable to create client for inference: " << err
                << std::endl;
            exit(1);
        }
    }


    std::vector<const tc::InferRequestedOutput*> createInferRequestedOutput(const std::vector<std::string>& output_names_)
    {
        std::vector<const tc::InferRequestedOutput*> outputs;
        tc::Error err;
        for (const auto& output_name : output_names_) {
            tc::InferRequestedOutput* output;
            err = tc::InferRequestedOutput::Create(&output, output_name);
            if (!err.IsOk()) {
                std::cerr << "unable to get output: " << err << std::endl;
                exit(1);
            }
            else
             outputs.push_back(output);
        }
        return outputs;    
    }

    tc::InferOptions createInferOptions(const std::string& modelName, const std::string& modelVersion) 
    {
        tc::InferOptions options(modelName);
        options.model_version_ = modelVersion;
        return options;
    }

    std::tuple<std::vector<std::vector<float>> , std::vector<std::vector<int64_t>>> getInferResults(
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

        std::vector<std::vector<float>> infer_results;
        std::vector<std::vector<int64_t> > infer_shapes;

        float* outputData;
        size_t outputByteSize;
        for (auto outputName : output_names)
        {
            std::vector<int64_t> infer_shape;
            std::vector<float> infer_result;

            result->RawData(
                outputName, (const uint8_t**)&outputData, &outputByteSize);

            tc::Error err = result->Shape(outputName, &infer_shape);
            infer_result = std::vector<float>(outputByteSize / sizeof(float));
            std::memcpy(infer_result.data(), outputData, outputByteSize);
            if (!err.IsOk())
            {
                std::cerr << "unable to get data for " << outputName << std::endl;
                exit(1);
            }
            infer_results.push_back(infer_result);
            infer_shapes.push_back(infer_shape);
        }

        return make_tuple(infer_results, infer_shapes);
    }


}