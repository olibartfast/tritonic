#include "Triton.hpp"

static size_t WriteCallback(char* ptr, size_t size, size_t nmemb, std::string& data) {
    size_t totalSize = size * nmemb;
    data.append(ptr, totalSize);
    return totalSize;
}

TritonModelInfo Triton::parseModelGrpc(const inference::ModelMetadataResponse& model_metadata, const inference::ModelConfigResponse& model_config) {
    TritonModelInfo info;

    // TO DO: Parse model metadata and config here
    return info;
}
TritonModelInfo Triton::parseModelHttp(const std::string& modelName, const std::string& url, const std::vector<std::vector<int64_t>>& input_sizes) {
    TritonModelInfo info;

    CURL* curl = curl_easy_init();
    if (!curl) {
        throw std::runtime_error("Failed to initialize libcurl.");
    }

    const auto modelConfigUrl = "http://" + url + ":8000/v2/models/" + modelName + "/config";

    curl_easy_setopt(curl, CURLOPT_URL, modelConfigUrl.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);

    std::string responseData;
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &responseData);

    CURLcode res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
        curl_easy_cleanup(curl);
        throw std::runtime_error("Failed to perform request: " + std::string(curl_easy_strerror(res)));
    }

    curl_easy_cleanup(curl);

    if (responseData.find("Request for unknown model") != std::string::npos) {
        throw std::runtime_error("Unknown model: " + modelName);
    }

    rapidjson::Document responseJson;
    responseJson.Parse(responseData.c_str());

    std::string platform = responseJson.HasMember("platform") && responseJson["platform"].IsString() 
                           ? responseJson["platform"].GetString() : "";

    info.max_batch_size_ = responseJson["max_batch_size"].GetInt();

    const auto& inputs = responseJson["input"].GetArray();
    for (size_t i = 0; i < inputs.Size(); ++i) {
        const auto& input = inputs[i];
        info.input_names.push_back(input["name"].GetString());
        
        std::string format = input["format"].GetString();
        if (format == "FORMAT_NONE") {
            format = (platform == "tensorflow_savedmodel") ? "FORMAT_NHWC" : "FORMAT_NCHW";
        }
        info.input_formats.push_back(format);

        std::vector<int64_t> shape;
        const auto& inputDims = input["dims"].GetArray();
        bool hasDynamicDim = false;
        for (const auto& dim : inputDims) {
            int64_t dimValue = dim.GetInt64();
            if (dimValue == -1) {
                hasDynamicDim = true;
            }
            shape.push_back(dimValue);
        }

        if (hasDynamicDim) {
            if (input_sizes.empty() || i >= input_sizes.size()) {
                throw std::runtime_error("Dynamic input dimension detected for input " + std::to_string(i) + 
                                         ", but no input sizes provided. Please specify input sizes.");
            }
            shape = input_sizes[i];
        } else if (!input_sizes.empty()) {
            std::cout << "Warning: Input sizes provided, but model does not have dynamic shapes. Ignoring provided input sizes." << std::endl;
        }

        // Handle batch size
        if (info.max_batch_size_ > 0 && shape.size() < 4) {
            shape.insert(shape.begin(), 1);  // Insert batch size of 1 at the beginning
        }

        info.input_shapes.push_back(shape);

        std::string datatype = input["data_type"].GetString();
        datatype.erase(0, 5); // Remove the "TYPE_" prefix
        info.input_datatypes.push_back(datatype);
        if (datatype == "FP32") {
            info.input_types.push_back(CV_32F);
        } else if (datatype == "INT32") {
            info.input_types.push_back(CV_32S);
        } else if (datatype == "INT64") {
            info.input_types.push_back(CV_32S);  // Map INT64 to CV_32S
            std::cerr << "Warning: INT64 type detected for input '" << info.input_names.back() 
                      << "'. Will be mapped to CV_32S." << std::endl;
        } else {
            throw std::runtime_error("Unsupported data type: " + datatype);
        }
    }

    for (const auto& output : responseJson["output"].GetArray()) {
        info.output_names.push_back(output["name"].GetString());
    }

    return info;
}

TritonModelInfo Triton::getModelInfo(const std::string& modelName, const std::string& url, const std::vector<std::vector<int64_t>>& input_sizes) {
    TritonModelInfo model_info;

    if (protocol_ == ProtocolType::HTTP) {
        model_info = parseModelHttp(modelName, url, input_sizes);
    } else if (protocol_ == ProtocolType::GRPC) {
        // TODO model_info = parseModelGrpc(modelName);
        model_info = parseModelHttp(modelName, url, input_sizes);
    } else {
        throw std::runtime_error("Unsupported protocol");
    }


    // Update input types based on the parsed information
    updateInputTypes();

    // Print the retrieved model information
    printModelInfo(model_info);
    this->model_info_ = model_info; 
    return model_info;
}

void Triton::updateInputTypes() 
{
    for (size_t i = 0; i < model_info_.input_shapes.size(); ++i) 
    {
        const auto& shape = model_info_.input_shapes[i];
        const auto& format = model_info_.input_formats[i];
        
        int inputType;
        
        if (format == "FORMAT_NCHW" || format == "FORMAT_NHWC") {
            if (shape.size() == 4) {
                int channels = (format == "FORMAT_NCHW") ? shape[1] : shape[3];
                if (channels == 1) {
                    inputType = model_info_.type1_;
                } else if (channels == 3) {
                    inputType = model_info_.type3_;
                } else {
                    inputType = CV_32F; // Default to float type if channel count is unexpected
                }
            } else if (shape.size() == 2) {
                inputType = model_info_.type1_; // Assume grayscale for 2D input
            } else {
                inputType = CV_32F; // Default to float type for other dimensions
            }
        } else {
            inputType = CV_32F; // Default to float type for other formats
        }
        
        model_info_.input_types[i] = inputType;
    }
}

// Updated implementation of setInputShapes
void Triton::setInputShapes(const std::vector<std::vector<int64_t>>& shapes) {
    if (shapes.size() != model_info_.input_shapes.size()) {
        throw std::runtime_error("Number of provided input shapes does not match model's input count");
    }

    for (size_t i = 0; i < shapes.size(); ++i) {
        if (shapes[i].size() != model_info_.input_shapes[i].size()) {
            throw std::runtime_error("Provided input shape does not match model's input dimension for input " + std::to_string(i));
        }

        const std::string& format = model_info_.input_formats[i];
        if (format == "FORMAT_NCHW" || format == "FORMAT_NHWC") {
            if (shapes[i].size() == 4) {
                int channels = (format == "FORMAT_NCHW") ? shapes[i][1] : shapes[i][3];
                if (channels != 1 && channels != 3) {
                    throw std::runtime_error("Invalid number of channels for " + format + " format in input " + std::to_string(i));
                }
            } else if (shapes[i].size() != 2) {
                throw std::runtime_error("Invalid shape for image input " + std::to_string(i));
            }
        } else if (format != "FORMAT_NONE") {
            throw std::runtime_error("Unsupported input format: " + format + " for input " + std::to_string(i));
        }

        model_info_.input_shapes[i] = shapes[i];
    }

    updateInputTypes();
}

// Kept the original setInputShape method for backward compatibility
void Triton::setInputShape(const std::vector<int64_t>& shape) {
    if (model_info_.input_shapes.empty()) {
        throw std::runtime_error("Model information not initialized");
    }

    setInputShapes({shape});
}

// Function to create Triton client based on the protocol
void Triton::createTritonClient()
{
    tc::Error err;
    if (protocol_ == ProtocolType::HTTP) {
        err = tc::InferenceServerHttpClient::Create(
            &triton_client_.httpClient, url_, verbose_);
    }
    else {
        err = tc::InferenceServerGrpcClient::Create(
            &triton_client_.grpcClient, url_, verbose_);
    }
    if (!err.IsOk()) {
        std::cerr << "error: unable to create client for inference: " << err
            << std::endl;
        exit(1);
    }

}

std::vector<const tc::InferRequestedOutput*> Triton::createInferRequestedOutput(const std::vector<std::string>& output_names_)
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

std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>> Triton::getInferResults(
    tc::InferResult* result,
    const size_t batch_size,
    const std::vector<std::string>& output_names)
{
    if (!result->RequestStatus().IsOk())
    {
        std::cerr << "inference failed with error: " << result->RequestStatus() << std::endl;
        exit(1);
    }

    std::vector<std::vector<TensorElement>> infer_results;
    std::vector<std::vector<int64_t>> infer_shapes;

    for (const auto& outputName : output_names)
    {
        std::vector<int64_t> infer_shape;
        std::vector<TensorElement> infer_result;

        const uint8_t* outputData;
        size_t outputByteSize;
        result->RawData(outputName, &outputData, &outputByteSize);

        tc::Error err = result->Shape(outputName, &infer_shape);
        if (!err.IsOk())
        {
            std::cerr << "unable to get shape for " << outputName << std::endl;
            exit(1);
        }

        // Determine the data type of the output
        std::string output_datatype;
        err = result->Datatype(outputName, &output_datatype);
        if (!err.IsOk())
        {
            std::cerr << "unable to get datatype for " << outputName << std::endl;
            exit(1);
        }

        // Convert raw data to TensorElement based on the datatype
        if (output_datatype == "FP32") {
            const float* floatData = reinterpret_cast<const float*>(outputData);
            size_t elementCount = outputByteSize / sizeof(float);
            infer_result.reserve(elementCount);
            for (size_t i = 0; i < elementCount; ++i) {
                infer_result.emplace_back(floatData[i]);
            }
        } else if (output_datatype == "INT32") {
            const int32_t* intData = reinterpret_cast<const int32_t*>(outputData);
            size_t elementCount = outputByteSize / sizeof(int32_t);
            infer_result.reserve(elementCount);
            for (size_t i = 0; i < elementCount; ++i) {
                infer_result.emplace_back(intData[i]);
            }
        } else if (output_datatype == "INT64") {
            const int64_t* longData = reinterpret_cast<const int64_t*>(outputData);
            size_t elementCount = outputByteSize / sizeof(int64_t);
            infer_result.reserve(elementCount);
            for (size_t i = 0; i < elementCount; ++i) {
                infer_result.emplace_back(longData[i]);
            }
        } else {
            std::cerr << "Unsupported datatype: " << output_datatype << std::endl;
            exit(1);
        }

        infer_results.push_back(std::move(infer_result));
        infer_shapes.push_back(std::move(infer_shape));
    }

    return std::make_tuple(infer_results, infer_shapes);
}
std::tuple<std::vector<std::vector<TensorElement>>, std::vector<std::vector<int64_t>>> Triton::infer(const std::vector<std::vector<uint8_t>>& input_data)
{
    tc::Error err;
    std::vector<tc::InferInput*> inputs;
    std::vector<const tc::InferRequestedOutput*> outputs = createInferRequestedOutput(model_info_.output_names);        
    tc::InferOptions options(model_name_);
    options.model_version_ = model_version_;

    // Check if input_data size matches expected number of inputs
    if (input_data.size() != model_info_.input_names.size()) {
        std::cerr << "Error: Mismatch in number of inputs. Expected " << model_info_.input_names.size() 
                  << ", but got " << input_data.size() << std::endl;
        exit(1);
    }

    // Create and populate inputs
    for (size_t i = 0; i < model_info_.input_names.size(); ++i) {
        tc::InferInput* input;
        err = tc::InferInput::Create(&input, model_info_.input_names[i], model_info_.input_shapes[i], model_info_.input_datatypes[i]);
        if (!err.IsOk()) {
            std::cerr << "Unable to create input " << model_info_.input_names[i] << ": " << err << std::endl;
            exit(1);
        }
        inputs.push_back(input);

        if (input_data[i].empty()) {
            std::cerr << "Warning: Empty input data for " << model_info_.input_names[i] << std::endl;
            continue;  // Skip appending empty data
        }

        err = input->AppendRaw(input_data[i]);
        if (!err.IsOk()) {
            std::cerr << "Failed setting input " << model_info_.input_names[i] << ": " << err << std::endl;
            exit(1);
        }

        std::cout << "Input " << model_info_.input_names[i] << " set with " << input_data[i].size() << " bytes of data" << std::endl;
    }


    tc::InferResult* result;
    std::unique_ptr<tc::InferResult> result_ptr;
    if (protocol_ == ProtocolType::HTTP) {
        err = triton_client_.httpClient->Infer(&result, options, inputs, outputs);
    }
    else {
        err = triton_client_.grpcClient->Infer(&result, options, inputs, outputs);
    }
    if (!err.IsOk()) {
        std::cerr << "Failed sending synchronous infer request: " << err << std::endl;
        exit(1);
    }

    auto [infer_results, infer_shapes] = getInferResults(result, model_info_.batch_size_, model_info_.output_names);
    result_ptr.reset(result);

    // Clean up inputs
    for (auto input : inputs) {
        delete input;
    }

    return std::make_tuple(std::move(infer_results), std::move(infer_shapes));
}