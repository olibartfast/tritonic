#include "Triton.hpp"

static auto logger = std::dynamic_pointer_cast<vision_infra::core::Logger>(
    vision_infra::core::LoggerManager::GetLogger("triton"));

static size_t WriteCallback(char* ptr, size_t size, size_t nmemb, std::string& data) {
    size_t totalSize = size * nmemb;
    data.append(ptr, totalSize);
    return totalSize;
}

TritonModelInfo Triton::parseModelGrpc(const inference::ModelMetadataResponse& model_metadata, const inference::ModelConfigResponse& model_config) {
    TritonModelInfo info;
    // Platform and max batch size
    std::string platform = model_metadata.platform();
    info.max_batch_size_ = model_config.config().max_batch_size();

    // Inputs
    int inputIndex = 0;
    for (const auto& input : model_metadata.inputs()) {
        info.input_names.push_back(input.name());
        // Format: try to get from config, fallback to FORMAT_NONE
        std::string format = "FORMAT_NONE";
        if (inputIndex < model_config.config().input_size()) {
            auto format_enum = model_config.config().input(inputIndex).format();
            switch (format_enum) {
                case inference::ModelInput::FORMAT_NCHW: format = "FORMAT_NCHW"; break;
                case inference::ModelInput::FORMAT_NHWC: format = "FORMAT_NHWC"; break;
                default: format = "FORMAT_NONE"; break;
            }
        }
        if (format == "FORMAT_NONE") {
            format = (platform == "tensorflow_savedmodel") ? "FORMAT_NHWC" : "FORMAT_NCHW";
        }
        info.input_formats.push_back(format);

        // Shape
        std::vector<int64_t> shape;
        bool hasDynamicDim = false;
        for (const auto& dim : input.shape()) {
            if (dim == -1) hasDynamicDim = true;
            shape.push_back(dim);
        }
        // No dynamic shape handling here (input_sizes not passed in this signature)
        if (info.max_batch_size_ > 0 && shape.size() < 4) {
            shape.insert(shape.begin(), 1); // Insert batch size
        }
        info.input_shapes.push_back(shape);

        // Data type
        std::string datatype = input.datatype();
        if (datatype.rfind("TYPE_", 0) == 0) datatype = datatype.substr(5);
        info.input_datatypes.push_back(datatype);
        if (datatype == "FP32") {
            info.input_types.push_back(CV_32F);
        } else if (datatype == "INT32") {
            info.input_types.push_back(CV_32S);
        } else if (datatype == "INT64") {
            info.input_types.push_back(CV_32S); // Map INT64 to CV_32S
        } else {
            throw std::runtime_error("Unsupported data type: " + datatype);
        }
        ++inputIndex;
    }

    // Outputs
    for (const auto& output : model_metadata.outputs()) {
        info.output_names.push_back(output.name());
    }
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
    curl_easy_cleanup(curl);

    if (res != CURLE_OK) {
        throw std::runtime_error("Failed to perform request: " + std::string(curl_easy_strerror(res)));
    }

    if (responseData.find("Request for unknown model") != std::string::npos) {
        throw std::runtime_error("Unknown model: " + modelName);
    }

    rapidjson::Document responseJson;
    responseJson.Parse(responseData.c_str());

    std::string platform = responseJson.HasMember("platform") && responseJson["platform"].IsString() 
                           ? responseJson["platform"].GetString() : "";

    info.max_batch_size_ = responseJson["max_batch_size"].GetInt();

    const auto& inputs = responseJson["input"].GetArray();
    size_t inputIndex = 0;  // Keep track of index for error messages and input_sizes access
    for (const auto& input : inputs) {
        logger->Info("Input " + std::to_string(inputIndex) + ": " + std::string(input["name"].GetString()));
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
            if (input_sizes.empty() || inputIndex >= input_sizes.size()) {
                throw std::runtime_error("Dynamic input dimension detected for input " + std::to_string(inputIndex) + 
                                        ", but no input sizes provided. Please specify input sizes.");
            }
            shape = input_sizes[inputIndex];
        } else if (!input_sizes.empty()) {
            logger->Warn("Input sizes provided, but model does not have dynamic shapes. Ignoring provided input sizes.");
        }

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
            logger->Warn("Warning: INT64 type detected for input '" + info.input_names.back() + "'. Will be mapped to CV_32S.");
        } else {
            throw std::runtime_error("Unsupported data type: " + datatype);
        }
        
        ++inputIndex;
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

    updateInputTypes();
    printModelInfo(model_info);
    this->model_info_ = model_info; 
    return model_info;
}

void Triton::updateInputTypes() {
    for (size_t i = 0; i < model_info_.input_shapes.size(); ++i) {
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
                    inputType = CV_32F;
                }
            } else if (shape.size() == 2) {
                inputType = model_info_.type1_;
            } else {
                inputType = CV_32F;
            }
        } else {
            inputType = CV_32F;
        }
        
        model_info_.input_types[i] = inputType;
    }
}
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

void Triton::setInputShape(const std::vector<int64_t>& shape) {
    if (model_info_.input_shapes.empty()) {
        throw std::runtime_error("Model information not initialized");
    }

    setInputShapes({shape});
}

void Triton::createTritonClient() {
    tc::Error err;
    if (protocol_ == ProtocolType::HTTP) {
        err = tc::InferenceServerHttpClient::Create(&triton_client_.httpClient, url_, verbose_);
    } else {
        err = tc::InferenceServerGrpcClient::Create(&triton_client_.grpcClient, url_, verbose_);
    }
    if (!err.IsOk()) {
        throw std::runtime_error("Unable to create client for inference: " + err.Message());
    }
}

std::vector<Tensor> Triton::getInferResults(
    tc::InferResult* result,
    const size_t batch_size,
    const std::vector<std::string>& output_names)
{
    if (!result->RequestStatus().IsOk()) {
        throw std::runtime_error("Inference failed with error: " + result->RequestStatus().Message());
    }

    std::vector<Tensor> tensors;
    tensors.reserve(output_names.size());

    for (const auto& outputName : output_names) {
        std::vector<int64_t> infer_shape;
        std::vector<TensorElement> infer_result;

        const uint8_t* outputData;
        size_t outputByteSize;
        result->RawData(outputName, &outputData, &outputByteSize);

        tc::Error err = result->Shape(outputName, &infer_shape);
        if (!err.IsOk()) {
            throw std::runtime_error("Unable to get shape for " + outputName + ": " + err.Message());
        }

        std::string output_datatype;
        err = result->Datatype(outputName, &output_datatype);
        if (!err.IsOk()) {
            throw std::runtime_error("Unable to get datatype for " + outputName + ": " + err.Message());
        }

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
            throw std::runtime_error("Unsupported datatype: " + output_datatype);
        }

        tensors.emplace_back(std::move(infer_result), std::move(infer_shape));
    }

    return tensors;
}

std::vector<Tensor> Triton::infer(const std::vector<std::vector<uint8_t>>& input_data) {
    tc::Error err;
    std::vector<std::unique_ptr<tc::InferInput>> inputs;
    std::vector<std::unique_ptr<tc::InferRequestedOutput>> outputs;
    
    // Create outputs with smart pointers
    for (const auto& output_name : model_info_.output_names) {
        tc::InferRequestedOutput* output;
        err = tc::InferRequestedOutput::Create(&output, output_name);
        if (!err.IsOk()) {
            throw std::runtime_error("Unable to get output: " + err.Message());
        }
        outputs.emplace_back(output);
    }
    
    // Create vector of raw pointers for the API call
    std::vector<const tc::InferRequestedOutput*> output_ptrs;
    output_ptrs.reserve(outputs.size());
    for (const auto& output : outputs) {
        output_ptrs.push_back(output.get());
    }
    
    tc::InferOptions options(model_name_);
    options.model_version_ = model_version_;

    if (input_data.size() != model_info_.input_names.size()) {
        throw std::runtime_error("Mismatch in number of inputs. Expected " + std::to_string(model_info_.input_names.size()) + 
                                 ", but got " + std::to_string(input_data.size()));
    }

    for (size_t i = 0; i < model_info_.input_names.size(); ++i) 
    {
        tc::InferInput* input;
        err = tc::InferInput::Create(&input, model_info_.input_names[i], model_info_.input_shapes[i], model_info_.input_datatypes[i]);
        if (!err.IsOk()) {
            throw std::runtime_error("Unable to create input " + model_info_.input_names[i] + ": " + err.Message());
        }
        inputs.emplace_back(input);

        if (input_data[i].empty()) {
            logger->Warn("Warning: Empty input data for " + model_info_.input_names[i]);
            continue;  // Skip appending empty data
        }

        err = input->AppendRaw(input_data[i]);
        if (!err.IsOk()) {
            throw std::runtime_error("Failed setting input " + model_info_.input_names[i] + ": " + err.Message());
        }

        logger->Debug("Input " + model_info_.input_names[i] + " set with " + std::to_string(input_data[i].size()) + " bytes of data");
    }

    // Create vector of raw pointers for the API call
    std::vector<tc::InferInput*> input_ptrs;
    input_ptrs.reserve(inputs.size());
    for (const auto& input : inputs) {
        input_ptrs.push_back(input.get());
    }

    tc::InferResult* result;
    std::unique_ptr<tc::InferResult> result_ptr;
    if (protocol_ == ProtocolType::HTTP) {
        err = triton_client_.httpClient->Infer(&result, options, input_ptrs, output_ptrs);
    } else {
        err = triton_client_.grpcClient->Infer(&result, options, input_ptrs, output_ptrs);
    }
    if (!err.IsOk()) {
        throw std::runtime_error("Failed sending synchronous infer request: " + err.Message());
    }

    auto tensors = getInferResults(result, model_info_.batch_size_, model_info_.output_names);
    result_ptr.reset(result);

    // Smart pointers automatically clean up inputs and outputs
    return tensors;
}

std::unique_ptr<SharedMemoryRegion> Triton::createSharedMemoryRegion(const std::string& name, size_t size) {
    if (shared_memory_type_ == SharedMemoryType::SYSTEM_SHARED_MEMORY) {
        return createSystemSharedMemoryRegion(name, size);
    } else {
        return createCudaSharedMemoryRegion(name, size);
    }
}

std::unique_ptr<SharedMemoryRegion> Triton::createSystemSharedMemoryRegion(const std::string& name, size_t size) {
    auto region = std::make_unique<SharedMemoryRegion>();
    region->name = name;
    region->key = "/" + name + "_" + std::to_string(getpid());
    region->size = size;
    region->type = SharedMemoryType::SYSTEM_SHARED_MEMORY;
    
    // Create shared memory region
    region->fd = shm_open(region->key.c_str(), O_CREAT | O_RDWR, 0666);
    if (region->fd == -1) {
        throw std::runtime_error("Failed to create system shared memory region: " + region->name);
    }
    
    // Set the size of the shared memory region
    if (ftruncate(region->fd, size) == -1) {
        close(region->fd);
        shm_unlink(region->key.c_str());
        throw std::runtime_error("Failed to set size of system shared memory region: " + region->name);
    }
    
    // Map the shared memory region
    region->ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, region->fd, 0);
    if (region->ptr == MAP_FAILED) {
        close(region->fd);
        shm_unlink(region->key.c_str());
        throw std::runtime_error("Failed to map system shared memory region: " + region->name);
    }
    
    logger->Info("Created system shared memory region '" + region->name + "' with key '" + region->key + "' and size " + std::to_string(region->size) + " bytes");
    
    return region;
}

std::unique_ptr<SharedMemoryRegion> Triton::createCudaSharedMemoryRegion(const std::string& name, size_t size) {
#ifdef TRITONIC_ENABLE_CUDA
    auto region = std::make_unique<SharedMemoryRegion>();
    region->name = name;
    region->key = name + "_cuda_" + std::to_string(getpid());
    region->size = size;
    region->type = SharedMemoryType::CUDA_SHARED_MEMORY;
    region->device_id = cuda_device_id_;
    
    // Set CUDA device
    cudaError_t cuda_err = cudaSetDevice(cuda_device_id_);
    if (cuda_err != cudaSuccess) {
        throw std::runtime_error("Failed to set CUDA device " + std::to_string(cuda_device_id_) + 
                                ": " + cudaGetErrorString(cuda_err));
    }
    
    // Allocate CUDA memory
    cuda_err = cudaMalloc(&region->ptr, size);
    if (cuda_err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate CUDA memory for region '" + region->name + 
                                "': " + cudaGetErrorString(cuda_err));
    }
    
    logger->Info("Created CUDA shared memory region '" + region->name + "' on device " + std::to_string(cuda_device_id_) + " with size " + std::to_string(region->size) + " bytes");
    
    return region;
#else
    throw std::runtime_error("CUDA shared memory requested but CUDA support not enabled. "
                           "Please compile with TRITONIC_ENABLE_CUDA=ON");
#endif
}

size_t Triton::calculateTensorSize(const std::vector<int64_t>& shape, const std::string& datatype) {
    size_t element_count = 1;
    for (int64_t dim : shape) {
        element_count *= dim;
    }
    
    size_t element_size;
    if (datatype == "FP32") {
        element_size = sizeof(float);
    } else if (datatype == "INT32") {
        element_size = sizeof(int32_t);
    } else if (datatype == "INT64") {
        element_size = sizeof(int64_t);
    } else if (datatype == "UINT8") {
        element_size = sizeof(uint8_t);
    } else {
        throw std::runtime_error("Unsupported datatype for shared memory: " + datatype);
    }
    
    return element_count * element_size;
}

void Triton::registerInputSharedMemory() {
    tc::Error err;
    
    // Clear existing input shared memory regions
    input_shm_regions_.clear();
    
    for (size_t i = 0; i < model_info_.input_names.size(); ++i) {
        const std::string& input_name = model_info_.input_names[i];
        const std::vector<int64_t>& shape = model_info_.input_shapes[i];
        const std::string& datatype = model_info_.input_datatypes[i];
        
        size_t tensor_size = calculateTensorSize(shape, datatype);
        std::string shm_name = "input_" + input_name + "_shm";
        
        auto region = createSharedMemoryRegion(shm_name, tensor_size);
        
        // Register with Triton server based on memory type
        if (shared_memory_type_ == SharedMemoryType::SYSTEM_SHARED_MEMORY) {
            if (protocol_ == ProtocolType::HTTP) {
                err = triton_client_.httpClient->RegisterSystemSharedMemory(
                    region->name, region->key, tensor_size);
            } else {
                err = triton_client_.grpcClient->RegisterSystemSharedMemory(
                    region->name, region->key, tensor_size);
            }
        } else {
#ifdef TRITONIC_ENABLE_CUDA
            if (protocol_ == ProtocolType::HTTP) {
                err = triton_client_.httpClient->RegisterCudaSharedMemory(
                    region->name, reinterpret_cast<CUdeviceptr>(region->ptr), 
                    region->device_id, tensor_size);
            } else {
                err = triton_client_.grpcClient->RegisterCudaSharedMemory(
                    region->name, reinterpret_cast<CUdeviceptr>(region->ptr), 
                    region->device_id, tensor_size);
            }
#endif
        }
        
        if (!err.IsOk()) {
            throw std::runtime_error("Failed to register input shared memory '" + 
                                   region->name + "': " + err.Message());
        }
        
        logger->Info("Registered " + std::string(shared_memory_type_ == SharedMemoryType::SYSTEM_SHARED_MEMORY ? "system" : "CUDA") + " input shared memory '" + region->name + "' for input '" + input_name + "'");
        input_shm_regions_.push_back(std::move(region));
    }
}

void Triton::registerOutputSharedMemory() {
    tc::Error err;
    
    // Clear existing output shared memory regions
    output_shm_regions_.clear();
    
    for (size_t i = 0; i < model_info_.output_names.size(); ++i) {
        const std::string& output_name = model_info_.output_names[i];
        
        // For output tensors, we need to estimate the size or get it from model metadata
        // For now, we'll use a default size and resize if needed
        size_t estimated_size = 1024 * 1024; // 1MB default
        std::string shm_name = "output_" + output_name + "_shm";
        
        auto region = createSharedMemoryRegion(shm_name, estimated_size);
        
        // Register with Triton server based on memory type
        if (shared_memory_type_ == SharedMemoryType::SYSTEM_SHARED_MEMORY) {
            if (protocol_ == ProtocolType::HTTP) {
                err = triton_client_.httpClient->RegisterSystemSharedMemory(
                    region->name, region->key, estimated_size);
            } else {
                err = triton_client_.grpcClient->RegisterSystemSharedMemory(
                    region->name, region->key, estimated_size);
            }
        } else {
#ifdef TRITONIC_ENABLE_CUDA
            if (protocol_ == ProtocolType::HTTP) {
                err = triton_client_.httpClient->RegisterCudaSharedMemory(
                    region->name, reinterpret_cast<CUdeviceptr>(region->ptr), 
                    region->device_id, estimated_size);
            } else {
                err = triton_client_.grpcClient->RegisterCudaSharedMemory(
                    region->name, reinterpret_cast<CUdeviceptr>(region->ptr), 
                    region->device_id, estimated_size);
            }
#endif
        }
        
        if (!err.IsOk()) {
            throw std::runtime_error("Failed to register output shared memory '" + 
                                   region->name + "': " + err.Message());
        }
        
        logger->Info("Registered " + std::string(shared_memory_type_ == SharedMemoryType::SYSTEM_SHARED_MEMORY ? "system" : "CUDA") + " output shared memory '" + region->name + "' for output '" + output_name + "'");
        output_shm_regions_.push_back(std::move(region));
    }
}

void Triton::unregisterSharedMemory() {
    tc::Error err;
    
    // Unregister input shared memory regions
    for (const auto& region : input_shm_regions_) {
        if (shared_memory_type_ == SharedMemoryType::SYSTEM_SHARED_MEMORY) {
            if (protocol_ == ProtocolType::HTTP) {
                err = triton_client_.httpClient->UnregisterSystemSharedMemory(region->name);
            } else {
                err = triton_client_.grpcClient->UnregisterSystemSharedMemory(region->name);
            }
        } else {
#ifdef TRITONIC_ENABLE_CUDA
            if (protocol_ == ProtocolType::HTTP) {
                err = triton_client_.httpClient->UnregisterCudaSharedMemory(region->name);
            } else {
                err = triton_client_.grpcClient->UnregisterCudaSharedMemory(region->name);
            }
#endif
        }
        
        if (!err.IsOk()) {
            logger->Warn("Failed to unregister input shared memory '" + region->name + "': " + err.Message());
        } else {
            logger->Info("Unregistered input shared memory '" + region->name + "'");
        }
    }
    
    // Unregister output shared memory regions
    for (const auto& region : output_shm_regions_) {
        if (shared_memory_type_ == SharedMemoryType::SYSTEM_SHARED_MEMORY) {
            if (protocol_ == ProtocolType::HTTP) {
                err = triton_client_.httpClient->UnregisterSystemSharedMemory(region->name);
            } else {
                err = triton_client_.grpcClient->UnregisterSystemSharedMemory(region->name);
            }
        } else {
#ifdef TRITONIC_ENABLE_CUDA
            if (protocol_ == ProtocolType::HTTP) {
                err = triton_client_.httpClient->UnregisterCudaSharedMemory(region->name);
            } else {
                err = triton_client_.grpcClient->UnregisterCudaSharedMemory(region->name);
            }
#endif
        }
        
        if (!err.IsOk()) {
            logger->Warn("Failed to unregister output shared memory '" + region->name + "': " + err.Message());
        } else {
            logger->Info("Unregistered output shared memory '" + region->name + "'");
        }
    }
    
    input_shm_regions_.clear();
    output_shm_regions_.clear();
}

std::vector<Tensor> Triton::inferWithSharedMemory(const std::vector<std::vector<uint8_t>>& input_data) {
    tc::Error err;
    std::vector<std::unique_ptr<tc::InferInput>> inputs;
    std::vector<std::unique_ptr<tc::InferRequestedOutput>> outputs;
    
    if (input_shm_regions_.empty()) {
        throw std::runtime_error("Input shared memory regions not registered. Call registerInputSharedMemory() first.");
    }
    
    if (input_data.size() != model_info_.input_names.size()) {
        throw std::runtime_error("Mismatch in number of inputs. Expected " + 
                                std::to_string(model_info_.input_names.size()) + 
                                ", but got " + std::to_string(input_data.size()));
    }
    
    // Copy input data to shared memory and create InferInput objects
    for (size_t i = 0; i < model_info_.input_names.size(); ++i) {
        const std::string& input_name = model_info_.input_names[i];
        const auto& region = input_shm_regions_[i];
        
        if (input_data[i].size() > region->size) {
            throw std::runtime_error("Input data size (" + std::to_string(input_data[i].size()) + 
                                    ") exceeds shared memory region size (" + 
                                    std::to_string(region->size) + ") for input " + input_name);
        }
        
        // Copy input data to shared memory
        if (shared_memory_type_ == SharedMemoryType::SYSTEM_SHARED_MEMORY) {
            std::memcpy(region->ptr, input_data[i].data(), input_data[i].size());
        } else {
#ifdef TRITONIC_ENABLE_CUDA
            cudaError_t cuda_err = cudaMemcpy(region->ptr, input_data[i].data(), 
                                            input_data[i].size(), cudaMemcpyHostToDevice);
            if (cuda_err != cudaSuccess) {
                throw std::runtime_error("Failed to copy data to CUDA shared memory for input " + 
                                       input_name + ": " + cudaGetErrorString(cuda_err));
            }
#endif
        }
        
        // Create InferInput with shared memory
        tc::InferInput* input;
        err = tc::InferInput::Create(&input, input_name, model_info_.input_shapes[i], 
                                   model_info_.input_datatypes[i]);
        if (!err.IsOk()) {
            throw std::runtime_error("Unable to create input " + input_name + ": " + err.Message());
        }
        
        // Set shared memory for input
        err = input->SetSharedMemory(region->name, input_data[i].size());
        if (!err.IsOk()) {
            delete input;
            throw std::runtime_error("Failed to set shared memory for input " + input_name + 
                                   ": " + err.Message());
        }
        
        inputs.emplace_back(input);
        logger->Info("Input '" + input_name + "' set with shared memory region '" + region->name + "'");
    }
    
    // Create outputs with shared memory (if available)
    for (size_t i = 0; i < model_info_.output_names.size(); ++i) {
        const std::string& output_name = model_info_.output_names[i];
        
        tc::InferRequestedOutput* output;
        err = tc::InferRequestedOutput::Create(&output, output_name);
        if (!err.IsOk()) {
            throw std::runtime_error("Unable to create output " + output_name + ": " + err.Message());
        }
        
        // Set shared memory for output if available
        if (i < output_shm_regions_.size()) {
            const auto& region = output_shm_regions_[i];
            err = output->SetSharedMemory(region->name, region->size);
            if (!err.IsOk()) {
                delete output;
                throw std::runtime_error("Failed to set shared memory for output " + output_name + 
                                       ": " + err.Message());
            }
            logger->Info("Output '" + output_name + "' set with shared memory region '" + region->name + "'");
        }
        
        outputs.emplace_back(output);
    }
    
    // Create vector of raw pointers for the API call
    std::vector<tc::InferInput*> input_ptrs;
    std::vector<const tc::InferRequestedOutput*> output_ptrs;
    
    input_ptrs.reserve(inputs.size());
    for (const auto& input : inputs) {
        input_ptrs.push_back(input.get());
    }
    
    output_ptrs.reserve(outputs.size());
    for (const auto& output : outputs) {
        output_ptrs.push_back(output.get());
    }
    
    tc::InferOptions options(model_name_);
    options.model_version_ = model_version_;
    
    // Perform inference
    tc::InferResult* result;
    std::unique_ptr<tc::InferResult> result_ptr;
    if (protocol_ == ProtocolType::HTTP) {
        err = triton_client_.httpClient->Infer(&result, options, input_ptrs, output_ptrs);
    } else {
        err = triton_client_.grpcClient->Infer(&result, options, input_ptrs, output_ptrs);
    }
    
    if (!err.IsOk()) {
        throw std::runtime_error("Failed sending synchronous infer request with shared memory: " + 
                                err.Message());
    }
    
    auto tensors = getInferResults(result, model_info_.batch_size_, 
                                  model_info_.output_names);
    result_ptr.reset(result);
    
    logger->Info("Inference with shared memory completed successfully");
    return tensors;
}

void Triton::setSharedMemoryType(SharedMemoryType type, int cuda_device) {
    shared_memory_type_ = type;
    cuda_device_id_ = cuda_device;
    
    // Unregister any existing shared memory regions
    unregisterSharedMemory();
    
    logger->Info("Shared memory type set to " + std::string(type == SharedMemoryType::SYSTEM_SHARED_MEMORY ? "System" : "CUDA"));
    if (type == SharedMemoryType::CUDA_SHARED_MEMORY) {
        logger->Info("CUDA device ID: " + std::to_string(cuda_device));
    }
}