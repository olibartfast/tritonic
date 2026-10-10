// Triton C++ backend "yolopreprocess" (specs/2026-10-01-gpu-preprocess-alternatives).
//
// mode=preprocess (variant V12): IMAGE (encoded JPEG bytes) -> nvJPEG decode -> shared CUDA
//   letterbox kernel -> PREPROCESSED FP32 [3,640,640] + ORIGINAL_SIZE INT64 [2] (H, W).
// mode=decode (variant V13 decode step): IMAGE -> nvJPEG decode -> DECODED UINT8 [H,W,3] RGB
//   + ORIGINAL_SIZE, so a TensorRT engine can letterbox in-graph.
//
// Both modes run on the instance's CUDA stream, reuse per-instance nvJPEG state and scratch
// buffers, and synchronize only that stream before responding. Invalid inputs get a
// per-request error response; the instance stays usable.

#include <cuda_runtime_api.h>
#include <nvjpeg.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "gray_to_rgb.hpp"
#include "triton/backend/backend_common.h"
#include "triton/backend/backend_model.h"
#include "triton/backend/backend_model_instance.h"
#include "triton/core/tritonbackend.h"
#include "tritonic/cuda/letterbox.hpp"

namespace triton {
namespace backend {
namespace yolopreprocess {

namespace {

constexpr int kCanvas = tritonic::cuda::kLetterboxCanvas;

TRITONSERVER_Error* CudaError(cudaError_t status, const char* what) {
    if (status == cudaSuccess) {
        return nullptr;
    }
    return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL,
                                 (std::string(what) + ": " + cudaGetErrorString(status)).c_str());
}

TRITONSERVER_Error* InvalidArg(const std::string& message) {
    return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, message.c_str());
}

template <typename T>
class DeviceBuffer {
public:
    ~DeviceBuffer() {
        cudaFree(ptr_);
    }

    TRITONSERVER_Error* Reserve(size_t count) {
        if (count <= capacity_) {
            return nullptr;
        }
        cudaFree(ptr_);
        ptr_ = nullptr;
        capacity_ = 0;
        RETURN_IF_ERROR(CudaError(cudaMalloc(reinterpret_cast<void**>(&ptr_), count * sizeof(T)),
                                  "cudaMalloc"));
        capacity_ = count;
        return nullptr;
    }
    T* get() const {
        return ptr_;
    }

private:
    T* ptr_{nullptr};
    size_t capacity_{0};
};

enum class Mode { Preprocess, Decode };

// Optional parameters are absent in the normal case; GetParameterValue reports that as
// an error object, which must be freed.
bool FindParameter(common::TritonJson::Value& parameters, const char* key, std::string* value) {
    TRITONSERVER_Error* error = GetParameterValue(parameters, key, value);
    if (error != nullptr) {
        TRITONSERVER_ErrorDelete(error);
        return false;
    }
    return true;
}

}  // namespace

//
// ModelState: configuration shared by all instances.
//
class ModelState : public BackendModel {
public:
    static TRITONSERVER_Error* Create(TRITONBACKEND_Model* triton_model, ModelState** state);

    Mode GetMode() const {
        return mode_;
    }
    int64_t MaxSide() const {
        return max_side_;
    }
    tritonic::cuda::PadRounding Rounding() const {
        return rounding_;
    }

private:
    explicit ModelState(TRITONBACKEND_Model* triton_model) : BackendModel(triton_model) {}
    TRITONSERVER_Error* ParseParameters();
    TRITONSERVER_Error* ValidateConfig();

    Mode mode_{Mode::Preprocess};
    int64_t max_side_{8192};
    tritonic::cuda::PadRounding rounding_{tritonic::cuda::kDefaultPadRounding};
};

TRITONSERVER_Error* ModelState::Create(TRITONBACKEND_Model* triton_model, ModelState** state) {
    try {
        *state = new ModelState(triton_model);
    } catch (const BackendModelException& ex) {
        RETURN_ERROR_IF_TRUE(ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
                             std::string("unexpected nullptr in BackendModelException"));
        RETURN_IF_ERROR(ex.err_);
    }
    RETURN_IF_ERROR((*state)->ParseParameters());
    return (*state)->ValidateConfig();
}

TRITONSERVER_Error* ModelState::ParseParameters() {
    common::TritonJson::Value parameters;
    if (!ModelConfig().Find("parameters", &parameters)) {
        return nullptr;
    }
    std::string value;
    if (FindParameter(parameters, "mode", &value)) {
        if (value == "preprocess") {
            mode_ = Mode::Preprocess;
        } else if (value == "decode") {
            mode_ = Mode::Decode;
        } else {
            return InvalidArg("parameter 'mode' must be 'preprocess' or 'decode', got '" + value +
                              "'");
        }
    }
    if (FindParameter(parameters, "max_side", &value)) {
        try {
            max_side_ = std::stoll(value);
        } catch (const std::exception&) {
            return InvalidArg("parameter 'max_side' must be an integer");
        }
        if (max_side_ <= 0 || max_side_ > 16384) {
            return InvalidArg("parameter 'max_side' must be in [1, 16384]");
        }
    }
    if (FindParameter(parameters, "pad_rounding", &value)) {
        if (value == "floor") {
            rounding_ = tritonic::cuda::PadRounding::Floor;
        } else if (value == "ceil") {
            rounding_ = tritonic::cuda::PadRounding::Ceil;
        } else {
            return InvalidArg("parameter 'pad_rounding' must be 'floor' or 'ceil'");
        }
    }
    return nullptr;
}

TRITONSERVER_Error* ModelState::ValidateConfig() {
    RETURN_ERROR_IF_FALSE(MaxBatchSize() == 1, TRITONSERVER_ERROR_INVALID_ARG,
                          std::string("yolopreprocess requires max_batch_size: 1"));

    common::TritonJson::Value inputs;
    RETURN_IF_ERROR(ModelConfig().MemberAsArray("input", &inputs));
    RETURN_ERROR_IF_FALSE(inputs.ArraySize() == 1, TRITONSERVER_ERROR_INVALID_ARG,
                          std::string("yolopreprocess expects exactly one input"));
    common::TritonJson::Value input;
    RETURN_IF_ERROR(inputs.IndexAsObject(0, &input));
    std::string name, datatype;
    RETURN_IF_ERROR(input.MemberAsString("name", &name));
    RETURN_IF_ERROR(input.MemberAsString("data_type", &datatype));
    RETURN_ERROR_IF_FALSE(name == "IMAGE" && datatype == "TYPE_UINT8",
                          TRITONSERVER_ERROR_INVALID_ARG,
                          std::string("yolopreprocess input must be IMAGE TYPE_UINT8"));

    const char* expected = mode_ == Mode::Preprocess ? "PREPROCESSED" : "DECODED";
    common::TritonJson::Value outputs;
    RETURN_IF_ERROR(ModelConfig().MemberAsArray("output", &outputs));
    bool tensor = false;
    bool size = false;
    for (size_t i = 0; i < outputs.ArraySize(); ++i) {
        common::TritonJson::Value output;
        RETURN_IF_ERROR(outputs.IndexAsObject(i, &output));
        RETURN_IF_ERROR(output.MemberAsString("name", &name));
        tensor |= name == expected;
        size |= name == "ORIGINAL_SIZE";
    }
    RETURN_ERROR_IF_FALSE(
        tensor && size, TRITONSERVER_ERROR_INVALID_ARG,
        std::string("yolopreprocess mode requires outputs ") + expected + " and ORIGINAL_SIZE");
    return nullptr;
}

//
// ModelInstanceState: per-instance CUDA and nvJPEG resources.
//
class ModelInstanceState : public BackendModelInstance {
public:
    static TRITONSERVER_Error* Create(ModelState* model_state,
                                      TRITONBACKEND_ModelInstance* triton_model_instance,
                                      ModelInstanceState** state);
    ~ModelInstanceState() override;

    void ProcessRequests(TRITONBACKEND_Request** requests, uint32_t request_count);

private:
    ModelInstanceState(ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance)
        : BackendModelInstance(model_state, triton_model_instance), model_state_(model_state) {}

    TRITONSERVER_Error* Init();
    TRITONSERVER_Error* Process(TRITONBACKEND_Request* request, TRITONBACKEND_Response* response);
    TRITONSERVER_Error* Decode(const uint8_t* data, size_t size, int components, int height,
                               int width, uint8_t* destination);
    TRITONSERVER_Error* WriteOriginalSize(TRITONBACKEND_Response* response, int height, int width);
    TRITONSERVER_Error* ResetJpegState();

    ModelState* model_state_;
    nvjpegHandle_t nvjpeg_{nullptr};
    nvjpegJpegState_t jpeg_state_{nullptr};
    std::vector<char> encoded_;
    int64_t original_size_[2]{0, 0};
    DeviceBuffer<uint8_t> decoded_;
    DeviceBuffer<uint8_t> gray_;
    DeviceBuffer<float> tensor_;
};

TRITONSERVER_Error* ModelInstanceState::Create(ModelState* model_state,
                                               TRITONBACKEND_ModelInstance* triton_model_instance,
                                               ModelInstanceState** state) {
    try {
        *state = new ModelInstanceState(model_state, triton_model_instance);
    } catch (const BackendModelInstanceException& ex) {
        RETURN_ERROR_IF_TRUE(ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
                             std::string("unexpected nullptr in BackendModelInstanceException"));
        RETURN_IF_ERROR(ex.err_);
    }
    return (*state)->Init();
}

TRITONSERVER_Error* ModelInstanceState::Init() {
    RETURN_ERROR_IF_FALSE(Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU,
                          TRITONSERVER_ERROR_INVALID_ARG,
                          std::string("yolopreprocess requires instance_group kind KIND_GPU"));
    RETURN_IF_ERROR(CudaError(cudaSetDevice(DeviceId()), "cudaSetDevice"));
    RETURN_ERROR_IF_FALSE(nvjpegCreateSimple(&nvjpeg_) == NVJPEG_STATUS_SUCCESS,
                          TRITONSERVER_ERROR_INTERNAL, std::string("nvjpegCreateSimple failed"));
    RETURN_ERROR_IF_FALSE(nvjpegJpegStateCreate(nvjpeg_, &jpeg_state_) == NVJPEG_STATUS_SUCCESS,
                          TRITONSERVER_ERROR_INTERNAL, std::string("nvjpegJpegStateCreate failed"));
    if (model_state_->GetMode() == Mode::Preprocess) {
        RETURN_IF_ERROR(tensor_.Reserve(3 * static_cast<size_t>(kCanvas) * kCanvas));
    }
    return nullptr;
}

ModelInstanceState::~ModelInstanceState() {
    cudaSetDevice(DeviceId());
    if (jpeg_state_) {
        nvjpegJpegStateDestroy(jpeg_state_);
    }
    if (nvjpeg_) {
        nvjpegDestroy(nvjpeg_);
    }
}

TRITONSERVER_Error* ModelInstanceState::ResetJpegState() {
    // A failed decode can leave the state mid-stream; start the next request clean.
    nvjpegJpegStateDestroy(jpeg_state_);
    jpeg_state_ = nullptr;
    RETURN_ERROR_IF_FALSE(nvjpegJpegStateCreate(nvjpeg_, &jpeg_state_) == NVJPEG_STATUS_SUCCESS,
                          TRITONSERVER_ERROR_INTERNAL, std::string("nvjpegJpegStateCreate failed"));
    return nullptr;
}

TRITONSERVER_Error* ModelInstanceState::Decode(const uint8_t* data, size_t size, int components,
                                               int height, int width, uint8_t* destination) {
    nvjpegImage_t image{};
    nvjpegStatus_t status;
    if (components == 3) {
        image.channel[0] = destination;
        image.pitch[0] = static_cast<size_t>(width) * 3;
        status = nvjpegDecode(nvjpeg_, jpeg_state_, data, size, NVJPEG_OUTPUT_RGBI, &image,
                              CudaStream());
    } else {
        RETURN_IF_ERROR(gray_.Reserve(static_cast<size_t>(width) * height));
        image.channel[0] = gray_.get();
        image.pitch[0] = static_cast<size_t>(width);
        status =
            nvjpegDecode(nvjpeg_, jpeg_state_, data, size, NVJPEG_OUTPUT_Y, &image, CudaStream());
        if (status == NVJPEG_STATUS_SUCCESS) {
            RETURN_IF_ERROR(
                CudaError(LaunchGrayToRgb(gray_.get(), height, width, destination, CudaStream()),
                          "gray to RGB"));
        }
    }
    if (status != NVJPEG_STATUS_SUCCESS) {
        RETURN_IF_ERROR(ResetJpegState());
        return InvalidArg("JPEG decode failed (nvjpeg status " +
                          std::to_string(static_cast<int>(status)) + ")");
    }
    return nullptr;
}

TRITONSERVER_Error* ModelInstanceState::WriteOriginalSize(TRITONBACKEND_Response* response,
                                                          int height, int width) {
    const int64_t shape[2] = {1, 2};
    TRITONBACKEND_Output* output;
    RETURN_IF_ERROR(TRITONBACKEND_ResponseOutput(response, &output, "ORIGINAL_SIZE",
                                                 TRITONSERVER_TYPE_INT64, shape, 2));
    void* buffer;
    TRITONSERVER_MemoryType memory_type = TRITONSERVER_MEMORY_CPU;
    int64_t memory_type_id = 0;
    RETURN_IF_ERROR(TRITONBACKEND_OutputBuffer(output, &buffer, sizeof(original_size_),
                                               &memory_type, &memory_type_id));
    original_size_[0] = height;
    original_size_[1] = width;
    if (memory_type == TRITONSERVER_MEMORY_GPU) {
        // original_size_ outlives the copy: the stream is synchronized before responding.
        return CudaError(cudaMemcpyAsync(buffer, original_size_, sizeof(original_size_),
                                         cudaMemcpyHostToDevice, CudaStream()),
                         "ORIGINAL_SIZE upload");
    }
    std::memcpy(buffer, original_size_, sizeof(original_size_));
    return nullptr;
}

TRITONSERVER_Error* ModelInstanceState::Process(TRITONBACKEND_Request* request,
                                                TRITONBACKEND_Response* response) {
    TRITONBACKEND_Input* input;
    RETURN_IF_ERROR(TRITONBACKEND_RequestInput(request, "IMAGE", &input));
    TRITONSERVER_DataType datatype;
    const int64_t* shape;
    uint32_t dims_count;
    uint64_t byte_size;
    uint32_t buffer_count;
    RETURN_IF_ERROR(TRITONBACKEND_InputProperties(input, nullptr, &datatype, &shape, &dims_count,
                                                  &byte_size, &buffer_count));
    if (datatype != TRITONSERVER_TYPE_UINT8) {
        return InvalidArg("IMAGE must be UINT8");
    }
    if (byte_size < 4) {
        return InvalidArg("IMAGE is empty or too short to be a JPEG");
    }
    const uint64_t max_side = static_cast<uint64_t>(model_state_->MaxSide());
    if (byte_size > max_side * max_side * 3) {
        return InvalidArg("IMAGE exceeds the configured size limit");
    }

    encoded_.resize(byte_size);
    size_t read_size = byte_size;
    RETURN_IF_ERROR(ReadInputTensor(request, "IMAGE", encoded_.data(), &read_size));
    const auto* data = reinterpret_cast<const uint8_t*>(encoded_.data());
    if (data[0] != 0xFF || data[1] != 0xD8) {
        return InvalidArg("IMAGE is not a JPEG (missing SOI marker)");
    }

    int components = 0;
    nvjpegChromaSubsampling_t subsampling;
    int widths[NVJPEG_MAX_COMPONENT] = {};
    int heights[NVJPEG_MAX_COMPONENT] = {};
    if (nvjpegGetImageInfo(nvjpeg_, data, read_size, &components, &subsampling, widths, heights) !=
        NVJPEG_STATUS_SUCCESS) {
        return InvalidArg("IMAGE is not a decodable JPEG");
    }
    const int width = widths[0];
    const int height = heights[0];
    if (components != 1 && components != 3) {
        return InvalidArg("unsupported JPEG with " + std::to_string(components) + " components");
    }
    if (width <= 0 || height <= 0 || width > model_state_->MaxSide() ||
        height > model_state_->MaxSide()) {
        return InvalidArg("JPEG dimensions " + std::to_string(width) + "x" +
                          std::to_string(height) + " outside the allowed range");
    }
    // Checked in both modes: in decode mode the downstream TensorRT letterbox would
    // otherwise receive an image it cannot resize.
    if (!tritonic::cuda::ComputeLetterboxGeometry(height, width, kCanvas,
                                                  model_state_->Rounding())) {
        return InvalidArg("JPEG aspect ratio " + std::to_string(width) + "x" +
                          std::to_string(height) + " is too extreme to letterbox");
    }

    const size_t decoded_bytes = static_cast<size_t>(width) * height * 3;
    if (model_state_->GetMode() == Mode::Decode) {
        const int64_t out_shape[4] = {1, height, width, 3};
        TRITONBACKEND_Output* output;
        RETURN_IF_ERROR(TRITONBACKEND_ResponseOutput(response, &output, "DECODED",
                                                     TRITONSERVER_TYPE_UINT8, out_shape, 4));
        void* buffer;
        TRITONSERVER_MemoryType memory_type = TRITONSERVER_MEMORY_GPU;
        int64_t memory_type_id = DeviceId();
        RETURN_IF_ERROR(TRITONBACKEND_OutputBuffer(output, &buffer, decoded_bytes, &memory_type,
                                                   &memory_type_id));
        if (memory_type == TRITONSERVER_MEMORY_GPU) {
            RETURN_IF_ERROR(
                Decode(data, read_size, components, height, width, static_cast<uint8_t*>(buffer)));
        } else {
            RETURN_IF_ERROR(decoded_.Reserve(decoded_bytes));
            RETURN_IF_ERROR(Decode(data, read_size, components, height, width, decoded_.get()));
            RETURN_IF_ERROR(CudaError(cudaMemcpyAsync(buffer, decoded_.get(), decoded_bytes,
                                                      cudaMemcpyDeviceToHost, CudaStream()),
                                      "DECODED download"));
        }
    } else {
        RETURN_IF_ERROR(decoded_.Reserve(decoded_bytes));
        RETURN_IF_ERROR(Decode(data, read_size, components, height, width, decoded_.get()));

        const int64_t out_shape[4] = {1, 3, kCanvas, kCanvas};
        const size_t tensor_bytes = 3 * static_cast<size_t>(kCanvas) * kCanvas * sizeof(float);
        TRITONBACKEND_Output* output;
        RETURN_IF_ERROR(TRITONBACKEND_ResponseOutput(response, &output, "PREPROCESSED",
                                                     TRITONSERVER_TYPE_FP32, out_shape, 4));
        void* buffer;
        TRITONSERVER_MemoryType memory_type = TRITONSERVER_MEMORY_GPU;
        int64_t memory_type_id = DeviceId();
        RETURN_IF_ERROR(TRITONBACKEND_OutputBuffer(output, &buffer, tensor_bytes, &memory_type,
                                                   &memory_type_id));
        const bool on_device = memory_type == TRITONSERVER_MEMORY_GPU;
        float* destination = on_device ? static_cast<float*>(buffer) : tensor_.get();
        const tritonic::cuda::LetterboxSource source{decoded_.get(), height, width,
                                                     static_cast<size_t>(width) * 3,
                                                     tritonic::cuda::ChannelOrder::RGB};
        RETURN_IF_ERROR(
            CudaError(tritonic::cuda::LaunchLetterbox(source, destination, CudaStream(), kCanvas,
                                                      model_state_->Rounding()),
                      "letterbox kernel"));
        if (!on_device) {
            RETURN_IF_ERROR(CudaError(cudaMemcpyAsync(buffer, tensor_.get(), tensor_bytes,
                                                      cudaMemcpyDeviceToHost, CudaStream()),
                                      "PREPROCESSED download"));
        }
    }

    RETURN_IF_ERROR(WriteOriginalSize(response, height, width));
    // Outputs must be complete before the response hands them to the next ensemble step.
    return CudaError(cudaStreamSynchronize(CudaStream()), "cudaStreamSynchronize");
}

void ModelInstanceState::ProcessRequests(TRITONBACKEND_Request** requests, uint32_t request_count) {
    LOG_IF_ERROR(CudaError(cudaSetDevice(DeviceId()), "cudaSetDevice"),
                 "failed to select the instance device");
    for (uint32_t r = 0; r < request_count; ++r) {
        TRITONBACKEND_Request* request = requests[r];
        uint64_t exec_start_ns = 0, compute_start_ns = 0, compute_end_ns = 0, exec_end_ns = 0;
        SET_TIMESTAMP(exec_start_ns);

        TRITONBACKEND_Response* response = nullptr;
        TRITONSERVER_Error* error = TRITONBACKEND_ResponseNew(&response, request);
        if (error == nullptr) {
            SET_TIMESTAMP(compute_start_ns);
            error = Process(request, response);
            SET_TIMESTAMP(compute_end_ns);
            LOG_IF_ERROR(
                TRITONBACKEND_ResponseSend(response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, error),
                "failed to send yolopreprocess response");
        } else {
            LOG_MESSAGE(TRITONSERVER_LOG_ERROR, (std::string("failed to create response: ") +
                                                 TRITONSERVER_ErrorMessage(error))
                                                    .c_str());
            compute_start_ns = compute_end_ns = exec_start_ns;
        }
        const bool success = error == nullptr;
        if (error != nullptr) {
            TRITONSERVER_ErrorDelete(error);
        }
        SET_TIMESTAMP(exec_end_ns);

        LOG_IF_ERROR(TRITONBACKEND_ModelInstanceReportStatistics(
                         TritonModelInstance(), request, success, exec_start_ns, compute_start_ns,
                         compute_end_ns, exec_end_ns),
                     "failed to report request statistics");
        LOG_IF_ERROR(TRITONBACKEND_ModelInstanceReportBatchStatistics(
                         TritonModelInstance(), 1, exec_start_ns, compute_start_ns, compute_end_ns,
                         exec_end_ns),
                     "failed to report batch statistics");
        LOG_IF_ERROR(TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL),
                     "failed to release request");
    }
}

extern "C" {

TRITONSERVER_Error* TRITONBACKEND_Initialize(TRITONBACKEND_Backend* backend) {
    uint32_t major = 0, minor = 0;
    RETURN_IF_ERROR(TRITONBACKEND_ApiVersion(&major, &minor));
    if (major != TRITONBACKEND_API_VERSION_MAJOR || minor < TRITONBACKEND_API_VERSION_MINOR) {
        return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_UNSUPPORTED,
                                     "yolopreprocess: triton backend API version unsupported");
    }
    return nullptr;
}

TRITONSERVER_Error* TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model* model) {
    ModelState* model_state = nullptr;
    TRITONSERVER_Error* error = ModelState::Create(model, &model_state);
    if (error != nullptr) {
        delete model_state;
        return error;
    }
    return TRITONBACKEND_ModelSetState(model, reinterpret_cast<void*>(model_state));
}

TRITONSERVER_Error* TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model* model) {
    void* state = nullptr;
    RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &state));
    delete reinterpret_cast<ModelState*>(state);
    return nullptr;
}

TRITONSERVER_Error* TRITONBACKEND_ModelInstanceInitialize(TRITONBACKEND_ModelInstance* instance) {
    TRITONBACKEND_Model* model;
    RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceModel(instance, &model));
    void* model_state = nullptr;
    RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &model_state));

    ModelInstanceState* instance_state = nullptr;
    TRITONSERVER_Error* error = ModelInstanceState::Create(
        reinterpret_cast<ModelState*>(model_state), instance, &instance_state);
    if (error != nullptr) {
        delete instance_state;
        return error;
    }
    return TRITONBACKEND_ModelInstanceSetState(instance, reinterpret_cast<void*>(instance_state));
}

TRITONSERVER_Error* TRITONBACKEND_ModelInstanceFinalize(TRITONBACKEND_ModelInstance* instance) {
    void* state = nullptr;
    RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &state));
    delete reinterpret_cast<ModelInstanceState*>(state);
    return nullptr;
}

TRITONSERVER_Error* TRITONBACKEND_ModelInstanceExecute(TRITONBACKEND_ModelInstance* instance,
                                                       TRITONBACKEND_Request** requests,
                                                       const uint32_t request_count) {
    void* state = nullptr;
    RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &state));
    reinterpret_cast<ModelInstanceState*>(state)->ProcessRequests(requests, request_count);
    return nullptr;
}

}  // extern "C"

}  // namespace yolopreprocess
}  // namespace backend
}  // namespace triton
