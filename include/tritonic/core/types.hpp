#pragma once

#include <string>
#include <variant>
#include <vector>

namespace tritonic::core {

using TensorElement = std::variant<float, int32_t, int64_t, uint8_t, std::string>;

struct Tensor {
    std::vector<TensorElement> data;
    std::vector<int64_t> shape;

    Tensor() = default;
    Tensor(std::vector<TensorElement> data_, std::vector<int64_t> shape_)
        : data(std::move(data_)), shape(std::move(shape_)) {}
};

// ---------------------------------------------------------------------------
// Chat types (OpenAI-compatible API)
// ---------------------------------------------------------------------------

struct Message {
    enum class Role { System, User, Assistant };
    Role role{Role::User};
    std::string content;
    std::vector<std::string> images;
};

struct ChatRequest {
    std::vector<Message> messages;
    std::string model;
    std::string detail{"low"};
    int   max_tokens{300};
    float temperature{1.0f};
    float top_p{1.0f};
    int   target_image_size{512};
};

struct ChatResponse {
    std::string text;
    bool  success{false};
    std::string error;
};

// ---------------------------------------------------------------------------
// Triton tensor backend request
// ---------------------------------------------------------------------------

struct TritonInferRequest {
    std::vector<std::vector<uint8_t>> input_data;
};

// ---------------------------------------------------------------------------
// Unified variant types (Strategy pattern I/O)
// ---------------------------------------------------------------------------

using BackendRequest  = std::variant<TritonInferRequest, ChatRequest>;
using BackendResponse = std::variant<std::vector<Tensor>, ChatResponse>;

}  // namespace tritonic::core
