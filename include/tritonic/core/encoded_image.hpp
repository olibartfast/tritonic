#pragma once

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "tritonic/triton/model_info.hpp"

namespace tritonic::core {

struct EncodedImageRequest {
    std::vector<std::vector<uint8_t>> inputs;
    std::vector<std::vector<int64_t>> shapes;
    int width;
    int height;
};

inline std::pair<int, int> ReadJpegDimensions(const std::vector<uint8_t>& bytes,
                                              const std::string& path) {
    if (bytes.size() < 4 || bytes[0] != 0xff || bytes[1] != 0xd8) {
        throw std::runtime_error("Encoded image is not a JPEG: " + path);
    }

    size_t position = 2;
    while (position + 3 < bytes.size()) {
        while (position < bytes.size() && bytes[position] != 0xff)
            ++position;
        while (position < bytes.size() && bytes[position] == 0xff)
            ++position;
        if (position >= bytes.size())
            break;

        const uint8_t marker = bytes[position++];
        if (marker == 0xd8 || marker == 0xd9 || marker == 0x01 ||
            (marker >= 0xd0 && marker <= 0xd7)) {
            continue;
        }
        if (position + 1 >= bytes.size())
            break;

        const size_t length = (static_cast<size_t>(bytes[position]) << 8) | bytes[position + 1];
        if (length < 2 || position + length > bytes.size())
            break;

        const bool startOfFrame =
            (marker >= 0xc0 && marker <= 0xc3) || (marker >= 0xc5 && marker <= 0xc7) ||
            (marker >= 0xc9 && marker <= 0xcb) || (marker >= 0xcd && marker <= 0xcf);
        if (startOfFrame) {
            if (length < 7)
                break;
            const int height = (static_cast<int>(bytes[position + 3]) << 8) | bytes[position + 4];
            const int width = (static_cast<int>(bytes[position + 5]) << 8) | bytes[position + 6];
            if (width > 0 && height > 0)
                return {width, height};
            break;
        }
        position += length;
    }

    throw std::runtime_error("Could not read JPEG dimensions: " + path);
}

inline EncodedImageRequest BuildEncodedImageRequest(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream.is_open()) {
        throw std::runtime_error("Could not open encoded image: " + path);
    }

    std::vector<uint8_t> bytes(std::istreambuf_iterator<char>(stream), {});
    if (bytes.empty()) {
        throw std::runtime_error("Encoded image is empty: " + path);
    }

    const auto size = static_cast<int64_t>(bytes.size());
    const auto [width, height] = ReadJpegDimensions(bytes, path);
    return {{std::move(bytes)}, {{1, size}}, width, height};
}

inline void ValidateEncodedImageModels(const tritonic::triton::ModelInfo& request_model,
                                       const tritonic::triton::ModelInfo& task_model) {
    if (request_model.input_names.size() != 1 || request_model.input_names[0] != "IMAGE" ||
        request_model.input_datatypes.size() != 1 || request_model.input_datatypes[0] != "UINT8") {
        throw std::runtime_error(
            "Encoded-image model must expose exactly one UINT8 input named IMAGE");
    }
    if (request_model.max_batch_size_ != 1) {
        throw std::runtime_error("Encoded-image model must declare max_batch_size: 1");
    }
    const size_t taskOutputCount = task_model.output_names.size();
    if (request_model.output_names.size() < taskOutputCount ||
        request_model.output_datatypes.size() < taskOutputCount ||
        request_model.output_shapes.size() < taskOutputCount ||
        !std::equal(task_model.output_names.begin(), task_model.output_names.end(),
                    request_model.output_names.begin()) ||
        !std::equal(task_model.output_datatypes.begin(), task_model.output_datatypes.end(),
                    request_model.output_datatypes.begin()) ||
        !std::equal(task_model.output_shapes.begin(), task_model.output_shapes.end(),
                    request_model.output_shapes.begin())) {
        throw std::runtime_error("Encoded-image model outputs do not match --task_model metadata");
    }
}

}  // namespace tritonic::core
