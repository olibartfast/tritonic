#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "neuriplo/tasks/core/result_types.hpp"
#include "tritonic/core/types.hpp"
#include "tritonic/triton/model_info.hpp"

namespace tritonic::core {

inline void ValidateGpuSegmentationModel(const tritonic::triton::ModelInfo& model) {
    if (model.input_names != std::vector<std::string>{"IMAGE"} ||
        model.input_datatypes != std::vector<std::string>{"UINT8"} || model.max_batch_size_ != 1) {
        throw std::runtime_error(
            "GPU-postprocessed model must expose one UINT8 IMAGE input with max batch 1");
    }
    struct Expected {
        const char* name;
        const char* type;
        std::vector<int64_t> suffix;
    };
    const std::vector<Expected> expected = {
        {"NUM_DETECTIONS", "INT32", {1}}, {"BOXES", "INT32", {100, 4}},
        {"SCORES", "FP32", {100}},        {"CLASSES", "INT32", {100}},
        {"MASK_OFFSETS", "INT64", {101}}, {"MASK_SHAPES", "INT32", {100, 2}},
        {"MASK_DATA", "UINT8", {-1}},
    };
    if (model.output_names.size() != expected.size())
        throw std::runtime_error("GPU-postprocessed model has an incomplete output contract");
    for (const auto& item : expected) {
        const auto found =
            std::find(model.output_names.begin(), model.output_names.end(), item.name);
        if (found == model.output_names.end())
            throw std::runtime_error(std::string("Missing GPU-postprocessed output: ") + item.name);
        const size_t index = static_cast<size_t>(found - model.output_names.begin());
        if (index >= model.output_datatypes.size() || index >= model.output_shapes.size() ||
            model.output_datatypes[index] != item.type ||
            model.output_shapes[index].size() < item.suffix.size() ||
            !std::equal(item.suffix.rbegin(), item.suffix.rend(),
                        model.output_shapes[index].rbegin())) {
            throw std::runtime_error(std::string("GPU-postprocessed output mismatch: ") +
                                     item.name);
        }
    }
}

inline int32_t RequireInt32(const TensorElement& value, const char* name) {
    if (!std::holds_alternative<int32_t>(value))
        throw std::runtime_error(std::string(name) + " is not INT32");
    return std::get<int32_t>(value);
}
inline int64_t RequireInt64(const TensorElement& value, const char* name) {
    if (!std::holds_alternative<int64_t>(value))
        throw std::runtime_error(std::string(name) + " is not INT64");
    return std::get<int64_t>(value);
}
inline float RequireFloat(const TensorElement& value, const char* name) {
    if (!std::holds_alternative<float>(value))
        throw std::runtime_error(std::string(name) + " is not FP32");
    return std::get<float>(value);
}
inline uint8_t RequireUint8(const TensorElement& value, const char* name) {
    if (!std::holds_alternative<uint8_t>(value))
        throw std::runtime_error(std::string(name) + " is not UINT8");
    return std::get<uint8_t>(value);
}

inline std::vector<neuriplo_tasks::Result> DecodeGpuSegmentationResults(
    const std::vector<Tensor>& tensors, const std::vector<std::string>& names, int frame_width,
    int frame_height) {
    if (tensors.size() != names.size() || frame_width <= 0 || frame_height <= 0)
        throw std::runtime_error("Invalid GPU segmentation result envelope");
    std::unordered_map<std::string, const Tensor*> map;
    for (size_t i = 0; i < tensors.size(); ++i)
        map.emplace(names[i], &tensors[i]);
    const auto get = [&map](const char* name) -> const Tensor& {
        const auto found = map.find(name);
        if (found == map.end())
            throw std::runtime_error(std::string("Missing GPU result: ") + name);
        return *found->second;
    };
    const auto& count_data = get("NUM_DETECTIONS").data;
    const auto& boxes = get("BOXES").data;
    const auto& scores = get("SCORES").data;
    const auto& classes = get("CLASSES").data;
    const auto& offsets = get("MASK_OFFSETS").data;
    const auto& shapes = get("MASK_SHAPES").data;
    const auto& masks = get("MASK_DATA").data;
    if (count_data.empty())
        throw std::runtime_error("NUM_DETECTIONS is empty");
    const int count = RequireInt32(count_data[0], "NUM_DETECTIONS");
    if (count < 0 || count > 100 || boxes.size() < 400 || scores.size() < 100 ||
        classes.size() < 100 || offsets.size() < 101 || shapes.size() < 200)
        throw std::runtime_error("GPU segmentation fixed output is invalid or truncated");

    std::vector<neuriplo_tasks::Result> results;
    results.reserve(static_cast<size_t>(count));
    int64_t previous = 0;
    for (int i = 0; i < count; ++i) {
        const size_t n = static_cast<size_t>(i);
        const int x = RequireInt32(boxes[n * 4], "BOXES");
        const int y = RequireInt32(boxes[n * 4 + 1], "BOXES");
        const int width = RequireInt32(boxes[n * 4 + 2], "BOXES");
        const int height = RequireInt32(boxes[n * 4 + 3], "BOXES");
        const float score = RequireFloat(scores[n], "SCORES");
        const int class_id = RequireInt32(classes[n], "CLASSES");
        const int mask_height = RequireInt32(shapes[n * 2], "MASK_SHAPES");
        const int mask_width = RequireInt32(shapes[n * 2 + 1], "MASK_SHAPES");
        const int64_t begin = RequireInt64(offsets[n], "MASK_OFFSETS");
        const int64_t end = RequireInt64(offsets[n + 1], "MASK_OFFSETS");
        if (!std::isfinite(score) || score < 0.0F || score > 1.0F || class_id < 0 || x < 0 ||
            y < 0 || width <= 0 || height <= 0 || x + width > frame_width ||
            y + height > frame_height || mask_width != width || mask_height != height ||
            begin != previous || end < begin ||
            end - begin != static_cast<int64_t>(width) * height ||
            end > static_cast<int64_t>(masks.size()))
            throw std::runtime_error("GPU segmentation emitted invalid or garbage geometry");
        neuriplo_tasks::InstanceSegmentation result;
        result.class_id = static_cast<float>(class_id);
        result.class_confidence = score;
        result.bbox = {x, y, width, height};
        result.mask_height = height;
        result.mask_width = width;
        result.mask_data.reserve(static_cast<size_t>(end - begin));
        bool nonzero = false;
        for (int64_t offset = begin; offset < end; ++offset) {
            const uint8_t value = RequireUint8(masks[static_cast<size_t>(offset)], "MASK_DATA");
            if (value != 0 && value != 255)
                throw std::runtime_error("GPU mask is not binary");
            nonzero = nonzero || value != 0;
            result.mask_data.push_back(value);
        }
        if (!nonzero)
            throw std::runtime_error("GPU segmentation emitted an empty mask");
        previous = end;
        results.emplace_back(std::move(result));
    }
    return results;
}

}  // namespace tritonic::core
