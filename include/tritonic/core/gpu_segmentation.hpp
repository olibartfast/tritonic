#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
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
        {"NUM_DETECTIONS", "INT32", {1}},
        {"BOXES", "INT32", {100, 4}},
        {"SCORES", "FP32", {100}},
        {"CLASSES", "INT32", {100}},
        {"INSTANCE_RING_OFFSETS", "INT64", {101}},
        {"RING_POINT_OFFSETS", "INT64", {-1}},
        {"POLYGON_POINTS", "INT32", {-1, 2}},
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

inline float PolygonSignedArea(const std::vector<neuriplo_tasks::vision::Point2f>& ring) {
    float area = 0.0F;
    for (size_t index = 0; index < ring.size(); ++index) {
        const auto& current = ring[index];
        const auto& next = ring[(index + 1) % ring.size()];
        area += current.x * next.y - next.x * current.y;
    }
    return area * 0.5F;
}

inline bool PolygonContainsPoint(const std::vector<neuriplo_tasks::vision::Point2f>& ring,
                                 const neuriplo_tasks::vision::Point2f& point) {
    bool inside = false;
    for (size_t current = 0, previous = ring.size() - 1; current < ring.size();
         previous = current++) {
        const auto& a = ring[current];
        const auto& b = ring[previous];
        const bool crosses = (a.y > point.y) != (b.y > point.y);
        if (crosses && point.x < (b.x - a.x) * (point.y - a.y) / (b.y - a.y) + a.x)
            inside = !inside;
    }
    return inside;
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
    const auto& instance_offsets = get("INSTANCE_RING_OFFSETS").data;
    const auto& ring_offsets = get("RING_POINT_OFFSETS").data;
    const auto& points = get("POLYGON_POINTS").data;
    if (count_data.empty())
        throw std::runtime_error("NUM_DETECTIONS is empty");
    const int count = RequireInt32(count_data[0], "NUM_DETECTIONS");
    if (count < 0 || count > 100 || boxes.size() < 400 || scores.size() < 100 ||
        classes.size() < 100 || instance_offsets.size() < 101)
        throw std::runtime_error("GPU polygon output is invalid or truncated");
    if (RequireInt64(instance_offsets[0], "INSTANCE_RING_OFFSETS") != 0)
        throw std::runtime_error("GPU polygon instance offsets must begin at zero");

    int64_t previous_rings = 0;
    for (int index = 0; index <= 100; ++index) {
        const int64_t value =
            RequireInt64(instance_offsets[static_cast<size_t>(index)], "INSTANCE_RING_OFFSETS");
        if (value < previous_rings || (index > count && value != previous_rings))
            throw std::runtime_error("GPU polygon instance offsets are invalid");
        previous_rings = value;
    }
    const int64_t total_rings =
        RequireInt64(instance_offsets[static_cast<size_t>(count)], "INSTANCE_RING_OFFSETS");
    if (total_rings < 0 || ring_offsets.size() < static_cast<size_t>(total_rings + 1) ||
        RequireInt64(ring_offsets[0], "RING_POINT_OFFSETS") != 0)
        throw std::runtime_error("GPU polygon ring offsets are invalid or truncated");

    int64_t previous_points = 0;
    for (int64_t ring = 0; ring <= total_rings; ++ring) {
        const int64_t value =
            RequireInt64(ring_offsets[static_cast<size_t>(ring)], "RING_POINT_OFFSETS");
        if (value < previous_points)
            throw std::runtime_error("GPU polygon point offsets are not monotonic");
        previous_points = value;
    }
    const int64_t total_points =
        RequireInt64(ring_offsets[static_cast<size_t>(total_rings)], "RING_POINT_OFFSETS");
    if (total_points < 0 || points.size() < static_cast<size_t>(total_points * 2))
        throw std::runtime_error("GPU polygon points are invalid or truncated");

    std::vector<neuriplo_tasks::Result> results;
    results.reserve(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i) {
        const size_t detection = static_cast<size_t>(i);
        const int x = RequireInt32(boxes[detection * 4], "BOXES");
        const int y = RequireInt32(boxes[detection * 4 + 1], "BOXES");
        const int width = RequireInt32(boxes[detection * 4 + 2], "BOXES");
        const int height = RequireInt32(boxes[detection * 4 + 3], "BOXES");
        const float score = RequireFloat(scores[detection], "SCORES");
        const int class_id = RequireInt32(classes[detection], "CLASSES");
        if (!std::isfinite(score) || score < 0.0F || score > 1.0F || class_id < 0 || x < 0 ||
            y < 0 || width <= 0 || height <= 0 || x + width > frame_width ||
            y + height > frame_height)
            throw std::runtime_error("GPU segmentation emitted invalid or garbage geometry");

        const int64_t ring_begin =
            RequireInt64(instance_offsets[detection], "INSTANCE_RING_OFFSETS");
        const int64_t ring_end =
            RequireInt64(instance_offsets[detection + 1], "INSTANCE_RING_OFFSETS");
        std::vector<neuriplo_tasks::SegmentationPolygon> polygons;
        std::vector<std::vector<neuriplo_tasks::vision::Point2f>> holes;
        for (int64_t ring_index = ring_begin; ring_index < ring_end; ++ring_index) {
            const int64_t point_begin =
                RequireInt64(ring_offsets[static_cast<size_t>(ring_index)], "RING_POINT_OFFSETS");
            const int64_t point_end = RequireInt64(
                ring_offsets[static_cast<size_t>(ring_index + 1)], "RING_POINT_OFFSETS");
            if (point_end - point_begin < 3)
                throw std::runtime_error("GPU polygon ring has fewer than three points");
            std::vector<neuriplo_tasks::vision::Point2f> ring;
            ring.reserve(static_cast<size_t>(point_end - point_begin));
            for (int64_t point_index = point_begin; point_index < point_end; ++point_index) {
                const int px =
                    RequireInt32(points[static_cast<size_t>(point_index * 2)], "POLYGON_POINTS");
                const int py = RequireInt32(points[static_cast<size_t>(point_index * 2 + 1)],
                                            "POLYGON_POINTS");
                if (px < x || px > x + width || py < y || py > y + height || px < 0 ||
                    px > frame_width || py < 0 || py > frame_height)
                    throw std::runtime_error("GPU polygon point is outside its detection geometry");
                ring.push_back({static_cast<float>(px), static_cast<float>(py)});
            }
            const float area = PolygonSignedArea(ring);
            if (area > 0.0F)
                polygons.push_back({std::move(ring), {}});
            else if (area < 0.0F)
                holes.push_back(std::move(ring));
            else
                throw std::runtime_error("GPU polygon ring has zero area");
        }

        for (auto& hole : holes) {
            size_t owner = polygons.size();
            float owner_area = std::numeric_limits<float>::max();
            for (size_t polygon = 0; polygon < polygons.size(); ++polygon) {
                const float area = PolygonSignedArea(polygons[polygon].exterior);
                if (area < owner_area &&
                    PolygonContainsPoint(polygons[polygon].exterior, hole.front())) {
                    owner = polygon;
                    owner_area = area;
                }
            }
            if (owner == polygons.size())
                throw std::runtime_error("GPU polygon hole has no containing exterior ring");
            polygons[owner].holes.push_back(std::move(hole));
        }
        if (polygons.empty())
            throw std::runtime_error("GPU segmentation emitted no polygons");

        neuriplo_tasks::InstanceSegmentation result;
        result.class_id = static_cast<float>(class_id);
        result.class_confidence = score;
        result.bbox = {x, y, width, height};
        result.polygons = std::move(polygons);
        results.emplace_back(std::move(result));
    }
    return results;
}

}  // namespace tritonic::core
