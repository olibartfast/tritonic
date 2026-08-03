#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "dali/pipeline/operator/operator.h"

namespace tritonic::dali_plugin {
namespace {
constexpr int kMaxDetections = 100;
constexpr float kInputSize = 640.0F;

template <typename T>
void ResizeOutput(::dali::Workspace& workspace, int index, const ::dali::TensorShape<>& shape) {
    auto& output = workspace.Output<::dali::GPUBackend>(index);
    output.set_type<T>();
    output.Resize(::dali::uniform_list_shape(1, shape));
}
}  // namespace

class Yolo26DetPostprocess final : public ::dali::Operator<::dali::GPUBackend> {
public:
    explicit Yolo26DetPostprocess(const ::dali::OpSpec& spec)
        : ::dali::Operator<::dali::GPUBackend>(spec),
          confidence_(spec.GetArgument<float>("confidence_threshold")),
          normalized_boxes_(spec.GetArgument<bool>("normalized_boxes")) {}

protected:
    bool SetupImpl(std::vector<::dali::OutputDesc>& output_desc,
                   const ::dali::Workspace&) override {
        output_desc.resize(4);
        return false;
    }

    void RunImpl(::dali::Workspace& workspace) override {
        const auto& detections_input = workspace.Input<::dali::GPUBackend>(0);
        const auto& original_size_input = workspace.Input<::dali::GPUBackend>(1);
        const auto stream = workspace.stream();

        const auto detections_shape = detections_input.tensor_shape(0);
        DALI_ENFORCE(detections_shape.size() == 2,
                     "Detection tensor must be 2-D [rows, width]");
        const int detection_rows = static_cast<int>(detections_shape[0]);
        const int detection_width = static_cast<int>(detections_shape[1]);
        // Both dimensions drive the copy below, so neither may be assumed: sizing the
        // buffer from a constant while reading the width from the shape lets a shorter
        // engine output read past the end of the device allocation.
        DALI_ENFORCE(detection_rows > 0, "Detection tensor has no rows");
        DALI_ENFORCE(detection_width >= 6,
                     "Detection row must hold at least x1,y1,x2,y2,score,class");
        std::vector<float> detections(static_cast<size_t>(detection_rows) * detection_width);
        int64_t original_size[2]{};
        CUDA_CALL(cudaMemcpyAsync(detections.data(), detections_input.raw_tensor(0),
                                  detections.size() * sizeof(float), cudaMemcpyDeviceToHost,
                                  stream));
        CUDA_CALL(cudaMemcpyAsync(original_size, original_size_input.raw_tensor(0),
                                  sizeof(original_size), cudaMemcpyDeviceToHost, stream));
        CUDA_CALL(cudaStreamSynchronize(stream));

        const int height = static_cast<int>(original_size[0]);
        const int width = static_cast<int>(original_size[1]);
        DALI_ENFORCE(width > 0 && height > 0, "Invalid original image size");
        const float gain = std::min(kInputSize / width, kInputSize / height);
        const float pad_width = (kInputSize - gain * width) / 2.0F;
        const float pad_height = (kInputSize - gain * height) / 2.0F;

        struct Detection {
            int box_x;
            int box_y;
            int box_width;
            int box_height;
            float confidence;
            int32_t class_id;
            int row_index;
        };
        std::vector<Detection> selected;
        selected.reserve(kMaxDetections);
        // Safe to stop at kMaxDetections mid-scan *only* because the YOLO26
        // end-to-end head emits its rows already sorted by descending score, so the
        // first kMaxDetections survivors are the highest-scoring ones. Do not copy
        // this loop to a raw anchor-grid head (YOLO11), where rows are in spatial
        // order and this silently drops the strongest detections.
        for (int row_index = 0;
             row_index < detection_rows && static_cast<int>(selected.size()) < kMaxDetections;
             ++row_index) {
            const float* row = detections.data() + row_index * detection_width;
            if (!std::isfinite(row[4]) || row[4] < confidence_)
                continue;
            const float denorm = normalized_boxes_ ? kInputSize : 1.0F;
            const float x1 =
                std::clamp((row[0] * denorm - pad_width) / gain, 0.0F, static_cast<float>(width));
            const float y1 =
                std::clamp((row[1] * denorm - pad_height) / gain, 0.0F, static_cast<float>(height));
            const float x2 =
                std::clamp((row[2] * denorm - pad_width) / gain, 0.0F, static_cast<float>(width));
            const float y2 =
                std::clamp((row[3] * denorm - pad_height) / gain, 0.0F, static_cast<float>(height));
            const int x = std::clamp(static_cast<int>(x1), 0, width - 1);
            const int y = std::clamp(static_cast<int>(y1), 0, height - 1);
            const int box_width = std::max(1, std::min(static_cast<int>(x2 - x1), width - x));
            const int box_height = std::max(1, std::min(static_cast<int>(y2 - y1), height - y));
            selected.push_back({x, y, box_width, box_height, row[4],
                                static_cast<int32_t>(row[5]), row_index});
        }
        const int32_t count = static_cast<int32_t>(selected.size());

        std::sort(selected.begin(), selected.end(),
                  [](const Detection& a, const Detection& b) { return a.confidence > b.confidence; });

        const int32_t output_count = std::min(count, kMaxDetections);
        std::vector<int32_t> boxes(kMaxDetections * 4, 0);
        std::vector<float> scores(kMaxDetections, 0.0F);
        std::vector<int32_t> classes(kMaxDetections, 0);
        for (int i = 0; i < output_count; ++i) {
            boxes[i * 4] = selected[i].box_x;
            boxes[i * 4 + 1] = selected[i].box_y;
            boxes[i * 4 + 2] = selected[i].box_width;
            boxes[i * 4 + 3] = selected[i].box_height;
            scores[i] = selected[i].confidence;
            classes[i] = selected[i].class_id;
        }

        ResizeOutput<int32_t>(workspace, 0, {1});
        ResizeOutput<int32_t>(workspace, 1, {kMaxDetections, 4});
        ResizeOutput<float>(workspace, 2, {kMaxDetections});
        ResizeOutput<int32_t>(workspace, 3, {kMaxDetections});
        auto copy_to_output = [&](int index, const void* source, size_t bytes) {
            CUDA_CALL(
                cudaMemcpyAsync(workspace.Output<::dali::GPUBackend>(index).raw_mutable_tensor(0),
                                source, bytes, cudaMemcpyHostToDevice, stream));
        };
        copy_to_output(0, &output_count, sizeof(output_count));
        copy_to_output(1, boxes.data(), boxes.size() * sizeof(int32_t));
        copy_to_output(2, scores.data(), scores.size() * sizeof(float));
        copy_to_output(3, classes.data(), classes.size() * sizeof(int32_t));
    }

private:
    float confidence_;
    // Whether the engine emits boxes in [0,1] rather than [0,640]. Previously inferred
    // from the column count, which silently scaled boxes by 640x for any engine whose
    // width did not match the assumption.
    bool normalized_boxes_;
};
}  // namespace tritonic::dali_plugin

DALI_REGISTER_OPERATOR(Yolo26DetPostprocess, ::tritonic::dali_plugin::Yolo26DetPostprocess,
                       ::dali::GPU);
DALI_SCHEMA(Yolo26DetPostprocess)
    .DocStr("YOLO26 detection bbox postprocessing")
    .NumInput(2)
    .NumOutput(4)
    .AddOptionalArg("confidence_threshold", "Detection confidence threshold", 0.5F)
    .AddOptionalArg("normalized_boxes",
                    "Engine emits boxes in [0,1] rather than input-pixel coordinates",
                    true);
