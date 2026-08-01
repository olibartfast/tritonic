#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "dali/pipeline/operator/operator.h"

namespace tritonic::dali_plugin {
namespace {

constexpr int kMaxDetections = 100;
constexpr int kMaxAnchors = 8400;
constexpr int kPrototypeChannels = 32;
constexpr int kPrototypeWidth = 160;
constexpr int kPrototypeHeight = 160;
constexpr float kInputSize = 640.0F;

__device__ float MaskSigmoidDot(const float* coefficients, const float* prototypes, int y, int x) {
  float sum = 0.0F;
  const int pixel = y * kPrototypeWidth + x;
  for (int channel = 0; channel < kPrototypeChannels; ++channel) {
    sum += coefficients[channel] *
           prototypes[channel * kPrototypeWidth * kPrototypeHeight + pixel];
  }
  return 1.0F / (1.0F + expf(-sum));
}

__global__ void BuildPackedMasks(const float* detections, int num_channels,
                                 const float* prototypes, const int32_t* selected,
                                 const int64_t* offsets, const int32_t* boxes, uint8_t* masks,
                                 int count, float threshold, int64_t total_pixels) {
  const int64_t output_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (output_index >= total_pixels) return;

  int detection = 0;
  while (detection + 1 < count && output_index >= offsets[detection + 1]) ++detection;
  const int width = boxes[detection * 4 + 2];
  const int height = boxes[detection * 4 + 3];
  const int64_t local_index = output_index - offsets[detection];
  const int dx = static_cast<int>(local_index % width);
  const int dy = static_cast<int>(local_index / width);
  const int anchor_idx = selected[detection];

  const float cx = detections[anchor_idx];
  const float cy = detections[kMaxAnchors + anchor_idx];
  const float dw = detections[2 * kMaxAnchors + anchor_idx];
  const float dh = detections[3 * kMaxAnchors + anchor_idx];
  const float raw_x1 = (cx - dw * 0.5F);
  const float raw_y1 = (cy - dh * 0.5F);
  const float raw_x2 = (cx + dw * 0.5F);
  const float raw_y2 = (cy + dh * 0.5F);
  const int proto_x1 = max(0, min(static_cast<int>(raw_x1 * 0.25F), kPrototypeWidth - 1));
  const int proto_y1 = max(0, min(static_cast<int>(raw_y1 * 0.25F), kPrototypeHeight - 1));
  const int proto_x2 = min(max(proto_x1 + 1, static_cast<int>(raw_x2 * 0.25F)), kPrototypeWidth);
  const int proto_y2 = min(max(proto_y1 + 1, static_cast<int>(raw_y2 * 0.25F)), kPrototypeHeight);
  const int source_width = max(1, proto_x2 - proto_x1);
  const int source_height = max(1, proto_y2 - proto_y1);
  const double fx = (static_cast<double>(dx) + 0.5) * source_width / width - 0.5;
  const double fy = (static_cast<double>(dy) + 0.5) * source_height / height - 0.5;
  const int raw_x0 = static_cast<int>(floor(fx));
  const int raw_y0 = static_cast<int>(floor(fy));
  const int x0 = proto_x1 + max(0, min(raw_x0, source_width - 1));
  const int y0 = proto_y1 + max(0, min(raw_y0, source_height - 1));
  const int x1 = proto_x1 + max(0, min(raw_x0 + 1, source_width - 1));
  const int y1 = proto_y1 + max(0, min(raw_y0 + 1, source_height - 1));
  const double wx = fx - raw_x0;
  const double wy = fy - raw_y0;

  float coeffs[kPrototypeChannels];
  for (int c = 0; c < kPrototypeChannels; ++c) {
    coeffs[c] = detections[(num_channels - kPrototypeChannels + c) * kMaxAnchors + anchor_idx];
  }
  const double top = MaskSigmoidDot(coeffs, prototypes, y0, x0) * (1.0 - wx) +
                     MaskSigmoidDot(coeffs, prototypes, y0, x1) * wx;
  const double bottom = MaskSigmoidDot(coeffs, prototypes, y1, x0) * (1.0 - wx) +
                        MaskSigmoidDot(coeffs, prototypes, y1, x1) * wx;
  const float value = static_cast<float>(top * (1.0 - wy) + bottom * wy);
  masks[output_index] = value > threshold ? 255 : 0;
}

float ComputeIoU(const int32_t* a, const int32_t* b) {
  const int ax1 = a[0], ay1 = a[1], ax2 = a[0] + a[2], ay2 = a[1] + a[3];
  const int bx1 = b[0], by1 = b[1], bx2 = b[0] + b[2], by2 = b[1] + b[3];
  const int inter_x1 = std::max(ax1, bx1);
  const int inter_y1 = std::max(ay1, by1);
  const int inter_x2 = std::min(ax2, bx2);
  const int inter_y2 = std::min(ay2, by2);
  if (inter_x2 <= inter_x1 || inter_y2 <= inter_y1) return 0.0F;
  const float inter_area = static_cast<float>(inter_x2 - inter_x1) * (inter_y2 - inter_y1);
  const float area_a = static_cast<float>(a[2]) * a[3];
  const float area_b = static_cast<float>(b[2]) * b[3];
  return inter_area / (area_a + area_b - inter_area);
}

std::vector<int32_t> ApplyNms(const std::vector<int32_t>& boxes,
                               const std::vector<int32_t>& classes,
                               const std::vector<float>& scores, int count, float nms_threshold) {
  std::vector<int> indices(count);
  for (int i = 0; i < count; ++i) indices[i] = i;
  std::sort(indices.begin(), indices.begin() + count,
            [&](int a, int b) { return scores[a] > scores[b]; });
  std::vector<bool> suppressed(count, false);
  for (int i = 0; i < count; ++i) {
    if (suppressed[indices[i]]) continue;
    for (int j = i + 1; j < count; ++j) {
      if (suppressed[indices[j]]) continue;
      if (classes[indices[i]] != classes[indices[j]]) continue;
      const float iou = ComputeIoU(&boxes[indices[i] * 4], &boxes[indices[j] * 4]);
      if (iou > nms_threshold) suppressed[indices[j]] = true;
    }
  }
  std::vector<int32_t> kept;
  for (int i = 0; i < count; ++i)
    if (!suppressed[indices[i]]) kept.push_back(indices[i]);
  return kept;
}

template <typename T>
void ResizeOutput(::dali::Workspace& workspace, int index, const ::dali::TensorShape<>& shape) {
  auto& output = workspace.Output<::dali::GPUBackend>(index);
  output.set_type<T>();
  output.Resize(::dali::uniform_list_shape(1, shape));
}

}  // namespace

class Yolo11SegMaskPostprocess final : public ::dali::Operator<::dali::GPUBackend> {
public:
  explicit Yolo11SegMaskPostprocess(const ::dali::OpSpec& spec)
      : ::dali::Operator<::dali::GPUBackend>(spec),
        confidence_(spec.GetArgument<float>("confidence_threshold")),
        mask_threshold_(spec.GetArgument<float>("mask_threshold")) {}

protected:
  bool SetupImpl(std::vector<::dali::OutputDesc>& output_desc,
                 const ::dali::Workspace&) override {
    output_desc.resize(6);
    return false;
  }

  void RunImpl(::dali::Workspace& workspace) override {
    const auto& detections_input = workspace.Input<::dali::GPUBackend>(0);
    const auto& prototypes_input = workspace.Input<::dali::GPUBackend>(1);
    const auto& original_size_input = workspace.Input<::dali::GPUBackend>(2);
    const auto stream = workspace.stream();

    const auto& det_shape = detections_input.tensor_shape(0);
    const int num_channels = det_shape[0];
    const int num_anchors = det_shape[1];
    DALI_ENFORCE(num_channels >= 4 + kPrototypeChannels,
                 "Detection channels too small for 4 bbox + N classes + 32 mask coeffs");
    const int num_classes = num_channels - 4 - kPrototypeChannels;

    std::vector<float> detections(static_cast<size_t>(num_channels) * num_anchors);
    int64_t original_size[2]{};
    CUDA_CALL(cudaMemcpyAsync(detections.data(), detections_input.raw_tensor(0),
                              detections.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CALL(cudaMemcpyAsync(original_size, original_size_input.raw_tensor(0),
                              sizeof(original_size), cudaMemcpyDeviceToHost, stream));
    CUDA_CALL(cudaStreamSynchronize(stream));

    const int height = static_cast<int>(original_size[0]);
    const int width = static_cast<int>(original_size[1]);
    DALI_ENFORCE(width > 0 && height > 0, "Invalid original image size");
    const float gain = std::min(kInputSize / width, kInputSize / height);
    const float pad_width = (kInputSize - gain * width) / 2.0F;
    const float pad_height = (kInputSize - gain * height) / 2.0F;

    std::vector<int32_t> filtered_indices;
    std::vector<int32_t> boxes(kMaxDetections * 4, 0);
    std::vector<float> scores(kMaxDetections, 0.0F);
    std::vector<int32_t> classes(kMaxDetections, 0);
    filtered_indices.reserve(kMaxDetections);

    for (int anchor = 0;
         anchor < num_anchors && static_cast<int>(filtered_indices.size()) < kMaxDetections;
         ++anchor) {
      const float cx = detections[anchor];
      const float cy = detections[kMaxAnchors + anchor];
      const float w = detections[2 * kMaxAnchors + anchor];
      const float h = detections[3 * kMaxAnchors + anchor];

      float max_score = -INFINITY;
      int best_class = 0;
      for (int c = 0; c < num_classes; ++c) {
        const float score = detections[(4 + c) * kMaxAnchors + anchor];
        if (score > max_score) { max_score = score; best_class = c; }
      }
      if (max_score < confidence_) continue;

      const float x1 = std::clamp((cx - w * 0.5F - pad_width) / gain, 0.0F, static_cast<float>(width));
      const float y1 = std::clamp((cy - h * 0.5F - pad_height) / gain, 0.0F, static_cast<float>(height));
      const float x2 = std::clamp((cx + w * 0.5F - pad_width) / gain, 0.0F, static_cast<float>(width));
      const float y2 = std::clamp((cy + h * 0.5F - pad_height) / gain, 0.0F, static_cast<float>(height));
      const int x = std::clamp(static_cast<int>(x1), 0, width - 1);
      const int y = std::clamp(static_cast<int>(y1), 0, height - 1);
      const int box_width = std::max(1, std::min(static_cast<int>(x2 - x1), width - x));
      const int box_height = std::max(1, std::min(static_cast<int>(y2 - y1), height - y));

      const int index = static_cast<int>(filtered_indices.size());
      filtered_indices.push_back(anchor);
      boxes[index * 4] = x;
      boxes[index * 4 + 1] = y;
      boxes[index * 4 + 2] = box_width;
      boxes[index * 4 + 3] = box_height;
      scores[index] = max_score;
      classes[index] = best_class;
    }

    const int raw_count = static_cast<int>(filtered_indices.size());
    const auto kept = ApplyNms(boxes, classes, scores, raw_count, 0.45F);
    const int32_t count = static_cast<int32_t>(kept.size());

    std::vector<int32_t> selected(count, 0);
    std::vector<int32_t> final_boxes(kMaxDetections * 4, 0);
    std::vector<float> final_scores(kMaxDetections, 0.0F);
    std::vector<int32_t> final_classes(kMaxDetections, 0);
    std::vector<int64_t> mask_offsets(kMaxDetections + 1, 0);
    for (int i = 0; i < count; ++i) {
      const int src = kept[i];
      selected[i] = filtered_indices[src];
      final_boxes[i * 4] = boxes[src * 4];
      final_boxes[i * 4 + 1] = boxes[src * 4 + 1];
      final_boxes[i * 4 + 2] = boxes[src * 4 + 2];
      final_boxes[i * 4 + 3] = boxes[src * 4 + 3];
      final_scores[i] = scores[src];
      final_classes[i] = classes[src];
      mask_offsets[i + 1] = mask_offsets[i] +
          static_cast<int64_t>(final_boxes[i * 4 + 2]) * final_boxes[i * 4 + 3];
    }
    for (int i = count + 1; i <= kMaxDetections; ++i) mask_offsets[i] = mask_offsets[count];
    const int64_t total_pixels = mask_offsets[count];

    ResizeOutput<int32_t>(workspace, 0, {1});
    ResizeOutput<int32_t>(workspace, 1, {kMaxDetections, 4});
    ResizeOutput<float>(workspace, 2, {kMaxDetections});
    ResizeOutput<int32_t>(workspace, 3, {kMaxDetections});
    auto copy_h2d = [&](int idx, const void* src, size_t bytes) {
      CUDA_CALL(cudaMemcpyAsync(
          workspace.Output<::dali::GPUBackend>(idx).raw_mutable_tensor(0), src, bytes,
          cudaMemcpyHostToDevice, stream));
    };
    copy_h2d(0, &count, sizeof(count));
    copy_h2d(1, final_boxes.data(), final_boxes.size() * sizeof(int32_t));
    copy_h2d(2, final_scores.data(), final_scores.size() * sizeof(float));
    copy_h2d(3, final_classes.data(), final_classes.size() * sizeof(int32_t));

    if (count == 0) {
      ResizeOutput<int64_t>(workspace, 4, {1});
      ResizeOutput<uint8_t>(workspace, 5, {1});
      const int64_t zero = 0;
      const uint8_t zero_byte = 0;
      copy_h2d(4, &zero, sizeof(zero));
      copy_h2d(5, &zero_byte, sizeof(zero_byte));
      return;
    }

    ResizeOutput<int64_t>(workspace, 4, {kMaxDetections + 1});
    ResizeOutput<uint8_t>(workspace, 5, {total_pixels});
    copy_h2d(4, mask_offsets.data(), mask_offsets.size() * sizeof(int64_t));

    const auto* detections_device = static_cast<const float*>(detections_input.raw_tensor(0));
    const auto* prototypes_device = static_cast<const float*>(prototypes_input.raw_tensor(0));
    int32_t* selected_device = nullptr;
    int64_t* mask_offsets_device = nullptr;
    int32_t* boxes_device = nullptr;
    uint8_t* mask_output = static_cast<uint8_t*>(
        workspace.Output<::dali::GPUBackend>(5).raw_mutable_tensor(0));
    CUDA_CALL(cudaMallocAsync(&selected_device, selected.size() * sizeof(int32_t), stream));
    CUDA_CALL(cudaMallocAsync(&mask_offsets_device, mask_offsets.size() * sizeof(int64_t), stream));
    CUDA_CALL(cudaMallocAsync(&boxes_device, final_boxes.size() * sizeof(int32_t), stream));
    CUDA_CALL(cudaMemcpyAsync(selected_device, selected.data(), selected.size() * sizeof(int32_t),
                              cudaMemcpyHostToDevice, stream));
    CUDA_CALL(cudaMemcpyAsync(mask_offsets_device, mask_offsets.data(),
                              mask_offsets.size() * sizeof(int64_t), cudaMemcpyHostToDevice,
                              stream));
    CUDA_CALL(cudaMemcpyAsync(boxes_device, final_boxes.data(), final_boxes.size() * sizeof(int32_t),
                              cudaMemcpyHostToDevice, stream));

    constexpr int threads = 256;
    const int blocks = static_cast<int>((total_pixels + threads - 1) / threads);
    BuildPackedMasks<<<blocks, threads, 0, stream>>>(
        detections_device, num_channels, prototypes_device, selected_device, mask_offsets_device,
        boxes_device, mask_output, count, mask_threshold_, total_pixels);
    CUDA_CALL(cudaGetLastError());

    CUDA_CALL(cudaFreeAsync(selected_device, stream));
    CUDA_CALL(cudaFreeAsync(mask_offsets_device, stream));
    CUDA_CALL(cudaFreeAsync(boxes_device, stream));
  }

private:
  float confidence_;
  float mask_threshold_;
};

}  // namespace tritonic::dali_plugin

DALI_REGISTER_OPERATOR(Yolo11SegMaskPostprocess, ::tritonic::dali_plugin::Yolo11SegMaskPostprocess,
                       ::dali::GPU);
DALI_SCHEMA(Yolo11SegMaskPostprocess)
    .DocStr("YOLO11 segmentation packed-mask postprocessing on CUDA (NMS + mask only)")
    .NumInput(3)
    .NumOutput(6)
    .AddOptionalArg("confidence_threshold", "Detection confidence threshold", 0.5F)
    .AddOptionalArg("mask_threshold", "Binary mask threshold", 0.5F);
