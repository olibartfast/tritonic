#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "dali/pipeline/operator/operator.h"

namespace tritonic::dali_plugin {
namespace {
constexpr int kMaxDetections = 100;
constexpr int kDetectionRows = 300;
constexpr int kDetectionWidth = 38;
constexpr int kPrototypeChannels = 32;
constexpr int kPrototypeWidth = 160;
constexpr int kPrototypeHeight = 160;
constexpr float kInputSize = 640.0F;

__device__ float SigmoidDot(const float* coefficients, const float* prototypes, int y, int x) {
  float sum = 0.0F;
  const int pixel = y * kPrototypeWidth + x;
  for (int channel = 0; channel < kPrototypeChannels; ++channel) {
    sum += coefficients[channel] *
           prototypes[channel * kPrototypeWidth * kPrototypeHeight + pixel];
  }
  return 1.0F / (1.0F + expf(-sum));
}

__global__ void BuildMasks(const float* detections, const float* prototypes,
                           const int32_t* selected, const int64_t* offsets,
                           const int32_t* boxes, uint8_t* masks, int count,
                           float threshold, int64_t total_pixels) {
  const int64_t output_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (output_index >= total_pixels)
    return;

  int detection = 0;
  while (detection + 1 < count && output_index >= offsets[detection + 1])
    ++detection;
  const int width = boxes[detection * 4 + 2];
  const int height = boxes[detection * 4 + 3];
  const int64_t local_index = output_index - offsets[detection];
  const int dx = static_cast<int>(local_index % width);
  const int dy = static_cast<int>(local_index / width);
  const float* row = detections + selected[detection] * kDetectionWidth;
  const float* coefficients = row + 6;

  const int proto_x1 = max(0, min(static_cast<int>(row[0] * 0.25F), kPrototypeWidth - 1));
  const int proto_y1 = max(0, min(static_cast<int>(row[1] * 0.25F), kPrototypeHeight - 1));
  const int proto_x2 = min(max(proto_x1 + 1, static_cast<int>(row[2] * 0.25F)),
                           kPrototypeWidth);
  const int proto_y2 = min(max(proto_y1 + 1, static_cast<int>(row[3] * 0.25F)),
                           kPrototypeHeight);
  const int source_width = proto_x2 - proto_x1;
  const int source_height = proto_y2 - proto_y1;
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
  const double top = SigmoidDot(coefficients, prototypes, y0, x0) * (1.0 - wx) +
                     SigmoidDot(coefficients, prototypes, y0, x1) * wx;
  const double bottom = SigmoidDot(coefficients, prototypes, y1, x0) * (1.0 - wx) +
                        SigmoidDot(coefficients, prototypes, y1, x1) * wx;
  const float value = static_cast<float>(top * (1.0 - wy) + bottom * wy);
  masks[output_index] = value > threshold ? 255 : 0;
}

template <typename T>
void ResizeOutput(::dali::Workspace& workspace, int index, const ::dali::TensorShape<>& shape) {
  auto& output = workspace.Output<::dali::GPUBackend>(index);
  output.set_type<T>();
  output.Resize(::dali::uniform_list_shape(1, shape));
}
}  // namespace

class Yolo26SegPostprocess final : public ::dali::Operator<::dali::GPUBackend> {
 public:
  explicit Yolo26SegPostprocess(const ::dali::OpSpec& spec)
      : ::dali::Operator<::dali::GPUBackend>(spec),
        confidence_(spec.GetArgument<float>("confidence_threshold")),
        mask_threshold_(spec.GetArgument<float>("mask_threshold")) {}

 protected:
  bool SetupImpl(std::vector<::dali::OutputDesc>& output_desc,
                 const ::dali::Workspace&) override {
    output_desc.resize(7);
    return false;
  }

  void RunImpl(::dali::Workspace& workspace) override {
    const auto& detections_input = workspace.Input<::dali::GPUBackend>(0);
    const auto& prototypes_input = workspace.Input<::dali::GPUBackend>(1);
    const auto& original_size_input = workspace.Input<::dali::GPUBackend>(2);
    const auto stream = workspace.stream();
    std::vector<float> detections(kDetectionRows * kDetectionWidth);
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

    std::vector<int32_t> selected;
    std::vector<int32_t> boxes(kMaxDetections * 4, 0);
    std::vector<float> scores(kMaxDetections, 0.0F);
    std::vector<int32_t> classes(kMaxDetections, 0);
    std::vector<int32_t> shapes(kMaxDetections * 2, 0);
    std::vector<int64_t> offsets(kMaxDetections + 1, 0);
    selected.reserve(kMaxDetections);
    for (int row_index = 0; row_index < kDetectionRows &&
                            static_cast<int>(selected.size()) < kMaxDetections;
         ++row_index) {
      const float* row = detections.data() + row_index * kDetectionWidth;
      if (!std::isfinite(row[4]) || row[4] < confidence_)
        continue;
      const float x1 = std::clamp((row[0] - pad_width) / gain, 0.0F,
                                  static_cast<float>(width));
      const float y1 = std::clamp((row[1] - pad_height) / gain, 0.0F,
                                  static_cast<float>(height));
      const float x2 = std::clamp((row[2] - pad_width) / gain, 0.0F,
                                  static_cast<float>(width));
      const float y2 = std::clamp((row[3] - pad_height) / gain, 0.0F,
                                  static_cast<float>(height));
      const int x = std::clamp(static_cast<int>(x1), 0, width - 1);
      const int y = std::clamp(static_cast<int>(y1), 0, height - 1);
      const int box_width = std::max(1, std::min(static_cast<int>(x2 - x1), width - x));
      const int box_height = std::max(1, std::min(static_cast<int>(y2 - y1), height - y));
      const int index = static_cast<int>(selected.size());
      selected.push_back(row_index);
      boxes[index * 4] = x;
      boxes[index * 4 + 1] = y;
      boxes[index * 4 + 2] = box_width;
      boxes[index * 4 + 3] = box_height;
      scores[index] = row[4];
      classes[index] = static_cast<int32_t>(row[5]);
      shapes[index * 2] = box_height;
      shapes[index * 2 + 1] = box_width;
      offsets[index + 1] = offsets[index] + static_cast<int64_t>(box_width) * box_height;
    }
    const int32_t count = static_cast<int32_t>(selected.size());
    for (int index = count + 1; index <= kMaxDetections; ++index)
      offsets[index] = offsets[count];
    const int64_t total_pixels = offsets[count];

    ResizeOutput<int32_t>(workspace, 0, {1});
    ResizeOutput<int32_t>(workspace, 1, {kMaxDetections, 4});
    ResizeOutput<float>(workspace, 2, {kMaxDetections});
    ResizeOutput<int32_t>(workspace, 3, {kMaxDetections});
    ResizeOutput<int64_t>(workspace, 4, {kMaxDetections + 1});
    ResizeOutput<int32_t>(workspace, 5, {kMaxDetections, 2});
    ResizeOutput<uint8_t>(workspace, 6, {std::max<int64_t>(1, total_pixels)});

    auto copy_to_output = [&](int index, const void* source, size_t bytes) {
      CUDA_CALL(cudaMemcpyAsync(workspace.Output<::dali::GPUBackend>(index).raw_mutable_tensor(0),
                                source, bytes, cudaMemcpyHostToDevice, stream));
    };
    copy_to_output(0, &count, sizeof(count));
    copy_to_output(1, boxes.data(), boxes.size() * sizeof(int32_t));
    copy_to_output(2, scores.data(), scores.size() * sizeof(float));
    copy_to_output(3, classes.data(), classes.size() * sizeof(int32_t));
    copy_to_output(4, offsets.data(), offsets.size() * sizeof(int64_t));
    copy_to_output(5, shapes.data(), shapes.size() * sizeof(int32_t));
    auto* mask_output = static_cast<uint8_t*>(
        workspace.Output<::dali::GPUBackend>(6).raw_mutable_tensor(0));
    if (total_pixels == 0) {
      CUDA_CALL(cudaMemsetAsync(mask_output, 0, 1, stream));
      return;
    }

    int32_t* selected_device = nullptr;
    CUDA_CALL(cudaMallocAsync(&selected_device, selected.size() * sizeof(int32_t), stream));
    CUDA_CALL(cudaMemcpyAsync(selected_device, selected.data(), selected.size() * sizeof(int32_t),
                              cudaMemcpyHostToDevice, stream));
    const auto* detections_device =
        static_cast<const float*>(detections_input.raw_tensor(0));
    const auto* prototypes_device =
        static_cast<const float*>(prototypes_input.raw_tensor(0));
    const auto* offsets_device = static_cast<const int64_t*>(
        workspace.Output<::dali::GPUBackend>(4).raw_tensor(0));
    const auto* boxes_device = static_cast<const int32_t*>(
        workspace.Output<::dali::GPUBackend>(1).raw_tensor(0));
    constexpr int threads = 256;
    const int blocks = static_cast<int>((total_pixels + threads - 1) / threads);
    BuildMasks<<<blocks, threads, 0, stream>>>(
        detections_device, prototypes_device, selected_device, offsets_device, boxes_device,
        mask_output, count, mask_threshold_, total_pixels);
    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaFreeAsync(selected_device, stream));
  }

 private:
  float confidence_;
  float mask_threshold_;
};
}  // namespace tritonic::dali_plugin

DALI_REGISTER_OPERATOR(Yolo26SegPostprocess,
                       ::tritonic::dali_plugin::Yolo26SegPostprocess, ::dali::GPU);
DALI_SCHEMA(Yolo26SegPostprocess)
    .DocStr("YOLO26 segmentation postprocessing on CUDA")
    .NumInput(3)
    .NumOutput(7)
    .AddOptionalArg("confidence_threshold", "Detection confidence threshold", 0.5F)
    .AddOptionalArg("mask_threshold", "Binary mask threshold", 0.5F);
