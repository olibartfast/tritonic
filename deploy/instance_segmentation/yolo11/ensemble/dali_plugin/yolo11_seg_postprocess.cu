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

__device__ float SigmoidDot(const float* coefficients, const float* prototypes, int y, int x) {
  float sum = 0.0F;
  const int pixel = y * kPrototypeWidth + x;
  for (int channel = 0; channel < kPrototypeChannels; ++channel) {
    sum += coefficients[channel] *
           prototypes[channel * kPrototypeWidth * kPrototypeHeight + pixel];
  }
  return 1.0F / (1.0F + expf(-sum));
}

__global__ void BuildMasks(const float* detections, int num_channels, const float* prototypes,
                           const int32_t* selected, const int64_t* offsets, const int32_t* boxes,
                           uint8_t* masks, int count, float threshold, int64_t total_pixels) {
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

  float coeffs_local[kPrototypeChannels];
  for (int c = 0; c < kPrototypeChannels; ++c) {
    coeffs_local[c] = detections[(num_channels - kPrototypeChannels + c) * kMaxAnchors + anchor_idx];
  }
  const double top = SigmoidDot(coeffs_local, prototypes, y0, x0) * (1.0 - wx) +
                     SigmoidDot(coeffs_local, prototypes, y0, x1) * wx;
  const double bottom = SigmoidDot(coeffs_local, prototypes, y1, x0) * (1.0 - wx) +
                        SigmoidDot(coeffs_local, prototypes, y1, x1) * wx;
  const float value = static_cast<float>(top * (1.0 - wy) + bottom * wy);
  masks[output_index] = value > threshold ? 255 : 0;
}

__device__ bool Foreground(const uint8_t* masks, int64_t mask_offset, int width, int height,
                           int x, int y) {
  return x >= 0 && x < width && y >= 0 && y < height &&
         masks[mask_offset + static_cast<int64_t>(y) * width + x] != 0;
}

__device__ bool BoundaryEdge(const uint8_t* masks, int64_t mask_offset, int width, int height,
                             int x, int y, int direction) {
  if (!Foreground(masks, mask_offset, width, height, x, y)) return false;
  if (direction == 0) return !Foreground(masks, mask_offset, width, height, x, y - 1);
  if (direction == 1) return !Foreground(masks, mask_offset, width, height, x + 1, y);
  if (direction == 2) return !Foreground(masks, mask_offset, width, height, x, y + 1);
  return !Foreground(masks, mask_offset, width, height, x - 1, y);
}

__device__ int64_t VisitedIndex(int64_t mask_offset, int width, int x, int y, int direction) {
  return (mask_offset + static_cast<int64_t>(y) * width + x) * 4 + direction;
}

__device__ void EdgeStart(int x, int y, int direction, int& vertex_x, int& vertex_y) {
  vertex_x = x + (direction == 1 || direction == 2 ? 1 : 0);
  vertex_y = y + (direction == 2 || direction == 3 ? 1 : 0);
}

__device__ void EdgeEnd(int x, int y, int direction, int& vertex_x, int& vertex_y) {
  vertex_x = x + (direction == 0 || direction == 1 ? 1 : 0);
  vertex_y = y + (direction == 1 || direction == 2 ? 1 : 0);
}

__device__ void CandidateCell(int vertex_x, int vertex_y, int direction, int& x, int& y) {
  if (direction == 0) { x = vertex_x; y = vertex_y; }
  else if (direction == 1) { x = vertex_x - 1; y = vertex_y; }
  else if (direction == 2) { x = vertex_x - 1; y = vertex_y - 1; }
  else { x = vertex_x; y = vertex_y - 1; }
}

__global__ void EnumerateBoundaryEdges(const uint8_t* masks, const int64_t* mask_offsets,
                                       const int32_t* boxes, int32_t* edge_cells,
                                       uint8_t* edge_directions, int32_t* edge_counts, int count,
                                       int64_t total_pixels) {
  const int64_t output_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (output_index >= total_pixels) return;

  int detection = 0;
  while (detection + 1 < count && output_index >= mask_offsets[detection + 1]) ++detection;
  const int width = boxes[detection * 4 + 2];
  const int height = boxes[detection * 4 + 3];
  const int64_t mask_offset = mask_offsets[detection];
  const int64_t local_index = output_index - mask_offset;
  const int x = static_cast<int>(local_index % width);
  const int y = static_cast<int>(local_index / width);
  const int64_t edge_base = mask_offset * 4;
  for (int direction = 0; direction < 4; ++direction) {
    if (!BoundaryEdge(masks, mask_offset, width, height, x, y, direction)) continue;
    const int slot = atomicAdd(edge_counts + detection, 1);
    const int64_t edge_index = edge_base + slot;
    edge_cells[edge_index * 2] = x;
    edge_cells[edge_index * 2 + 1] = y;
    edge_directions[edge_index] = static_cast<uint8_t>(direction);
  }
}

__global__ void TracePolygons(const uint8_t* masks, const int64_t* mask_offsets,
                              const int32_t* boxes, const int32_t* edge_cells,
                              const uint8_t* edge_directions, const int32_t* edge_counts,
                              uint8_t* visited, int32_t* point_scratch, int32_t* ring_counts,
                              int32_t* ring_point_counts, int32_t* errors, int count) {
  const int detection = blockIdx.x;
  if (detection >= count || threadIdx.x != 0) return;

  const int box_x = boxes[detection * 4];
  const int box_y = boxes[detection * 4 + 1];
  const int width = boxes[detection * 4 + 2];
  const int height = boxes[detection * 4 + 3];
  const int64_t mask_offset = mask_offsets[detection];
  const int64_t point_base = mask_offset * 4;
  const int64_t ring_base = mask_offset;
  int ring_count = 0;
  int point_count = 0;

  for (int edge = 0; edge < edge_counts[detection]; ++edge) {
    const int64_t edge_index = point_base + edge;
    const int x = edge_cells[edge_index * 2];
    const int y = edge_cells[edge_index * 2 + 1];
    const int initial_direction = edge_directions[edge_index];
    const int64_t initial_visited = VisitedIndex(mask_offset, width, x, y, initial_direction);
    if (visited[initial_visited] != 0) continue;

    int current_x = x;
    int current_y = y;
    int current_direction = initial_direction;
    int start_x = 0, start_y = 0;
    EdgeStart(current_x, current_y, current_direction, start_x, start_y);
    const int ring_point_begin = point_count;
    point_scratch[(point_base + point_count) * 2] = box_x + start_x;
    point_scratch[(point_base + point_count) * 2 + 1] = box_y + start_y;
    ++point_count;

    const int max_steps = width * height * 4;
    bool closed = false;
    for (int step = 0; step < max_steps; ++step) {
      visited[VisitedIndex(mask_offset, width, current_x, current_y, current_direction)] = 1;
      int end_x = 0, end_y = 0;
      EdgeEnd(current_x, current_y, current_direction, end_x, end_y);
      if (end_x == start_x && end_y == start_y) { closed = true; break; }

      const int turns[4] = {1, 0, 3, 2};
      int next_x = 0, next_y = 0, next_direction = -1;
      for (int turn_index = 0; turn_index < 4; ++turn_index) {
        const int candidate_direction = (current_direction + turns[turn_index]) % 4;
        int candidate_x = 0, candidate_y = 0;
        CandidateCell(end_x, end_y, candidate_direction, candidate_x, candidate_y);
        if (candidate_x < 0 || candidate_x >= width || candidate_y < 0 || candidate_y >= height ||
            !BoundaryEdge(masks, mask_offset, width, height, candidate_x, candidate_y,
                          candidate_direction) ||
            visited[VisitedIndex(mask_offset, width, candidate_x, candidate_y,
                                 candidate_direction)] != 0)
          continue;
        next_x = candidate_x; next_y = candidate_y; next_direction = candidate_direction; break;
      }
      if (next_direction < 0) { errors[detection] = 1; return; }
      if (next_direction != current_direction) {
        point_scratch[(point_base + point_count) * 2] = box_x + end_x;
        point_scratch[(point_base + point_count) * 2 + 1] = box_y + end_y;
        ++point_count;
      }
      current_x = next_x; current_y = next_y; current_direction = next_direction;
    }
    const int ring_points = point_count - ring_point_begin;
    if (!closed || ring_points < 3) { errors[detection] = 2; return; }
    ring_point_counts[ring_base + ring_count] = ring_points;
    ++ring_count;
  }
  if (ring_count == 0) { errors[detection] = 3; return; }
  ring_counts[detection] = ring_count;
}

__device__ int64_t HullCross(const int32_t* points, int64_t base, int origin, int a, int b) {
  const int64_t ox = points[(base + origin) * 2];
  const int64_t oy = points[(base + origin) * 2 + 1];
  const int64_t ax = points[(base + a) * 2];
  const int64_t ay = points[(base + a) * 2 + 1];
  const int64_t bx = points[(base + b) * 2];
  const int64_t by = points[(base + b) * 2 + 1];
  return (ax - ox) * (by - oy) - (ay - oy) * (bx - ox);
}

__device__ int64_t HullDistanceSquared(const int32_t* points, int64_t base, int origin, int point) {
  const int64_t dx = points[(base + point) * 2] - points[(base + origin) * 2];
  const int64_t dy = points[(base + point) * 2 + 1] - points[(base + origin) * 2 + 1];
  return dx * dx + dy * dy;
}

__global__ void ConvexHullRings(const int32_t* contours, int32_t* hulls,
                                const int64_t* mask_offsets, const int32_t* ring_counts,
                                int32_t* ring_point_counts, int32_t* errors, int count) {
  const int detection = blockIdx.x;
  if (detection >= count || threadIdx.x != 0 || errors[detection] != 0) return;

  const int64_t point_base = mask_offsets[detection] * 4;
  const int64_t ring_base = mask_offsets[detection];
  int64_t source_point = point_base;
  int64_t destination_point = point_base;
  for (int ring = 0; ring < ring_counts[detection]; ++ring) {
    const int source_count = ring_point_counts[ring_base + ring];
    int64_t source_area_twice = 0;
    for (int point = 0; point < source_count; ++point) {
      const int next = (point + 1) % source_count;
      source_area_twice += static_cast<int64_t>(contours[(source_point + point) * 2]) *
                               contours[(source_point + next) * 2 + 1] -
                           static_cast<int64_t>(contours[(source_point + next) * 2]) *
                               contours[(source_point + point) * 2 + 1];
    }

    int start = 0;
    for (int point = 1; point < source_count; ++point) {
      const int32_t x = contours[(source_point + point) * 2];
      const int32_t y = contours[(source_point + point) * 2 + 1];
      const int32_t start_x = contours[(source_point + start) * 2];
      const int32_t start_y = contours[(source_point + start) * 2 + 1];
      if (x < start_x || (x == start_x && y < start_y)) start = point;
    }

    int current = start;
    int hull_count = 0;
    do {
      hulls[(destination_point + hull_count) * 2] = contours[(source_point + current) * 2];
      hulls[(destination_point + hull_count) * 2 + 1] =
          contours[(source_point + current) * 2 + 1];
      ++hull_count;
      int next = -1;
      for (int candidate = 0; candidate < source_count; ++candidate) {
        if (candidate == current) continue;
        if (next < 0) { next = candidate; continue; }
        const int64_t cross = HullCross(contours, source_point, current, next, candidate);
        if (cross < 0 ||
            (cross == 0 && HullDistanceSquared(contours, source_point, current, candidate) >
                               HullDistanceSquared(contours, source_point, current, next)))
          next = candidate;
      }
      current = next;
    } while (current != start && hull_count <= source_count);

    if (hull_count < 3 || hull_count > source_count) { errors[detection] = 4; return; }

    int64_t hull_area_twice = 0;
    for (int point = 0; point < hull_count; ++point) {
      const int next = (point + 1) % hull_count;
      hull_area_twice += static_cast<int64_t>(hulls[(destination_point + point) * 2]) *
                             hulls[(destination_point + next) * 2 + 1] -
                         static_cast<int64_t>(hulls[(destination_point + next) * 2]) *
                             hulls[(destination_point + point) * 2 + 1];
    }
    if ((source_area_twice < 0) != (hull_area_twice < 0)) {
      for (int left = 0, right = hull_count - 1; left < right; ++left, --right) {
        const int32_t left_x = hulls[(destination_point + left) * 2];
        const int32_t left_y = hulls[(destination_point + left) * 2 + 1];
        hulls[(destination_point + left) * 2] = hulls[(destination_point + right) * 2];
        hulls[(destination_point + left) * 2 + 1] = hulls[(destination_point + right) * 2 + 1];
        hulls[(destination_point + right) * 2] = left_x;
        hulls[(destination_point + right) * 2 + 1] = left_y;
      }
    }
    ring_point_counts[ring_base + ring] = hull_count;
    source_point += source_count;
    destination_point += hull_count;
  }
}

float ComputeIoU(const int32_t* a, const int32_t* b) {
  const int ax1 = a[0], ay1 = a[1], ax2 = a[0] + a[2], ay2 = a[1] + a[3];
  const int bx1 = b[0], by1 = b[1], bx2 = b[0] + b[2], by2 = b[1] + b[3];
  const int inter_x1 = std::max(ax1, bx1);
  const int inter_y1 = std::max(ay1, by1);
  const int inter_x2 = std::min(ax2, bx2);
  const int inter_y2 = std::min(ay2, by2);
  if (inter_x2 <= inter_x1 || inter_y2 <= inter_y1) return 0.0F;
  const float inter_area =
      static_cast<float>(inter_x2 - inter_x1) * (inter_y2 - inter_y1);
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

class Yolo11SegPostprocess final : public ::dali::Operator<::dali::GPUBackend> {
public:
  explicit Yolo11SegPostprocess(const ::dali::OpSpec& spec)
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
      std::vector<int64_t> instance_ring_offsets(kMaxDetections + 1, 0);
      const int64_t zero = 0;
      const int32_t zero_point[2]{0, 0};
      ResizeOutput<int64_t>(workspace, 4, {kMaxDetections + 1});
      ResizeOutput<int64_t>(workspace, 5, {1});
      ResizeOutput<int32_t>(workspace, 6, {1, 2});
      copy_h2d(4, instance_ring_offsets.data(), instance_ring_offsets.size() * sizeof(int64_t));
      copy_h2d(5, &zero, sizeof(zero));
      copy_h2d(6, zero_point, sizeof(zero_point));
      return;
    }

    int32_t* selected_device = nullptr;
    uint8_t* mask_scratch = nullptr;
    uint8_t* visited = nullptr;
    int32_t* edge_cells = nullptr;
    uint8_t* edge_directions = nullptr;
    int32_t* edge_counts_device = nullptr;
    int32_t* point_scratch = nullptr;
    int32_t* ring_counts_device = nullptr;
    int32_t* ring_point_counts_device = nullptr;
    int32_t* errors_device = nullptr;
    const int64_t max_points = total_pixels * 4;
    CUDA_CALL(cudaMallocAsync(&selected_device, selected.size() * sizeof(int32_t), stream));
    CUDA_CALL(cudaMallocAsync(&mask_scratch, total_pixels * sizeof(uint8_t), stream));
    CUDA_CALL(cudaMallocAsync(&visited, max_points * sizeof(uint8_t), stream));
    CUDA_CALL(cudaMallocAsync(&edge_cells, max_points * 2 * sizeof(int32_t), stream));
    CUDA_CALL(cudaMallocAsync(&edge_directions, max_points * sizeof(uint8_t), stream));
    CUDA_CALL(cudaMallocAsync(&edge_counts_device, kMaxDetections * sizeof(int32_t), stream));
    CUDA_CALL(cudaMallocAsync(&point_scratch, max_points * 2 * sizeof(int32_t), stream));
    CUDA_CALL(cudaMallocAsync(&ring_counts_device, kMaxDetections * sizeof(int32_t), stream));
    CUDA_CALL(cudaMallocAsync(&ring_point_counts_device, total_pixels * sizeof(int32_t), stream));
    CUDA_CALL(cudaMallocAsync(&errors_device, kMaxDetections * sizeof(int32_t), stream));
    CUDA_CALL(cudaMemcpyAsync(selected_device, selected.data(), selected.size() * sizeof(int32_t),
                              cudaMemcpyHostToDevice, stream));
    CUDA_CALL(cudaMemsetAsync(visited, 0, max_points * sizeof(uint8_t), stream));
    CUDA_CALL(cudaMemsetAsync(edge_counts_device, 0, kMaxDetections * sizeof(int32_t), stream));
    CUDA_CALL(cudaMemsetAsync(ring_counts_device, 0, kMaxDetections * sizeof(int32_t), stream));
    CUDA_CALL(cudaMemsetAsync(ring_point_counts_device, 0, total_pixels * sizeof(int32_t), stream));
    CUDA_CALL(cudaMemsetAsync(errors_device, 0, kMaxDetections * sizeof(int32_t), stream));

    const auto* detections_device = static_cast<const float*>(detections_input.raw_tensor(0));
    const auto* prototypes_device = static_cast<const float*>(prototypes_input.raw_tensor(0));
    int64_t* mask_offsets_device = nullptr;
    int32_t* boxes_device = nullptr;
    CUDA_CALL(cudaMallocAsync(&mask_offsets_device, mask_offsets.size() * sizeof(int64_t), stream));
    CUDA_CALL(cudaMallocAsync(&boxes_device, final_boxes.size() * sizeof(int32_t), stream));
    CUDA_CALL(cudaMemcpyAsync(mask_offsets_device, mask_offsets.data(),
                              mask_offsets.size() * sizeof(int64_t), cudaMemcpyHostToDevice,
                              stream));
    CUDA_CALL(cudaMemcpyAsync(boxes_device, final_boxes.data(), final_boxes.size() * sizeof(int32_t),
                              cudaMemcpyHostToDevice, stream));

    constexpr int threads = 256;
    const int blocks = static_cast<int>((total_pixels + threads - 1) / threads);
    BuildMasks<<<blocks, threads, 0, stream>>>(detections_device, num_channels, prototypes_device,
                                                selected_device, mask_offsets_device, boxes_device,
                                                mask_scratch, count, mask_threshold_, total_pixels);
    CUDA_CALL(cudaGetLastError());
    EnumerateBoundaryEdges<<<blocks, threads, 0, stream>>>(
        mask_scratch, mask_offsets_device, boxes_device, edge_cells, edge_directions,
        edge_counts_device, count, total_pixels);
    CUDA_CALL(cudaGetLastError());
    TracePolygons<<<count, 1, 0, stream>>>(mask_scratch, mask_offsets_device, boxes_device,
                                           edge_cells, edge_directions, edge_counts_device, visited,
                                           point_scratch, ring_counts_device,
                                           ring_point_counts_device, errors_device, count);
    CUDA_CALL(cudaGetLastError());
    ConvexHullRings<<<count, 1, 0, stream>>>(point_scratch, edge_cells, mask_offsets_device,
                                             ring_counts_device, ring_point_counts_device,
                                             errors_device, count);
    CUDA_CALL(cudaGetLastError());

    std::vector<int32_t> ring_counts(kMaxDetections, 0);
    std::vector<int32_t> ring_point_counts(static_cast<size_t>(total_pixels), 0);
    std::vector<int32_t> errors(kMaxDetections, 0);
    CUDA_CALL(cudaMemcpyAsync(ring_counts.data(), ring_counts_device,
                              ring_counts.size() * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CALL(cudaMemcpyAsync(ring_point_counts.data(), ring_point_counts_device,
                              ring_point_counts.size() * sizeof(int32_t), cudaMemcpyDeviceToHost,
                              stream));
    CUDA_CALL(cudaMemcpyAsync(errors.data(), errors_device, errors.size() * sizeof(int32_t),
                              cudaMemcpyDeviceToHost, stream));
    CUDA_CALL(cudaStreamSynchronize(stream));

    std::vector<int64_t> instance_ring_offsets(kMaxDetections + 1, 0);
    for (int detection = 0; detection < count; ++detection) {
      DALI_ENFORCE(errors[detection] == 0 && ring_counts[detection] > 0,
                   "CUDA polygon tracing failed");
      instance_ring_offsets[detection + 1] =
          instance_ring_offsets[detection] + ring_counts[detection];
    }
    for (int detection = count + 1; detection <= kMaxDetections; ++detection)
      instance_ring_offsets[detection] = instance_ring_offsets[count];
    const int64_t total_rings = instance_ring_offsets[count];
    std::vector<int64_t> ring_point_offsets(static_cast<size_t>(total_rings + 1), 0);
    int64_t ring_index = 0;
    for (int detection = 0; detection < count; ++detection) {
      const int64_t scratch_ring_base = mask_offsets[detection];
      for (int ring = 0; ring < ring_counts[detection]; ++ring) {
        const int32_t points = ring_point_counts[scratch_ring_base + ring];
        DALI_ENFORCE(points >= 3, "CUDA polygon tracing emitted a degenerate ring");
        ring_point_offsets[static_cast<size_t>(ring_index + 1)] =
            ring_point_offsets[static_cast<size_t>(ring_index)] + points;
        ++ring_index;
      }
    }
    const int64_t total_points = ring_point_offsets.back();

    ResizeOutput<int64_t>(workspace, 4, {kMaxDetections + 1});
    ResizeOutput<int64_t>(workspace, 5, {total_rings + 1});
    ResizeOutput<int32_t>(workspace, 6, {total_points, 2});
    copy_h2d(4, instance_ring_offsets.data(), instance_ring_offsets.size() * sizeof(int64_t));
    copy_h2d(5, ring_point_offsets.data(), ring_point_offsets.size() * sizeof(int64_t));
    auto* point_output =
        static_cast<int32_t*>(workspace.Output<::dali::GPUBackend>(6).raw_mutable_tensor(0));
    int64_t destination_point = 0;
    for (int detection = 0; detection < count; ++detection) {
      int64_t detection_points = 0;
      const int64_t scratch_ring_base = mask_offsets[detection];
      for (int ring = 0; ring < ring_counts[detection]; ++ring)
        detection_points += ring_point_counts[scratch_ring_base + ring];
      const int64_t source_point = mask_offsets[detection] * 4;
      CUDA_CALL(cudaMemcpyAsync(point_output + destination_point * 2, edge_cells + source_point * 2,
                                detection_points * 2 * sizeof(int32_t), cudaMemcpyDeviceToDevice,
                                stream));
      destination_point += detection_points;
    }

    CUDA_CALL(cudaFreeAsync(selected_device, stream));
    CUDA_CALL(cudaFreeAsync(mask_scratch, stream));
    CUDA_CALL(cudaFreeAsync(visited, stream));
    CUDA_CALL(cudaFreeAsync(edge_cells, stream));
    CUDA_CALL(cudaFreeAsync(edge_directions, stream));
    CUDA_CALL(cudaFreeAsync(edge_counts_device, stream));
    CUDA_CALL(cudaFreeAsync(point_scratch, stream));
    CUDA_CALL(cudaFreeAsync(ring_counts_device, stream));
    CUDA_CALL(cudaFreeAsync(ring_point_counts_device, stream));
    CUDA_CALL(cudaFreeAsync(errors_device, stream));
    CUDA_CALL(cudaFreeAsync(mask_offsets_device, stream));
    CUDA_CALL(cudaFreeAsync(boxes_device, stream));
  }

private:
  float confidence_;
  float mask_threshold_;
};

}  // namespace tritonic::dali_plugin

DALI_REGISTER_OPERATOR(Yolo11SegPostprocess, ::tritonic::dali_plugin::Yolo11SegPostprocess,
                       ::dali::GPU);
DALI_SCHEMA(Yolo11SegPostprocess)
    .DocStr("YOLO11 segmentation polygon postprocessing on CUDA (NMS + mask->polygon)")
    .NumInput(3)
    .NumOutput(7)
    .AddOptionalArg("confidence_threshold", "Detection confidence threshold", 0.5F)
    .AddOptionalArg("mask_threshold", "Binary mask threshold", 0.5F);
