#pragma once
#include "common.hpp"
#include "RFDetr.hpp"

class RFDetrSeg : public RFDetr {
public:
    RFDetrSeg(const TritonModelInfo& model_info);
    
    TaskType getTaskType() override { return TaskType::InstanceSegmentation; }
    
    std::vector<Result> postprocess(const cv::Size& frame_size, 
                                    const std::vector<std::vector<TensorElement>>& infer_results,
                                    const std::vector<std::vector<int64_t>>& infer_shapes) override;

private:
    std::optional<size_t> masks_idx_;
};
