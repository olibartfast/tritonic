#include "YOLOSeg.hpp"


cv::Rect YOLOSeg::getSegPadSize(const size_t inputW,
const size_t inputH,
const cv::Size& inputSize)
{
    std::vector<int> padSize;
    float w, h, x, y;
    float r_w = inputW / (inputSize.width * 1.0);
    float r_h = inputH / (inputSize.height * 1.0);
    if (r_h > r_w)
    {
        w = inputW;
        h = r_w * inputSize.height;
        x = 0;
        y = (inputH - h) / 2;
    }
    else
    {
        w = r_h * inputSize.width;
        h = inputH;
        x = (inputW - w) / 2;
        y = 0;
    }
    return cv::Rect(x, y, w,h);
}


std::vector<Result> YOLOSeg::postprocess(const cv::Size& frame_size, 
                                                  const std::vector<std::vector<float>>& infer_results, 
                                                  const std::vector<std::vector<int64_t>>& infer_shapes) 
    {
        std::vector<Result> results;
        std::vector<cv::Rect> boxes;
        std::vector<float> confs;
        std::vector<int> classIds;
        std::vector<std::vector<float>> picked_proposals;
        const auto confThreshold = 0.5f;
        const auto iouThreshold = 0.4f;
        const auto& infer_shape = infer_shapes[1]; 
        const auto& infer_result = infer_results[1]; 

        const auto& mask_shape = infer_shapes[0];
        const auto& mask_result = infer_results[0];

        // yolov5/v6/v7
        if(infer_shape[2] < infer_shape[1])
        {
            const int numClasses = infer_shape[2] - 5 - 32;
            const int maskDim = mask_shape[1];  // Dimension of the mask embeddings
            for (auto it = infer_result.begin(); it != infer_result.end(); it += (numClasses + 5 + maskDim))
            {
                float clsConf = it[4];
                if (clsConf > confThreshold)
                {
                    // Find the best class (highest confidence)
                    int bestClassId = 0;
                    float bestClassConf = 0;
                    for (int i = 0; i < numClasses; ++i)
                    {
                        if (it[5 + i] > bestClassConf)
                        {
                            bestClassConf = it[5 + i];
                            bestClassId = i;
                        }
                    }

                    boxes.emplace_back(get_rect(frame_size, std::vector<float>(it, it + 4)));
                    float confidence = clsConf * bestClassConf;
                    confs.emplace_back(confidence);
                    classIds.emplace_back(bestClassId);
                    
                    // Fill picked_proposals with mask coefficients
                    picked_proposals.emplace_back(std::vector<float>(it + 5 + numClasses, it + 5 + numClasses + maskDim));
                }
            }
        }
        else // yolov8/v9 and yolo11
        {
            const int numClasses = infer_shape[1] - 32 - 4;
            std::vector<std::vector<float>> output(infer_shape[1], std::vector<float>(infer_shape[2]));

            // Construct output matrix
            for (int i = 0; i < infer_shape[1]; i++) {
                for (int j = 0; j < infer_shape[2]; j++) {
                    output[i][j] = infer_result[i * infer_shape[2] + j];
                }
            }

            // Transpose output matrix
            std::vector<std::vector<float>> transposedOutput(infer_shape[2], std::vector<float>(infer_shape[1]));
            for (int i = 0; i < infer_shape[1]; i++) {
                for (int j = 0; j < infer_shape[2]; j++) {
                    transposedOutput[j][i] = output[i][j];
                }
            }

            // Get all the YOLO proposals
            for (int i = 0; i < infer_shape[2]; i++) {
                const auto& row = transposedOutput[i];
                const float* bboxesPtr = row.data();
                const float* scoresPtr = bboxesPtr + 4;
                auto maxSPtr = std::max_element(scoresPtr, scoresPtr + numClasses);
                float score = *maxSPtr;
                if (score > confThreshold) {
                    boxes.emplace_back(get_rect(frame_size, std::vector<float>(bboxesPtr, bboxesPtr + 4)));
                    int label = maxSPtr - scoresPtr;
                    confs.emplace_back(score);
                    classIds.emplace_back(label);
                    picked_proposals.emplace_back(std::vector<float>(scoresPtr + numClasses, scoresPtr + numClasses + mask_shape[1]));
                }
            }
        }

        // Perform NMS
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confs, confThreshold, iouThreshold, indices);

        // Process mask results
        int sc = mask_shape[1], sh = mask_shape[2], sw = mask_shape[3];
        cv::Mat protos(sc, sh * sw, CV_32F, const_cast<float*>(mask_result.data()));
        cv::Rect segPadRect = getSegPadSize(input_width_, input_height_, frame_size);
        cv::Rect roi(int((float)segPadRect.x / input_width_ * sw), 
                     int((float)segPadRect.y / input_height_ * sh), 
                     int(sw - segPadRect.x / 2), 
                     int(sh - segPadRect.y / 2));

        cv::Mat maskProposals;
        for (int idx : indices) {
            maskProposals.push_back(cv::Mat(picked_proposals[idx]).t());
        }

        if (!indices.empty()) {
            cv::Mat masks = (maskProposals * protos).t();
            masks = masks.reshape(indices.size(), {sh, sw});
            std::vector<cv::Mat> maskChannels;
            cv::split(masks, maskChannels);

            for (size_t i = 0; i < indices.size(); ++i) {
                int idx = indices[i];
                InstanceSegmentation seg;
                seg.bbox = boxes[idx];
                seg.class_confidence = confs[idx];
                seg.class_id = classIds[idx];

                const float mask_thresh = 0.5f;
                cv::Mat mask;

                // Sigmoid
                cv::exp(-maskChannels[i], mask);
                mask = 1.0 / (1.0 + mask); // 160*160

                // Ensure roi is within mask boundaries
                cv::Rect safeRoi = roi & cv::Rect(0, 0, mask.cols, mask.rows);
                if (safeRoi.width > 0 && safeRoi.height > 0) {
                    mask = mask(safeRoi);
                } else {
                    // If roi is completely outside, skip this instance
                    continue;
                }

                cv::resize(mask, mask, frame_size, cv::INTER_NEAREST);

                // Ensure bbox is within frame boundaries
                cv::Rect safeBbox = seg.bbox & cv::Rect(0, 0, frame_size.width, frame_size.height);
                if (safeBbox.width > 0 && safeBbox.height > 0) {
                    mask = mask(safeBbox);
                    mask = mask > mask_thresh;
                    
                    // Store mask data and dimensions
                    seg.mask_data.assign(mask.data, mask.data + mask.total());
                    seg.mask_height = mask.rows;
                    seg.mask_width = mask.cols;

                    results.push_back(seg);
                }
            }
        }

        return results;
    }