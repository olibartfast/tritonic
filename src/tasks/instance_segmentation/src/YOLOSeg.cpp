#include "YOLOv8Seg.hpp"




   std::tuple<std::vector<Detection>, Mask> postprocess(const float*  output0, const float*  output1, const  std::vector<int64_t>& shape0,  const std::vector<int64_t>& shape1, const cv::Size& frame_size)
        {

            const auto offset = 4;
            const auto num_classes = shape0[1] - offset - shape1[1];
            std::vector<std::vector<float>> output0_matrix(shape0[1], std::vector<float>(shape0[2]));

            // Construct output matrix
            for (size_t i = 0; i < shape0[1]; ++i) {
                for (size_t j = 0; j < shape0[2]; ++j) {
                    output0_matrix[i][j] = output0[i * shape0[2] + j];
                }
            }

            std::vector<std::vector<float>> transposed_output0(shape0[2], std::vector<float>(shape0[1]));

            // Transpose output matrix
            for (int i = 0; i < shape0[1]; ++i) {
                for (int j = 0; j < shape0[2]; ++j) {
                    transposed_output0[j][i] = output0_matrix[i][j];
                }
            }

            std::vector<cv::Rect> boxes;
            std::vector<float> confs;
            std::vector<int> classIds;
            const auto conf_threshold = 0.25f;
            const auto iou_threshold = 0.4f;
            
            std::vector<std::vector<float>> picked_proposals;

            // Get all the YOLO proposals
            for (int i = 0; i < shape0[2]; ++i) {
                const auto& row = transposed_output0[i];
                const float* bboxesPtr = row.data();
                const float* scoresPtr = bboxesPtr + 4;
                auto maxSPtr = std::max_element(scoresPtr, scoresPtr + num_classes);
                float score = *maxSPtr;
                if (score > conf_threshold) {
                    boxes.emplace_back(get_rect(frame_size, std::vector<float>(bboxesPtr, bboxesPtr + 4)));
                    int label = maxSPtr - scoresPtr;
                    confs.emplace_back(score);
                    classIds.emplace_back(label);
                    picked_proposals.emplace_back(std::vector<float>(scoresPtr + num_classes, scoresPtr + num_classes + shape1[1]));
                }
            }

            // Perform Non Maximum Suppression and draw predictions.
            std::vector<int> indices;
            cv::dnn::NMSBoxes(boxes, confs, conf_threshold, iou_threshold, indices);
            std::vector<Detection> detections;
            Mask segMask;
            int sc, sh, sw;
            std::tie(sc, sh, sw) = std::make_tuple(static_cast<int>(shape1[1]), static_cast<int>(shape1[2]), static_cast<int>(shape1[3]));
            cv::Mat(std::vector<float>(output1, output1 + sc * sh * sw)).reshape(0, { sc, sw * sh }).copyTo(segMask.protos);        
            cv::Rect segPadRect = getSegPadSize(input_width_, input_height_, frame_size);
            cv::Rect roi(int((float)segPadRect.x / input_width_ * sw), int((float)segPadRect.y / input_height_ * sh), int(sw - segPadRect.x / 2), int(sh - segPadRect.y / 2));
            segMask.maskRoi = roi; 
            cv::Mat maskProposals;
            for (int i = 0; i < indices.size(); i++)
            {
                Detection det;
                int idx = indices[i];
                det.label_id = classIds[idx];
                det.bbox = boxes[idx];
                det.score = confs[idx];
                detections.emplace_back(det);
                maskProposals.push_back(cv::Mat(picked_proposals[idx]).t());
            }
            maskProposals.copyTo(segMask.maskProposals);
            return std::make_tuple(detections, segMask);
        }

        virtual std::tuple<std::vector<Detection>, Mask> infer(const cv::Mat& image) = 0;
};
