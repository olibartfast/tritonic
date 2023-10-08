#include "YoloNas.hpp"

// Override the preprocess function
std::vector<uint8_t> YoloNas::preprocess(
    const cv::Mat& img, const std::string& format, int img_type1, int img_type3,
    size_t img_channels, const cv::Size& img_size) 
{
    std::vector<uint8_t> input_data;
    cv::Mat sample;
    cv::cvtColor(img, sample,  cv::COLOR_BGR2RGB);
    sample.convertTo(
        sample, (img_channels == 3) ? img_type3 : img_type1);
    sample.convertTo(sample, CV_32FC3, 1.f / 255.f);
    cv::resize(sample, sample, cv::Size(img_size.width, img_size.height));

    // Allocate a buffer to hold all image elements.
    size_t img_byte_size = sample.total() * sample.elemSize();
    size_t pos = 0;
    input_data.resize(img_byte_size);

    // (format.compare("FORMAT_NCHW") == 0)
    //
    // For CHW formats must split out each channel from the matrix and
    // order them as BBBB...GGGG...RRRR. To do this split the channels
    // of the image directly into 'input_data'. The BGR channels are
    // backed by the 'input_data' vector so that ends up with CHW
    // order of the data.
    std::vector<cv::Mat> input_bgr_channels;
    for (size_t i = 0; i < img_channels; ++i)
    {
        input_bgr_channels.emplace_back(
            img_size.height, img_size.width, img_type1, &(input_data[pos]));
        pos += input_bgr_channels.back().total() *
            input_bgr_channels.back().elemSize();
    }

    cv::split(sample, input_bgr_channels);

    if (pos != img_byte_size)
    {
        std::cerr << "unexpected total size of channels " << pos << ", expecting "
            << img_byte_size << std::endl;
        exit(1);
    }

    return input_data;
}

// Override the postprocess function
std::vector<Result> YoloNas::postprocess(const cv::Size& frame_size, std::vector<std::vector<float>>& infer_results, 
const std::vector<std::vector<int64_t>>& infer_shapes) 
{

    std::vector<Result> detections;
    std::vector<cv::Rect> boxes;
    std::vector<float> confs;
    std::vector<int> classIds;
    const auto confThreshold = 0.5f;
    const auto iouThreshold = 0.4f;
    float* scores_result = infer_results[0].data();
    float* detection_result = infer_results[1].data();
    const auto scores_shape = infer_shapes[0];
    const auto detection_shape = infer_shapes[1];
    
    const int numClasses =  scores_shape[2];
    const auto rows = detection_shape[1];
    const auto boxes_size =  detection_shape[2];
    for (int i = 0; i < rows; ++i) 
    {
        cv::Mat scores(1, numClasses, CV_32FC1, scores_result);
        cv::Point class_id;
        double maxClassScore;
        minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);
        if (maxClassScore >= confThreshold) 
        {
            confs.push_back(maxClassScore);
            classIds.push_back(class_id.x);
            float r_w = (frame_size.width * 1.0) / input_width_;
            float r_h = (frame_size.height * 1.0) / input_height_ ;
            std::vector<float> bbox(&detection_result[0], &detection_result[4]);

            int left = (int)(bbox[0] * r_w);
            int top = (int)(bbox[1] * r_h);
            int width = (int)((bbox[2] - bbox[0]) * r_w);
            int height = (int)((bbox[3] - bbox[1]) * r_h);
            boxes.push_back(cv::Rect(left, top, width, height));
        }
        // Jump to the next column.
        scores_result += numClasses;
        detection_result += boxes_size;
    }      

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confs, confThreshold, iouThreshold, indices);

    for (int idx : indices)
    {
        Detection d;
        d.bbox = cv::Rect(boxes[idx]);
        d.class_confidence = confs[idx];
        d.class_id = classIds[idx];
        detections.emplace_back(d);
    }        
    return detections;
}
