#include "Yolo.hpp"

Yolo::Yolo(int input_width, int input_height) : TaskInterface(input_width, input_height) {
    input_width_ = input_width;
    input_height_ = input_height;
}

cv::Rect Yolo::get_rect(const cv::Size& imgSz, const std::vector<float>& bbox)
{
    int l, r, t, b;
    float r_w = input_width_ / (imgSz.width * 1.0);
    float r_h = input_height_ / (imgSz.height * 1.0);
    if (r_h > r_w) {
        l = bbox[0] - bbox[2] / 2.f;
        r = bbox[0] + bbox[2] / 2.f;
        t = bbox[1] - bbox[3] / 2.f - (input_height_ - r_w * imgSz.height) / 2;
        b = bbox[1] + bbox[3] / 2.f - (input_height_ - r_w * imgSz.height) / 2;
        l = l / r_w;
        r = r / r_w;
        t = t / r_w;
        b = b / r_w;
    }
    else {
        l = bbox[0] - bbox[2] / 2.f - (input_width_ - r_h * imgSz.width) / 2;
        r = bbox[0] + bbox[2] / 2.f - (input_width_ - r_h * imgSz.width) / 2;
        t = bbox[1] - bbox[3] / 2.f;
        b = bbox[1] + bbox[3] / 2.f;
        l = l / r_h;
        r = r / r_h;
        t = t / r_h;
        b = b / r_h;
    }
    return cv::Rect(l, t, r - l, b - t);
}

auto Yolo::getBestClassInfo(std::vector<float>::iterator it, const size_t& numClasses)
{
    int idxMax = 5;
    float maxConf = 0;

    for (int i = 5; i < numClasses + 5; i++)
    {
        if (it[i] > maxConf)
        {
            maxConf = it[i];
            idxMax = i - 5;
        }
    }
    return std::make_tuple(maxConf, idxMax);
}

std::vector<Result> Yolo::postprocess(const cv::Size& frame_size, std::vector<std::vector<float>>& infer_results, 
const std::vector<std::vector<int64_t>>& infer_shapes) 
{
    
    std::vector<Result> detections;
    std::vector<cv::Rect> boxes;
    std::vector<float> confs;
    std::vector<int> classIds;
    const auto confThreshold = 0.5f;
    const auto iouThreshold = 0.4f;
    const auto infer_shape = infer_shapes.front(); 
    auto infer_result = infer_results.front(); 

    if(infer_shape[2]  < infer_shape[1])
    {
        const int numClasses =  infer_shape[2] - 5;
        for (auto it = infer_result.begin(); it != infer_result.end(); it += (numClasses + 5))
        {
            float clsConf = it[4];
            if (clsConf > confThreshold)
            {
                auto[objConf, classId] = Yolo::getBestClassInfo(it, numClasses);
                boxes.emplace_back(Yolo::get_rect(frame_size, std::vector<float>(it, it + 4)));
                float confidence = clsConf * objConf;
                confs.emplace_back(confidence);
                classIds.emplace_back(classId);              
            }
        }

    }
    else
    {
        const int numClasses =  infer_shape[1] - 4;
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
                boxes.emplace_back(Yolo::get_rect(frame_size, std::vector<float>(bboxesPtr, bboxesPtr + 4)));
                int label = maxSPtr - scoresPtr;
                confs.emplace_back(score);
                classIds.emplace_back(label);
            }
        }
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


std::vector<uint8_t> Yolo::preprocess(
    const cv::Mat& img, const std::string& format, int img_type1, int img_type3,
    size_t img_channels, const cv::Size& img_size) 
{
    // Image channels are in BGR order. Currently model configuration
    // data doesn't provide any information as to the expected channel
    // orderings (like RGB, BGR). We are going to assume that RGB is the
    // most likely ordering and so change the channels to that ordering.
    std::vector<uint8_t> input_data;
    cv::Mat sample;
    cv::cvtColor(img, sample,  cv::COLOR_BGR2RGB);
    sample.convertTo(
        sample, (img_channels == 3) ? img_type3 : img_type1);
    int w, h, x, y;
    float r_w = input_width_ / (sample.cols * 1.0);
    float r_h = input_height_ / (sample.rows * 1.0);
    if (r_h > r_w)
    {
        w = input_width_;
        h = r_w * sample.rows;
        x = 0;
        y = (input_height_ - h) / 2;
    }
    else
    {
        w = r_h * sample.cols;
        h = input_height_;
        x = (input_width_ - w) / 2;
        y = 0;
    }
    cv::Mat re(h, w, CV_8UC3);
    cv::resize(sample, re, re.size(), 0, 0, cv::INTER_CUBIC);
    cv::Mat out(input_height_, input_width_, CV_8UC3, cv::Scalar(128, 128, 128));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
    out.convertTo(sample, CV_32FC3, 1.f / 255.f);


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
