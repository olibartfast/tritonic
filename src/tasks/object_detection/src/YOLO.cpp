#include "YOLO.hpp"
#include "Logger.hpp"

YOLO::YOLO(const TritonModelInfo& model_info) : TaskInterface(model_info)
{
    input_width_ = model_info.input_shapes[0][3];
    input_height_ = model_info.input_shapes[0][2];
}

cv::Rect YOLO::get_rect(const cv::Size& imgSz, const std::vector<float>& bbox)
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

std::vector<std::vector<uint8_t>> YOLO::preprocess(const std::vector<cv::Mat>& imgs)
{
    if (imgs.empty()) {
        throw std::runtime_error("Input image vector is empty");
    }

    cv::Mat img = imgs.front();
    if (img.empty()) {
        throw std::runtime_error("Input image is empty");
    }
    std::vector<std::vector<uint8_t>> input_data(model_info_.input_shapes.size());

    for (size_t i = 0; i < model_info_.input_shapes.size(); ++i) {
        const auto& input_name = model_info_.input_names[i];
        const auto& input_shape = model_info_.input_shapes[i];
        const auto& input_format = model_info_.input_formats[i];
        const auto& input_type = model_info_.input_types[i];

        if (input_shape.size() >= 3) {
            // This is likely an image input
            const auto input_size = cv::Size(input_width_, input_height_);
            input_data[i] = preprocess_image(img, input_format, model_info_.type1_, model_info_.type3_, img.channels(), input_size);
        } else {
            // For other types of inputs, you might need to add more cases
            // or use a default handling method
            throw std::runtime_error("Unhandled input");
        }
    }
    return input_data;
}

std::tuple<float, int> YOLO::getBestClassInfo(const std::vector<TensorElement>& data, size_t startIdx, const size_t& numClasses)
{
    int idxMax = 0;
    float maxConf = 0;

    auto get_float = [](const TensorElement& elem) {
        return std::visit([](auto&& arg) -> float { return static_cast<float>(arg); }, elem);
    };

    for (size_t i = 5; i < numClasses + 5; i++)
    {
        float conf = get_float(data[startIdx + i]);
        if (conf > maxConf)
        {
            maxConf = conf;
            idxMax = i - 5;
        }
    }
    return std::make_tuple(maxConf, idxMax);
}

std::vector<Result> YOLO::postprocess(const cv::Size& frame_size, const std::vector<std::vector<TensorElement>>& infer_results, 
    const std::vector<std::vector<int64_t>>& infer_shapes) 
{
    auto& logger = Logger::getInstance();
    std::vector<Result> detections;
    std::vector<cv::Rect> boxes;
    std::vector<float> confs;
    std::vector<int> classIds;
    const auto confThreshold = 0.5f;
    const auto iouThreshold = 0.4f;
    const auto& infer_shape = infer_shapes.front(); 
    const auto& infer_result = infer_results.front(); 

    logger.debug("YOLO Postprocess Debug:");
    logger.debugf("  Input shape: [{}, {}, {}]", infer_shape[0], infer_shape[1], infer_shape[2]);
    logger.debugf("  Confidence threshold: {}", confThreshold);
    logger.debugf("  Frame size: {}x{}", frame_size.width, frame_size.height); 

    auto get_float = [](const TensorElement& elem) {
        return std::visit([](auto&& arg) -> float { return static_cast<float>(arg); }, elem);
    };

    // yolov5/v6/v7
    if(infer_shape[2] < infer_shape[1])
    {
        const int numClasses = infer_shape[2] - 5;
        const int stride = numClasses + 5;
        
        for (size_t i = 0; i < infer_result.size(); i += stride)
        {
            float clsConf = get_float(infer_result[i + 4]);
            if (clsConf > confThreshold)
            {
                auto [objConf, classId] = getBestClassInfo(infer_result, i, numClasses);
                
                std::vector<float> box_coords;
                box_coords.reserve(4);
                for (int j = 0; j < 4; ++j) {
                    box_coords.push_back(get_float(infer_result[i + j]));
                }
                boxes.emplace_back(get_rect(frame_size, box_coords));
                
                float confidence = clsConf * objConf;
                confs.emplace_back(confidence);
                classIds.emplace_back(classId);              
            }
        }
    }
    else // yolov8/v9/v11/v12
    {
        const int numClasses =  infer_shape[1] - 4;
        std::vector<std::vector<float>> output(infer_shape[1], std::vector<float>(infer_shape[2]));

        // Construct output matrix
        for (int i = 0; i < infer_shape[1]; i++) {
            for (int j = 0; j < infer_shape[2]; j++) {
                output[i][j] = get_float(infer_result[i * infer_shape[2] + j]);
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
        int detectionCount = 0;
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
                detectionCount++;
                if (detectionCount <= 5) { // Only show first 5 detections for debugging
                    logger.debugf("  Detection {}: class={}, score={}", detectionCount, label, score);
                }
            }
        }
        logger.debugf("  Total detections before NMS: {}", detectionCount);
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confs, confThreshold, iouThreshold, indices);
    
    logger.debugf("  Total detections after NMS: {}", indices.size());

    for (int idx : indices)
    {
        Detection d;
        d.bbox = cv::Rect(boxes[idx]);
        d.class_confidence = confs[idx];
        d.class_id = classIds[idx];
        detections.emplace_back(d);
        logger.debugf("  Final detection: class={}, confidence={}, bbox=[{},{},{},{}]", 
                      d.class_id, d.class_confidence, d.bbox.x, d.bbox.y, d.bbox.width, d.bbox.height);
    }        
    return detections; 
}

std::vector<uint8_t> YOLO::preprocess_image(
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
