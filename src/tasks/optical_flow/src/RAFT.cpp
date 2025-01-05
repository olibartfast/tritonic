#include "RAFT.hpp"

std::vector<std::vector<uint8_t>>  RAFT::preprocess(const std::vector<cv::Mat>& imgs)
{
    if (imgs.empty()) {
        throw std::runtime_error("Input image vector is empty");
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
            input_data[i] = preprocess_image(imgs[i], input_format, model_info_.type1_, model_info_.type3_, imgs[i].channels(), input_size);
        } else {
            // For other types of inputs, you might need to add more cases
            // or use a default handling method
            throw std::runtime_error("Unhandled input");
        }
    }
    return input_data;
}


std::vector<Result> RAFT::postprocess(const cv::Size& frame_size, 
                                const std::vector<std::vector<TensorElement>>& infer_results,
                                const std::vector<std::vector<int64_t>>& infer_shapes) {
    if (infer_results.empty() || infer_shapes.empty()) {
        throw std::runtime_error("Empty inference results");
    }

    // Assume the flow is in the first output tensor
    const auto& flow_data = infer_results[0];
    const auto& flow_shape = infer_shapes[0];

    if (flow_shape.size() != 4) { // Expecting [1, 2, H, W]
        throw std::runtime_error("Unexpected flow shape");
    }

    int height = flow_shape[2];
    int width = flow_shape[3];

    // Convert TensorElement vector to cv::Mat
    cv::Mat flow_mat(height, width, CV_32FC2);
    float* flow_ptr = flow_mat.ptr<float>();
    for (size_t i = 0; i < flow_data.size(); i += 2) {
        flow_ptr[i] = std::get<float>(flow_data[i]);     // u component
        flow_ptr[i+1] = std::get<float>(flow_data[i+1]); // v component
    }

    // Calculate magnitude and angle
    cv::Mat magnitude, angle;
    cv::cartToPolar(flow_mat.col(0), flow_mat.col(1), magnitude, angle);

    // Normalize magnitude
    double mag_max;
    cv::minMaxLoc(magnitude, nullptr, &mag_max);
    if (mag_max > 0) {
        magnitude /= mag_max;
    }

    // Convert angle to [0, 1] range
    angle *= (1.0 / (2 * CV_PI));
    angle += 0.5;

    // Create OpticalFlow result
    OpticalFlow result;
    result.flow = flow_mat;
    result.max_displacement = static_cast<float>(mag_max);

    // Resize flow to original frame size if necessary
    if (frame_size != flow_mat.size()) {
        cv::resize(result.flow, result.flow, frame_size);
    }

    return {result};
}


std::vector<uint8_t> RAFT::preprocess_image(
    const cv::Mat& img, const std::string& format, int img_type1, int img_type3,
    size_t img_channels, const cv::Size& img_size) 
{
    std::vector<uint8_t> input_data;
    cv::Mat sample;
    cv::cvtColor(img, sample,  cv::COLOR_BGR2RGB);
    sample.convertTo(
        sample, (img_channels == 3) ? img_type3 : img_type1);
    sample.convertTo(sample, CV_32FC3, 1.f / 127.5f, -1.f);
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