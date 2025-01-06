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



static cv::Mat makeColorwheel() {
    const int RY = 15;
    const int YG = 6;
    const int GC = 4;
    const int CB = 11;
    const int BM = 13;
    const int MR = 6;

    int ncols = RY + YG + GC + CB + BM + MR;
    cv::Mat colorwheel(ncols, 1, CV_8UC3);

    int col = 0;
    // RY
    for (int i = 0; i < RY; ++i, ++col) {
        colorwheel.at<cv::Vec3b>(col, 0) = cv::Vec3b(255, 255 * i / RY, 0);
    }
    // YG
    for (int i = 0; i < YG; ++i, ++col) {
        colorwheel.at<cv::Vec3b>(col, 0) = cv::Vec3b(255 - 255 * i / YG, 255, 0);
    }
    // GC
    for (int i = 0; i < GC; ++i, ++col) {
        colorwheel.at<cv::Vec3b>(col, 0) = cv::Vec3b(0, 255, 255 * i / GC);
    }
    // CB
    for (int i = 0; i < CB; ++i, ++col) {
        colorwheel.at<cv::Vec3b>(col, 0) = cv::Vec3b(0, 255 - 255 * i / CB, 255);
    }
    // BM
    for (int i = 0; i < BM; ++i, ++col) {
        colorwheel.at<cv::Vec3b>(col, 0) = cv::Vec3b(255 * i / BM, 0, 255);
    }
    // MR
    for (int i = 0; i < MR; ++i, ++col) {
        colorwheel.at<cv::Vec3b>(col, 0) = cv::Vec3b(255, 0, 255 - 255 * i / MR);
    }

    return colorwheel;
}

static cv::Mat visualizeFlow(const cv::Mat& flow_mat) {

    cv::Mat flow_parts[2];
    cv::split(flow_mat, flow_parts);
    cv::Mat u = flow_parts[0], v = flow_parts[1];

    cv::Mat magnitude, angle;
    cv::cartToPolar(u, v, magnitude, angle);

    // Normalize magnitude
    double mag_max;
    cv::minMaxLoc(magnitude, 0, &mag_max);
    if (mag_max > 0) {
        magnitude /= mag_max;
    }

    // Convert angle to [0, 1] range
    angle *= (1.0 / (2 * CV_PI));
    angle += 0.5;

    // Apply color wheel
    cv::Mat colorwheel = makeColorwheel();
    const int ncols = colorwheel.rows;
    cv::Mat flow_color(flow_mat.size(), CV_8UC3);

    for (int i = 0; i < flow_mat.rows; ++i) {
        for (int j = 0; j < flow_mat.cols; ++j) {
            float mag = magnitude.at<float>(i, j);
            float ang = angle.at<float>(i, j);

            int k0 = static_cast<int>(ang * (ncols - 1));
            int k1 = (k0 + 1) % ncols;
            float f = (ang * (ncols - 1)) - k0;

            cv::Vec3b col0 = colorwheel.at<cv::Vec3b>(k0);
            cv::Vec3b col1 = colorwheel.at<cv::Vec3b>(k1);

            cv::Vec3b color;
            for (int ch = 0; ch < 3; ++ch) {
                float col = (1 - f) * col0[ch] + f * col1[ch];
                if (mag <= 1) {
                    col = 255 - mag * (255 - col);
                } else {
                    col *= 0.75;
                }
                color[ch] = static_cast<uchar>(col);
            }

            flow_color.at<cv::Vec3b>(i, j) = color;
        }
    }

    return flow_color;
}

std::vector<Result> RAFT::postprocess(const cv::Size& frame_size, 
                                const std::vector<std::vector<TensorElement>>& infer_results,
                                const std::vector<std::vector<int64_t>>& infer_shapes) {
    if (infer_results.empty() || infer_shapes.empty()) {
        throw std::runtime_error("Empty inference results");
    }

    // Assume the flow is in the first output tensor
    const auto& flow_data = infer_results[output_idx_.value()];
    const auto& flow_shape = infer_shapes[output_idx_.value()];

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
    result.flow = visualizeFlow(flow_mat);
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