#include "RAFT.hpp"

std::vector<uint8_t> RAFT::preprocess_image(
    const cv::Mat& img, const std::string& format, int img_type1, int img_type3,
    size_t img_channels, const cv::Size& img_size)
{
    std::vector<uint8_t> input_data;
    cv::Mat sample;

    // 1. Convert to RGB
    cv::cvtColor(img, sample, cv::COLOR_BGR2RGB);

    // 2. Resize (with better interpolation)
    if (img.cols > img_size.width || img.rows > img_size.height) {
        cv::resize(sample, sample, img_size, 0, 0, cv::INTER_AREA); // Downsizing
    } else {
        cv::resize(sample, sample, img_size, 0, 0, cv::INTER_LINEAR); // Upscaling or same size
    }

    // 3. Convert to CV_32F and Normalize
    sample.convertTo(sample, CV_32F, 1.0 / 255.0);
    sample = (sample - 0.5) / 0.5;

    // 4. Channel Splitting (CHW) - Only if your model requires it
    if (format.compare("FORMAT_NCHW") == 0) {
        size_t img_byte_size = sample.total() * sample.elemSize();
        size_t pos = 0;
        input_data.resize(img_byte_size);

        std::vector<cv::Mat> input_rgb_channels; // Using RGB since we converted earlier
        for (size_t i = 0; i < img_channels; ++i) {
            input_rgb_channels.emplace_back(
                img_size.height, img_size.width, CV_32FC1, &(input_data[pos])); // Assuming CV_32F
            pos += input_rgb_channels.back().total() *
                   input_rgb_channels.back().elemSize();
        }

        cv::split(sample, input_rgb_channels);

        if (pos != img_byte_size) {
            std::cerr << "Unexpected total size of channels: " << pos
                      << ", expecting: " << img_byte_size << std::endl;
            exit(1);
        }
    } else {
        // Handle other formats (e.g., HWC) or throw an error if not supported
        std::cerr << "Unsupported format: " << format << std::endl;
        exit(1);
    }

    return input_data;
}
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
    // Constants for color wheel
    const int RY = 15, YG = 6, GC = 4, CB = 11, BM = 13, MR = 6;
    const int ncols = RY + YG + GC + CB + BM + MR;
    cv::Mat colorwheel(ncols, 1, CV_8UC3);

    int col = 0;
    // RY
    for (int i = 0; i < RY; ++i, ++col) {
        colorwheel.at<cv::Vec3b>(col) = cv::Vec3b(255, 255 * i / RY, 0);
    }
    // YG
    for (int i = 0; i < YG; ++i, ++col) {
        colorwheel.at<cv::Vec3b>(col) = cv::Vec3b(255 - 255 * i / YG, 255, 0);
    }
    // GC
    for (int i = 0; i < GC; ++i, ++col) {
        colorwheel.at<cv::Vec3b>(col) = cv::Vec3b(0, 255, 255 * i / GC);
    }
    // CB
    for (int i = 0; i < CB; ++i, ++col) {
        colorwheel.at<cv::Vec3b>(col) = cv::Vec3b(0, 255 - 255 * i / CB, 255);
    }
    // BM
    for (int i = 0; i < BM; ++i, ++col) {
        colorwheel.at<cv::Vec3b>(col) = cv::Vec3b(255 * i / BM, 0, 255);
    }
    // MR
    for (int i = 0; i < MR; ++i, ++col) {
        colorwheel.at<cv::Vec3b>(col) = cv::Vec3b(255, 0, 255 - 255 * i / MR);
    }
    return colorwheel;
}


static cv::Mat flowToColor(const cv::Mat& flow_mat) {
    cv::Mat flow_parts[2];
    cv::split(flow_mat, flow_parts);
    cv::Mat u = flow_parts[0], v = flow_parts[1];

    // Compute magnitude and angle
    cv::Mat magnitude, angle;
    cv::cartToPolar(u, v, magnitude, angle);

    // Normalize magnitude
    double mag_max;
    cv::minMaxLoc(magnitude, nullptr, &mag_max);
    if (mag_max > 0) {
        magnitude /= mag_max;
    }

    // Convert angle to [0, 1] range
    angle *= (1.0 / (2 * CV_PI));
    angle += 0.5; // Shift angle to [0, 2π] for color wheel mapping

    // Apply color wheel
    cv::Mat colorwheel = makeColorwheel();
    const int ncols = colorwheel.rows;
    cv::Mat flow_color(flow_mat.size(), CV_8UC3);

    for (int i = 0; i < flow_mat.rows; ++i) {
        for (int j = 0; j < flow_mat.cols; ++j) {
            float mag = magnitude.at<float>(i, j);
            float ang = angle.at<float>(i, j);

            // Find nearest colors in the wheel
            int k0 = static_cast<int>(ang * (ncols - 1));
            int k1 = (k0 + 1) % ncols;
            float f = (ang * (ncols - 1)) - k0;

            // Get colors from the wheel
            cv::Vec3b col0 = colorwheel.at<cv::Vec3b>(k0);
            cv::Vec3b col1 = colorwheel.at<cv::Vec3b>(k1);

            // Interpolate colors
            cv::Vec3b color;
            for (int ch = 0; ch < 3; ++ch) {
                float channel_value = (1 - f) * col0[ch] + f * col1[ch];
                // Apply magnitude modulation
                if (mag <= 1) {
                    channel_value = 255 - mag * (255 - channel_value);
                } else {
                    channel_value *= 0.75;
                }
                color[ch] = static_cast<uchar>(channel_value);
            }

            flow_color.at<cv::Vec3b>(i, j) = color;
        }
    }

    return flow_color;
}

std::vector<Result> RAFT::postprocess(const cv::Size& frame_size,
                                          const std::vector<std::vector<TensorElement>>& infer_results,
                                          const std::vector<std::vector<int64_t>>& infer_shapes) {
    // Validate inputs
    if (infer_results.empty() || infer_shapes.empty()) {
        throw std::runtime_error("Empty inference results");
    }


    const auto& flow_data = infer_results.front();
    const auto& flow_shape = infer_shapes.front();

    if (flow_shape.size() != 4 || flow_shape[1] != 2) {
        throw std::runtime_error("Unexpected flow shape. Expected [1,2,H,W]");
    }

    const int height = flow_shape[2];
    const int width = flow_shape[3];
    const size_t expected_size = height * width * 2;

    if (flow_data.size() != expected_size) {
        throw std::runtime_error("Flow data size mismatch");
    }

    // **Dynamically determine U and V channel offsets (Example)**
    // This is a placeholder. You'll need to adapt this based on how your
    // model/inference engine orders the output channels.
    // You might need to use information from model_info_ or other metadata.
    const int u_channel_offset = 0; // Example: U channel starts at the beginning
    const int v_channel_offset = height * width; // Example: V channel starts after U

    // Create flow matrix
    cv::Mat flow_mat(height, width, CV_32FC2);
    float* flow_ptr = flow_mat.ptr<float>();

    // Reconstruct the flow field considering channel ordering
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            // U channel (horizontal flow)
            flow_ptr[y * width * 2 + x * 2] =
                std::get<float>(flow_data[u_channel_offset + y * width + x]);
            // V channel (vertical flow)
            flow_ptr[y * width * 2 + x * 2 + 1] =
                std::get<float>(flow_data[v_channel_offset + y * width + x]);
        }
    }

    // Calculate maximum displacement
    cv::Mat magnitude, angle;
    std::vector<cv::Mat> flow_parts;
    cv::split(flow_mat, flow_parts);
    cv::cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle);
    double max_displacement;
    cv::minMaxLoc(magnitude, nullptr, &max_displacement);

    // Create colored visualization
    cv::Mat flow_color = flowToColor(flow_mat);

    // Create result
    OpticalFlow result;
    result.raw_flow = flow_mat;
    result.flow = flow_color;
    result.max_displacement = max_displacement;

    // Resize if necessary
    if (frame_size != flow_mat.size()) {
        cv::resize(result.flow, result.flow, frame_size, 0, 0, cv::INTER_LINEAR);
        cv::resize(result.raw_flow, result.raw_flow, frame_size, 0, 0, cv::INTER_LINEAR);
    }

    return {result}; // Return a vector containing the OpticalFlow result
}