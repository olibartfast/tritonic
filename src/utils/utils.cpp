#include "utils.hpp"
#include <cmath>

std::string ToLower(const std::string& str) {
    std::string lowerStr = str;
    std::transform(lowerStr.begin(), lowerStr.end(), lowerStr.begin(), [](unsigned char c) {
        return std::tolower(c);
    });
    return lowerStr;
}

// Function to split a string by a delimiter and return a vector of strings
std::vector<std::string> split(const std::string& s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

std::vector<std::vector<int64_t>> parseInputSizes(const std::string& input) {
    std::vector<std::vector<int64_t>> sizes;
    std::vector<std::string> inputs = split(input, ';'); // Split by ';'

    for (const auto& input_str : inputs) {
        std::vector<std::string> dims = split(input_str, ','); // Split by ','
        std::vector<int64_t> dimensions;
        for (const auto& dim : dims) {
            dimensions.push_back(std::stoi(dim)); // Convert to int
        }
        sizes.push_back(dimensions);
    }

    return sizes;
}

// Function to check if the file has an image extension
bool IsImageFile(const std::string& fileName) {
    static const std::set<std::string> imageExtensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"};
    
    // Extract the file extension
    std::string extension = fileName.substr(fileName.find_last_of("."));
    
    // Convert the extension to lowercase and check if it's in the set of image extensions
    return imageExtensions.find(ToLower(extension)) != imageExtensions.end();
}

std::vector<cv::Scalar> generateRandomColors(size_t size) {
    std::vector<cv::Scalar> colors;
    colors.reserve(size); // Reserve space to avoid reallocations

    // Create a random number generator
    std::random_device rd;  // Seed for the random number engine
    std::mt19937 gen(rd()); // Mersenne Twister engine
    std::uniform_int_distribution<> dist(0, 255); // Range for BGR values

    // Generate random BGR colors
    for (size_t i = 0; i < size; ++i) {
        cv::Scalar color(dist(gen), dist(gen), dist(gen)); // Random B, G, R values
        colors.push_back(color);
    }

    return colors;
}

void draw_label(cv::Mat& input_image, const std::string& label, float confidence, int left, int top)
{
    const float FONT_SCALE = 0.7;
    const int FONT_FACE = cv::FONT_HERSHEY_DUPLEX; // Change font type to what you think is better for you
    const int THICKNESS = 2;
    cv::Scalar YELLOW = cv::Scalar(0, 255, 255);

    // Display the label and confidence at the top of the bounding box.
    int baseLine;
    std::string confidenceStr = std::to_string(confidence).substr(0, 4);
    std::string display_text = label + ": " + confidenceStr;
    cv::Size label_size = cv::getTextSize(display_text, FONT_FACE, FONT_SCALE, THICKNESS, &baseLine);
    top = cv::max(top, label_size.height);

    // Top left corner.
    cv::Point tlc = cv::Point(left, top);
    // Bottom right corner.
    cv::Point brc = cv::Point(left + label_size.width, top + label_size.height + baseLine);

    // Draw black rectangle.
    cv::rectangle(input_image, tlc, brc, cv::Scalar(255, 0, 255), cv::FILLED);

    // Put the label and confidence on the black rectangle.
    cv::putText(input_image, display_text, cv::Point(left, top + label_size.height), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS);
}


cv::Mat makeColorwheel() {
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

cv::Mat visualizeFlow(const cv::Mat& flow_mat) {

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