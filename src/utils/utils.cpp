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

