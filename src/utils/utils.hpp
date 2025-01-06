#pragma once
#include "common.hpp"

std::vector<std::string> split(const std::string& s, char delimiter);
std::vector<std::vector<int64_t>> parseInputSizes(const std::string& input);

std::string ToLower(const std::string& str);
bool IsImageFile(const std::string& fileName);
std::vector<cv::Scalar> generateRandomColors(size_t size);

void draw_label(cv::Mat& input_image, const std::string& label, float confidence, int left, int top);

cv::Mat makeColorwheel();
cv::Mat visualizeFlow(const cv::Mat& flow);
