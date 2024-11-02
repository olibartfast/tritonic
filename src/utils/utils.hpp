#pragma once
#include "common.hpp"

std::string ToLower(const std::string& str);
bool IsImageFile(const std::string& fileName);
std::vector<cv::Scalar> generateRandomColors(size_t size);

void draw_label(cv::Mat& input_image, const std::string& label, float confidence, int left, int top);
