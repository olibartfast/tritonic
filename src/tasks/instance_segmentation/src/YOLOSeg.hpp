#pragma once
#include "common.hpp"

struct Detection {
    cv::Rect bbox;
    float score;
    float label_id;
};

struct Mask {
    cv::Mat maskProposals;
    cv::Mat protos;
    cv::Rect maskRoi;
};

