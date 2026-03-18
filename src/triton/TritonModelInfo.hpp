#pragma once
#include <opencv2/core.hpp>
#include <string>
#include <vector>

struct TritonModelInfo {
    std::vector<std::string> output_names;
    std::vector<std::string> input_names;
    std::vector<std::vector<int64_t>> input_shapes;
    std::vector<std::string> input_formats;
    std::vector<std::string> input_datatypes;
    std::vector<int> input_types;

    int type1_{CV_32FC1};
    int type3_{CV_32FC3};
    int max_batch_size_{0};
    int batch_size_{1};
};