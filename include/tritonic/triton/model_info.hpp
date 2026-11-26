#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace tritonic::triton {

enum class InputType : int {
    UInt8 = 0,
    Int8,
    UInt16,
    Int16,
    Int32,
    Float32,
    Float64,
    String,
};

struct ModelInfo {
    using InputType = tritonic::triton::InputType;

    std::vector<std::string> output_names;
    std::vector<std::string> output_datatypes;
    std::vector<std::vector<int64_t>> output_shapes;
    std::vector<std::string> input_names;
    std::vector<std::vector<int64_t>> input_shapes;
    std::vector<std::string> input_formats;
    std::vector<std::string> input_datatypes;
    std::vector<InputType> input_types;

    InputType type1_{InputType::Float32};
    InputType type3_{InputType::Float32};
    int max_batch_size_{0};
    int batch_size_{1};
};

}  // namespace tritonic::triton
