#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <algorithm> 
#include <random>
#include <vector>
#include <string>
#include <variant>

// Common types and lightweight includes that don't require heavy dependencies
// Heavy dependencies like grpc_client.h should only be included where needed

using TensorElement = std::variant<float, int32_t, int64_t, uint8_t>;

/**
 * @brief Tensor structure combining data and shape
 */
struct Tensor {
    std::vector<TensorElement> data;
    std::vector<int64_t> shape;
    
    Tensor() = default;
    Tensor(std::vector<TensorElement> data_, std::vector<int64_t> shape_)
        : data(std::move(data_)), shape(std::move(shape_)) {}
};