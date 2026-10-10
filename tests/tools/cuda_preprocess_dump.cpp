// Runs the shared CUDA letterbox kernel (src/cuda/letterbox.cu) on JPEG files and writes
// the FP32 CHW tensors in the same layout as yolo_preprocess_baseline, for tensor parity
// (spec V-5) and compute-sanitizer runs (spec V-14).
#include "tritonic/cuda/letterbox.hpp"

#include <cuda_runtime_api.h>
#include <opencv2/imgcodecs.hpp>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void Check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

template <typename T>
struct DeviceBuffer {
    T* ptr = nullptr;
    explicit DeviceBuffer(size_t bytes) {
        Check(cudaMalloc(reinterpret_cast<void**>(&ptr), bytes), "cudaMalloc");
    }
    ~DeviceBuffer() {
        cudaFree(ptr);
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

}  // namespace

int main(int argc, char** argv) {
    using tritonic::cuda::PadRounding;
    PadRounding rounding = tritonic::cuda::kDefaultPadRounding;
    int first = 1;
    if (argc > 1 && std::string(argv[1]).rfind("--pad-rounding=", 0) == 0) {
        const std::string value = std::string(argv[1]).substr(15);
        if (value != "floor" && value != "ceil") {
            std::cerr << "--pad-rounding must be floor or ceil\n";
            return 2;
        }
        rounding = value == "ceil" ? PadRounding::Ceil : PadRounding::Floor;
        first = 2;
    }
    if (argc - first < 2) {
        std::cerr << "usage: cuda_preprocess_dump [--pad-rounding=floor|ceil] OUTPUT_DIR IMAGE "
                     "[IMAGE ...]\n";
        return 2;
    }

    const std::filesystem::path outputDir(argv[first]);
    std::filesystem::create_directories(outputDir);
    constexpr int canvas = tritonic::cuda::kLetterboxCanvas;
    const size_t tensorElements = 3 * static_cast<size_t>(canvas) * canvas;

    try {
        cudaStream_t stream = nullptr;
        Check(cudaStreamCreate(&stream), "cudaStreamCreate");
        DeviceBuffer<float> output(tensorElements * sizeof(float));
        std::vector<float> host(tensorElements);

        for (int index = first + 1; index < argc; ++index) {
            const std::filesystem::path source(argv[index]);
            // OpenCV decodes to BGR; the kernel converts channel order itself.
            const cv::Mat image = cv::imread(source.string(), cv::IMREAD_COLOR);
            if (image.empty()) {
                throw std::runtime_error("unable to decode image: " + source.string());
            }

            const size_t rowBytes = static_cast<size_t>(image.cols) * 3;
            DeviceBuffer<uint8_t> input(rowBytes * image.rows);
            Check(cudaMemcpy2DAsync(input.ptr, rowBytes, image.data, image.step, rowBytes,
                                    image.rows, cudaMemcpyHostToDevice, stream),
                  "upload");
            Check(tritonic::cuda::LaunchLetterbox({input.ptr, image.rows, image.cols, rowBytes,
                                                   tritonic::cuda::ChannelOrder::BGR},
                                                  output.ptr, stream, canvas, rounding),
                  "LaunchLetterbox");
            Check(cudaMemcpyAsync(host.data(), output.ptr, tensorElements * sizeof(float),
                                  cudaMemcpyDeviceToHost, stream),
                  "download");
            Check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

            const auto destination = outputDir / (source.stem().string() + ".fp32");
            std::ofstream stream_out(destination, std::ios::binary);
            stream_out.write(reinterpret_cast<const char*>(host.data()),
                             static_cast<std::streamsize>(tensorElements * sizeof(float)));
            if (!stream_out) {
                throw std::runtime_error("unable to write output: " + destination.string());
            }
            std::cout << source.string() << '\t' << image.cols << '\t' << image.rows << '\t'
                      << destination.string() << '\t' << tensorElements * sizeof(float) << '\n';
        }
        cudaStreamDestroy(stream);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
