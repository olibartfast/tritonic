#include "tritonic/cuda/device.hpp"

#include <stdexcept>
#include <string>

#include <cuda_runtime_api.h>

namespace tritonic::cuda {

void RequireCudaDevice(int deviceId) {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess) {
        throw std::runtime_error(
            "--preprocess_device=cuda needs an NVIDIA GPU and driver, but CUDA reports: " +
            std::string(cudaGetErrorString(err)) +
            ". Use --preprocess_device=cpu on this machine.");
    }

    if (deviceCount == 0) {
        throw std::runtime_error(
            "--preprocess_device=cuda needs an NVIDIA GPU and driver, but CUDA reports: "
            "no devices. Use --preprocess_device=cpu on this machine.");
    }

    if (deviceId < 0 || deviceId >= deviceCount) {
        throw std::runtime_error(
            "--preprocess_device=cuda requested device " + std::to_string(deviceId) + " but only " +
            std::to_string(deviceCount) +
            " device(s) available. Use --cuda_device_id to select a valid device.");
    }
}

}  // namespace tritonic::cuda
