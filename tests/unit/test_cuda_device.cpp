#ifdef TRITONIC_ENABLE_CUDA

#include <gtest/gtest.h>

#include "tritonic/cuda/device.hpp"

TEST(CudaDeviceTest, RejectsOutOfRangeDevice) {
    // Device 9999 is extremely unlikely to exist; expect rejection.
    EXPECT_THROW(tritonic::cuda::RequireCudaDevice(9999), std::runtime_error);
}

TEST(CudaDeviceTest, RejectsNegativeDevice) {
    // Negative device IDs are invalid.
    EXPECT_THROW(tritonic::cuda::RequireCudaDevice(-1), std::runtime_error);
}

#endif  // TRITONIC_ENABLE_CUDA
