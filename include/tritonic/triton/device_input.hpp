#pragma once

#include <cstddef>
#include <vector>

#include "tritonic/core/types.hpp"

namespace tritonic::triton {

/**
 * Optional capability of an ITriton client: inference whose single input already lives in
 * device memory registered with the server (CUDA shared memory), so the caller writes the
 * input tensor in place instead of handing over a host buffer.
 *
 * Kept separate from ITriton so that existing implementations and mocks are unaffected;
 * callers discover it with std::dynamic_pointer_cast.
 */
class IDeviceInputInference {
public:
    virtual ~IDeviceInputInference() = default;

    // Device pointer to the registered region backing input 0, created on first use.
    // Throws if the model's input 0 needs more than `bytes`, or does not match it.
    virtual void* deviceInputBuffer(size_t bytes) = 0;

    // Infers from the region returned by deviceInputBuffer(). The caller must have finished
    // writing it (the producing stream synchronized) before calling.
    virtual std::vector<core::Tensor> inferFromDeviceInput() = 0;
};

}  // namespace tritonic::triton
