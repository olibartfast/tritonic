#pragma once

namespace tritonic::cuda {

/**
 * Validates that a CUDA device is available and usable.
 *
 * Queries the CUDA runtime for available devices and verifies that the requested device ID
 * is valid. Called at startup when --preprocess_device=cuda is specified to fail early with
 * a clear error message rather than crash during preprocessing on CPU-only machines.
 *
 * Throws std::runtime_error if:
 * - CUDA device query fails
 * - No devices are present
 * - The requested device ID is out of range
 */
void RequireCudaDevice(int deviceId);

}  // namespace tritonic::cuda
