#pragma once

#include <string>
#include "tritonic/core/types.hpp"

namespace tritonic::core {

/**
 * Common Strategy interface for all inference backends.
 *
 * Both the Triton tensor backend and the OpenAI-compatible Chat backend
 * implement this interface, enabling dependency injection, mockability,
 * and clean modularity.
 *
 * Design patterns: Strategy (App holds IInferenceBackend*), Facade (each
 * concrete backend hides its transport details).
 */
class IInferenceBackend {
public:
    virtual ~IInferenceBackend() = default;

    virtual BackendResponse infer(const BackendRequest& request) = 0;

    virtual std::string backendName() const noexcept = 0;
};

}  // namespace tritonic::core
