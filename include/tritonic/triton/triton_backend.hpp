#pragma once

#include <memory>
#include <stdexcept>
#include "tritonic/core/interfaces.hpp"
#include "tritonic/triton/itriton.hpp"

namespace tritonic::triton {

/**
 * Adapter: bridges ITriton (binary tensor protocol) into
 * core::IInferenceBackend (the common Strategy interface).
 *
 * Exposes client() for Triton-specific lifecycle calls (getModelInfo,
 * isServerLive, …) without leaking those onto IInferenceBackend.
 *
 * Design pattern: Adapter.  Header-only (all methods trivially small).
 */
class TritonBackend final : public core::IInferenceBackend {
public:
    explicit TritonBackend(std::shared_ptr<ITriton> triton)
        : triton_(std::move(triton)) {}

    core::BackendResponse infer(const core::BackendRequest& request) override {
        if (const auto* req = std::get_if<core::TritonInferRequest>(&request)) {
            return triton_->infer(req->input_data);
        }
        throw std::invalid_argument(
            "TritonBackend: received ChatRequest — use ChatBackend instead");
    }

    std::string backendName() const noexcept override { return "triton"; }

    ITriton& client() noexcept { return *triton_; }
    const ITriton& client() const noexcept { return *triton_; }

private:
    std::shared_ptr<ITriton> triton_;
};

}  // namespace tritonic::triton
