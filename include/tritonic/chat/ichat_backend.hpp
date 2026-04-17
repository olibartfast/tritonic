#pragma once

#include "tritonic/core/types.hpp"

namespace tritonic::chat {

/**
 * Strategy interface for OpenAI-compatible Chat Completions backends.
 * Used by ChatSession (which needs the typed, stateless infer() call).
 */
class IChatBackend {
public:
    virtual ~IChatBackend() = default;

    virtual core::ChatResponse infer(const core::ChatRequest& request) = 0;
};

}  // namespace tritonic::chat
