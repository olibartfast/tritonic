#pragma once

#include <string>
#include "IChatBackend.hpp"
#include "common/IInferenceBackend.hpp"

/**
 * Concrete OpenAI-compatible Chat Completions client.
 *
 * Works with any server that implements /v1/chat/completions:
 *   Ollama    -- http://localhost:11434/v1/chat/completions
 *   llama.cpp -- http://localhost:8080/v1/chat/completions
 *   SGLang    -- http://localhost:30000/v1/chat/completions
 *   vLLM      -- http://localhost:8000/v1/chat/completions
 *   OpenAI    -- https://api.openai.com/v1/chat/completions
 *   Together  -- https://api.together.xyz/v1/chat/completions
 *
 * Provider selection is purely a matter of --api_endpoint; no subclassing.
 *
 * Design patterns:
 *   Facade    — hides CURL, base64 encoding, JSON building from callers.
 *   Strategy  — implements IInferenceBackend (for App) and IChatBackend
 *               (for ChatSession, which needs the typed interface).
 */
class ChatBackend : public IInferenceBackend, public IChatBackend {
public:
    /**
     * @param endpoint  Full URL, e.g. "http://localhost:11434/v1/chat/completions"
     * @param api_key   Bearer token; may be empty for unauthenticated local servers.
     */
    explicit ChatBackend(std::string endpoint, std::string api_key = {});
    ~ChatBackend() override = default;

    // IChatBackend — typed, used by ChatSession
    ChatResponse infer(const ChatRequest& request) override;

    // IInferenceBackend — variant-based, used by App
    BackendResponse infer(const BackendRequest& request) override;
    std::string backendName() const noexcept override {
        return "chat";
    }

private:
    std::string endpoint_;
    std::string api_key_;

    static std::string encodeImageToBase64(const std::string& image_path, int target_size);
    static bool isUrl(const std::string& s) noexcept;
    static std::string escapeJson(const std::string& s);
    static std::string buildRequestBody(const ChatRequest& request);
    static ChatResponse parseResponse(const std::string& raw_json);

    static std::size_t writeCallback(void* contents, std::size_t size, std::size_t nmemb,
                                     std::string* out);
};
