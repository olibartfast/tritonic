#include <gtest/gtest.h>
#include <cstdlib>
#include <string>
#include "chat/ChatBackend.hpp"
#include "chat/ChatSession.hpp"

/**
 * Integration tests for ChatBackend with real API endpoints.
 *
 * These tests require a live API endpoint and valid credentials.
 * Set environment variables before running:
 *   - CHAT_API_ENDPOINT (e.g., "https://openrouter.ai/api/v1/chat/completions")
 *   - CHAT_API_KEY (optional, for authenticated endpoints)
 *   - CHAT_MODEL (e.g., "google/gemma-2-9b-it")
 *
 * Example:
 *   export CHAT_API_ENDPOINT="https://openrouter.ai/api/v1/chat/completions"
 *   export CHAT_API_KEY="sk-..."
 *   export CHAT_MODEL="google/gemma-2-9b-it"
 *   ./build/tritonic_integration_tests --gtest_filter="ChatBackendIntegration.*"
 */

class ChatBackendIntegration : public ::testing::Test {
protected:
    static std::string getEnv(const char* name, const std::string& default_value = "") {
        const char* val = std::getenv(name);
        return val ? std::string(val) : default_value;
    }

    static bool isConfigured() {
        return !getEnv("CHAT_API_ENDPOINT").empty();
    }

    void SetUp() override {
        if (!isConfigured()) {
            GTEST_SKIP()
                << "Skipping integration test: CHAT_API_ENDPOINT not set. "
                << "Set environment variables to run integration tests:\n"
                << "  export CHAT_API_ENDPOINT="
                   "\"https://your-api-endpoint/v1/chat/completions\"\n"
                << "  export CHAT_API_KEY=\"your-api-key\" (optional)\n"
                << "  export CHAT_MODEL=\"model-name\" (optional, defaults to empty)";
        }

        endpoint_ = getEnv("CHAT_API_ENDPOINT");
        api_key_ = getEnv("CHAT_API_KEY");
        model_ = getEnv("CHAT_MODEL", "");
    }

    std::string endpoint_;
    std::string api_key_;
    std::string model_;
};

TEST_F(ChatBackendIntegration, SimpleTextQuery) {
    ASSERT_FALSE(endpoint_.empty());

    ChatBackend backend(endpoint_, api_key_);

    ChatRequest request;
    request.model = model_;
    request.max_tokens = 50;
    request.temperature = 0.7f;
    request.messages.push_back({Message::Role::User, "What is 2+2? Answer with just the number."});

    ChatResponse response = backend.infer(request);

    EXPECT_TRUE(response.success) << "Error: " << response.error;
    EXPECT_FALSE(response.text.empty()) << "Response text should not be empty";

    // Basic validation: response should contain "4"
    if (response.success) {
        std::cout << "Response: " << response.text << std::endl;
        EXPECT_NE(response.text.find("4"), std::string::npos)
            << "Expected response to contain '4', got: " << response.text;
    }
}

TEST_F(ChatBackendIntegration, SessionMultipleTurns) {
    ASSERT_FALSE(endpoint_.empty());

    auto backend = std::make_shared<ChatBackend>(endpoint_, api_key_);
    ChatSession session(backend);

    // First turn
    ChatResponse r1 = session.send("Hello! My name is Alice.", {}, model_, 50);
    EXPECT_TRUE(r1.success) << "First turn failed: " << r1.error;
    EXPECT_FALSE(r1.text.empty());

    if (r1.success) {
        std::cout << "Turn 1 Response: " << r1.text << std::endl;
    }

    // Second turn - model should remember the name
    ChatResponse r2 = session.send("What is my name?", {}, model_, 50);
    EXPECT_TRUE(r2.success) << "Second turn failed: " << r2.error;

    if (r2.success) {
        std::cout << "Turn 2 Response: " << r2.text << std::endl;
        // Response should mention "Alice" (case-insensitive)
        std::string lower_response = r2.text;
        std::transform(lower_response.begin(), lower_response.end(), lower_response.begin(),
                       ::tolower);
        EXPECT_NE(lower_response.find("alice"), std::string::npos)
            << "Expected model to remember name 'Alice' from previous turn";
    }

    // Verify history
    EXPECT_GE(session.history().size(), 4u);  // 2 user messages + 2 assistant responses
}

TEST_F(ChatBackendIntegration, SystemPromptRespected) {
    ASSERT_FALSE(endpoint_.empty());

    auto backend = std::make_shared<ChatBackend>(endpoint_, api_key_);
    ChatSession session(backend);

    // Set a system prompt that instructs the model to always respond in a specific way
    session.setSystemPrompt(
        "You are a helpful assistant. Always start your response with 'UNDERSTOOD:'");

    ChatResponse response = session.send("Tell me about C++", {}, model_, 100);

    EXPECT_TRUE(response.success) << "Request failed: " << response.error;

    if (response.success) {
        std::cout << "Response with system prompt: " << response.text << std::endl;
        // Check if response follows the system prompt (this is a soft check as not all models
        // strictly follow instructions)
        if (response.text.find("UNDERSTOOD:") == std::string::npos) {
            std::cout << "Note: Model did not strictly follow system prompt" << std::endl;
        }
    }
}

TEST_F(ChatBackendIntegration, MaxTokensLimitsResponse) {
    ASSERT_FALSE(endpoint_.empty());

    ChatBackend backend(endpoint_, api_key_);

    ChatRequest request;
    request.model = model_;
    request.max_tokens = 10;  // Very short response
    request.temperature = 0.7f;
    request.messages.push_back(
        {Message::Role::User, "Write a long essay about the history of computers."});

    ChatResponse response = backend.infer(request);

    EXPECT_TRUE(response.success) << "Error: " << response.error;

    if (response.success) {
        std::cout << "Short response (max 10 tokens): " << response.text << std::endl;

        // Count approximate tokens (split by spaces as rough estimate)
        size_t token_estimate = 0;
        for (char c : response.text) {
            if (c == ' ' || c == '\n')
                ++token_estimate;
        }

        // Response should be relatively short due to token limit
        EXPECT_LE(token_estimate, 30u) << "Response seems too long for max_tokens=10 (estimated "
                                        << token_estimate << " tokens)";
    }
}

TEST_F(ChatBackendIntegration, InvalidModelHandledGracefully) {
    ASSERT_FALSE(endpoint_.empty());

    ChatBackend backend(endpoint_, api_key_);

    ChatRequest request;
    request.model = "this-model-definitely-does-not-exist-12345";
    request.max_tokens = 50;
    request.messages.push_back({Message::Role::User, "Hello"});

    ChatResponse response = backend.infer(request);

    // Should fail gracefully
    EXPECT_FALSE(response.success) << "Expected failure for non-existent model";
    EXPECT_FALSE(response.error.empty()) << "Error message should be provided";

    std::cout << "Expected error for invalid model: " << response.error << std::endl;
}

// This test is commented out because it requires a vision-capable model
// and may not work with all endpoints
/*
TEST_F(ChatBackendIntegration, DISABLED_ImageInference) {
    ASSERT_FALSE(endpoint_.empty());

    ChatBackend backend(endpoint_, api_key_);

    // Create a simple test image
    cv::Mat test_image(100, 100, CV_8UC3, cv::Scalar(255, 0, 0));  // Blue image
    std::string test_image_path = "/tmp/test_chat_image.jpg";
    cv::imwrite(test_image_path, test_image);

    ChatRequest request;
    request.model = model_;
    request.max_tokens = 100;
    request.messages.push_back(
        {Message::Role::User, "What color is this image?", {test_image_path}});

    ChatResponse response = backend.infer(request);

    EXPECT_TRUE(response.success) << "Error: " << response.error;
    if (response.success) {
        std::cout << "Vision response: " << response.text << std::endl;
    }

    // Cleanup
    std::remove(test_image_path.c_str());
}
*/
