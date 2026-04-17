#pragma once

#include <memory>
#include <string>
#include <vector>
#include "CommonTypes.hpp"
#include "IChatBackend.hpp"

/**
 * Stateful conversation manager for OpenAI-compatible chat backends.
 *
 * Responsibilities:
 *  - Accumulates the full message history across turns (Memento pattern).
 *  - Applies a sliding-window trim to stay within context budgets, while
 *    never evicting the system prompt or the pinned anchor context.
 *  - Delegates each single HTTP call to IChatBackend::infer() (the stateless
 *    Facade), keeping transport logic out of this class.
 *
 * Relationship to ChatBackend / IInferenceBackend:
 *
 *   App  ──► IInferenceBackend ──► ChatBackend  (single-turn, no history)
 *   CLI  ──► ChatSession       ──► IChatBackend (multi-turn, owns history)
 *
 * Sliding window strategy (Strategy C — pinned context):
 *   [system] [pinned_context] [u0 a0] [u1 a1] … [uN]
 *   The system message and pinned context are never dropped.
 *   When history exceeds max_history_turns_, the oldest [uX aX] pair is
 *   removed until the budget is satisfied.
 *
 * Usage example (code review):
 *   ChatSession session(backend, 20);
 *   session.setSystemPrompt("You are a C++ code reviewer.");
 *   session.pinContext(readFile("MyClass.cpp"));   // never evicted
 *   auto r1 = session.send("Review this code");
 *   auto r2 = session.send("Explain issue #2");
 *   auto r3 = session.send("Write a fix for the memory leak");
 */
class ChatSession {
public:
    /**
     * @param backend          IChatBackend implementation (ChatBackend, mock, …)
     * @param max_history_turns  Maximum number of user+assistant *pairs* to keep
     *                           before the oldest pair is evicted.  Does not count
     *                           the system message or pinned context.
     */
    explicit ChatSession(std::shared_ptr<IChatBackend> backend, int max_history_turns = 20);

    /** Set the system-level instruction.  Replaces any previous system prompt.
     *  Must be called before the first send() for best results. */
    void setSystemPrompt(const std::string& prompt);

    /** Pin an anchor context (e.g. a code snippet, document) that will never
     *  be evicted by the sliding window.  Injected as a User message
     *  immediately after the system prompt and marked immutable. */
    void pinContext(const std::string& text);

    /**
     * Send a user message and append the model's reply to history.
     *
     * @param user_message  Plain-text prompt for this turn.
     * @param images        Optional image paths / URLs (multimodal turn).
     * @param model         Model name; empty = use whatever was set last.
     * @param max_tokens    Per-turn token budget (0 = use session default).
     */
    ChatResponse send(const std::string& user_message, const std::vector<std::string>& images = {},
                      const std::string& model = {}, int max_tokens = 0);

    /** Reset history (system prompt and pinned context are preserved). */
    void clear();

    /** Read-only view of the current history (for testing / logging). */
    const std::vector<Message>& history() const noexcept {
        return history_;
    }

private:
    std::shared_ptr<IChatBackend> backend_;
    std::vector<Message> history_;  // accumulated turns (excludes pinned)
    std::string system_prompt_;
    std::string pinned_context_;
    int max_history_turns_;
    std::string model_;
    int default_max_tokens_{512};
    float temperature_{1.0f};
    float top_p_{1.0f};
    int target_image_size_{512};
    std::string detail_{"low"};

    /** Build a full ChatRequest from current state + new user turn. */
    ChatRequest buildRequest(const Message& user_turn, const std::string& model,
                             int max_tokens) const;

    /**
     * Drop the oldest user+assistant pair when history exceeds the budget.
     * System prompt and pinned context slots are never touched.
     */
    void trimHistory();
};
