#include "ChatSession.hpp"

#include <stdexcept>

ChatSession::ChatSession(std::shared_ptr<IChatBackend> backend, int max_history_turns)
    : backend_(std::move(backend)), max_history_turns_(max_history_turns) {
    if (!backend_) throw std::invalid_argument("ChatSession: backend must not be null");
    if (max_history_turns_ < 1)
        throw std::invalid_argument("ChatSession: max_history_turns must be >= 1");
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

void ChatSession::setSystemPrompt(const std::string& prompt) {
    system_prompt_ = prompt;
}

void ChatSession::pinContext(const std::string& text) {
    pinned_context_ = text;
}

void ChatSession::clear() {
    history_.clear();
    // system_prompt_ and pinned_context_ are intentionally kept
}

// ---------------------------------------------------------------------------
// Core: send a turn
// ---------------------------------------------------------------------------

ChatResponse ChatSession::send(const std::string& user_message,
                               const std::vector<std::string>& images,
                               const std::string& model,
                               int max_tokens) {
    Message user_turn;
    user_turn.role    = Message::Role::User;
    user_turn.content = user_message;
    user_turn.images  = images;

    ChatRequest req = buildRequest(user_turn, model, max_tokens);
    ChatResponse resp = backend_->infer(req);

    if (resp.success) {
        // Commit turn to history only on success
        history_.push_back(std::move(user_turn));
        history_.emplace_back(Message{Message::Role::Assistant, resp.text, {}});
        trimHistory();
    }

    return resp;
}

// ---------------------------------------------------------------------------
// Request builder
// ---------------------------------------------------------------------------

ChatRequest ChatSession::buildRequest(const Message& user_turn,
                                      const std::string& model,
                                      int max_tokens) const {
    ChatRequest req;
    req.model             = model.empty() ? model_ : model;
    req.max_tokens        = max_tokens > 0 ? max_tokens : default_max_tokens_;
    req.temperature       = temperature_;
    req.top_p             = top_p_;
    req.target_image_size = target_image_size_;
    req.detail            = detail_;

    // 1. System prompt
    if (!system_prompt_.empty()) {
        req.messages.push_back({Message::Role::System, system_prompt_, {}});
    }

    // 2. Pinned anchor context (injected as User turn, never evicted)
    if (!pinned_context_.empty()) {
        req.messages.push_back({Message::Role::User, pinned_context_, {}});
    }

    // 3. Rolling history (already trimmed to budget)
    req.messages.insert(req.messages.end(), history_.begin(), history_.end());

    // 4. Current user turn (not yet in history_)
    req.messages.push_back(user_turn);

    return req;
}

// ---------------------------------------------------------------------------
// Sliding-window trim (Strategy C: pinned context never dropped)
// ---------------------------------------------------------------------------

void ChatSession::trimHistory() {
    // Each "turn" is a user + assistant message pair (2 messages).
    // Drop oldest pairs until we are within budget.
    while (static_cast<int>(history_.size()) > max_history_turns_ * 2) {
        // Remove oldest user message
        history_.erase(history_.begin());
        // Remove its paired assistant response if present
        if (!history_.empty()) {
            history_.erase(history_.begin());
        }
    }
}
