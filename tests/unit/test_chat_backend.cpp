#include <gtest/gtest.h>
#include <functional>

#include "chat/ChatSession.hpp"
#include "chat/IChatBackend.hpp"

namespace {

ChatResponse ok(const std::string& text) {
    return {.text = text, .success = true, .error = {}};
}

ChatResponse fail(const std::string& e) {
    return {.text = {}, .success = false, .error = e};
}

class StubChatBackend : public IChatBackend {
public:
    std::function<ChatResponse(const ChatRequest&)> handler;
    int call_count{0};
    ChatRequest last_request;

    ChatResponse infer(const ChatRequest& request) override {
        ++call_count;
        last_request = request;
        if (handler) {
            return handler(request);
        }
        return fail("no handler configured");
    }
};

}  // namespace

TEST(IChatBackendTest, SingleTurnSuccess) {
    StubChatBackend backend;
    backend.handler = [](const ChatRequest&) -> ChatResponse { return ok("Paris"); };

    ChatRequest req;
    req.messages.push_back({Message::Role::User, "Capital of France?"});
    ChatResponse r = backend.infer(req);
    EXPECT_TRUE(r.success);
    EXPECT_EQ(r.text, "Paris");
}

TEST(IChatBackendTest, ServerErrorPropagates) {
    StubChatBackend backend;
    backend.handler = [](const ChatRequest&) -> ChatResponse { return fail("Connection refused"); };

    ChatRequest req;
    req.messages.push_back({Message::Role::User, "hello"});
    ChatResponse r = backend.infer(req);
    EXPECT_FALSE(r.success);
    EXPECT_FALSE(r.error.empty());
}

TEST(ChatSessionTest, SingleTurnAddsToHistory) {
    auto backend = std::make_shared<StubChatBackend>();
    backend->handler = [](const ChatRequest&) -> ChatResponse { return ok("Hello there"); };

    ChatSession session(backend);
    ChatResponse r = session.send("Hi");
    EXPECT_TRUE(r.success);
    ASSERT_EQ(session.history().size(), 2u);
    EXPECT_EQ(session.history()[0].role, Message::Role::User);
    EXPECT_EQ(session.history()[1].role, Message::Role::Assistant);
    EXPECT_EQ(session.history()[1].content, "Hello there");
}

TEST(ChatSessionTest, FailedTurnDoesNotAddToHistory) {
    auto backend = std::make_shared<StubChatBackend>();
    backend->handler = [](const ChatRequest&) -> ChatResponse { return fail("timeout"); };

    ChatSession session(backend);
    session.send("Hi");
    EXPECT_TRUE(session.history().empty());
}

TEST(ChatSessionTest, SystemPromptInjectedFirst) {
    auto backend = std::make_shared<StubChatBackend>();
    backend->handler = [](const ChatRequest& req) -> ChatResponse {
        EXPECT_FALSE(req.messages.empty());
        if (req.messages.empty()) {
            return fail("empty messages");
        }
        EXPECT_EQ(req.messages[0].role, Message::Role::System);
        EXPECT_EQ(req.messages[0].content, "You are a C++ reviewer.");
        return ok("looks good");
    };

    ChatSession session(backend);
    session.setSystemPrompt("You are a C++ reviewer.");
    session.send("Review this");
}

TEST(ChatSessionTest, PinnedContextInjectedAfterSystem) {
    auto backend = std::make_shared<StubChatBackend>();
    backend->handler = [](const ChatRequest& req) -> ChatResponse {
        EXPECT_GE(req.messages.size(), 3u);
        if (req.messages.size() < 3u) {
            return fail("insufficient messages");
        }
        EXPECT_EQ(req.messages[0].role, Message::Role::System);
        EXPECT_EQ(req.messages[1].content, "int x = 42;");
        EXPECT_EQ(req.messages[2].role, Message::Role::User);
        return ok("The code looks fine");
    };

    ChatSession session(backend);
    session.setSystemPrompt("You are a code reviewer.");
    session.pinContext("int x = 42;");
    session.send("Review this code");
}

TEST(ChatSessionTest, PinnedContextNeverEvicted) {
    auto backend = std::make_shared<StubChatBackend>();
    backend->handler = [](const ChatRequest&) -> ChatResponse { return ok("ok"); };

    ChatSession session(backend, /*max_history_turns=*/1);
    session.setSystemPrompt("sys");
    session.pinContext("pinned code");
    session.send("turn 1");
    session.send("turn 2");
    session.send("turn 3");

    backend->handler = [](const ChatRequest& req) -> ChatResponse {
        bool found = false;
        for (const auto& m : req.messages) {
            if (m.content == "pinned code") {
                found = true;
                break;
            }
        }
        EXPECT_TRUE(found);
        return ok("ok");
    };

    session.send("turn 4");
}

TEST(ChatSessionTest, HistoryTrimsAtMaxTurns) {
    auto backend = std::make_shared<StubChatBackend>();
    backend->handler = [](const ChatRequest&) -> ChatResponse { return ok("reply"); };

    ChatSession session(backend, /*max_history_turns=*/2);
    session.send("turn 1");
    session.send("turn 2");
    session.send("turn 3");

    EXPECT_LE(session.history().size(), 4u);
}

TEST(ChatSessionTest, ClearResetsHistory) {
    auto backend = std::make_shared<StubChatBackend>();
    backend->handler = [](const ChatRequest&) -> ChatResponse { return ok("answer"); };

    ChatSession session(backend);
    session.setSystemPrompt("sys");
    session.send("hello");
    EXPECT_EQ(session.history().size(), 2u);

    session.clear();
    EXPECT_TRUE(session.history().empty());

    backend->handler = [](const ChatRequest& req) -> ChatResponse {
        EXPECT_FALSE(req.messages.empty());
        if (req.messages.empty()) {
            return fail("empty messages");
        }
        EXPECT_EQ(req.messages[0].role, Message::Role::System);
        return ok("hi");
    };

    session.send("hello again");
}

TEST(ChatSessionTest, ImagesAttachedToUserMessage) {
    auto backend = std::make_shared<StubChatBackend>();
    backend->handler = [](const ChatRequest& req) -> ChatResponse {
        EXPECT_FALSE(req.messages.empty());
        if (req.messages.empty()) {
            return fail("empty messages");
        }
        const auto& last = req.messages.back();
        EXPECT_EQ(last.role, Message::Role::User);
        EXPECT_EQ(last.images.size(), 2u);
        return ok("I see two images");
    };

    ChatSession session(backend);
    session.send("describe", {"img1.jpg", "img2.jpg"});
}
