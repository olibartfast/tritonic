#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "chat/IChatBackend.hpp"
#include "chat/ChatSession.hpp"
#include "mocks/MockChatBackend.hpp"

using ::testing::_;
using ::testing::Return;
using ::testing::SizeIs;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static ChatResponse ok(const std::string& text) {
    return {.text = text, .success = true, .error = {}};
}
static ChatResponse fail(const std::string& e) {
    return {.text = {}, .success = false, .error = e};
}

// ---------------------------------------------------------------------------
// IChatBackend contract via mock
// ---------------------------------------------------------------------------

TEST(IChatBackendTest, SingleTurnSuccess) {
    MockChatBackend mock;
    EXPECT_CALL(mock, infer(_)).WillOnce(Return(ok("Paris")));

    ChatRequest req;
    req.messages.push_back({Message::Role::User, "Capital of France?"});
    ChatResponse r = mock.infer(req);
    EXPECT_TRUE(r.success);
    EXPECT_EQ(r.text, "Paris");
}

TEST(IChatBackendTest, ServerErrorPropagates) {
    MockChatBackend mock;
    EXPECT_CALL(mock, infer(_)).WillOnce(Return(fail("Connection refused")));

    ChatRequest req;
    req.messages.push_back({Message::Role::User, "hello"});
    ChatResponse r = mock.infer(req);
    EXPECT_FALSE(r.success);
    EXPECT_FALSE(r.error.empty());
}

// ---------------------------------------------------------------------------
// ChatSession: basic turn accumulation
// ---------------------------------------------------------------------------

TEST(ChatSessionTest, SingleTurnAddsToHistory) {
    auto mock = std::make_shared<MockChatBackend>();
    EXPECT_CALL(*mock, infer(_)).WillOnce(Return(ok("Hello there")));

    ChatSession session(mock);
    ChatResponse r = session.send("Hi");
    EXPECT_TRUE(r.success);
    EXPECT_EQ(session.history().size(), 2u);
    EXPECT_EQ(session.history()[0].role, Message::Role::User);
    EXPECT_EQ(session.history()[1].role, Message::Role::Assistant);
    EXPECT_EQ(session.history()[1].content, "Hello there");
}

TEST(ChatSessionTest, FailedTurnDoesNotAddToHistory) {
    auto mock = std::make_shared<MockChatBackend>();
    EXPECT_CALL(*mock, infer(_)).WillOnce(Return(fail("timeout")));

    ChatSession session(mock);
    session.send("Hi");
    EXPECT_TRUE(session.history().empty());
}

// ---------------------------------------------------------------------------
// ChatSession: system prompt and pinned context injected correctly
// ---------------------------------------------------------------------------

TEST(ChatSessionTest, SystemPromptInjectedFirst) {
    auto mock = std::make_shared<MockChatBackend>();

    EXPECT_CALL(*mock, infer(_))
        .WillOnce([](const ChatRequest& req) {
            EXPECT_FALSE(req.messages.empty());
            EXPECT_EQ(req.messages[0].role, Message::Role::System);
            EXPECT_EQ(req.messages[0].content, "You are a C++ reviewer.");
            return ok("looks good");
        });

    ChatSession session(mock);
    session.setSystemPrompt("You are a C++ reviewer.");
    session.send("Review this");
}

TEST(ChatSessionTest, PinnedContextInjectedAfterSystem) {
    auto mock = std::make_shared<MockChatBackend>();

    EXPECT_CALL(*mock, infer(_))
        .WillOnce([](const ChatRequest& req) {
            ASSERT_GE(req.messages.size(), 3u);
            EXPECT_EQ(req.messages[0].role, Message::Role::System);
            EXPECT_EQ(req.messages[1].content, "int x = 42;");
            EXPECT_EQ(req.messages[2].role, Message::Role::User);
            return ok("The code looks fine");
        });

    ChatSession session(mock);
    session.setSystemPrompt("You are a code reviewer.");
    session.pinContext("int x = 42;");
    session.send("Review this code");
}

TEST(ChatSessionTest, PinnedContextNeverEvicted) {
    auto mock = std::make_shared<MockChatBackend>();
    ChatSession session(mock, /*max_history_turns=*/1);
    session.setSystemPrompt("sys");
    session.pinContext("pinned code");

    EXPECT_CALL(*mock, infer(_)).WillRepeatedly(Return(ok("ok")));
    session.send("turn 1");
    session.send("turn 2");
    session.send("turn 3");

    EXPECT_CALL(*mock, infer(_))
        .WillOnce([](const ChatRequest& req) {
            bool found = false;
            for (const auto& m : req.messages) {
                if (m.content == "pinned code") { found = true; break; }
            }
            EXPECT_TRUE(found);
            return ok("ok");
        });
    session.send("turn 4");
}

// ---------------------------------------------------------------------------
// ChatSession: sliding window trim
// ---------------------------------------------------------------------------

TEST(ChatSessionTest, HistoryTrimsAtMaxTurns) {
    auto mock = std::make_shared<MockChatBackend>();
    EXPECT_CALL(*mock, infer(_)).WillRepeatedly(Return(ok("reply")));

    ChatSession session(mock, /*max_history_turns=*/2);
    session.send("turn 1");
    session.send("turn 2");
    session.send("turn 3");

    EXPECT_LE(session.history().size(), 4u);
}

// ---------------------------------------------------------------------------
// ChatSession: clear resets turns, preserves system prompt
// ---------------------------------------------------------------------------

TEST(ChatSessionTest, ClearResetsHistory) {
    auto mock = std::make_shared<MockChatBackend>();
    EXPECT_CALL(*mock, infer(_)).WillRepeatedly(Return(ok("answer")));

    ChatSession session(mock);
    session.setSystemPrompt("sys");
    session.send("hello");
    EXPECT_EQ(session.history().size(), 2u);

    session.clear();
    EXPECT_TRUE(session.history().empty());

    EXPECT_CALL(*mock, infer(_))
        .WillOnce([](const ChatRequest& req) {
            EXPECT_EQ(req.messages[0].role, Message::Role::System);
            return ok("hi");
        });
    session.send("hello again");
}

// ---------------------------------------------------------------------------
// ChatSession: multimodal turn
// ---------------------------------------------------------------------------

TEST(ChatSessionTest, ImagesAttachedToUserMessage) {
    auto mock = std::make_shared<MockChatBackend>();

    EXPECT_CALL(*mock, infer(_))
        .WillOnce([](const ChatRequest& req) {
            const auto& last = req.messages.back();
            EXPECT_EQ(last.role, Message::Role::User);
            EXPECT_THAT(last.images, SizeIs(2));
            return ok("I see two images");
        });

    ChatSession session(mock);
    session.send("describe", {"img1.jpg", "img2.jpg"});
}
