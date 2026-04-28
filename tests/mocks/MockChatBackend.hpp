#pragma once

#include <gmock/gmock.h>
#include "chat/IChatBackend.hpp"

class MockChatBackend : public IChatBackend {
public:
    MOCK_METHOD(ChatResponse, infer, (const ChatRequest&), (override));
};
