#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "App.hpp"
#include "ITriton.hpp"
#include "ILogger.hpp"
#include "Config.hpp"
#include "MockTriton.hpp"

using ::testing::_;
using ::testing::Return;
using ::testing::NiceMock;

class FakeLogger : public ILogger {
public:
    void setLogLevel(LogLevel) override {}
    void setLogFile(const std::string&) override {}
    void setConsoleOutput(bool) override {}
    void log(LogLevel, const std::string&) override {}
    void debug(const std::string&) override {}
    void info(const std::string&) override {}
    void warn(const std::string&) override {}
    void error(const std::string&) override {}
    void fatal(const std::string&) override {}
};

class AppTest : public ::testing::Test {
protected:
    void SetUp() override {
        mockTriton = std::make_shared<MockTriton>();
        logger = std::make_shared<FakeLogger>();
        config = std::make_shared<Config>();
        
        // Setup default valid config
        config->model_name = "test_model";
        config->model_type = "classification";
        config->server_address = "localhost";
        config->port = 8000;
        config->source = "test.jpg";
        config->labels_file = "labels.txt";
    }

    std::shared_ptr<MockTriton> mockTriton;
    std::shared_ptr<FakeLogger> logger;
    std::shared_ptr<Config> config;
};

TEST_F(AppTest, RunFailsIfTritonClientCreationFails) {
    ON_CALL(*mockTriton, createTritonClient())
        .WillByDefault(::testing::Throw(std::runtime_error("Connection failed")));
    EXPECT_CALL(*mockTriton, createTritonClient()).Times(1);
    
    // EXPECT_CALL(*mockLogger, error(_)).Times(1);

    App app(mockTriton, config, logger);
    int result = app.run();
    
    EXPECT_EQ(result, 1);
}

// More tests can be added here to verify App behavior
