#include <gtest/gtest.h>
#include "Config.hpp"

class ConfigTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup test configuration
        valid_config_.server_address = "localhost";
        valid_config_.port = 8000;
        valid_config_.model_name = "test_model";
        valid_config_.model_type = "yolov8";
        valid_config_.source = "test.jpg";
    }

    Config valid_config_;
};

TEST_F(ConfigTest, ValidConfigIsValid) {
    EXPECT_TRUE(valid_config_.isValid());
    EXPECT_TRUE(valid_config_.getValidationErrors().empty());
}

TEST_F(ConfigTest, EmptyConfigIsInvalid) {
    Config empty_config;
    EXPECT_FALSE(empty_config.isValid());
    EXPECT_FALSE(empty_config.getValidationErrors().empty());
}

TEST_F(ConfigTest, ModelNameWithPathIsInvalid) {
    Config config = valid_config_;
    
    config.model_name = "path/to/model";
    EXPECT_FALSE(config.isValid());
    
    config.model_name = "path\\to\\model";
    EXPECT_FALSE(config.isValid());
    
    config.model_name = "/absolute/path";
    EXPECT_FALSE(config.isValid());
}

TEST_F(ConfigTest, ValidModelNameIsValid) {
    Config config = valid_config_;
    
    config.model_name = "valid_model_name";
    EXPECT_TRUE(config.isValid());
    
    config.model_name = "model123";
    EXPECT_TRUE(config.isValid());
    
    config.model_name = "model_with_underscores";
    EXPECT_TRUE(config.isValid());
}

TEST_F(ConfigTest, InvalidPortIsInvalid) {
    Config config = valid_config_;
    
    config.port = 0;
    EXPECT_FALSE(config.isValid());
    
    config.port = -1;
    EXPECT_FALSE(config.isValid());
    
    config.port = 99999;
    EXPECT_FALSE(config.isValid());
}

TEST_F(ConfigTest, ValidPortIsValid) {
    Config config = valid_config_;
    
    config.port = 8000;
    EXPECT_TRUE(config.isValid());
    
    config.port = 8080;
    EXPECT_TRUE(config.isValid());
    
    config.port = 9999;
    EXPECT_TRUE(config.isValid());
}
