#include <gtest/gtest.h>
#include "Config.hpp"
#include "ConfigManager.hpp"

// InferenceConfig defaults

TEST(InferenceConfigTest, DefaultValues) {
    InferenceConfig config;
    EXPECT_EQ(config.GetServerAddress(), "localhost");
    EXPECT_EQ(config.GetPort(), 8000);
    EXPECT_EQ(config.GetProtocol(), "http");
    EXPECT_FALSE(config.GetVerbose());
    EXPECT_TRUE(config.GetModelName().empty());
    EXPECT_TRUE(config.GetModelType().empty());
    EXPECT_TRUE(config.GetSource().empty());
    EXPECT_TRUE(config.GetLabelsFile().empty());
    EXPECT_EQ(config.GetBatchSize(), 1);
    EXPECT_FALSE(config.GetShowFrame());
    EXPECT_TRUE(config.GetWriteFrame());
    EXPECT_FLOAT_EQ(config.GetConfidenceThreshold(), 0.5f);
    EXPECT_FLOAT_EQ(config.GetNmsThreshold(), 0.4f);
    EXPECT_EQ(config.GetSharedMemoryType(), "none");
    EXPECT_EQ(config.GetCudaDeviceId(), 0);
    EXPECT_EQ(config.GetLogLevel(), "info");
    EXPECT_TRUE(config.GetLogFile().empty());
    EXPECT_TRUE(config.GetInputSizes().empty());
}

TEST(InferenceConfigTest, ServerSettersGetters) {
    InferenceConfig config;
    config.SetServerAddress("192.168.1.100");
    config.SetPort(8001);
    config.SetProtocol("grpc");
    config.SetVerbose(true);
    EXPECT_EQ(config.GetServerAddress(), "192.168.1.100");
    EXPECT_EQ(config.GetPort(), 8001);
    EXPECT_EQ(config.GetProtocol(), "grpc");
    EXPECT_TRUE(config.GetVerbose());
}

TEST(InferenceConfigTest, ModelSettersGetters) {
    InferenceConfig config;
    config.SetModelName("yolov8n");
    config.SetModelType("yolo");
    config.SetModelVersion("1");
    EXPECT_EQ(config.GetModelName(), "yolov8n");
    EXPECT_EQ(config.GetModelType(), "yolo");
    EXPECT_EQ(config.GetModelVersion(), "1");
}

TEST(InferenceConfigTest, ProcessingSettersGetters) {
    InferenceConfig config;
    config.SetSource("/data/video.mp4");
    config.SetLabelsFile("/data/coco.names");
    config.SetShowFrame(true);
    config.SetWriteFrame(false);
    config.SetConfidenceThreshold(0.7f);
    config.SetNmsThreshold(0.3f);
    EXPECT_EQ(config.GetSource(), "/data/video.mp4");
    EXPECT_EQ(config.GetLabelsFile(), "/data/coco.names");
    EXPECT_TRUE(config.GetShowFrame());
    EXPECT_FALSE(config.GetWriteFrame());
    EXPECT_FLOAT_EQ(config.GetConfidenceThreshold(), 0.7f);
    EXPECT_FLOAT_EQ(config.GetNmsThreshold(), 0.3f);
}

TEST(InferenceConfigTest, InputSizesSingleInput) {
    InferenceConfig config;
    config.SetInputSizes({{3, 640, 640}});
    ASSERT_EQ(config.GetInputSizes().size(), 1u);
    EXPECT_EQ(config.GetInputSizes()[0], (std::vector<int64_t>{3, 640, 640}));
}

TEST(InferenceConfigTest, InputSizesMultipleInputs) {
    InferenceConfig config;
    config.SetInputSizes({{3, 640, 640}, {2}});
    ASSERT_EQ(config.GetInputSizes().size(), 2u);
    EXPECT_EQ(config.GetInputSizes()[0], (std::vector<int64_t>{3, 640, 640}));
    EXPECT_EQ(config.GetInputSizes()[1], (std::vector<int64_t>{2}));
}

TEST(InferenceConfigTest, LoggingSettersGetters) {
    InferenceConfig config;
    config.SetLogLevel("debug");
    config.SetLogFile("/tmp/tritonic.log");
    EXPECT_EQ(config.GetLogLevel(), "debug");
    EXPECT_EQ(config.GetLogFile(), "/tmp/tritonic.log");
}

TEST(InferenceConfigTest, MoveConstruct) {
    InferenceConfig a;
    a.SetModelName("mymodel");
    a.SetPort(9999);
    InferenceConfig b = std::move(a);
    EXPECT_EQ(b.GetModelName(), "mymodel");
    EXPECT_EQ(b.GetPort(), 9999);
}

// ConfigManager

TEST(ConfigManagerTest, HelpReturnsNullptr) {
    ConfigManager mgr;
    const char* argv[] = {"tritonic", "--help"};
    auto config = mgr.LoadFromCommandLine(2, argv);
    EXPECT_EQ(config, nullptr);
}

TEST(ConfigManagerTest, DefaultsWithNoArgs) {
    ConfigManager mgr;
    const char* argv[] = {"tritonic"};
    auto config = mgr.LoadFromCommandLine(1, argv);
    ASSERT_NE(config, nullptr);
    EXPECT_EQ(config->GetServerAddress(), "localhost");
    EXPECT_EQ(config->GetPort(), 8000);
    EXPECT_EQ(config->GetProtocol(), "http");
    EXPECT_FALSE(config->GetVerbose());
    EXPECT_TRUE(config->GetWriteFrame());
    EXPECT_FALSE(config->GetShowFrame());
}

TEST(ConfigManagerTest, ParsesModelArgs) {
    ConfigManager mgr;
    const char* argv[] = {
        "tritonic",
        "--model=yolov8n",
        "--model_type=yolo",
        "--labelsFile=/data/coco.names",
        "--source=/data/image.jpg"
    };
    int argc = sizeof(argv) / sizeof(argv[0]);
    auto config = mgr.LoadFromCommandLine(argc, argv);
    ASSERT_NE(config, nullptr);
    EXPECT_EQ(config->GetModelName(), "yolov8n");
    EXPECT_EQ(config->GetModelType(), "yolo");
    EXPECT_EQ(config->GetLabelsFile(), "/data/coco.names");
    EXPECT_EQ(config->GetSource(), "/data/image.jpg");
}

TEST(ConfigManagerTest, ParsesServerArgs) {
    ConfigManager mgr;
    const char* argv[] = {
        "tritonic",
        "--serverAddress=10.0.0.1",
        "--port=8001",
        "--protocol=grpc",
        "--verbose=true"
    };
    int argc = sizeof(argv) / sizeof(argv[0]);
    auto config = mgr.LoadFromCommandLine(argc, argv);
    ASSERT_NE(config, nullptr);
    EXPECT_EQ(config->GetServerAddress(), "10.0.0.1");
    EXPECT_EQ(config->GetPort(), 8001);
    EXPECT_EQ(config->GetProtocol(), "grpc");
    EXPECT_TRUE(config->GetVerbose());
}

TEST(ConfigManagerTest, ParsesInputSizesSingle) {
    ConfigManager mgr;
    const char* argv[] = {"tritonic", "--input_sizes=3,640,640"};
    auto config = mgr.LoadFromCommandLine(2, argv);
    ASSERT_NE(config, nullptr);
    ASSERT_EQ(config->GetInputSizes().size(), 1u);
    EXPECT_EQ(config->GetInputSizes()[0], (std::vector<int64_t>{3, 640, 640}));
}

TEST(ConfigManagerTest, ParsesInputSizesMultiple) {
    ConfigManager mgr;
    const char* argv[] = {"tritonic", "--input_sizes=3,640,640;2"};
    auto config = mgr.LoadFromCommandLine(2, argv);
    ASSERT_NE(config, nullptr);
    ASSERT_EQ(config->GetInputSizes().size(), 2u);
    EXPECT_EQ(config->GetInputSizes()[0], (std::vector<int64_t>{3, 640, 640}));
    EXPECT_EQ(config->GetInputSizes()[1], (std::vector<int64_t>{2}));
}

TEST(ConfigManagerTest, ParsesConfidenceAndNms) {
    ConfigManager mgr;
    const char* argv[] = {
        "tritonic",
        "--confidence_threshold=0.7",
        "--nms_threshold=0.3"
    };
    int argc = sizeof(argv) / sizeof(argv[0]);
    auto config = mgr.LoadFromCommandLine(argc, argv);
    ASSERT_NE(config, nullptr);
    EXPECT_FLOAT_EQ(config->GetConfidenceThreshold(), 0.7f);
    EXPECT_FLOAT_EQ(config->GetNmsThreshold(), 0.3f);
}
