#include <cctype>
#include <iostream>
#include <optional>
#include <sstream>
#include <string_view>
#include "App.hpp"
#include "Config.hpp"
#include "ConfigManager.hpp"
#include "Logger.hpp"
#include "Triton.hpp"
#include "chat/ChatBackend.hpp"
#include "chat/ChatSession.hpp"
#include "chat/IChatBackend.hpp"

namespace {

std::string ToLower(std::string value) {
    for (char& ch : value) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return value;
}

std::optional<std::string> ResolveApiEndpoint(const InferenceConfig& config) {
    if (!config.GetApiEndpoint().empty()) {
        return config.GetApiEndpoint();
    }

    const std::string service = ToLower(config.GetApiService());
    if (service.empty()) {
        return std::nullopt;
    }

    if (service == "openai") {
        return "https://api.openai.com/v1/chat/completions";
    }
    if (service == "openrouter") {
        return "https://openrouter.ai/api/v1/chat/completions";
    }
    if (service == "together") {
        return "https://api.together.xyz/v1/chat/completions";
    }
    if (service == "zai") {
        return "https://api.z.ai/api/paas/v4/chat/completions";
    }
    throw std::invalid_argument("Unsupported --api_service value: " + config.GetApiService());
}

std::string ResolveApiKeyEnv(const InferenceConfig& config) {
    if (!config.GetApiKeyEnv().empty()) {
        return config.GetApiKeyEnv();
    }

    const std::string service = ToLower(config.GetApiService());
    if (service == "openrouter") {
        return "OPENROUTER_API_KEY";
    }
    if (service == "together") {
        return "TOGETHER_API_KEY";
    }
    if (service == "zai") {
        return "ZAI_API_KEY";
    }
    if (service == "openai") {
        return "OPENAI_API_KEY";
    }
    return {};
}

bool IsKnownTextOnlyModel(std::string_view model_name) {
    return model_name == "glm-5.1";
}

}  // namespace

int main(int argc, const char* argv[]) {
    try {
        // Initialize logging
        auto logger = std::dynamic_pointer_cast<Logger>(LoggerManager::GetLogger("tritonic"));
        logger->SetLevel(LogLevel::INFO);
        logger->EnableConsoleOutput(true);

        // Load configuration
        std::unique_ptr<InferenceConfig> config;
        try {
            ConfigManager configManager;
            config = configManager.LoadFromCommandLine(argc, argv);
        } catch (const std::invalid_argument& e) {
            logger->Error("Command line configuration error: " + std::string(e.what()));
            return 1;
        }

        if (!config) {
            return 0;  // Help requested
        }

        // Configure logger
        if (!config->GetLogFile().empty()) {
            logger->SetOutputFile(config->GetLogFile());
        }

        if (config->GetLogLevel() == "debug")
            logger->SetLevel(LogLevel::DEBUG);
        else if (config->GetLogLevel() == "warn")
            logger->SetLevel(LogLevel::WARN);
        else if (config->GetLogLevel() == "error")
            logger->SetLevel(LogLevel::ERROR);
        else
            logger->SetLevel(LogLevel::INFO);

        // Log configuration
        logger->Info("Configuration:");
        logger->Info("  Server Address: " + config->GetServerAddress());
        logger->Info("  Port: " + std::to_string(config->GetPort()));
        logger->Info("  Protocol: " + config->GetProtocol());
        logger->Info("  Model Name: " + config->GetModelName());
        logger->Info("  Model Version: " + config->GetModelVersion());
        logger->Info("  Model Type: " + config->GetModelType());
        logger->Info("  Source: " + config->GetSource());
        logger->Info("  Labels File: " + config->GetLabelsFile());
        logger->Info("  Verbose: " + std::string(config->GetVerbose() ? "true" : "false"));
        logger->Info("  Show Frame: " + std::string(config->GetShowFrame() ? "true" : "false"));
        logger->Info("  Write Frame: " + std::string(config->GetWriteFrame() ? "true" : "false"));
        if (!config->GetTextPrompt().empty() || !config->GetTextInput().empty()) {
            logger->Info("  Text Prompt: " + config->GetTextPrompt());
            logger->Info("  Text Input: " + config->GetTextInput());
            logger->Info("  Max Tokens: " + std::to_string(config->GetMaxTokens()));
            logger->Info("  Temperature: " + std::to_string(config->GetTemperature()));
            logger->Info("  Top-P: " + std::to_string(config->GetTopP()));
        }

        // ---------------------------------------------------------------
        // Select backend and run
        // ---------------------------------------------------------------
        if (config->GetBackend() == "chat") {
            // --- Chat (OpenAI-compatible) backend ---
            logger->Info("Backend: chat (OpenAI-compatible)");

            const auto resolvedEndpoint = ResolveApiEndpoint(*config);
            if (!resolvedEndpoint) {
                logger->Error("--api_endpoint or --api_service is required when --backend=chat");
                return 1;
            }

            const std::string apiEndpoint = *resolvedEndpoint;
            logger->Info("  Endpoint: " + apiEndpoint);

            const std::string apiKeyEnv = ResolveApiKeyEnv(*config);
            if (!config->GetApiService().empty()) {
                logger->Info("  API Service: " + config->GetApiService());
            }

            std::string api_key;
            if (!apiKeyEnv.empty()) {
                const char* val = std::getenv(apiKeyEnv.c_str());
                if (val) {
                    api_key = val;
                } else {
                    logger->Warn("Env var '" + apiKeyEnv +
                                 "' is not set — proceeding without API key");
                }
            }

            auto chatBackend = std::make_shared<ChatBackend>(apiEndpoint, api_key,
                                                             config->GetInferenceTimeoutMs());

            // Collect images from --source (comma-separated paths or URLs)
            std::vector<std::string> source_images;
            std::istringstream src_stream(config->GetSource());
            std::string token;
            while (std::getline(src_stream, token, ',')) {
                if (!token.empty())
                    source_images.push_back(token);
            }

            if (!source_images.empty() && IsKnownTextOnlyModel(config->GetModelName())) {
                logger->Error("Model '" + config->GetModelName() +
                              "' is text-only. For multimodal GLM requests, use GLM-4.6V "
                              "with --api_service=zai instead.");
                return 1;
            }

            if (config->GetInteractive()) {
                // --- Interactive / multi-turn mode (ChatSession) ---
                logger->Info("Interactive chat mode. Type 'exit' or Ctrl-D to quit.");
                ChatSession session(chatBackend);
                if (!config->GetTextPrompt().empty()) {
                    session.setSystemPrompt(config->GetTextPrompt());
                }

                std::string line;
                while (true) {
                    std::cout << "You> " << std::flush;
                    if (!std::getline(std::cin, line))
                        break;
                    if (line == "exit" || line == "quit")
                        break;
                    if (line.empty())
                        continue;

                    // Images only on the first turn if provided via --source
                    std::vector<std::string> turn_images;
                    if (!source_images.empty()) {
                        turn_images = source_images;
                        source_images.clear();  // subsequent turns are text only
                    }

                    ChatResponse resp = session.send(line, turn_images, config->GetModelName(),
                                                     config->GetMaxTokens());
                    if (resp.success) {
                        std::cout << "Bot> " << resp.text << '\n';
                    } else {
                        logger->Error("Chat error: " + resp.error);
                    }
                }
            } else {
                // --- Single-turn mode ---
                if (config->GetTextPrompt().empty()) {
                    logger->Error("--text_prompt is required for --backend=chat single-turn mode");
                    return 1;
                }

                Message user_msg;
                user_msg.role = Message::Role::User;
                user_msg.content = config->GetTextPrompt();
                user_msg.images = source_images;

                ChatRequest req;
                req.messages.push_back(std::move(user_msg));
                req.model = config->GetModelName();
                req.max_tokens = config->GetMaxTokens();
                req.temperature = config->GetTemperature();
                req.top_p = config->GetTopP();
                req.target_image_size = config->GetTargetImageSize();

                logger->Info("Sending chat request to " + apiEndpoint);
                ChatResponse resp = chatBackend->infer(req);

                if (resp.success) {
                    std::cout << resp.text << '\n';
                } else {
                    logger->Error("Chat backend error: " + resp.error);
                    return 1;
                }
            }
            return 0;
        }

        // --- Triton backend (default) ---
        int port = config->GetPort();
        ProtocolType protocol =
            config->GetProtocol() == "grpc" ? ProtocolType::GRPC : ProtocolType::HTTP;

        // Auto-switch port if protocol is GRPC and port is default HTTP port
        if (protocol == ProtocolType::GRPC && port == 8000) {
            logger->Warn(
                "Protocol is GRPC but port is 8000 (default HTTP). Switching to 8001 (default "
                "GRPC).");
            port = 8001;
        }

        std::string url = config->GetServerAddress() + ":" + std::to_string(port);

        auto triton = std::make_shared<Triton>(
            url, protocol, config->GetModelName(), config->GetModelVersion(), config->GetVerbose(),
            SharedMemoryType::SYSTEM_SHARED_MEMORY, 0, config->GetInferenceTimeoutMs());

        // Create and run App
        std::shared_ptr<InferenceConfig> configPtr = std::move(config);

        App app(triton, configPtr, logger);
        return app.run();

    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
}
