#include <iostream>
#include <sstream>
#include "App.hpp"
#include "Config.hpp"
#include "ConfigManager.hpp"
#include "Logger.hpp"
#include "Triton.hpp"
#include "chat/ChatBackend.hpp"
#include "chat/ChatSession.hpp"
#include "chat/IChatBackend.hpp"

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
            logger->Info("  Endpoint: " + config->GetApiEndpoint());

            if (config->GetApiEndpoint().empty()) {
                logger->Error("--api_endpoint is required when --backend=chat");
                return 1;
            }

            std::string api_key;
            if (!config->GetApiKeyEnv().empty()) {
                const char* val = std::getenv(config->GetApiKeyEnv().c_str());
                if (val) {
                    api_key = val;
                } else {
                    logger->Warn("Env var '" + config->GetApiKeyEnv() +
                                 "' is not set — proceeding without API key");
                }
            }

            auto chatBackend = std::make_shared<ChatBackend>(config->GetApiEndpoint(), api_key);

            // Collect images from --source (comma-separated paths or URLs)
            std::vector<std::string> source_images;
            std::istringstream src_stream(config->GetSource());
            std::string token;
            while (std::getline(src_stream, token, ',')) {
                if (!token.empty())
                    source_images.push_back(token);
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

                logger->Info("Sending chat request to " + config->GetApiEndpoint());
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

        auto triton = std::make_shared<Triton>(url, protocol, config->GetModelName(),
                                               config->GetModelVersion(), config->GetVerbose());

        // Create and run App
        std::shared_ptr<InferenceConfig> configPtr = std::move(config);

        App app(triton, configPtr, logger);
        return app.run();

    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
}
