#pragma once
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

class InferenceConfig {
public:
    InferenceConfig() = default;
    virtual ~InferenceConfig() = default;

    InferenceConfig(const InferenceConfig&) = delete;
    InferenceConfig& operator=(const InferenceConfig&) = delete;
    InferenceConfig(InferenceConfig&&) = default;
    InferenceConfig& operator=(InferenceConfig&&) = default;

    // Server
    const std::string& GetServerAddress() const noexcept {
        return server_address_;
    }
    void SetServerAddress(const std::string& v) {
        server_address_ = v;
    }
    int GetPort() const noexcept {
        return port_;
    }
    void SetPort(int v) {
        port_ = v;
    }
    const std::string& GetProtocol() const noexcept {
        return protocol_;
    }
    void SetProtocol(const std::string& v) {
        protocol_ = v;
    }
    bool GetVerbose() const noexcept {
        return verbose_;
    }
    void SetVerbose(bool v) noexcept {
        verbose_ = v;
    }

    // Model
    const std::string& GetModelName() const noexcept {
        return model_name_;
    }
    void SetModelName(const std::string& v) {
        model_name_ = v;
    }
    const std::string& GetModelVersion() const noexcept {
        return model_version_;
    }
    void SetModelVersion(const std::string& v) {
        model_version_ = v;
    }
    const std::string& GetModelType() const noexcept {
        return model_type_;
    }
    void SetModelType(const std::string& v) {
        model_type_ = v;
    }
    const std::vector<std::vector<int64_t>>& GetInputSizes() const noexcept {
        return input_sizes_;
    }
    void SetInputSizes(const std::vector<std::vector<int64_t>>& v) {
        input_sizes_ = v;
    }

    // Input/Output
    const std::string& GetSource() const noexcept {
        return source_;
    }
    void SetSource(const std::string& v) {
        source_ = v;
    }
    const std::string& GetLabelsFile() const noexcept {
        return labels_file_;
    }
    void SetLabelsFile(const std::string& v) {
        labels_file_ = v;
    }
    int GetBatchSize() const noexcept {
        return batch_size_;
    }
    void SetBatchSize(int v) {
        batch_size_ = v;
    }

    // Processing
    bool GetShowFrame() const noexcept {
        return show_frame_;
    }
    void SetShowFrame(bool v) noexcept {
        show_frame_ = v;
    }
    bool GetWriteFrame() const noexcept {
        return write_frame_;
    }
    void SetWriteFrame(bool v) noexcept {
        write_frame_ = v;
    }
    float GetConfidenceThreshold() const noexcept {
        return confidence_threshold_;
    }
    void SetConfidenceThreshold(float v) noexcept {
        confidence_threshold_ = v;
    }
    float GetNmsThreshold() const noexcept {
        return nms_threshold_;
    }
    void SetNmsThreshold(float v) noexcept {
        nms_threshold_ = v;
    }

    // Shared memory
    const std::string& GetSharedMemoryType() const noexcept {
        return shared_memory_type_;
    }
    void SetSharedMemoryType(const std::string& v) {
        shared_memory_type_ = v;
    }
    int GetCudaDeviceId() const noexcept {
        return cuda_device_id_;
    }
    void SetCudaDeviceId(int v) noexcept {
        cuda_device_id_ = v;
    }

    // Logging
    const std::string& GetLogLevel() const noexcept {
        return log_level_;
    }
    void SetLogLevel(const std::string& v) {
        log_level_ = v;
    }
    const std::string& GetLogFile() const noexcept {
        return log_file_;
    }
    void SetLogFile(const std::string& v) {
        log_file_ = v;
    }

    // Multimodal (unused by tritonic but preserved for API compatibility)
    bool GetEnableMultimodal() const noexcept {
        return enable_multimodal_;
    }
    void SetEnableMultimodal(bool v) noexcept {
        enable_multimodal_ = v;
    }
    const std::string& GetTextInput() const noexcept {
        return text_input_;
    }
    void SetTextInput(const std::string& v) {
        text_input_ = v;
    }
    const std::string& GetAudioInput() const noexcept {
        return audio_input_;
    }
    void SetAudioInput(const std::string& v) {
        audio_input_ = v;
    }
    const std::string& GetTextPrompt() const noexcept {
        return text_prompt_;
    }
    void SetTextPrompt(const std::string& v) {
        text_prompt_ = v;
    }
    const std::string& GetModalityCombination() const noexcept {
        return modality_combination_;
    }
    void SetModalityCombination(const std::string& v) {
        modality_combination_ = v;
    }
    float GetTextWeight() const noexcept {
        return text_weight_;
    }
    void SetTextWeight(float v) noexcept {
        text_weight_ = v;
    }
    float GetImageWeight() const noexcept {
        return image_weight_;
    }
    void SetImageWeight(float v) noexcept {
        image_weight_ = v;
    }
    float GetAudioWeight() const noexcept {
        return audio_weight_;
    }
    void SetAudioWeight(float v) noexcept {
        audio_weight_ = v;
    }

private:
    std::string server_address_{"localhost"};
    int port_{8000};
    std::string protocol_{"http"};
    bool verbose_{false};

    std::string model_name_;
    std::string model_version_;
    std::string model_type_;
    std::vector<std::vector<int64_t>> input_sizes_;

    std::string source_;
    std::string labels_file_;
    int batch_size_{1};

    bool show_frame_{false};
    bool write_frame_{true};
    float confidence_threshold_{0.5f};
    float nms_threshold_{0.4f};

    std::string shared_memory_type_{"none"};
    int cuda_device_id_{0};

    std::string log_level_{"info"};
    std::string log_file_;

    bool enable_multimodal_{false};
    std::string text_input_;
    std::string audio_input_;
    std::string text_prompt_;
    std::string modality_combination_{"concat"};
    float text_weight_{1.0f};
    float image_weight_{1.0f};
    float audio_weight_{1.0f};
};
