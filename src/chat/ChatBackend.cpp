#include "ChatBackend.hpp"

#include <curl/curl.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <sstream>
#include <stdexcept>

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

ChatBackend::ChatBackend(std::string endpoint, std::string api_key)
    : endpoint_(std::move(endpoint)), api_key_(std::move(api_key)) {
    if (endpoint_.empty()) {
        throw std::invalid_argument("ChatBackend: endpoint must not be empty");
    }
}

// ---------------------------------------------------------------------------
// Public interface — IChatBackend (typed, used by ChatSession)
// ---------------------------------------------------------------------------

ChatResponse ChatBackend::infer(const ChatRequest& request) {
    std::string body = buildRequestBody(request);

    std::string response_buf;
    std::string error_buf(CURL_ERROR_SIZE, '\0');

    CURL* curl = curl_easy_init();
    if (!curl) {
        return {.text = {}, .success = false, .error = "curl_easy_init() failed"};
    }

    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    if (!api_key_.empty()) {
        std::string auth = "Authorization: Bearer " + api_key_;
        headers = curl_slist_append(headers, auth.c_str());
    }

    curl_easy_setopt(curl, CURLOPT_URL, endpoint_.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, static_cast<long>(body.size()));
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_buf);
    curl_easy_setopt(curl, CURLOPT_ERRORBUFFER, error_buf.data());
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 120L);

    CURLcode res = curl_easy_perform(curl);

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK) {
        std::string err = error_buf.data()[0] ? error_buf.data() : curl_easy_strerror(res);
        return {.text = {}, .success = false, .error = "CURL error: " + err};
    }

    return parseResponse(response_buf);
}

// IInferenceBackend — variant-based dispatch (used by App)
BackendResponse ChatBackend::infer(const BackendRequest& request) {
    if (const auto* req = std::get_if<ChatRequest>(&request)) {
        return infer(*req);
    }
    throw std::invalid_argument(
        "ChatBackend::infer() received a TritonInferRequest — "
        "use TritonBackend for binary tensor inference");
}

// ---------------------------------------------------------------------------
// Request builder
// ---------------------------------------------------------------------------

/* static */
std::string ChatBackend::buildRequestBody(const ChatRequest& request) {
    auto roleStr = [](Message::Role r) -> const char* {
        switch (r) {
            case Message::Role::System:
                return "system";
            case Message::Role::Assistant:
                return "assistant";
            default:
                return "user";
        }
    };

    // Serialise each message in the history
    std::ostringstream messages;
    messages << '[';
    bool first_msg = true;
    for (const auto& msg : request.messages) {
        if (!first_msg)
            messages << ',';
        first_msg = false;

        messages << "{\"role\":\"" << roleStr(msg.role) << "\",";

        if (msg.images.empty()) {
            // Plain text message — content is a simple string
            messages << "\"content\":\"" << escapeJson(msg.content) << "\"}";
        } else {
            // Multimodal message — content is an array of content blocks
            messages << "\"content\":[";
            messages << "{\"type\":\"text\",\"text\":\"" << escapeJson(msg.content) << "\"}";
            for (const auto& img : msg.images) {
                messages << ',';
                if (isUrl(img)) {
                    messages << "{\"type\":\"image_url\",\"image_url\":{" << "\"url\":\""
                             << escapeJson(img) << "\"," << "\"detail\":\""
                             << escapeJson(request.detail) << "\"}}";
                } else {
                    std::string b64 = encodeImageToBase64(img, request.target_image_size);
                    messages << "{\"type\":\"image_url\",\"image_url\":{"
                             << "\"url\":\"data:image/jpeg;base64," << b64 << "\","
                             << "\"detail\":\"" << escapeJson(request.detail) << "\"}}";
                }
            }
            messages << "]}";
        }
    }
    messages << ']';

    // Top-level request object
    std::ostringstream body;
    body << '{';
    if (!request.model.empty()) {
        body << "\"model\":\"" << escapeJson(request.model) << "\",";
    }
    body << "\"messages\":" << messages.str() << ",\"max_tokens\":" << request.max_tokens
         << ",\"temperature\":" << request.temperature << ",\"top_p\":" << request.top_p << '}';

    return body.str();
}

// ---------------------------------------------------------------------------
// Response parser — minimal: just extract choices[0].message.content
// ---------------------------------------------------------------------------

/* static */
ChatResponse ChatBackend::parseResponse(const std::string& raw_json) {
    // Lightweight extraction without a full JSON library dependency.
    // Looks for: "content":"<value>"  in the response object.
    // The OpenAI spec always puts it as a plain string (not nested array) here.
    auto extract = [&](const std::string& key) -> std::string {
        std::string needle = "\"" + key + "\":\"";
        auto pos = raw_json.find(needle);
        if (pos == std::string::npos)
            return {};
        pos += needle.size();
        std::string value;
        while (pos < raw_json.size()) {
            if (raw_json[pos] == '\\' && pos + 1 < raw_json.size()) {
                char esc = raw_json[pos + 1];
                switch (esc) {
                    case 'n':
                        value += '\n';
                        break;
                    case 'r':
                        value += '\r';
                        break;
                    case 't':
                        value += '\t';
                        break;
                    default:
                        value += esc;
                        break;
                }
                pos += 2;
            } else if (raw_json[pos] == '"') {
                break;
            } else {
                value += raw_json[pos++];
            }
        }
        return value;
    };

    // Check for API-level error
    auto err_msg = extract("message");
    if (raw_json.find("\"error\"") != std::string::npos && !err_msg.empty()) {
        return {.text = {}, .success = false, .error = err_msg};
    }

    std::string text = extract("content");
    if (text.empty()) {
        return {.text = {}, .success = false, .error = "No content in response: " + raw_json};
    }
    return {.text = text, .success = true, .error = {}};
}

// ---------------------------------------------------------------------------
// Image helpers
// ---------------------------------------------------------------------------

/* static */
std::string ChatBackend::encodeImageToBase64(const std::string& image_path, int target_size) {
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (image.empty()) {
        throw std::runtime_error("ChatBackend: cannot read image: " + image_path);
    }

    // Resize preserving aspect ratio, then square-pad with black
    float aspect = static_cast<float>(image.cols) / static_cast<float>(image.rows);
    int new_w, new_h;
    if (aspect >= 1.0f) {
        new_w = target_size;
        new_h = static_cast<int>(static_cast<float>(target_size) / aspect);
    } else {
        new_h = target_size;
        new_w = static_cast<int>(static_cast<float>(target_size) * aspect);
    }

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(new_w, new_h));

    int top = (target_size - new_h) / 2;
    int bottom = target_size - new_h - top;
    int left = (target_size - new_w) / 2;
    int right = target_size - new_w - left;

    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, top, bottom, left, right, cv::BORDER_CONSTANT,
                       cv::Scalar(0, 0, 0));

    std::vector<uint8_t> buf;
    cv::imencode(".jpg", padded, buf);

    // Base64 encode
    static constexpr char kTable[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    std::string out;
    out.reserve(((buf.size() + 2) / 3) * 4);
    for (std::size_t i = 0; i < buf.size(); i += 3) {
        uint32_t b = static_cast<uint32_t>(buf[i]) << 16;
        if (i + 1 < buf.size())
            b |= static_cast<uint32_t>(buf[i + 1]) << 8;
        if (i + 2 < buf.size())
            b |= static_cast<uint32_t>(buf[i + 2]);

        out += kTable[(b >> 18) & 0x3F];
        out += kTable[(b >> 12) & 0x3F];
        out += (i + 1 < buf.size()) ? kTable[(b >> 6) & 0x3F] : '=';
        out += (i + 2 < buf.size()) ? kTable[(b >> 0) & 0x3F] : '=';
    }
    return out;
}

/* static */
bool ChatBackend::isUrl(const std::string& s) noexcept {
    return s.rfind("http://", 0) == 0 || s.rfind("https://", 0) == 0;
}

/* static */
std::string ChatBackend::escapeJson(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (unsigned char c : s) {
        switch (c) {
            case '\\':
                out += "\\\\";
                break;
            case '"':
                out += "\\\"";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                if (c < 0x20) {
                    char buf[7];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += static_cast<char>(c);
                }
                break;
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// CURL write callback
// ---------------------------------------------------------------------------

/* static */
std::size_t ChatBackend::writeCallback(void* contents, std::size_t size, std::size_t nmemb,
                                       std::string* out) {
    std::size_t total = size * nmemb;
    out->append(static_cast<char*>(contents), total);
    return total;
}
