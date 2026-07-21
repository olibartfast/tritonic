# Tritonic Feature Roadmap

Based on Triton Inference Server v2.65.0–v2.69.0 new features.
Each step is atomic: one PR, one testable unit of work.

---

## Phase 1 — Async Inference + Request Cancellation

### 1.1 — Define async types in core types header
**File:** `include/tritonic/core/types.hpp`
**Action:** Add:
- `InferCallback = std::function<void(BackendResponse)>`
- `CancelToken` — a shared_ptr to a cancellable context (opaque, backend-owned)
- `AsyncInferHandle` — struct holding `CancelToken` and a future/promise

### 1.2 — Extend IInferenceBackend with async + cancel
**File:** `include/tritonic/core/interfaces.hpp`
**Action:** Add pure virtual methods:
- `virtual AsyncInferHandle inferAsync(const BackendRequest&, InferCallback) = 0;`
- `virtual void cancelRequest(const CancelToken&) = 0;`

### 1.3 — Extend ITriton with async + cancel
**File:** `include/tritonic/triton/itriton.hpp`
**Action:** Add pure virtual methods:
- `virtual AsyncInferHandle inferAsync(const std::vector<std::vector<uint8_t>>&, InferCallback) = 0;`
- `virtual void cancelInference(const CancelToken&) = 0;`

### 1.4 — Implement async infer in Triton backend (gRPC)
**File:** `src/triton/Triton.hpp`, `src/triton/Triton.cpp`
**Action:**
- Store a `std::unordered_map<CancelToken, grpc::ClientContext*>` for cancellation.
- Use `InferenceServerGrpcClient::AsyncInfer()` which takes a callback.
- `cancelInference()` calls `grpc::ClientContext::TryCancel()`.
- Add a `std::thread` pool or use `InferResult` polling for the HTTP path (HTTP has no native async — simulate with `std::async` wrapping `RunInfer`).

### 1.5 — Implement async + cancel stubs in ChatBackend
**File:** `src/chat/ChatBackend.hpp`, `src/chat/ChatBackend.cpp`
**Action:**
- `inferAsync()` wraps the existing `infer()` in `std::async` + callback (CURL is synchronous).
- `cancelRequest()` sets a `std::atomic<bool>` abort flag checked in the CURL write callback.

### 1.6 — Update App to use async infer loop
**File:** `src/main/App.cpp`
**Action:**
- Add `--async` CLI flag (`Config.cpp` / `Config.hpp`).
- In the frame-processing loop: if async mode, call `inferAsync()` with a callback that feeds postprocessing, instead of blocking `infer()`.
- Wire SIGINT handler to call `cancelRequest()`.

### 1.7 — Unit tests
**Files:** `tests/unit/test_triton_async.cpp`, `tests/unit/test_chat_async.cpp`
**Action:**
- MockTriton: verify callback invoked, cancellation sets error.
- MockChatBackend: verify callback invoked, mid-request abort sets cancelled status.

**Verification:** `ctest --output-on-failure`

---

## Phase 2 — Streaming / Decoupled Model Inference

### 2.1 — Define streaming types
**File:** `include/tritonic/core/types.hpp`
**Action:** Add:
- `StreamChunk = std::variant<std::vector<uint8_t>, std::string>` (raw tensor data or text).
- `StreamCallback = std::function<void(StreamChunk)>`
- `StreamHandle` — struct with `CancelToken` + `std::future<void>` that completes when stream ends.

### 2.2 — Extend ITriton with stream infer
**File:** `include/tritonic/triton/itriton.hpp`
**Action:** Add:
- `virtual StreamHandle inferStream(const std::vector<std::vector<uint8_t>>&, StreamCallback) = 0;`

### 2.3 — Implement gRPC streaming in Triton backend
**File:** `src/triton/Triton.hpp`, `src/triton/Triton.cpp`
**Action:**
- Use `InferenceServerGrpcClient::StartStream()` with a `grpc::ClientReaderWriter`.
- Spawn a reader thread that pumps chunks to `StreamCallback`.
- Expose the writer side for cancellation via `CancelToken`.

### 2.4 — Implement streaming in ChatBackend
**File:** `src/chat/ChatBackend.hpp`, `src/chat/ChatBackend.cpp`
**Action:**
- Add `inferStream()` that parses SSE (Server-Sent Events) from OpenAI streaming endpoint (`"stream": true`).
- Use CURL `CURLOPT_WRITEFUNCTION` to parse `data: {...}\n\n` chunks, invoke `StreamCallback` per token.

### 2.5 — Streaming REPL mode
**File:** `src/main/client.cpp`
**Action:**
- Add `--stream` CLI flag.
- In interactive chat mode, pass streaming callback that prints tokens as they arrive.

**Verification:** Manual test with vLLM/Llama.cpp streaming endpoint + `--stream --interactive`.

---

## Phase 3 — Inference Timeout Configuration

### 3.1 — Add timeout to InferenceConfig
**File:** `include/tritonic/infra/config.hpp`
**Action:** Add `int inference_timeout_ms = 0;` (0 = no timeout).

### 3.2 — Add --timeout CLI flag
**File:** `src/main/ConfigManager.cpp`
**Action:** Add `--inference_timeout=<ms>` parser entry.

### 3.3 — Thread timeout through Triton infer
**File:** `src/triton/Triton.hpp`, `src/triton/Triton.cpp`
**Action:**
- Pass `options.client_timeout` from config to `InferOptions`.
- Replace hardcoded 30s model-ready loop with configurable timeout (`max_retries = timeout_ms / 1000`).
- In async mode: wrap callback in a timer that auto-cancels via `CancelToken`.

### 3.4 — Thread timeout through ChatBackend
**File:** `src/chat/ChatBackend.hpp`, `src/chat/ChatBackend.cpp`
**Action:** Set `CURLOPT_TIMEOUT_MS` on the CURL handle from config.

**Verification:** CI test with a mock server that delays responses > timeout, assert error response.

---

## Phase 4 — Client-Side Metrics

### 4.1 — Add metrics collector class
**File (new):** `include/tritonic/infra/metrics.hpp`
**Action:**
- `class MetricsCollector` with counters: `total_requests`, `total_success`, `total_fail`, `total_cancelled`.
- Latency histogram: `first_response_latency_ms`, `total_inference_latency_ms`.
- Thread-safe (std::atomic + mutex for histogram).

### 4.2 — Instrument ITriton infer paths
**File:** `src/triton/Triton.cpp`
**Action:**
- Wrap `infer()` and `inferAsync()` with timer RAII guards that push to `MetricsCollector`.
- Log metrics on graceful shutdown or SIGUSR1.

### 4.3 — Pull server-side metrics via HTTP
**File:** `src/triton/Triton.cpp`
**Action:**
- Add `Metrics getServerMetrics()` to `ITriton`.
- HTTP GET `/v2/metrics` → parse prometheus text format → `struct Metrics`.
- Expose via `--metrics` CLI flag (print to stdout or write to file).

### 4.4 — Metrics output
**File:** `src/main/App.cpp`
**Action:**
- Add `--metrics_file=<path>` flag.
- On shutdown, serialize metrics as JSON to file.

**Verification:** Run a session, check metrics file has non-zero counters.

---

## Phase 5 — Batch Inference

### 5.1 — Add batch size to InferenceConfig
**File:** `include/tritonic/infra/config.hpp`
**Action:** Add `int batch_size = 1;`

### 5.2 — Add --batch CLI flag
**File:** `src/main/ConfigManager.cpp`
**Action:** Add `--batch_size=<N>`.

### 5.3 — Batch-frame accumulator in App
**File:** `src/main/App.cpp`
**Action:**
- In the frame loop, accumulate `batch_size` frames into a vector before calling `infer()`.
- Pass batched input data → single infer call → split results back to per-frame postprocessing.
- Leverage Triton's dynamic batching (set model `max_batch_size`).

### 5.4 — Model info validation
**File:** `src/triton/Triton.cpp`
**Action:**
- After `getModelInfo()`, validate `max_batch_size >= config.batch_size`.
- Warn if batching unsupported by model.

**Verification:** Test with a batched YOLO model, verify per-frame detections match single-frame baseline.

---

## Phase 6 — Model Ensemble Awareness

Triton exposes an ensemble as a normal model, so no client-side ensemble request
or response API is required. The implementation work is server-side GPU
preprocessing plus a Tritonic encoded-image input path.

The atomic implementation plan is in
[YOLO Ensemble Inference Roadmap](ENSEMBLE_INFERENCE_ROADMAP.md). It delivers a
DALI preprocessing ensemble first, then separately gates native C++/CUDA
preprocessing and measured GPU postprocessing.

---

## Phase 7 — Chat Backend Model Control

### 7.1 — Extend IChatBackend with model control
**File:** `include/tritonic/chat/ichat_backend.hpp`
**Action:** Add:
- `virtual bool isModelReady(const std::string& model) = 0;`
- `virtual void loadModel(const std::string& model) = 0;`
- `virtual void unloadModel(const std::string& model) = 0;`
- `virtual std::vector<std::string> listModels() = 0;`

### 7.2 — Implement in ChatBackend via Triton OpenAI frontend
**File:** `src/chat/ChatBackend.hpp`, `src/chat/ChatBackend.cpp`
**Action:**
- HTTP calls to `/v2/models/{model}/ready`, `/v2/repository/models/{model}/load`, `/v2/repository/models/{model}/unload`.
- `listModels()` → `GET /v2/repository/index`.

### 7.3 — CLI integration
**File:** `src/main/client.cpp`
**Action:**
- Add `--list_models` flag (loads, prints list, exits).
- Add `--load_model=<name>` and `--unload_model=<name>` flags.
- Interactive mode: `/load`, `/unload`, `/models` chat commands.

**Verification:** Test with a Triton server running the OpenAI frontend.

---

## Phase 8 — Polish

### 8.1 — BFloat16 tensor type support
**File:** `include/tritonic/core/types.hpp`, `src/triton/Triton.cpp`
**Action:**
- Add `DataType::BF16` to type enum.
- Map in `getOpenCVTypeString()` / tensor conversion paths.
- Add `BVLC` byte ordering for Little-Endian Brain Float (Triton standard).

### 8.2 — Shared memory flag adaptation
**File:** `src/triton/Triton.cpp`
**Action:**
- On shared memory registration failure, detect error message containing "allow_client_shm".
- Log clear warning: "Server requires --allow-client-shm=true".
- Add `--allow_client_shm` CLI flag that passes `allow_client_shm=true` to server config if tritonic manages the server process.

### 8.3 — Hardening: input size validation
**File:** `src/main/App.cpp`, `src/triton/Triton.cpp`
**Action:**
- Validate input tensor byte size before sending (guard against > model max input size).
- Cap at model's configured `max_batch_size * tensor_byte_size`.

### 8.4 — Hardening: chunked input limit
**File:** `src/triton/Triton.cpp`
**Action:**
- Mirror Triton v2.69.0 HTTP hardening: cap request body at configurable limit (default 256MB).

**Verification:** Full CI pass + manual smoke test with shared memory enabled Triton server.

---

## Summary of File Changes

| Phase | New files | Modified files |
|-------|-----------|----------------|
| 1 | `tests/unit/test_triton_async.cpp`, `tests/unit/test_chat_async.cpp` | `core/types.hpp`, `core/interfaces.hpp`, `triton/itriton.hpp`, `Triton.hpp`, `Triton.cpp`, `ChatBackend.hpp`, `ChatBackend.cpp`, `App.cpp`, `ConfigManager.cpp`, `config.hpp`, `MockTriton.hpp`, `MockChatBackend.hpp` |
| 2 | — | `core/types.hpp`, `triton/itriton.hpp`, `Triton.hpp`, `Triton.cpp`, `ChatBackend.hpp`, `ChatBackend.cpp`, `client.cpp` |
| 3 | — | `config.hpp`, `ConfigManager.cpp`, `Triton.hpp`, `Triton.cpp`, `ChatBackend.hpp`, `ChatBackend.cpp` |
| 4 | `infra/metrics.hpp`, `infra/metrics.cpp` | `Triton.cpp`, `App.cpp`, `ConfigManager.cpp` |
| 5 | — | `config.hpp`, `ConfigManager.cpp`, `App.cpp`, `Triton.cpp` |
| 6 | — | `core/types.hpp`, `triton/itriton.hpp`, `Triton.cpp`, `ConfigManager.cpp` |
| 7 | — | `ichat_backend.hpp`, `ChatBackend.hpp`, `ChatBackend.cpp`, `client.cpp` |
| 8 | — | `core/types.hpp`, `Triton.cpp`, `App.cpp` |
