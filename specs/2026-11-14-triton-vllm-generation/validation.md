# Validation — Native Triton vLLM text generation

Requirements: [requirements.md](requirements.md) · Plan:
[plan.md](plan.md) · Design: [design.md](design.md)

This file defines checks before implementation. **No implementation, build,
unit, loopback, or GPU smoke check has been run for this feature.** Source
contract verification was performed against the linked r26.10 files; it is not
runtime compatibility evidence. All V-2 through V-7 evidence is pending.

## Requirements-to-evidence matrix

| ID | Requirements | Check | Pass criterion |
|---|---|---|---|
| V-1 | R-8–R-12, R-14 | Review r26.10 source links and record the existing configure/build/test baseline for `WITH_CHAT_BACKEND=ON/OFF`. | Wire facts in requirements agree with pinned source. Existing OFF failures are recorded before changes and are not attributed to this feature. No global version pin changes. |
| V-2 | R-8–R-13 | Focused backend/unit tests from T-4. | Valid URL and encoded model; exact JSON contract; UTF-8/control/quote escaping; empty and exact output; failures for malformed/trailing JSON, wrong schema, error object and non-200; response limit accepts 16 MiB and rejects byte 16 MiB + 1; diagnostics ≤1024 bytes; status preserved when observed; zero timeout and positive timeout behavior. |
| V-3 | R-1–R-7, R-13 | Config and CLI tests from T-8. | Required values and ranges enforced; invalid config makes no request; default values are accepted; unsupported options are rejected with explicit-presence and aliases covered; no unknown-backend fallback; feature-off diagnostic; success stdout contains only output + newline and errors are nonzero with no partial stdout. |
| V-4 | R-9–R-13, R-15 | Local ephemeral HTTP-server integration test from T-9/T-10. | Capture POST path, headers and semantic JSON; successful output matches `text_output` exactly; failure scenarios do not become text; timeout, disconnect, size cap and recovery pass deterministically without external services. |
| V-5 | R-14 | Build and CTest matrix below. | All four combinations configure/build/test after the scoped existing chat-OFF guard repair. Feature-OFF selection reports unavailable; feature-ON path is present; existing vision/chat checks pass. Record any unrelated pre-existing baseline failure. |
| V-6 | R-16 | Opt-in exact-image GPU smoke below. | Readiness succeeds; direct curl and TritonIC calls both return generated text without echoing the prompt; unknown model fails nonzero; next valid call succeeds. Evidence includes immutable image digest, host/GPU/driver, model ID/revision, and repository config. **Pending until run.** |
| V-7 | R-17 | Review changed docs and run link checks. | Commands and limitations match R-1–R-7; all relative links resolve and absolute GitHub links respond successfully. |

## CPU build and test matrix

From repository root, with Triton client SDK dependencies installed as
described in `AGENTS.md`:

```bash
export TritonClientBuild_DIR="$PWD/triton_client_libs/install"

cmake -S . -B build-gen-chat -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
  -DWITH_TRITON_GENERATE_BACKEND=ON -DWITH_CHAT_BACKEND=ON
cmake --build build-gen-chat -j2
ctest --test-dir build-gen-chat --output-on-failure

cmake -S . -B build-gen-nochat -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
  -DWITH_TRITON_GENERATE_BACKEND=ON -DWITH_CHAT_BACKEND=OFF
cmake --build build-gen-nochat -j2
ctest --test-dir build-gen-nochat --output-on-failure

cmake -S . -B build-nogen-chat -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
  -DWITH_TRITON_GENERATE_BACKEND=OFF -DWITH_CHAT_BACKEND=ON
cmake --build build-nogen-chat -j2
ctest --test-dir build-nogen-chat --output-on-failure

cmake -S . -B build-nogen-nochat -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
  -DWITH_TRITON_GENERATE_BACKEND=OFF -DWITH_CHAT_BACKEND=OFF
cmake --build build-nogen-nochat -j2
ctest --test-dir build-nogen-nochat --output-on-failure
```

Use separate directories so cached CMake option values cannot mask a missing
source guard. Record configure/build/test exit codes and counts for each
combination. First run the same ON/OFF combinations on the unmodified tree for
V-1; retain those failures as baseline evidence.

## CLI checks

Use the built binary and the loopback fixture to verify:

```bash
./build-gen-chat/tritonic --backend=triton-generate --help
```

The help text must explain `--backend=triton-generate`, host/port, model,
prompt, HTTP-only behavior, and the build option. A successful CLI fixture
must emit only generated text plus a trailing newline on stdout. Capture stdout,
stderr and exit status separately. For invalid config, assert no connection
arrives at the fixture. For feature-off, assert the diagnostic names
`WITH_TRITON_GENERATE_BACKEND`.

Representative validation cases include missing model/prompt; unknown backend;
gRPC; URI syntax in server address; invalid port; empty/path-traversing model;
max_tokens 0; negative/non-finite temperature; top-p 0, >1 or non-finite;
negative timeout; explicitly supplied stop words or repetition penalty even at
default value; unsupported source/image/audio/interactive flags; accepted
`write_frame=true`, `batch_size=1`, CPU preprocessing and other documented
inert defaults; nondefault confidence/NMS or inactive modality values fail.

## Loopback HTTP cases

Run the C++ loopback test from CTest on an ephemeral localhost port, with no
external Triton server. Assert that a prompt containing quotes, backslashes,
newlines and non-ASCII UTF-8 is represented correctly after JSON parsing at the
server fixture. Assert the URL encodes a valid model as one segment and refuses
invalid model names before the request is sent.

The fixture must return each of: valid nonempty output, valid empty output,
valid output with surrounding whitespace, embedded NUL escaped in JSON,
non-200 error JSON, non-200 non-JSON, HTTP 200 Triton `error`, malformed JSON,
trailing garbage, missing `text_output`, non-string `text_output`, invalid
UTF-8, disconnect after headers, delay past a configured timeout, exactly
16 MiB, and 16 MiB + 1. For the exact-boundary cases, use a valid JSON object
with trailing JSON whitespace padded to the target byte length; the +1 case
adds one more whitespace byte. This ensures the cap check is tested separately
from JSON parsing. After a failed response, send another valid request through
the same backend and verify success. Check that no path retries and no redirect
is followed.

## Opt-in Triton 26.10 GPU smoke

Run only on a host where the selected model fits the available GPU. This is an
operator-run compatibility check, not part of CPU CI. Use the upstream
`r26.10` vLLM sample repository as a starting point. Its sample uses the small
`facebook/opt-125m` model, `gpu_memory_utilization: 0.1`, eager execution, and
`KIND_MODEL`; suitability still depends on the host and installed image.
Record the immutable Hugging Face revision used for the model (or the exact
local model artifact digest). Do not replace that value with a moving branch
name in the evidence.

1. Record commit, UTC time, `nvidia-smi` GPU/driver output, Docker image digest
   for `nvcr.io/nvidia/tritonserver:26.10-vllm-python-py3`, model ID and
   revision, and the complete `config.pbtxt` / `model.json` used.
2. Prepare a repository named `vllm_model` from the r26.10 sample, placing
   `config.pbtxt` and `1/model.json` under that model directory. Set the model
   ID and immutable revision for the chosen artifact. The sample config uses
   `backend: "vllm"` and `instance_group` with `kind: KIND_MODEL`.
3. Start the plain server on its normal HTTP port, without launching the
   optional OpenAI frontend:

   ```bash
   docker run --rm --gpus all --net=host \
     -v "$PWD/model_repository:/models:ro" \
     nvcr.io/nvidia/tritonserver:26.10-vllm-python-py3 \
     tritonserver --model-repository=/models
   ```

4. Wait for both `curl -fsS http://localhost:8000/v2/health/ready` and
   `curl -fsS http://localhost:8000/v2/models/vllm_model/ready` to succeed.
   Send a direct request:

   ```bash
   curl -fsS -X POST http://localhost:8000/v2/models/vllm_model/generate \
     -H 'Content-Type: application/json' -H 'Accept: application/json' \
     -d '{"text_input":"Reply with the word READY.","stream":false,"exclude_input_in_output":true,"parameters":{"max_tokens":16,"temperature":0.0,"top_p":1.0}}'
   ```

5. Send the equivalent client request and save stdout, stderr and exit status:

   ```bash
   ./build-gen-chat/tritonic \
     --backend=triton-generate \
     --serverAddress=localhost --port=8000 --protocol=http \
     --model=vllm_model \
     --text_prompt='Reply with the word READY.' \
     --max_tokens=16 --temperature=0 --top_p=1
   ```

   Both responses must contain generated text without the input prompt being
   prepended. The text need not be deterministic unless the model/runtime
   provides deterministic sampling for the recorded settings.
6. Call a nonexistent model and verify a nonzero client result. Then repeat the
   valid call and confirm it succeeds, showing the failed request did not leave
   the client unusable.
7. Store the exact commands, outputs, image digest, GPU/driver, model revision,
   model config and TritonIC commit in the packet’s `evidence/` directory. Mark
   V-6 pass only after reviewing that record; otherwise leave it pending with
   the observed blocker.

## Link verification

Resolve each relative link in the changed documentation from its containing
directory. Check each absolute GitHub link with `curl -sI` (or a successful
fetch) and record any unreachable links. The r26.10 source pages cited in
requirements were opened during spec preparation; future documentation edits
must recheck the final URLs.

## Evidence status

At spec authoring time, V-1 is supported by source inspection of the official
r26.10 model, request utility, sample model config, and Triton Generate
extension. The local feature baseline has not been run during this spec task.
V-2–V-7 are planned checks with no result recorded. In particular, there is no
claim that this client has yet run against a live Triton vLLM server.
