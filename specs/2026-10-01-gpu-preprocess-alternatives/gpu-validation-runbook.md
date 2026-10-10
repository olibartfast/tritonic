# GPU validation runbook — GPU preprocessing alternatives

How to reproduce the GPU evidence in [validation.md](validation.md). There are two paths:

| Path | Needs | Covers | Cost |
|---|---|---|---|
| A. Google Colab | Colab GPU runtime (no Docker) | T-2, T-3, V-5 (kernel), V-14 (kernel) | Colab compute units, ~25 min on a T4 |
| B. Docker GPU host (NVIDIA Brev or any box) | NVIDIA driver ≥ R580 (or data-center GPU with forward compat), Docker + NVIDIA container toolkit, ≥ 150 GB disk | Everything: builds, spikes, V-5 … V-15 | ~2 h; A10G `g5.2xlarge` on Brev ≈ $1.45/h |

Pins for both paths (NGC 26.08, same as `olibartfast/rf-detr-cpp-inference` `versions.env`):
`tritonserver:26.08-py3`, `-py3-sdk`, `tensorrt:26.08-py3`; TensorRT 11.2.1.2 (built
against CUDA 13.3); DALI 2.2.0; client libraries `r26.08`.

---

## Path A — Google Colab (partial gate)

**What it proves:** the DALI conventions (pad rounding, resampling), TensorRT's acceptance of the
in-graph letterbox, the shared CUDA kernel's parity with DALI, and the kernel's memory safety.
**What it cannot prove:** anything Triton-dependent (backend loading, ensembles, V2's CUDA
shared memory, detection parity, soak, benchmarks), because Colab has no Docker.

1. On the development machine, build the upload bundle (the notebook needs only these files):

   ```bash
   zip -r tritonic.zip integration-tests/gpu-preprocess \
       deploy/object_detection/yolo26/ensemble/generate_pipeline.py \
       deploy/object_detection/yolo26/ensemble/trt_pre include/tritonic/cuda \
       src/cuda/letterbox.cu tests/tools/cuda_preprocess_dump.cpp benchmarks/common \
       data/images/{bus,horses,person,mug}.jpg -x '*__pycache__*'
   git rev-parse --short HEAD > COMMIT && zip tritonic.zip COMMIT
   ```

2. In Colab: **Runtime → Change runtime type → T4 GPU** (any NVIDIA GPU works; T4 is the cheapest),
   then **Connect**.
3. Open the **Files** panel and upload `tritonic.zip` and
   `integration-tests/gpu-preprocess/colab_validation.ipynb` to `/content`. Wait until the upload
   spinner stops: a command that runs before an upload has finished silently uses the old file.
4. Either open the uploaded notebook and **Run all**, or run it from any notebook cell:

   ```python
   import json; from IPython import get_ipython
   nb = json.load(open('/content/colab_validation.ipynb'))
   print('CELL_RESULTS', [get_ipython().run_cell(''.join(c['source'])).success
                          for c in nb['cells'] if c['cell_type'] == 'code'])
   ```

5. Read the line between `SUMMARY_JSON_BEGIN` and `SUMMARY_JSON_END`, and copy it into the
   validation evidence log. Detailed reports are under `/content/results/`; Colab deletes them
   when the runtime ends.
6. **Runtime → Disconnect and delete runtime** to stop using compute units.

Timing on the 2026-10-09 run: the CUDA 13.3 toolkit apt download (~3 GB) took about 12 minutes at
3–6 MB/s and dominates the run; the checks themselves take about 2 minutes. The notebook installs
`cuda-compat` itself when the driver is older than R580 (that run's T4 had 580.82, so no compat
was needed).

---

## Path B — Docker GPU host via NVIDIA Brev (full gate)

### 1. Create the instance

```bash
brev login                                  # once; interactive
brev search --gpu-name A10G --stoppable --sort price
brev create tritonic-gate --type g5.2xlarge --min-disk 200
brev ls                                     # wait for BUILD=COMPLETED and SHELL=READY
```

* `--min-disk 200`: the default disk is 10 GB, while the three NGC 26.08 images (~45 GB) plus
  builds need about 80 GB.
* A10G (sm_86) matches an RTX 30-series card, so its benchmark numbers are the most comparable;
  an L4 (`g2-standard-8`, $1.02/h) is cheaper and is just as good for correctness.
* `brev create` can print `Ready` while the console still shows `BUILDING`: SSH only works once
  `brev ls` shows `SHELL READY` (~10–15 minutes after creation on 2026-10-09).
* **Check the disk before pulling images.** On the AWS `g5.2xlarge` (Deep Learning AMI, Ubuntu 22.04,
  driver 595.91 / CUDA 13.2) `--min-disk 200` did not grow `/`: the root disk is 117 GB with
  64 GB free, and the extra space is a 391 GB local NVMe volume at `/opt/dlami/nvme` that is
  wiped on stop. Docker uses the containerd snapshotter, so move both stores onto it:

  ```bash
  brev exec tritonic-gate "set -e
    sudo systemctl stop docker docker.socket containerd
    for d in docker containerd; do sudo mkdir -p /opt/dlami/nvme/\$d
      sudo rsync -a /var/lib/\$d/ /opt/dlami/nvme/\$d/
      sudo mount --bind /opt/dlami/nvme/\$d /var/lib/\$d; done
    sudo systemctl start containerd docker
    sudo mkdir -p /opt/dlami/nvme/work && sudo chown \$(id -u):\$(id -g) /opt/dlami/nvme/work"
  ```

  and use `/opt/dlami/nvme/work` instead of `~` below. The A10G's R595 driver runs the 26.08
  containers (CUDA 13.4) through data-center forward compatibility.

### 2. Copy the code (no push needed)

```bash
git ls-files --cached --others --exclude-standard \
  | grep -vE '^data/(tritonic\.jpeg|tritonic_logo_2\.png|images/raft/|processed_examples/)' \
  | zip -q tritonic-gate.zip -@
brev copy tritonic-gate.zip tritonic-gate:~/
brev exec tritonic-gate "rm -rf ~/tritonic && mkdir ~/tritonic && cd ~/tritonic && unzip -q ~/tritonic-gate.zip"
```

### 3. Run the gate

```bash
brev exec tritonic-gate "cd ~/tritonic && tmux new -d -s gate 'integration-tests/gpu-preprocess/run_gpu_host_gate.sh ~/gate-results > ~/gate.log 2>&1'"
brev exec tritonic-gate "cat ~/gate-results/STATUS; tail -3 ~/gate.log"   # poll
```

Start the run detached (`tmux new -d`, or `nohup … &` wrapped in `timeout`): `brev exec` stays
attached to a plain background job until it exits. The image pulls took about 3.5 minutes on
the `g5.2xlarge`.

`run_gpu_host_gate.sh` runs every step in the 26.08 containers, so the host needs only Docker,
the NVIDIA runtime, `curl` and `python3`. Each step logs to `gate-results/logs/<step>.log` and is
recorded as PASS or FAIL in `gate-results/STATUS`. A failing step does not stop the run.

| Step | Does | Evidence |
|---|---|---|
| `env` | pulls the three images, `nvidia-smi`, `env_probe.sh` | V-0 |
| `libs` | extracts the client libraries; starts the long-lived SDK work container (`--network host --ipc host --pid host`, needed for CUDA IPC) | — |
| `build` | `build-sdk` (default) and `build-cuda` (`TRITONIC_ENABLE_CUDA=ON`), both with `ctest` | V-3, V-4, V-6 |
| `model` | `yolo export model=yolo26n.pt format=onnx`; FP16 engine with `trtexec` | — |
| `stage` | `setup_model_repository.sh --engine … --trt-pre-onnx …` (DALI plugin, pipelines, backend, V13 engine) | C-4 for real |
| `triton` | starts `tritonserver`; fails unless every model is `READY` | backend loads |
| `spikes` | `run_spikes.sh` (T-2 with `trtexec`, T-3) | T-2, T-3 |
| `tensors` | DALI vs V12 server output and vs the kernel tool | V-5 |
| `gates` | detection parity (B1 vs B0; V2, V12, V13 vs B1), decode, invalid inputs, soak | V-7 … V-11, V-13 |
| `bench` | `run.py` (HTTP + gRPC, 6 fixtures, 100 iterations) and `perf.py` (concurrency 1/4/8) | V-1, V-2, V-12 |
| `sanitize` | kernel and V2 client in the SDK container; server memcheck on the host | V-14 |
| `report` | `compare.py` → `comparison.md` | V-15 |

To rerun a subset, set `STEPS`, e.g. `STEPS="triton gates"`. Other overrides: `CUDA_ARCH`, `YOLO_MODEL`,
`SOAK_SEQUENTIAL`, `SOAK_CONCURRENT`, `BENCH_ITERATIONS`, `LABEL`.

### 4. Fetch the results and delete the instance

```bash
brev copy tritonic-gate:~/gate-results.tar.gz ./
brev delete tritonic-gate        # or `brev stop` to keep the pulled images for a rerun
```

Then write `recommendation.md` into the results directory, rerun
`python3 benchmarks/yolo26-det/compare.py <results>`, and record the evidence in
[validation.md](validation.md).

### Any other Docker GPU host

Steps 2–4 without `brev`: copy the bundle by any means, unzip it, and run
`integration-tests/gpu-preprocess/run_gpu_host_gate.sh`. The script assumes nothing
Brev-specific.

---

## Pitfalls seen so far

* `.gitignore`'s `build*` silently ignores new `build_*.sh` scripts; each one needs an exception
  (as for `build_plugin.sh`, `build_backend.sh`, `build_trt_pre_engine.sh`).
* Triton 26.08 rejects client shared-memory registration ("Client shared memory is disabled") unless the server runs with `--allow-client-shm=true`; the driver, `sanitize.sh` and the deploy README pass it.
* A DALI-backend model fails when a request names only some of its outputs ("Number of outputs expected by the requests (1) does not match ... (2)"); request all outputs or none.
* TensorRT 11.2 accepts UINT8 tensors only as `Cast` inputs: cast before any layout change.
* TensorRT 11 builds strongly typed networks only: `trtexec --fp16` is gone ("Unknown option") and
  `--stronglyTyped` is a no-op. FP16 engines come from FP16-typed ONNX with FP32 I/O
  (`deploy/object_detection/yolo26/ensemble/trt_pre/to_fp16_onnx.py`, `keep_io_types=True`). For V13
  the YOLO graph is converted before the letterbox is prepended, so the letterbox stays FP32.
  **However**, on the YOLO26 `end2end=True` export this conversion leaves a `Concat` with mixed
  Half/Float inputs, which the strongly typed parser rejects. The gate therefore builds FP32
  engines (`PRECISION=fp32`, the driver default) for every variant; FP16 is a tracked follow-up.
* Ultralytics 8.4 (8.4.174 on 2026-10-09) exports YOLO26 with the raw `[1,84,8400]` head unless
  `end2end=True` is passed; the repository contract is the NMS-free `output0 [1,300,6]`.
* `pip install ultralytics` in the TensorRT image pulls GUI OpenCV, which fails on missing
  `libxcb.so.1`; replace it with `opencv-python-headless`.
* `pkill -f <pattern>` run through `brev exec` also matches the remote shell's own command line
  (which contains the pattern) and kills the session; use `tmux kill-session` instead.
* When a TensorRT Python-API harness uploads with CuPy, do the upload and `execute_async_v3`
  on the same stream; separate non-blocking streams race (order-dependent wrong outputs).
* The Triton SDK image ships its own gtest/gmock headers and abseil; the CMake changes in commit
  `c7d86b5` are needed for builds against 26.08.
* In the SDK image `LIBRARY_PATH` has an empty entry, so a bare `g++` run from the repository
  root reads `specs/` as a GCC specs file. CMake builds are unaffected.
