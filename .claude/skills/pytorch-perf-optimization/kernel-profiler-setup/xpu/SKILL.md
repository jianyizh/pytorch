---
name: kernel-profiler-setup-xpu
description: XPU-specific implementation of kernel-profiler-setup using Intel unitrace. Check, build, and configure unitrace for collecting Intel XPU hardware counters and traces.
---

# Intel unitrace Setup

**Goal:** Ensure `unitrace` is available and ready to profile Intel GPU workloads.

## XPU-specific procedure

### 1. Source oneAPI environment

```bash
source /opt/intel/oneapi/setvars.sh
```

Required environment variables after sourcing: `CMPLR_ROOT`, `ONEAPI_ROOT`.

### 2. Check if unitrace is already available

```bash
which unitrace 2>/dev/null && unitrace --version 2>/dev/null || echo "UNITRACE_NOT_FOUND"
```

If found, skip to step 5 (verify).

### 3. Check prerequisites

```bash
which g++ 2>/dev/null || which icpx 2>/dev/null || echo "NO_CXX_COMPILER"
cmake --version 2>/dev/null || echo "NO_CMAKE"
python3 --version 2>/dev/null || echo "NO_PYTHON"
```

Required: CMake >= 3.22, C++17 compiler, Python >= 3.9.

### 4. Clone and build (only if not found)

```bash
cd /tmp
git clone https://github.com/intel/pti-gpu.git
cd pti-gpu/tools/unitrace
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_WITH_MPI=0 ..
make -j$(nproc)
export PATH=/tmp/pti-gpu/tools/unitrace/build:$PATH
```

### 5. Verify

```bash
unitrace --help | head -5
unitrace --device-list
```

### 6. Set observation permissions (if needed)

```bash
echo 0 > /proc/sys/dev/xe/observation_paranoid 2>/dev/null
```

### 7. Write output files

Collect the info via SSH, then write files locally and SCP:

```bash
# Gather info from remote
UNITRACE_PATH=$(ssh $SSH_TARGET "$ENV_SETUP && which unitrace")
UNITRACE_VERSION=$(ssh $SSH_TARGET "$ENV_SETUP && unitrace --version 2>&1 | head -1")
DEVICE_LIST=$(ssh $SSH_TARGET "$ENV_SETUP && unitrace --device-list 2>&1")
```

Write the log file locally to `/tmp/opencode/01_kernel_profiler_setup.log`, then SCP to `$RUN_DIR`.

Write the JSON file locally to `/tmp/opencode/01_kernel_profiler_setup.json` with this content:

```json
{
  "step": "01_kernel_profiler_setup",
  "profiler_available": true,
  "vendor_tool": "unitrace",
  "profiler_path": "<UNITRACE_PATH>",
  "version": "<UNITRACE_VERSION>",
  "supported_modes": ["timeline", "hw_counters", "stall_sampling"],
  "run_dir": "$RUN_DIR"
}
```

Then SCP both files to `$RUN_DIR` on the remote machine.

## Common unitrace commands (reference for downstream steps)

```bash
# Per-kernel aggregate counters
unitrace -q -i 20 -g ComputeBasic        ./bench > compute_basic.log
unitrace -q -i 20 -g VectorEngineProfile ./bench > ve_profile.log
unitrace -q -i 20 -g VectorEngineStalls  ./bench > ve_stalls.log

# Stall sampling (per IP)
unitrace --stall-sampling -i 20 ./bench > stalls.log

# Device timeline
unitrace -d ./bench
```

## Cautions

- Source `/opt/intel/oneapi/setvars.sh` before running unitrace if Level Zero/SYCL dependencies are not in the system path.
- On some systems, set `echo 0 > /proc/sys/dev/xe/observation_paranoid` to allow HW metrics.
