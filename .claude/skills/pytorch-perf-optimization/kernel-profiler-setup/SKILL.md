---
name: kernel-profiler-setup
description: Device-agnostic setup for kernel-level GPU profilers. Ensures the vendor profiler is available and ready to collect hardware counters or kernel timelines.
---

# Kernel Profiler Setup

**Goal:** Make a suitable kernel profiler available and ready to collect GPU kernel metrics and timelines.

## REQUIRED INPUTS

| Input | Source | Notes |
|-------|--------|-------|
| `RUN_DIR` | Orchestrator (created before this step) | Shared directory for all step outputs |
| `target_device` | User config | Accelerator vendor: `xpu`, `cuda`, `rocm` |
| Remote access info | User config | SSH credentials, env setup commands if device is remote |

No prior step JSON files are needed -- this is Step 1.

## Execution context

All actions in this step run on the **target device** (REMOTE if applicable). Use SSH for every command. Write files locally first, then SCP to `$RUN_DIR`.

## Procedure

### 1. Identify the vendor tool

| Accelerator | Profiler | What it provides |
|-------------|----------|------------------|
| Intel XPU | `unitrace` (PTI-GPU) | HW counters, timelines, stall sampling |
| NVIDIA GPU | Nsight Systems / Compute | NVTX timelines, SMSP counters, roofline |
| AMD GPU | `rocprof` / Omniperf | HW counters, kernel dispatch tracing |

### 2. Check availability

```bash
which <profiler-binary> || echo NOT_FOUND
<profiler-binary> --version
```

### 3. Check prerequisites

Required: C++/SYCL/CUDA/ROCm runtime as appropriate. Many profilers need root/admin or relaxed observation mode.

### 4. Build or install if missing

Use official packages or source.

### 5. Verify with smoke test

Run a help command and a device-list to confirm the profiler works on the target device. Do not proceed until the smoke test passes.

## REQUIRED OUTPUTS

Write these files to `$RUN_DIR` before returning:

### `$RUN_DIR/01_kernel_profiler_setup.json`

```json
{
  "step": "01_kernel_profiler_setup",
  "profiler_available": true,
  "vendor_tool": "<unitrace|ncu|rocprof>",
  "profiler_path": "<absolute path to binary>",
  "version": "<version string>",
  "supported_modes": ["timeline", "hw_counters", "stall_sampling"],
  "run_dir": "<$RUN_DIR>"
}
```

All fields are mandatory. `profiler_available` must be `true` to proceed.

### `$RUN_DIR/01_kernel_profiler_setup.log`

Human-readable setup verification including:
- Timestamp
- Vendor tool name and version
- Device list output
- Smoke test result

## VERIFICATION

**Run all verification commands via SSH on the target machine (they access `$RUN_DIR` which is remote). Write any output files locally to `/tmp/opencode/` first, then SCP to `$RUN_DIR`.**

Before returning, run these commands and confirm they succeed:

```bash
test -f $RUN_DIR/01_kernel_profiler_setup.json && echo "JSON OK" || echo "JSON MISSING"
test -f $RUN_DIR/01_kernel_profiler_setup.log && echo "LOG OK" || echo "LOG MISSING"
python3 -c "
import json, sys
d = json.load(open('$RUN_DIR/01_kernel_profiler_setup.json'))
required = ['profiler_available', 'vendor_tool', 'profiler_path', 'version', 'run_dir']
missing = [k for k in required if k not in d]
assert not missing, f'Missing fields: {missing}'
assert d['profiler_available'] == True, 'profiler_available must be true'
print('VERIFICATION PASSED')
"
```

## For XPU

See the XPU-specific sub-skill for Intel `unitrace` installation, oneAPI sourcing, and verification commands.
