---
name: kernel-occupancy
description: Compute the theoretical occupancy of a GPU kernel from its launch-time resource usage (work-group size, subgroup/SIMD width, register usage, shared local memory). Determines whether the kernel uses enough of the device thread-context resources.
---

# Kernel Occupancy Analysis

**Goal:** Determine the **theoretical occupancy** of the dominant kernel -- the fraction of the device's available thread contexts actually resident per compute unit (an SM / streaming multiprocessor / Xe-core) during execution. Low occupancy means the kernel is not filling the hardware's thread/scheduling resources, which can cap latency hiding, MLP, and memory-compute overlap.

This step runs immediately after Step 3 (profiler parser) and BEFORE arithmetic intensity / Roofline (Step 5). Occupancy gives the scheduler-level explanation for why a kernel is slow; AI/Roofline gives the theoretical bound.

## REQUIRED INPUTS

| Input | Source | Notes |
|-------|--------|-------|
| `$RUN_DIR/01_kernel_profiler_setup.json` | Step 1 | vendor_tool, device |
| `$RUN_DIR/03_kernel_profiler_parser.json` | Step 3 | dominant_kernel_name |
| Launch parameters for the dominant kernel | Step 1 timeline log (see vendor sub-skill) | WG size, subgroup/simd width, register usage, shared-memory usage |

Read the prior JSON and the launch-parameter dump from the timeline log.

## Execution context

| Action | Where |
|--------|-------|
| Read prior step JSON files | REMOTE |
| Read the timeline log launch parameters | REMOTE |
| Compute occupancy | LOCAL (arithmetic from launch params) |
| Write step JSON/log to `$RUN_DIR` | REMOTE |
| Verification | REMOTE |

## Procedure

### 1. Collect launch parameters

From the timeline log produced in Step 1 (see the vendor sub-skill for the exact command,
e.g. a verbose timeline mode on XPU), extract for the dominant kernel:
- **Work-group (WG) size**: number of work-items launched together
- **Subgroup / SIMD size**: width of a hardware thread
- **Register usage** per thread
- **Shared Local Memory (SLM / shared memory)** used per WG
- **Number of workgroups** launched (global size / WG size)
- Whether the kernel uses **barriers**

### 2. Compute occupancy

The occupancy is the fraction of the **hardware thread slots** on a compute unit that the
kernel actually occupies. It is determined by several independent resource limits: thread
slots, shared memory, and (if barriers are used) barrier registers. Common limits, with the
generic formula the vendor sub-skill instantiates:

- `MAX_THREADS_PER_CORE`: hardware thread slots per compute unit. Each thread slot == one
  subgroup (hardware thread).
- `THREADS_PER_WG = ceil(workgroup_size / subgroup_size)`: thread slots one WG consumes.
- `limit_threads = floor(MAX_THREADS_PER_CORE / THREADS_PER_WG)`: max resident WGs by thread slots.
- `limit_smem = floor(SMEM_PER_CORE / smem_per_wg)` (`= MAX_WGS` if smem_per_wg == 0): max resident WGs by shared memory.
- `limit_wg = max_WGs_per_core`: hardware cap on resident WGs.
- `limit_barrier = MAX_BARRIER_REGISTERS` (only if kernel uses barriers): max resident WGs by barrier registers.

```
num_wg = min(limit_threads, limit_smem, limit_wg, limit_barrier, total_WGs_launched(optional))
occupancy = THREADS_PER_WG * num_wg / MAX_THREADS_PER_CORE   (0.0 to 1.0)
```

The device nearly always has multiple concurrent WGs filling the thread slots, so `occupancy = total_resident_threads / MAX_THREADS_PER_CORE`.

Note: some architectures further constrain thread slots by register pressure (e.g. a
"large register file" mode can halve thread slots, or exceeding the budget can force limited
residency). Cross-check this with Step 5 (register spill / register-file analysis).

### 3. Classify occupancy

| Occupancy | Classification | Implication |
|-----------|----------------|-------------|
| `>= 0.6` | Good | Not thread/scheduler-limited |
| `0.3 - 0.6` | Moderate | Partial; may limit MLP/latency hiding |
| `< 0.3` | Low | Likely limits latency hiding, MLP, overlap; check WG size, registers, SLM |

**Caveat:** higher occupancy does not always mean higher performance (e.g., a kernel that trades occupancy for more SLM/reuse may still be faster). Treat occupancy as a diagnostic for *why* a bound exists, not a target to maximize blindly.

## REQUIRED OUTPUTS

### `$RUN_DIR/04_kernel_occupancy.json`

```json
{
  "step": "04_kernel_occupancy",
  "dominant_kernel_name": "<kernel_name>",
  "device": "<device_name>",
  "launch_params": {
    "workgroup_size": <int>,
    "subgroup_size": <int>,
    "registers_per_thread": <int>,
    "smem_per_workgroup_bytes": <int>,
    "num_workgroups": <int>,
    "uses_barriers": <bool>
  },
  "resource_limits": {
    "max_threads_per_core": <int>,
    "threads_per_wg": <int>,
    "limit_threads": <int>,
    "limit_smem": <int>,
    "max_wgs_per_core": <int>,
    "limit_barrier": <int>
  },
  "occupancy": <float 0..1>,
  "classification": "<Good|Moderate|Low>",
  "workgroup_size_recommendation": "<string>",
  "run_dir": "<$RUN_DIR>"
}
```

### `$RUN_DIR/04_kernel_occupancy.log`

Human-readable breakdown showing each resource limit and the occupancy computation.

## VERIFICATION

**Run all verification commands via SSH on the target machine (they access `$RUN_DIR` which is remote).**

```bash
test -f $RUN_DIR/04_kernel_occupancy.json && echo "JSON OK" || echo "JSON MISSING"
python3 -c "
import json
d = json.load(open('$RUN_DIR/04_kernel_occupancy.json'))
required = ['launch_params', 'occupancy', 'classification']
missing = [k for k in required if k not in d]
assert not missing, f'Missing fields: {missing}'
assert 0.0 <= d['occupancy'] <= 1.0, f'Bad occupancy: {d[\"occupancy\"]}'
assert 'workgroup_size' in d['launch_params']
print(f'VERIFICATION PASSED: occupancy={d[\"occupancy\"]:.2f}')
"
```

## Vendor-specific details

See the vendor sub-skill (e.g., `xpu/SKILL.md`) for the exact hardware resource tables and the occupancy formula.