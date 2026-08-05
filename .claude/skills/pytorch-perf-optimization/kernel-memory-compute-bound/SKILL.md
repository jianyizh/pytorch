---
name: kernel-memory-compute-bound
description: Classify a device-bound kernel as memory-bound or compute-bound using the Roofline model (AI vs ridge point) and compute the theoretical minimum runtime.
---

# Kernel Memory-vs-Compute Bound Analysis

**Goal:** Given AI, peak FLOPS, and peak memory bandwidth, classify the kernel and compute its theoretical minimum runtime.

## REQUIRED INPUTS

| Input | Source | Notes |
|-------|--------|-------|
| `$RUN_DIR/01_kernel_profiler_setup.json` | Step 1 | run_dir |
| `$RUN_DIR/02_host_vs_device_bound.json` | Step 2 | t_op_us, t_dev_us, dominant_kernel_name |
| `$RUN_DIR/03_kernel_profiler_parser.json` | Step 3 | median_gpu_time_ns |
| `$RUN_DIR/04_kernel_arithmetic_intensity.json` | Step 4 | total_flops, total_bytes, AI, compute_path |
| Device peak numbers | Spec table or measurement | Peak FLOPS (path/dtype-matched), peak memory BW |

Read all prior JSON files.

## Execution context

| Action | Where |
|--------|-------|
| Read prior step JSON files | REMOTE |
| Query device name (`torch.accelerator.device_name()`) | REMOTE |
| Look up spec table / compute ridge point | LOCAL (arithmetic) |
| Write step JSON/log to `$RUN_DIR` | REMOTE |
| Verification | REMOTE |

## Procedure

### 1. Choose the right peaks

Two dimensions must match the kernel:

1. **Compute path**: `matrix` peak for GEMM/conv/attention, `vector` peak for element-wise/reductions/pooling.
2. **dtype**: use the peak for the actual execution dtype (FP32, FP16, BF16, INT8).

Priority for sourcing peak numbers:
1. Vendor spec / datasheet
2. On-device query (clock, compute-unit count, memory bus width)
3. Micro-benchmark for achievable peak (preferred when vendor numbers are optimistic)

### 2. Convert to base units

```python
peak_flops = peak_flops_tflops * 1e12   # FLOP/s
peak_bw    = peak_bw_gbps * 1e9         # B/s
```

### 3. Compute ridge point and classify

```python
ridge = peak_flops / peak_bw   # FLOP/Byte

if AI < ridge:
    bound_type = "Memory-Bound"
    time_theory_s = total_bytes / peak_bw
else:
    bound_type = "Compute-Bound"
    time_theory_s = total_flops / peak_flops

time_compute_ms = (total_flops / peak_flops) * 1000
time_memory_ms  = (total_bytes / peak_bw) * 1000
time_theory_ms  = max(time_compute_ms, time_memory_ms)

near_ridge = 0.5 <= (AI / ridge) <= 2.0
```

## REQUIRED OUTPUTS

### `$RUN_DIR/05_kernel_memory_compute_bound.json`

```json
{
  "step": "05_kernel_memory_compute_bound",
  "device_name": "<device>",
  "compute_path": "<matrix|vector>",
  "dtype": "<dtype>",
  "peak_flops_tflops": <float>,
  "peak_bw_gbps": <float>,
  "peak_source": "<vendor spec|query|measurement>",
  "ridge_point": <float>,
  "AI": <float>,
  "AI_over_ridge": <float>,
  "bound_type": "<Memory-Bound|Compute-Bound>",
  "near_ridge": <bool>,
  "time_compute_ms": <float>,
  "time_memory_ms": <float>,
  "time_theory_ms": <float>,
  "measured_time_ms": <float>,
  "measured_over_theory": <float>,
  "run_dir": "<$RUN_DIR>"
}
```

### `$RUN_DIR/05_kernel_memory_compute_bound.log`

Human-readable Roofline report.

## VERIFICATION

**Run all verification commands via SSH on the target machine (they access `$RUN_DIR` which is remote). Write any output files locally to `/tmp/opencode/` first, then SCP to `$RUN_DIR`.**

```bash
test -f $RUN_DIR/05_kernel_memory_compute_bound.json && echo "JSON OK" || echo "JSON MISSING"
python3 -c "
import json
d = json.load(open('$RUN_DIR/05_kernel_memory_compute_bound.json'))
required = ['bound_type', 'ridge_point', 'time_theory_ms', 'peak_flops_tflops', 'peak_bw_gbps']
missing = [k for k in required if k not in d]
assert not missing, f'Missing fields: {missing}'
assert d['bound_type'] in ('Memory-Bound', 'Compute-Bound')
print(f'VERIFICATION PASSED: {d[\"bound_type\"]}, theory={d[\"time_theory_ms\"]:.3f}ms')
"
```

## Vendor-specific details

See the vendor sub-skill (e.g., `xpu/SKILL.md`) for device-specific peak number tables and measurement methods.
