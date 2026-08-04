---
name: instructions-measurement
description: Device-agnostic measurement of pipeline/instruction throughput utilization. Determines whether a kernel is instruction-bound and identifies the dominant pipeline.
---

# Instructions Measurement

**Goal:** Determine whether a kernel is limited by instruction throughput rather than memory bandwidth, and identify the dominant pipeline.

## REQUIRED INPUTS

| Input | Source | Notes |
|-------|--------|-------|
| `$RUN_DIR/01_kernel_profiler_setup.json` | Step 1 | vendor_tool |
| `$RUN_DIR/02_host_vs_device_bound.json` | Step 2 | dominant_kernel_name |
| `$RUN_DIR/03_kernel_profiler_parser.json` | Step 3 | raw_log_files (VectorEngineProfile), parsed data |
| `$RUN_DIR/04_kernel_arithmetic_intensity.json` | Step 4 | total_bytes |
| `$RUN_DIR/05_kernel_memory_compute_bound.json` | Step 5 | peak_bw_gbps |
| `$RUN_DIR/06_memory_load_store.json` | Step 6 | T_mem_ms, peak_bw_gbps, measured bytes |

Read ALL prior JSON files. This step MUST run after Step 6.

## Execution context

| Action | Where |
|--------|-------|
| Read prior step JSON files | REMOTE |
| Read/parse raw VectorEngineProfile and VectorEngineStalls logs from `$RUN_DIR` | REMOTE |
| Compute per-pipe times | LOCAL (arithmetic from extracted counters) |
| Write step JSON/log to `$RUN_DIR` | REMOTE |
| Verification | REMOTE |

## Background

A kernel cannot run faster than either:

```
T_mem  = projected_bytes / peak_memory_bw
T_pipe = dynamic_operations(pipe) / peak_ops_per_second(pipe)
```

On a SIMD GPU, count **dynamic lane-operations** (slots). One SIMD32 instruction = 32 slots. Pipelines operate concurrently, so the kernel time is set by the most heavily loaded pipe:

```
T_instruction = max over all pipes (T_pipe)
T_lower       = max(T_instruction, T_mem)
```

Empirical overlap rule:
- T_instruction > 0.8 * T_mem -> instruction-bound
- T_mem > 0.8 * T_instruction -> memory-bound
- Otherwise -> transitional

## Procedure

### 1. Extract per-pipe instruction counters

From VectorEngineProfile (already collected in Step 3), extract post-warmup median for the dominant kernel:
- Per-pipe executed slot counts
- Any MATH/special-function counters (may need to be added separately)

### 2. Determine peak ops/s for each pipe

```
peak = execution_units * ops_per_cycle_per_unit * clock_frequency
```

### 3. Compute per-pipe time

```python
T_pipe = dynamic_slots_for_pipe / peak_slots_per_second
T_instruction = max(T_pipe for all pipes)
```

### 4. Compare with T_mem from Step 6

Use T_mem from the memory-load-store step. Classify per the 0.8x overlap rule.

### 5. Identify dominant pipe sub-counters

Break down the dominant pipe by sub-counters (e.g., INT32 vs INT64 vs MATH for ALU1) to understand the specific bottleneck.

## REQUIRED OUTPUTS

### `$RUN_DIR/07_instructions_measurement.json`

```json
{
  "step": "07_instructions_measurement",
  "dominant_kernel_name": "<kernel_name>",
  "device_xve_count": <int>,
  "device_freq_ghz": <float>,
  "peak_slots_per_s": <float>,
  "per_pipe": {
    "ALU0": {"slots": <int>, "T_ms": <float>},
    "ALU1": {"slots": <int>, "T_ms": <float>},
    "MATH": {"slots": <int>, "T_ms": <float>},
    "SEND": {"slots": <int>, "T_ms": <float>},
    "DPAS": {"slots": <int>, "T_ms": <float>}
  },
  "sub_counters": {
    "FP16": <int>, "FP32": <int>,
    "INT32": <int>, "INT64": <int>,
    "BITCONV": <int>, "CONTROL": <int>
  },
  "T_instruction_ms": <float>,
  "dominant_pipe": "<ALU0|ALU1|DPAS|SEND>",
  "T_mem_ms": <float>,
  "primary_bound": "<ALU0|ALU1|DPAS|SEND|memory>",
  "T_actual_ms": <float>,
  "T_actual_over_T_lower": <float>,
  "stall_breakdown": {
    "ALUWR_pct": <float>,
    "SBID_pct": <float>,
    "PIPESTALL_pct": <float>,
    "INSTFETCH_pct": <float>,
    "CONTROL_pct": <float>
  },
  "run_dir": "<$RUN_DIR>"
}
```

### `$RUN_DIR/07_instructions_measurement.log`

Human-readable per-pipe analysis report.

## VERIFICATION

**Run all verification commands via SSH on the target machine (they access `$RUN_DIR` which is remote). Write any output files locally to `/tmp/opencode/` first, then SCP to `$RUN_DIR`.**

```bash
test -f $RUN_DIR/07_instructions_measurement.json && echo "JSON OK" || echo "JSON MISSING"
python3 -c "
import json
d = json.load(open('$RUN_DIR/07_instructions_measurement.json'))
required = ['T_instruction_ms', 'dominant_pipe', 'T_mem_ms', 'primary_bound', 'per_pipe']
missing = [k for k in required if k not in d]
assert not missing, f'Missing fields: {missing}'
print(f'VERIFICATION PASSED: dominant_pipe={d[\"dominant_pipe\"]}, primary_bound={d[\"primary_bound\"]}')
"
```

## Common levers by dominant pipe

| Dominant pipe | Typical cause | Lever |
|---------------|---------------|-------|
| FP / vector | Redundant FP math | Remove redundancy, lower precision, native math |
| Integer / scalar | Index decomposition, 64-bit arithmetic | IntDivider, precompute indices, narrow pointers |
| Matrix/tensor | GEMM tile mismatch | Tune tile shape |
| Load/store / message | Memory instruction count | Revisit memory measurement |

## Cautions

- Count **dynamic** operations (slots), not static binary size.
- Do NOT sum all pipes; the slowest pipe wins.
- In the transition region, a small change can flip the bound.

## For XPU

See the XPU-specific sub-skill for XVE counter names, pipe model, and peak slot rate.
