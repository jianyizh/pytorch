---
name: host-vs-device-bound
description: Device-agnostic classification of whether a PyTorch workload is host-bound or device-bound. Uses the kernel profiler to get kernel names and durations, compares with steady-state wall time.
---

# Host-vs-Device Bound Analysis

**Goal:** Decide whether the bottleneck is on the host or on the device, and identify the dominant kernel(s).

## REQUIRED INPUTS

| Input | Source | Notes |
|-------|--------|-------|
| `$RUN_DIR/01_kernel_profiler_setup.json` | Step 1 | Must contain `profiler_available=true`, `vendor_tool`, `run_dir` |
| Op config | User / orchestrator | `op_name`, `shapes`, `dtype`, `device` |
| Benchmark script or info to create one | User / orchestrator | Reproducible way to invoke the op |

Read and parse `01_kernel_profiler_setup.json` first. Fail if `profiler_available` is not `true`.

## Execution context

| Action | Where |
|--------|-------|
| Read `01_kernel_profiler_setup.json` | REMOTE (file is on target machine) |
| Write benchmark script to `$RUN_DIR` | REMOTE |
| Run benchmark (bench mode) | REMOTE |
| Run profiler (profile mode) | REMOTE |
| Grep kernel source in PyTorch repo | LOCAL |
| Write step JSON/log to `$RUN_DIR` | REMOTE |
| Verification | REMOTE |

## Mental model

```
U = t_dev / t_op
```

- `t_op` -- steady-state per-call wall time (untraced loop, sync only at boundaries).
- `t_dev` -- total device kernel time of one call, measured by the kernel profiler.

| U | Classification |
|---|----------------|
| `U >= 0.9` | Device Bound |
| `U <= 0.5` | Host Bound |
| `0.5 < U < 0.9` | Mixed |

## Procedure

### 1. Create the benchmark script

Write a benchmark script to `$RUN_DIR/bench_<op_name>.py` with two modes:
- `profile` mode: warmup + N iterations for profiling (no timing code, just calls)
- `bench` mode: tight untraced loop with `time.perf_counter` for t_op measurement

Rules for the benchmark:
- Allocate inputs once, outside the timed loop. Use random data.
- Use `torch.no_grad()` unless the backward op is the target.
- No `.item()`, `print()`, or host readback inside the loop.
- Use `torch.accelerator.synchronize()` or device-specific sync.
- Tune `n` so each trial is 1-5 seconds. Start with n=500.
- Use 50 warmup calls, 7 trials, report median.

### 2. Measure t_op (untraced)

Run the benchmark in `bench` mode WITHOUT the profiler:

```bash
python $RUN_DIR/bench_<op>.py bench
```

Record `median_t_op_us` from the output.

### 3. Profile to get t_dev and kernel names

Run under the vendor profiler to get per-kernel device durations.

Parse the profiler output to find:
- The dominant kernel name(s) (consuming most of t_dev)
- The post-warmup median kernel duration

**Save the raw profiler output to `$RUN_DIR/` before parsing.**

### 4. Compute U and classify

```python
U = t_dev / t_op
```

Apply the classification table. Record the dominant kernel name.

### 5. Search for kernel source

Grep the PyTorch source tree for the base kernel name (strip template parameters):

```bash
grep -rn "KernelBaseName" aten/src/ATen/native/
grep -rn "KernelBaseName" third_party/torch-xpu-ops/src/
```

Record the source file and line number if found.

## REQUIRED OUTPUTS

### `$RUN_DIR/02_host_vs_device_bound.json`

```json
{
  "step": "02_host_vs_device_bound",
  "op": "<op_name>",
  "shapes": "<shape description>",
  "dtype": "<dtype>",
  "device": "<device>",
  "t_op_us": <float>,
  "t_dev_us": <float>,
  "U": <float>,
  "classification": "<Device Bound|Host Bound|Mixed>",
  "dominant_kernel_name": "<full kernel name from profiler>",
  "kernel_source_file": "<file:line or null>",
  "kernel_properties": {
    "compiled": "<AOT|JIT>",
    "simd_width": <int>,
    "num_args": <int>
  },
  "run_dir": "<$RUN_DIR>"
}
```

All fields are mandatory.

### `$RUN_DIR/02_host_vs_device_bound.log`

Human-readable summary with all measured values.

### Raw profiler output

Save the raw profiler log to `$RUN_DIR/` (e.g., `unitrace_timeline.log`).

### Benchmark script

The benchmark script should already be at `$RUN_DIR/bench_<op>.py`.

## VERIFICATION

**Run all verification commands via SSH on the target machine (they access `$RUN_DIR` which is remote). Write any output files locally to `/tmp/opencode/` first, then SCP to `$RUN_DIR`.**

```bash
test -f $RUN_DIR/02_host_vs_device_bound.json && echo "JSON OK" || echo "JSON MISSING"
test -f $RUN_DIR/02_host_vs_device_bound.log && echo "LOG OK" || echo "LOG MISSING"
test -f $RUN_DIR/bench_*.py && echo "BENCH OK" || echo "BENCH MISSING"
python3 -c "
import json
d = json.load(open('$RUN_DIR/02_host_vs_device_bound.json'))
required = ['t_op_us', 't_dev_us', 'U', 'classification', 'dominant_kernel_name']
missing = [k for k in required if k not in d]
assert not missing, f'Missing fields: {missing}'
assert d['classification'] in ('Device Bound', 'Host Bound', 'Mixed'), f'Bad classification: {d[\"classification\"]}'
print(f'VERIFICATION PASSED: U={d[\"U\"]:.3f} -> {d[\"classification\"]}')
"
```

## Common causes and levers

| Classification | Likely cause | Lever |
|----------------|--------------|-------|
| Host-bound | Python/ATen overhead, sync | Fuse ops, `torch.compile`, remove `.item()`/sync |
| Device-bound | Kernel execution dominates | Proceed to Step 3 (profiler parser) |
| Mixed | Partial overlap | Report both; try larger shapes or fusion |

## Pitfalls

- Synchronizing after every call treats the result as latency, not throughput.
- Timing a single call instead of N back-to-back calls.
- Including warmup calls in the timed window.
- Taking t_op from the profiled run (profiler overhead inflates t_op).

## For XPU

See the XPU-specific sub-skill for `unitrace -d` commands and log parsing.
