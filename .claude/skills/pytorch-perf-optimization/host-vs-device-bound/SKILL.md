---
name: host-vs-device-bound
description: Device-agnostic classification of whether a PyTorch workload is host-bound or device-bound. Uses torch.profiler to get kernel names and durations, compares with steady-state wall time.
---

# Host-vs-Device Bound Analysis

**Goal:** Decide whether the bottleneck is on the host or on the device, and identify the dominant kernel(s).

## REQUIRED INPUTS

| Input | Source | Notes |
|-------|--------|-------|
| `$RUN_DIR/01_kernel_profiler_setup.json` | Step 1 | `run_dir`, `vendor_tool` |
| Op config | User / orchestrator | `op_name`, `shapes`, `dtype`, `device` |
| Benchmark script or info to create one | User / orchestrator | Reproducible way to invoke the op |

Read and parse `01_kernel_profiler_setup.json` first.

## Execution context

| Action | Where |
|--------|-------|
| Read `01_kernel_profiler_setup.json` | REMOTE |
| Write benchmark script to `$RUN_DIR` | REMOTE |
| Run benchmark (bench mode, for t_op) | REMOTE |
| Run benchmark (profile mode, torch.profiler, for t_dev) | REMOTE |
| Grep kernel source in PyTorch repo | LOCAL |
| Write step JSON/log to `$RUN_DIR` | REMOTE |
| Verification | REMOTE |

## Mental model

```
U = t_dev / t_op
```

- `t_op` -- steady-state per-call wall time (untraced loop, sync only at boundaries).
- `t_dev` -- total device kernel time of one call, from `torch.profiler`.

| U | Classification |
|---|----------------|
| `U >= 0.9` | Device Bound |
| `U <= 0.5` | Host Bound |
| `0.5 < U < 0.9` | Mixed |

## Procedure

### 1. Create the benchmark script

Write a benchmark script to `$RUN_DIR/bench_<op_name>.py` with three modes:

- `bench` mode: tight untraced loop for t_op measurement
- `profile` mode: uses `torch.profiler` to get kernel names and device time
- `warmup_profile` mode (for vendor profiler in later steps): warmup + N bare calls

The script must have three modes (`bench`, `profile`, `warmup_profile`):

```python
import torch, time, statistics, sys, json

# === Op setup (adapt to the specific op) ===
device = "<device>"
# ... allocate inputs, create op ...
def call():
    pass  # invoke the op

mode = sys.argv[1] if len(sys.argv) > 1 else "bench"

if mode == "bench":
    # Choose N so total time per trial is ~1 second (e.g., N=100 for a 10ms kernel)
    n = 100
    warmup = 10
    trials = 5

    for _ in range(warmup):
        call()
    torch.accelerator.synchronize()

    vals = []
    for _ in range(trials):
        torch.accelerator.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            call()
        torch.accelerator.synchronize()
        vals.append((time.perf_counter() - t0) / n)

    t_op = statistics.median(vals)
    print(f"median_t_op_us={t_op*1e6:.3f}")
    print(f"stdev_us={statistics.stdev(vals)*1e6:.3f}")
    print(f"n={n} trials={trials}")

elif mode == "profile":
    # Warmup
    for _ in range(10):
        call()
    torch.accelerator.synchronize()

    # Select device activity based on accelerator type
    activities = [torch.profiler.ProfilerActivity.CPU]
    if hasattr(torch.profiler.ProfilerActivity, device.upper()):
        activities.append(getattr(torch.profiler.ProfilerActivity, device.upper()))

    with torch.profiler.profile(activities=activities, record_shapes=True) as prof:
        for _ in range(20):
            call()
        torch.accelerator.synchronize()

    # Print kernel table sorted by device time
    print(prof.key_averages().table(sort_by="self_device_time_total", row_limit=10))

    # Export detailed JSON for parsing
    events = [{"name": e.key, "self_device_time_us": e.self_device_time_total,
               "count": e.count, "avg_device_time_us": e.self_device_time_total / max(e.count, 1)}
              for e in prof.key_averages() if e.self_device_time_total > 0]
    events.sort(key=lambda x: x["self_device_time_us"], reverse=True)
    print("===PROFILE_JSON===")
    print(json.dumps(events, indent=2))

elif mode == "warmup_profile":
    # Bare calls for vendor profiler in later steps
    for _ in range(5):
        call()
    torch.accelerator.synchronize()
    for _ in range(20):
        call()
    torch.accelerator.synchronize()
    print("WARMUP_PROFILE_DONE")
```

Rules for the benchmark:
- Allocate inputs once, outside the timed loop. Use random data.
- Use `torch.no_grad()` unless the backward op is the target.
- No `.item()`, `print()`, or host readback inside the loop.
- Use `torch.accelerator.synchronize()` for device sync.
- Tune `n` so each trial is ~1 second. For a 10ms kernel, n=100 is enough.

### 2. Measure t_op (untraced)

Run the benchmark in `bench` mode:

```bash
python $RUN_DIR/bench_<op>.py bench
```

Record `median_t_op_us` from the output.

### 3. Profile to get t_dev and kernel names

Run in `profile` mode to use `torch.profiler`:

```bash
python $RUN_DIR/bench_<op>.py profile > $RUN_DIR/torch_profiler_output.log 2>&1
```

Parse the output:
1. Find the `===PROFILE_JSON===` marker.
2. Parse the JSON array of kernel events.
3. The top kernel by `self_device_time_us` is the dominant kernel.
4. Compute `t_dev` = dominant kernel's `avg_device_time_us` (or sum if multiple kernels per op call).
5. Skip warmup: torch.profiler already handles this via the context manager scope.

### 4. Compute U and classify

```python
U = t_dev / t_op
```

Apply the classification table. Record the dominant kernel name.

### 5. Search for kernel source

Grep the PyTorch source tree for the base kernel name (strip template parameters and mangling):

```bash
grep -rn "KernelBaseName" aten/src/ATen/native/
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
  "dominant_kernel_name": "<full kernel name from torch.profiler>",
  "kernel_source_file": "<file:line or null>",
  "run_dir": "<$RUN_DIR>"
}
```

All fields are mandatory.

### `$RUN_DIR/02_host_vs_device_bound.log`

Human-readable summary with all measured values.

### `$RUN_DIR/torch_profiler_output.log`

Raw torch.profiler output (the full stdout from profile mode).

### Benchmark script

The benchmark script at `$RUN_DIR/bench_<op>.py`.

## VERIFICATION

**Run all verification commands via SSH on the target machine (they access `$RUN_DIR` which is remote). Write any output files locally to `/tmp/opencode/` first, then SCP to `$RUN_DIR`.**

```bash
test -f $RUN_DIR/02_host_vs_device_bound.json && echo "JSON OK" || echo "JSON MISSING"
test -f $RUN_DIR/02_host_vs_device_bound.log && echo "LOG OK" || echo "LOG MISSING"
test -f $RUN_DIR/bench_*.py && echo "BENCH OK" || echo "BENCH MISSING"
test -f $RUN_DIR/torch_profiler_output.log && echo "PROFILER LOG OK" || echo "PROFILER LOG MISSING"
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

## For vendor-specific details

See the vendor-specific sub-skill (e.g., `xpu/SKILL.md`) for additional kernel property extraction using the vendor profiler.
