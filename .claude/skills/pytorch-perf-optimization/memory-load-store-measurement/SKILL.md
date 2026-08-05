---
name: memory-load-store-measurement
description: Device-agnostic measurement and analysis of kernel memory traffic, bandwidth, and amplification. Validates whether a kernel is truly memory-bound by comparing measured vs projected traffic.
---

# Memory Load/Store Measurement

**Goal:** Determine whether a kernel is limited by memory bandwidth, cache traffic, or latency, and quantify traffic amplification.

## REQUIRED INPUTS

| Input | Source | Notes |
|-------|--------|-------|
| `$RUN_DIR/01_kernel_profiler_setup.json` | Step 1 | vendor_tool |
| `$RUN_DIR/02_host_vs_device_bound.json` | Step 2 | dominant_kernel_name, t_dev_us |
| `$RUN_DIR/03_kernel_profiler_parser.json` | Step 3 | raw_log_files (timing/memory group), median_gpu_time_ns |
| `$RUN_DIR/04_kernel_arithmetic_intensity.json` | Step 4 | total_bytes (projected) |
| `$RUN_DIR/05_kernel_memory_compute_bound.json` | Step 5 | bound_type, time_theory_ms, peak_bw_gbps |

Read ALL prior JSON files. Parse the timing/memory raw log for memory counters.

## Execution context

| Action | Where |
|--------|-------|
| Read prior step JSON files | REMOTE |
| Read/parse raw timing/memory log from `$RUN_DIR` | REMOTE |
| Run peak bandwidth benchmark | REMOTE |
| Compute bandwidth/amplification metrics | LOCAL (arithmetic from extracted numbers) |
| Write step JSON/log to `$RUN_DIR` | REMOTE |
| Verification | REMOTE |

## Procedure

### 1. Extract memory counters from parsed data

From the timing/memory counter log (already collected in Step 3), extract for the dominant kernel (post-warmup median):
- Measured DRAM read bytes
- Measured DRAM write bytes
- Kernel duration (ns)
- Cache access counters (if available)

The exact counter names are vendor-specific (see vendor sub-skill). Use the normalized `summary` from Step 3 JSON (`dram_read_bytes`, `dram_write_bytes`, `gpu_time_ns`).

### 2. Measure achievable peak bandwidth

Run a large in-place copy benchmark that exceeds last-level cache:

```python
import torch, time

device = torch.accelerator.current_accelerator()
total_mem = torch.accelerator.current_device_properties().total_memory
n = int(total_mem * 0.8 / 3 / 4)  # float32
a = torch.randn(n, device=device)
b = torch.randn(n, device=device)
c = torch.randn(n, device=device)

for _ in range(10):
    a.neg_(); b.neg_(); c.neg_()
torch.accelerator.synchronize()

results = []
for trial in range(5):
    torch.accelerator.synchronize()
    t0 = time.perf_counter()
    reps = 30
    for _ in range(reps):
        a.neg_(); b.neg_(); c.neg_()
    torch.accelerator.synchronize()
    t1 = time.perf_counter()
    total_bytes = (a.nbytes + b.nbytes + c.nbytes) * reps * 2
    results.append(total_bytes / (t1 - t0) / 1e9)

import statistics
peak_bw = statistics.median(results)
```

Use random data (not zeros) to avoid hardware compression distortion.

### 3. Compute bandwidth and amplification

```python
measured_read = dram_read_bytes    # from Step 3 normalized summary
measured_write = dram_write_bytes
gpu_time_s = gpu_time_ns * 1e-9

dram_read_bw  = measured_read / gpu_time_s
dram_write_bw = measured_write / gpu_time_s
dram_total_bw = (measured_read + measured_write) / gpu_time_s

total_util = dram_total_bw / (peak_bw * 1e9)

# projected bytes from Step 5 (split into read/write if known)
projected_read = total_bytes_input
projected_write = total_bytes_output

read_amplification  = measured_read / projected_read
write_amplification = measured_write / projected_write
```

### 4. Compare with memory lower bound

```python
T_mem = (projected_read + projected_write) / (peak_bw * 1e9)  # seconds
T_actual = gpu_time_s
ratio = T_actual / T_mem
```

| Ratio | Interpretation |
|-------|----------------|
| 1.0-1.2 | Near the memory lower bound |
| >1.4 | Another bottleneck; memory is NOT the primary limiter |

### 5. Classify

When BW is near peak and T_actual is near T_mem, the kernel is DRAM bandwidth-bound.

When BW is below peak, determine why from the combination of amplification and stalls:

| BW utilization | Amplification | Stalls | Classification |
|---------------|---------------|--------|---------------|
| High (>70%) | ~1.0 | Normal | **DRAM bandwidth-bound** -- memory bus is saturated |
| Medium | High (>>1.0) | Medium | **Uncoalesced / poor locality** -- strided or scattered access inflates traffic; each cache-line fetch only partially used |
| Low (<50%) | ~1.0 | High memory-dep stalls | **Insufficient memory-level parallelism** -- too few concurrent memory requests in flight to fill the bus; need more independent loads |
| Low (<50%) | ~1.0 | High ALU-dep/pipe stalls | **Poor compute-memory overlap** -- compute instructions block the pipeline, preventing the GPU from issuing memory requests while waiting; memory and compute are serialized instead of overlapped |
| Low (<50%) | High (>>1.0) | Varies | **Uncoalesced access + low MLP (memory-level parallelism)** -- scattered access pattern AND insufficient concurrency; fix coalescing first |

These are common patterns, not exhaustive. A kernel can have both poor coalescing and insufficient MLP, or may exhibit a scenario not listed here. If the measured data does not fit any row cleanly, describe the actual numbers and let the downstream analysis (Step 7 instruction measurement) disambiguate.

## REQUIRED OUTPUTS

### `$RUN_DIR/06_memory_load_store.json`

```json
{
  "step": "06_memory_load_store",
  "dominant_kernel_name": "<kernel_name>",
  "gpu_time_ns": <float>,
  "measured_read_bytes": <int>,
  "measured_write_bytes": <int>,
  "projected_read_bytes": <int>,
  "projected_write_bytes": <int>,
  "dram_read_bw_gbps": <float>,
  "dram_write_bw_gbps": <float>,
  "dram_total_bw_gbps": <float>,
  "peak_bw_gbps": <float>,
  "bw_utilization": <float>,
  "read_amplification": <float>,
  "write_amplification": <float>,
  "T_mem_ms": <float>,
  "T_actual_ms": <float>,
  "T_actual_over_T_mem": <float>,
  "memory_bound_subtype": "<DRAM BW bound|uncoalesced access|insufficient MLP|poor compute-memory overlap>",
  "run_dir": "<$RUN_DIR>"
}
```

### `$RUN_DIR/06_memory_load_store.log`

Human-readable memory traffic report.

## VERIFICATION

**Run all verification commands via SSH on the target machine (they access `$RUN_DIR` which is remote). Write any output files locally to `/tmp/opencode/` first, then SCP to `$RUN_DIR`.**

```bash
test -f $RUN_DIR/06_memory_load_store.json && echo "JSON OK" || echo "JSON MISSING"
python3 -c "
import json
d = json.load(open('$RUN_DIR/06_memory_load_store.json'))
required = ['measured_read_bytes', 'measured_write_bytes', 'dram_total_bw_gbps', 'peak_bw_gbps', 'bw_utilization', 'read_amplification', 'write_amplification', 'T_mem_ms']
missing = [k for k in required if k not in d]
assert not missing, f'Missing fields: {missing}'
print(f'VERIFICATION PASSED: BW util={d[\"bw_utilization\"]:.1%}, read_amp={d[\"read_amplification\"]:.2f}x')
"
```

## Levers

| Classification | Lever |
|---------------|-------|
| DRAM bandwidth-bound | Reduce bytes (lower precision, tiling, fuse kernels) |
| Uncoalesced / poor locality | Improve access pattern (contiguous loads, pack FP16 into d32+, tile spatially), reduce amplification |
| Insufficient MLP (memory-level parallelism) | Increase occupancy, add independent loads (vectorize / unroll), reduce dependent memory chains, prefetch |
| Poor compute-memory overlap | Reduce instruction count on the blocking pipe (see Step 7), vectorize to amortize index math, expose more independent work between dependent loads |
| Uneven read/write ratio | Eliminate temporary tensors, fuse producers/consumers |

## Vendor-specific details

See the vendor sub-skill (e.g., `xpu/SKILL.md`) for device-specific counter names, compression behavior, and stall interpretation.
