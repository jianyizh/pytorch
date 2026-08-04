---
name: kernel-profiler-parser
description: Device-agnostic guidelines for parsing kernel-profiler output and collecting hardware counter logs. Collects raw metric logs and parses them into structured data.
---

# Kernel Profiler Output Parsing

**Goal:** Collect hardware counter logs from the vendor profiler, then parse them into clean, analysis-ready structured data. Save both raw logs and parsed results to $RUN_DIR.

## REQUIRED INPUTS

| Input | Source | Notes |
|-------|--------|-------|
| `$RUN_DIR/01_kernel_profiler_setup.json` | Step 1 | `vendor_tool`, `profiler_path` |
| `$RUN_DIR/02_host_vs_device_bound.json` | Step 2 | `dominant_kernel_name` for filtering |
| Benchmark script | Step 2 | `$RUN_DIR/bench_<op>.py` with `profile` mode |

Read and parse both JSON files first.

## Execution context

| Action | Where |
|--------|-------|
| Read prior step JSON files | REMOTE |
| Run profiler to collect metric logs | REMOTE |
| Save raw logs to `$RUN_DIR` | REMOTE |
| Parse raw logs | REMOTE (parse from saved files, not stdout) |
| Write parsed JSON and step JSON/log | REMOTE |
| Verification | REMOTE |

## Procedure

### 1. Collect raw counter logs

Run the profiler with each required metric group. **Save raw output to files in $RUN_DIR first, then parse from those files.**

For each metric group:
```bash
<profiler> <flags> -g <GroupName> python $RUN_DIR/bench_<op>.py profile > $RUN_DIR/<group_name>_raw.log 2>&1
```

Typical groups to collect:
- Basic compute/memory counters (timing, DRAM bytes, cache)
- Per-pipe instruction counters (ALU0, ALU1, SEND, etc.)
- Stall breakdown counters
- Stall sampling (per-IP samples)

### 2. Parse each log

Use a proper parser, not ad-hoc regex:
- Handle quoted fields (kernel names contain commas, angle brackets, spaces)
- Locate the data section (skip app stdout, find the metrics header)
- Parse right-to-left for CSV where kernel names contain commas

### 3. Filter and aggregate

- Filter to the dominant kernel from Step 2.
- Skip the first instance (warmup).
- Compute median of post-warmup instances for each counter.

### 4. Save parsed results

Write parsed data as JSON files in `$RUN_DIR/`.

## REQUIRED OUTPUTS

### `$RUN_DIR/03_kernel_profiler_parser.json`

```json
{
  "step": "03_kernel_profiler_parser",
  "dominant_kernel_name": "<kernel_name>",
  "raw_log_files": {
    "ComputeBasic": "$RUN_DIR/compute_basic_raw.log",
    "VectorEngineProfile": "$RUN_DIR/ve_profile_raw.log",
    "VectorEngineStalls": "$RUN_DIR/ve_stalls_raw.log"
  },
  "parsed_counter_files": {
    "ComputeBasic": "$RUN_DIR/compute_basic_parsed.json",
    "VectorEngineProfile": "$RUN_DIR/ve_profile_parsed.json",
    "VectorEngineStalls": "$RUN_DIR/ve_stalls_parsed.json"
  },
  "median_gpu_time_ns": <float>,
  "summary": {
    "GpuTime_ns": <float>,
    "GPU_MEMORY_BYTE_READ": <int>,
    "GPU_MEMORY_BYTE_WRITE": <int>
  },
  "run_dir": "<$RUN_DIR>"
}
```

### `$RUN_DIR/03_kernel_profiler_parser.log`

Human-readable summary of parsed data.

### Raw log files

All raw profiler logs saved in `$RUN_DIR/` (e.g., `compute_basic_raw.log`).

### Parsed JSON files

Per-group parsed data in `$RUN_DIR/` (e.g., `compute_basic_parsed.json`).

## VERIFICATION

**Run all verification commands via SSH on the target machine (they access `$RUN_DIR` which is remote). Write any output files locally to `/tmp/opencode/` first, then SCP to `$RUN_DIR`.**

```bash
test -f $RUN_DIR/03_kernel_profiler_parser.json && echo "JSON OK" || echo "JSON MISSING"
test -f $RUN_DIR/03_kernel_profiler_parser.log && echo "LOG OK" || echo "LOG MISSING"
python3 -c "
import json
d = json.load(open('$RUN_DIR/03_kernel_profiler_parser.json'))
required = ['dominant_kernel_name', 'raw_log_files', 'parsed_counter_files', 'median_gpu_time_ns']
missing = [k for k in required if k not in d]
assert not missing, f'Missing fields: {missing}'
# Verify raw logs exist
for name, path in d['raw_log_files'].items():
    assert open(path).readable(), f'Cannot read raw log: {path}'
print('VERIFICATION PASSED')
"
```

## For XPU

See the XPU-specific sub-skill for unitrace metric groups and the right-to-left CSV parsing technique.
