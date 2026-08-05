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

Run the vendor profiler with each required metric group. **Save raw output to files in $RUN_DIR first, then parse from those files.**

The exact command syntax depends on the vendor tool (see the vendor sub-skill for commands). The generic pattern is:

```bash
<vendor_profiler_command> python $RUN_DIR/bench_<op>.py warmup_profile > $RUN_DIR/<group_name>_raw.log 2>&1
```

Collect these categories of counters (vendor-specific names vary):

| Category | What it measures | Example metrics |
|----------|-----------------|-----------------|
| Timing + memory traffic | Kernel duration, DRAM bytes read/written, cache stats | Vendor-specific |
| Per-pipe instructions | Dynamic instruction/slot counts per execution pipe | Vendor-specific |
| Stall breakdown | Stall reasons (memory dependency, pipe conflict, etc.) | Vendor-specific |
| Stall sampling (optional) | Per-instruction-address stall samples | Vendor-specific |

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
    "<group_1>": "$RUN_DIR/<group_1>_raw.log",
    "<group_2>": "$RUN_DIR/<group_2>_raw.log"
  },
  "parsed_counter_files": {
    "<group_1>": "$RUN_DIR/<group_1>_parsed.json",
    "<group_2>": "$RUN_DIR/<group_2>_parsed.json"
  },
  "median_gpu_time_ns": <float>,
  "summary": {
    "gpu_time_ns": <float>,
    "dram_read_bytes": <int>,
    "dram_write_bytes": <int>
  },
  "run_dir": "<$RUN_DIR>"
}
```

The `raw_log_files` and `parsed_counter_files` keys are vendor-specific group names (e.g., `ComputeBasic` on XPU, or `memory_throughput` on CUDA). The `summary` uses generic field names that all vendors must normalize to.

### `$RUN_DIR/03_kernel_profiler_parser.log`

Human-readable summary of parsed data.

### Raw log files

All raw profiler logs saved in `$RUN_DIR/` (named by metric group).

### Parsed JSON files

Per-group parsed data in `$RUN_DIR/` (named by metric group).

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

## Vendor-specific details

See the vendor sub-skill (e.g., `xpu/SKILL.md`) for vendor-specific metric groups and log parsing techniques.
