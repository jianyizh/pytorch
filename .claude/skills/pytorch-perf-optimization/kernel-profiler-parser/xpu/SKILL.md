---
name: kernel-profiler-parser-xpu
description: XPU-specific implementation of kernel-profiler-parser for Intel unitrace. Collects and parses unitrace hardware-counter CSV logs.
---

# Unitrace Hardware-Counter Metrics Collection and Parsing

**Goal:** Collect raw unitrace metric logs for all required groups, then parse them into structured data handling commas inside kernel names.

## Collection commands

Run each metric group and save raw output to `$RUN_DIR`:

```bash
unitrace -q -i 20 -g ComputeBasic        python $RUN_DIR/bench_<op>.py profile > $RUN_DIR/compute_basic_raw.log 2>&1
unitrace -q -i 20 -g VectorEngineProfile python $RUN_DIR/bench_<op>.py profile > $RUN_DIR/ve_profile_raw.log 2>&1
unitrace -q -i 20 -g VectorEngineStalls  python $RUN_DIR/bench_<op>.py profile > $RUN_DIR/ve_stalls_raw.log 2>&1
unitrace --stall-sampling -i 20           python $RUN_DIR/bench_<op>.py profile > $RUN_DIR/stall_sampling_raw.log 2>&1
```

**Critical: save to files FIRST, then parse from files. Never parse from stdout only.**

## Metric groups and what they provide

| Group | Flag | Key counters |
|-------|------|-------------|
| ComputeBasic | `-g ComputeBasic` | `GpuTime[ns]`, `GPU_MEMORY_BYTE_READ/WRITE`, `LOAD_STORE_CACHE_*`, `XVE_ACTIVE`, `XVE_STALL` |
| VectorEngineProfile | `-g VectorEngineProfile` | `XVE_INST_EXECUTED_ALU0_ALL`, `ALU1_ALL`, `SEND_ALL`, `MATH`, `FP16`, `FP32`, `INT32`, `INT64`, `BITCONV`, `CONTROL_ALL` |
| VectorEngineStalls | `-g VectorEngineStalls` | `XVE_STALL_ALUWR`, `SBID`, `PIPESTALL`, `INSTFETCH`, `CONTROL`, `BARRIER`, `SENDWR` |
| EuStallSampling | `--stall-sampling` | Per-IP stall samples |

## The CSV parsing problem

Unitrace output begins with application stdout, then a section marker and header:

```
=== Device #0 Metrics ===
Kernel,GlobalInstanceId,SubDeviceId,GpuTime[ns],...
"kernel_name<float, 3>[...]",1,0,12345,...
```

Kernel names contain commas, so `split(',')` misaligns columns.

## The fix: parse right-to-left

All columns except the first (Kernel) are numeric. Given N header columns:

```python
fields = row.split(',')
metric_values = fields[-(num_columns - 1):]
kernel_name = ','.join(fields[:len(fields) - (num_columns - 1)])
kernel_name = kernel_name.strip().strip('"')
```

## Inline parser

```python
import re, json

def parse_unitrace_metrics(log_text):
    lines = log_text.splitlines()
    header_line = None
    data_start = None
    for i, line in enumerate(lines):
        if re.match(r"^=== Device #\d+ Metrics ===$", line.strip()):
            for j in range(i + 1, len(lines)):
                if lines[j].strip():
                    header_line = lines[j].strip()
                    data_start = j + 1
                    break
            break
    if header_line is None:
        raise ValueError("No metrics section found")

    header_fields = [h.strip() for h in header_line.split(',')]
    num_columns = len(header_fields)

    rows = []
    for line in lines[data_start:]:
        line = line.strip()
        if not line or line.startswith('=== Device #'):
            continue
        fields = line.split(',')
        if len(fields) < num_columns:
            continue
        metric_values = fields[-(num_columns - 1):]
        kernel_name = ','.join(fields[:len(fields) - (num_columns - 1)])
        kernel_name = kernel_name.strip().strip('"')
        row = {"Kernel": kernel_name}
        for col_name, value in zip(header_fields[1:], metric_values):
            row[col_name] = value.strip()
        rows.append(row)
    return header_fields, rows
```

## Post-parse processing

1. Filter rows to the dominant kernel name (substring match).
2. Skip the first instance (GlobalInstanceId == first occurrence) as warmup.
3. Compute median of remaining instances for each numeric counter.
4. Save the median summary and all rows as JSON.

## Best practices

- Skip the first instance of each kernel as warmup.
- Median across >5 post-warmup iterations is more robust than mean.
- Strip surrounding quotes from kernel names and whitespace from column headers.
