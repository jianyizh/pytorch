---
name: kernel-profiler-parser-xpu
description: XPU-specific implementation of kernel-profiler-parser for Intel unitrace. Collects and parses unitrace hardware-counter CSV logs using the bundled parse_hw_metrics.py script.
---

# Unitrace Hardware-Counter Metrics Collection and Parsing

**Goal:** Collect raw unitrace metric logs for all required groups, then parse them using the bundled `parse_hw_metrics.py` script.

## Bundled script

This skill includes a parser script at:

```
.claude/skills/pytorch-perf-optimization/kernel-profiler-parser/xpu/scripts/parse_hw_metrics.py
```

**You MUST copy this script to the remote `$RUN_DIR` before parsing.** It handles the unitrace CSV format where kernel names contain commas.

### Setup: copy script to remote

```bash
# LOCAL: the script is in the skill directory relative to the repo root
SCRIPT_PATH=".claude/skills/pytorch-perf-optimization/kernel-profiler-parser/xpu/scripts/parse_hw_metrics.py"

# SCP to remote $RUN_DIR
scp $SCRIPT_PATH $SSH_TARGET:$RUN_DIR/parse_hw_metrics.py
```

## Step 1: Collect raw counter logs

Run each metric group via SSH and save raw output to `$RUN_DIR`:

```bash
ssh $SSH_TARGET "$ENV_SETUP && unitrace -q -i 20 -g ComputeBasic python $RUN_DIR/bench_<op>.py warmup_profile > $RUN_DIR/compute_basic_raw.log 2>&1"
ssh $SSH_TARGET "$ENV_SETUP && unitrace -q -i 20 -g VectorEngineProfile python $RUN_DIR/bench_<op>.py warmup_profile > $RUN_DIR/ve_profile_raw.log 2>&1"
ssh $SSH_TARGET "$ENV_SETUP && unitrace -q -i 20 -g VectorEngineStalls python $RUN_DIR/bench_<op>.py warmup_profile > $RUN_DIR/ve_stalls_raw.log 2>&1"
ssh $SSH_TARGET "$ENV_SETUP && unitrace --stall-sampling -i 20 python $RUN_DIR/bench_<op>.py warmup_profile > $RUN_DIR/stall_sampling_raw.log 2>&1"
```

**Critical: save to files FIRST, then parse from files.**

## Step 2: Parse each log using the script

Run the script on the remote machine to parse each raw log:

```bash
# Parse ComputeBasic (memory/timing counters)
ssh $SSH_TARGET "python $RUN_DIR/parse_hw_metrics.py $RUN_DIR/compute_basic_raw.log --kernel '<dominant_kernel_substring>' --summary --output $RUN_DIR/compute_basic_parsed.json"

# Parse VectorEngineProfile (per-pipe instruction counters)
ssh $SSH_TARGET "python $RUN_DIR/parse_hw_metrics.py $RUN_DIR/ve_profile_raw.log --kernel '<dominant_kernel_substring>' --summary --output $RUN_DIR/ve_profile_parsed.json"

# Parse VectorEngineStalls (stall breakdown)
ssh $SSH_TARGET "python $RUN_DIR/parse_hw_metrics.py $RUN_DIR/ve_stalls_raw.log --kernel '<dominant_kernel_substring>' --summary --output $RUN_DIR/ve_stalls_parsed.json"
```

The `--summary` flag computes post-warmup medians. The `--kernel` flag filters to the dominant kernel.

## Step 3: Read parsed results and build the step JSON

Read the parsed JSON files from remote and extract the normalized summary fields:

```python
# From compute_basic_parsed.json:
gpu_time_ns = parsed["GpuTime[ns]"]
dram_read_bytes = parsed["GPU_MEMORY_BYTE_READ[bytes]"]
dram_write_bytes = parsed["GPU_MEMORY_BYTE_WRITE[bytes]"]
```

Write the step JSON (`03_kernel_profiler_parser.json`) with these normalized fields in the `summary` dict.

## Metric groups reference

| Group | Flag | Key counters |
|-------|------|-------------|
| ComputeBasic | `-g ComputeBasic` | `GpuTime[ns]`, `GPU_MEMORY_BYTE_READ[bytes]`, `GPU_MEMORY_BYTE_WRITE[bytes]`, `LOAD_STORE_CACHE_*`, `XVE_ACTIVE[%]`, `XVE_STALL[%]` |
| VectorEngineProfile | `-g VectorEngineProfile` | `XVE_INST_EXECUTED_ALU0_ALL`, `ALU1_ALL`, `SEND_ALL`, `MATH`, `FP16`, `FP32`, `INT32`, `INT64`, `BITCONV`, `CONTROL_ALL` |
| VectorEngineStalls | `-g VectorEngineStalls` | `XVE_STALL_ALUWR[%]`, `SBID[%]`, `PIPESTALL[%]`, `INSTFETCH[%]`, `CONTROL[%]`, `BARRIER[%]`, `SENDWR[%]` |
| EuStallSampling | `--stall-sampling` | Per-IP stall samples |

## Counter name normalization (XPU -> generic)

When writing the Step 3 JSON `summary`, map XPU counter names to generic fields:

| XPU counter | Generic field in JSON |
|-------------|----------------------|
| `GpuTime[ns]` | `gpu_time_ns` |
| `GPU_MEMORY_BYTE_READ[bytes]` | `dram_read_bytes` |
| `GPU_MEMORY_BYTE_WRITE[bytes]` | `dram_write_bytes` |

## The CSV parsing problem (why the script is needed)

Unitrace output begins with application stdout, then a section marker and header:

```
=== Device #0 Metrics ===
Kernel,GlobalInstanceId,SubDeviceId,GpuTime[ns],...
"kernel_name<float, 3>[...]",1,0,12345,...
```

Kernel names contain commas, so naive `split(',')` misaligns columns. The script handles this by parsing right-to-left: all columns except the first are numeric, so it splits from the right.

## Script CLI reference

```bash
python parse_hw_metrics.py <log_file> [options]

Options:
  --kernel <substring>   Filter to kernels matching this substring
  --skip-warmup <N>      Skip first N instances of each kernel (default: 1)
  --format <json|csv>    Output format (default: json)
  --summary              Print median summary for the dominant kernel
  --output <path>        Write output to file (default: stdout)
```
