---
name: pytorch-perf-optimization
description: Single entry point for optimizing a PyTorch kernel or operator on any accelerator. Use when the user says "Optimize this PyTorch kernel/op", asks why a kernel is slow, or asks about host vs device bound, arithmetic intensity, Roofline memory/compute bound, hardware-profiler counters, stalls, memory/instruction bottlenecks, or assembly-to-source mapping. Orchestrates a device-agnostic workflow and dispatches to vendor-specific sub-skills under pytorch-perf-optimization/.
---

# PyTorch Performance Optimization Workflow

This skill is the **single entry point** for diagnosing and optimizing a PyTorch kernel/operator. You are the **orchestrator**. You do NOT execute steps yourself. You dispatch each step to a sub-agent, verify its output, and pass accumulated state to the next step.

## Critical rules

1. **Do NOT load all sub-skills upfront.** Load each sub-skill file only when you reach that step.
2. **Do NOT execute step logic yourself.** Launch a sub-agent for each step using the Task tool.
3. **Each sub-agent gets: the sub-skill content + ALL previous step JSON results + user config.**
4. **After each sub-agent returns, verify the REQUIRED OUTPUT files exist on disk before proceeding.**
5. **Steps are strictly sequential.** Do not skip, reorder, or parallelize steps.

## User prompts that activate this workflow

- "Optimize this PyTorch kernel/op."
- "Why is my kernel slow?"
- "Is this host-bound or device-bound?"
- "Memory-bound or compute-bound?"
- "What is the arithmetic intensity / Roofline classification?"
- "How do I collect/parse hardware metrics?"
- "This kernel is limited by memory / instructions / stalls."
- "Map hot assembly back to source code."

## Required inputs from the user

Gather these before starting. Ask the user if any are missing.

| Input | Why it matters |
|-------|----------------|
| `op_name` | Operator being analyzed (e.g. `softmax`, `groupnorm`, `gemm`) |
| `shapes` | Input/output tensor shapes |
| `dtype` | Main dtype (fp16/bf16/fp32/int8) and accumulation dtype if different |
| `device` | Target accelerator (`xpu`, `cuda`, etc.) and exact SKU if known |
| `benchmark script` | Reproducible way to run the op (or enough info to generate one) |
| `remote access` | SSH credentials / environment setup commands if the device is remote |

## Remote execution protocol

When the target device is on a remote machine, the orchestrator must build these values from user-provided credentials before Step 0:

### `SSH_TARGET`

The `user@host` string for the remote machine.

### `SSH_PASSWORD`

The password (if using password auth).

### `ENV_SETUP`

The chain of environment commands to run on the remote machine before any real command:

```bash
# Example (adapt to user's actual environment):
ENV_SETUP="source ~/.bashrc && conda activate myenv && export PATH=/path/to/tools:\$PATH"
```

### Running commands remotely

To run a single-line command on the remote device:

```bash
sshpass -p '$SSH_PASSWORD' ssh -o StrictHostKeyChecking=no $SSH_TARGET "$ENV_SETUP && <command>"
```

### Writing files to remote `$RUN_DIR`

**NEVER use SSH heredoc or `cat >` over SSH to write multi-line content.** Nested quoting between SSH, bash, and file content will break.

Instead, always use this two-step pattern:

1. Write the file locally to `/tmp/opencode/` using the Write tool (or bash `cat > /tmp/opencode/...`).
2. Copy it to the remote machine using `scp`:

```bash
sshpass -p '$SSH_PASSWORD' scp -o StrictHostKeyChecking=no /tmp/opencode/<filename> $SSH_TARGET:$RUN_DIR/<filename>
```

This applies to:
- Benchmark scripts (`bench_*.py`)
- Step JSON files (`NN_*.json`)
- Step log files (`NN_*.log`)
- Any helper scripts

### Reading files from remote `$RUN_DIR`

```bash
sshpass -p '$SSH_PASSWORD' ssh -o StrictHostKeyChecking=no $SSH_TARGET "cat $RUN_DIR/<filename>"
```

### What runs where

| Action | Where | How |
|--------|-------|-----|
| Reading sub-skill SKILL.md files | LOCAL | Read tool on local repo |
| Reading PyTorch kernel source code | LOCAL | Read/Grep tool on local repo |
| Arithmetic / Roofline computation | LOCAL | Python in sub-agent or local shell |
| Writing benchmark scripts | LOCAL then SCP | Write tool -> `/tmp/opencode/` -> `scp` to remote |
| Running benchmarks / profiler | REMOTE | SSH command |
| Writing step JSON/log to `$RUN_DIR` | LOCAL then SCP | Write tool -> `/tmp/opencode/` -> `scp` to remote |
| Gate verification | REMOTE | SSH + `cat` + `python3 -c` |
| Reading step JSON for next sub-agent | REMOTE | SSH + `cat` |

### If the device is local

If the device is on the same machine, skip SSH and SCP. Write files directly to `$RUN_DIR`. Run commands directly. Sub-skills work identically either way.

## Step 0 -- Detect device and create shared run directory

### 0a. Detect the current accelerator device

Run this on the target machine to confirm the device type and name:

```bash
python -c "
import torch
dev = torch.accelerator.current_accelerator()
print(f'device_type={dev.type}')
print(f'device_name={torch.accelerator.device_name(0)}')
"
```

Record `device_type` (e.g., `xpu`, `cuda`) and `device_name` (e.g., `Intel(R) Arc(TM) B580 Graphics`). These determine which vendor-specific sub-skills to load.

### 0b. Create shared run directory

1. Create `/tmp/opencode/` locally if it doesn't exist (for staging files before SCP).
2. Create `$RUN_DIR` on the target machine:

```bash
# Local staging:
mkdir -p /tmp/opencode

# Remote RUN_DIR:
sshpass -p '$SSH_PASSWORD' ssh -o StrictHostKeyChecking=no $SSH_TARGET "$ENV_SETUP && RUN_DIR=/tmp/opencode_runs/<op_name>_\$(date +%Y%m%d_%H%M%S) && mkdir -p \$RUN_DIR && echo \$RUN_DIR"
```

Record the `RUN_DIR` path from the output. All downstream steps use this exact path.

## Step 1 -- kernel-profiler-setup

1. **Load the sub-skill:**
   - Read `.claude/skills/pytorch-perf-optimization/kernel-profiler-setup/SKILL.md`
   - If device is XPU, also read `.claude/skills/pytorch-perf-optimization/kernel-profiler-setup/xpu/SKILL.md`
2. **Launch sub-agent** with a prompt containing:
   - The sub-skill content (generic + vendor-specific)
   - `RUN_DIR` path, `SSH_TARGET`, `SSH_PASSWORD`, `ENV_SETUP`
   - Device type
3. **Gate -- verify before proceeding:**
   ```bash
   sshpass -p '$SSH_PASSWORD' ssh $SSH_TARGET "cat $RUN_DIR/01_kernel_profiler_setup.json"
   # Must contain: profiler_available=true, vendor_tool, run_dir
   ```
4. Read the JSON content and carry it forward.

## Step 2 -- host-vs-device-bound

1. **Load the sub-skill:**
   - Read `.claude/skills/pytorch-perf-optimization/host-vs-device-bound/SKILL.md`
   - If device is XPU, also read `.claude/skills/pytorch-perf-optimization/host-vs-device-bound/xpu/SKILL.md`
2. **Launch sub-agent** with:
   - The sub-skill content
   - Step 1 JSON results
   - Op config (op_name, shapes, dtype, device)
   - `RUN_DIR` path and SSH credentials (`SSH_TARGET`, `SSH_PASSWORD`, `ENV_SETUP`)
   - Benchmark script (or instructions to create one)
3. **Gate -- verify:**
   ```bash
   sshpass -p '$SSH_PASSWORD' ssh $SSH_TARGET "cat $RUN_DIR/02_host_vs_device_bound.json"
   # Must contain: t_op_us, t_dev_us, U, classification, dominant_kernel_name
   ```
4. Read the JSON. **Branch:**
   - `classification == "Host Bound"` --> skip to Step 9 with host-bound levers.
   - `classification == "Device Bound"` or `"Mixed"` --> continue to Step 3.

## Step 3 -- kernel-profiler-parser

1. **Load the sub-skill:**
   - Read `.claude/skills/pytorch-perf-optimization/kernel-profiler-parser/SKILL.md`
   - If device is XPU, also read `.claude/skills/pytorch-perf-optimization/kernel-profiler-parser/xpu/SKILL.md`
2. **Launch sub-agent** with:
   - The sub-skill content
   - Step 1 + Step 2 JSON results
   - `RUN_DIR` path and SSH credentials
   - The dominant kernel name from Step 2
3. **Gate -- verify:**
   ```bash
   sshpass -p '$SSH_PASSWORD' ssh $SSH_TARGET "cat $RUN_DIR/03_kernel_profiler_parser.json"
   # Must contain: dominant_kernel_name, parsed_counter_files (dict of group -> path)
   ```
4. Read the JSON and carry forward.

## Step 4 -- kernel-arithmetic-intensity

1. **Load the sub-skill:**
   - Read `.claude/skills/pytorch-perf-optimization/kernel-arithmetic-intensity/SKILL.md`
   - If device is XPU, also read `.claude/skills/pytorch-perf-optimization/kernel-arithmetic-intensity/xpu/SKILL.md`
2. **Launch sub-agent** with:
   - The sub-skill content
   - ALL previous step JSON results (Steps 1-3)
   - Op config (op_name, shapes, dtype)
   - `RUN_DIR` path and SSH credentials
   - Path to the local PyTorch source tree for kernel source inspection
3. **Gate -- verify:**
   ```bash
   sshpass -p '$SSH_PASSWORD' ssh $SSH_TARGET "cat $RUN_DIR/04_kernel_arithmetic_intensity.json"
   # Must contain: total_flops, total_bytes, AI, compute_path
   ```

## Step 5 -- kernel-memory-compute-bound

1. **Load the sub-skill:**
   - Read `.claude/skills/pytorch-perf-optimization/kernel-memory-compute-bound/SKILL.md`
   - If device is XPU, also read `.claude/skills/pytorch-perf-optimization/kernel-memory-compute-bound/xpu/SKILL.md`
2. **Launch sub-agent** with:
   - The sub-skill content
   - ALL previous step JSON results (Steps 1-4)
   - `RUN_DIR` path and SSH credentials
3. **Gate -- verify:**
   ```bash
   sshpass -p '$SSH_PASSWORD' ssh $SSH_TARGET "cat $RUN_DIR/05_kernel_memory_compute_bound.json"
   # Must contain: bound_type, ridge_point, time_theory_ms, time_compute_ms, time_memory_ms
   ```

## Step 6 -- memory-load-store-measurement

1. **Load the sub-skill:**
   - Read `.claude/skills/pytorch-perf-optimization/memory-load-store-measurement/SKILL.md`
   - If device is XPU, also read `.claude/skills/pytorch-perf-optimization/memory-load-store-measurement/xpu/SKILL.md`
2. **Launch sub-agent** with:
   - The sub-skill content
   - ALL previous step JSON results (Steps 1-5)
   - `RUN_DIR` path and SSH credentials
3. **Gate -- verify:**
   ```bash
   sshpass -p '$SSH_PASSWORD' ssh $SSH_TARGET "cat $RUN_DIR/06_memory_load_store.json"
   # Must contain: measured_read_bytes, measured_write_bytes, dram_total_bw_gbps,
   #   bw_utilization, read_amplification, write_amplification, T_mem_ms, peak_bw_gbps
   ```

## Step 7 -- instructions-measurement

**This step MUST run after Step 6** (it needs T_mem from Step 6 for comparison).

1. **Load the sub-skill:**
   - Read `.claude/skills/pytorch-perf-optimization/instructions-measurement/SKILL.md`
   - If device is XPU, also read `.claude/skills/pytorch-perf-optimization/instructions-measurement/xpu/SKILL.md`
2. **Launch sub-agent** with:
   - The sub-skill content
   - ALL previous step JSON results (Steps 1-6)
   - `RUN_DIR` path and SSH credentials
3. **Gate -- verify:**
   ```bash
   sshpass -p '$SSH_PASSWORD' ssh $SSH_TARGET "cat $RUN_DIR/07_instructions_measurement.json"
   # Must contain: T_instruction_ms, dominant_pipe, per_pipe, T_mem_ms, primary_bound
   ```

## Step 8 -- asm-source-mapping

Always run this step. It maps hot instructions back to source code using the dominant pipe and stall info from Steps 6/7.

1. **Load the sub-skill:**
   - Read `.claude/skills/pytorch-perf-optimization/asm-source-mapping/SKILL.md`
   - If device is XPU, also read `.claude/skills/pytorch-perf-optimization/asm-source-mapping/xpu/SKILL.md`
2. **Launch sub-agent** with:
   - The sub-skill content
   - ALL previous step JSON results (Steps 1-7)
   - `RUN_DIR` path and SSH credentials
   - Path to the local PyTorch source tree
   - The dominant pipe and stall info from Steps 6/7
3. **Gate -- verify:**
   ```bash
   sshpass -p '$SSH_PASSWORD' ssh $SSH_TARGET "cat $RUN_DIR/08_asm_source_mapping.json"
   # Must contain: scenario, hot_source_locations, parallelization_analysis
   ```

## Step 9 -- Final report

This step does NOT use a sub-skill file. You (the orchestrator) produce the final report directly.

1. Read ALL step JSON files from `$RUN_DIR` (01, 02, 03, 04, 05, 06, 07, 08).
2. Produce the report using the output template below.
3. Save the report to `$RUN_DIR/09_final_report.md`.

## Sub-agent prompt template

When launching a sub-agent for Step N, construct the prompt as:

```
You are executing Step N of the PyTorch Performance Optimization workflow.

## Your task
<paste the sub-skill SKILL.md content here>
<if vendor-specific, also paste the xpu/SKILL.md content>

## Previous step results
<paste the JSON content of ALL previous steps: 01, 02, ..., N-1>

## User configuration
- op_name: ...
- shapes: ...
- dtype: ...
- device: ...
- RUN_DIR: <exact path on target machine>
- Source tree: <local path to PyTorch repo>

## Remote execution protocol

The target device is on a remote machine.

SSH_TARGET: <user@host>
SSH_PASSWORD: <password>
ENV_SETUP: <environment setup command chain>

### To run a command on the remote device

Use the Bash tool:
```
sshpass -p '<SSH_PASSWORD>' ssh -o StrictHostKeyChecking=no <SSH_TARGET> "<ENV_SETUP> && <your_command>"
```

### To write a file to remote $RUN_DIR

NEVER use SSH heredoc or cat-over-SSH for multi-line content. Instead:
1. Use the Write tool to write the file locally to /tmp/opencode/<filename>
2. Use the Bash tool to scp it:
```
sshpass -p '<SSH_PASSWORD>' scp -o StrictHostKeyChecking=no /tmp/opencode/<filename> <SSH_TARGET>:$RUN_DIR/<filename>
```

### To read a file from remote $RUN_DIR

Use the Bash tool:
```
sshpass -p '<SSH_PASSWORD>' ssh -o StrictHostKeyChecking=no <SSH_TARGET> "cat $RUN_DIR/<filename>"
```

### To read local source code

Use the Read or Grep tool directly on the local filesystem. The PyTorch source
tree is at: <source_tree_path>

### Summary of where things run

- Profiling, benchmarking, peak BW measurement: REMOTE (SSH)
- Writing benchmark scripts, JSON, logs: LOCAL (Write tool) then SCP to remote
- Reading kernel source code (.cpp/.h): LOCAL (Read/Grep tool)
- Arithmetic (FLOPs, Bytes, AI, Roofline): LOCAL
- Verification: REMOTE (SSH + python3 -c)

If SSH_TARGET is empty, the device is local. Run all commands directly and
write files directly to $RUN_DIR.

## Critical requirements
1. Save ALL raw profiler outputs to files in $RUN_DIR BEFORE parsing them.
   Run as: ssh ... "<env> && <profiler_cmd> > $RUN_DIR/output.log 2>&1"
2. Write REQUIRED OUTPUT JSON and log files via Write tool + SCP.
3. Run VERIFICATION commands via SSH at the end.
4. Return a summary of your findings including key numeric results.
```

## Output template (Step 9)

```
## Performance Optimization Report -- <op_name> on <device>

### 1. Bound classification
- Host / Device: <U value and classification>
- AI: <value> FLOP/Byte (compute_path=<matrix|vector>)
- Roofline: <Memory-Bound | Compute-Bound> (time_theory = <ms>)
- Measured: <ms>

### 2. Hardware validation
- T_mem (projected): <ms>
- Measured DRAM BW: <GB/s>, utilization: <pct>
- Read/write amplification: <values>
- Dominant pipe: <FP | INT | MEMORY | ...>
- Dominant stall: <stall type>

### 3. Hotspot
- Hot kernel name: <...>
- Source mapping: <file:line / asm region>

### 4. Recommended lever(s) in priority order
1. ...
2. ...

### 5. Next measurement
<exact command to re-run after applying levers>
```

## Lever reference (for Step 9)

This table is a starting point, not an exhaustive lookup. You MUST reason about the specific kernel's data layout, parallelization strategy, and which computations are shared across adjacent threads. The best optimization often comes from restructuring how work is distributed, not from micro-optimizing individual operations.

| Dominant bound | Levers |
|----------------|--------|
| Host-bound | fuse, queue more work, remove sync, compile |
| DRAM bandwidth | reduce precision, tiling, eliminate temporaries |
| Traffic amplification | improve coalescing, contiguous loads, d32 packing |
| Poor compute-memory overlap | increase occupancy, prefetch, reduce dependent chains, expose more independent work |
| Insufficient MLP | increase occupancy, vectorize loads, reduce dependent memory chains |
| Uncoalesced access | improve coalescing, contiguous loads, d32 packing |
| FP ALU | remove redundant FP, lower precision, native math |
| INT/MATH ALU | vectorize across shared-coordinate dim (amortize index math), `IntDivider`, narrow 64-bit pointers, reparameterize grid |
| Matrix/tensor | GEMM tile / tensor-core / matrix-engine tuning |

Note on load width: when DRAM utilization is already low (clean, coalesced traffic, `T_actual / T_mem > 1.4`), wider loads (e.g., `float4`) mainly reduce instruction count and amortize index math -- they do **not** increase peak DRAM bandwidth.

## Best practices

- Always measure before guessing. The Roofline step takes seconds; it prevents wasted micro-optimization.
- Distinguish **bandwidth-bound** from **latency-bound** before applying memory levers.
- Use random data for memory-bandwidth measurements; zero/constant tensors may compress and lie.
- Skip the first iteration as warmup when parsing logs.
- Report theoretical lower bounds alongside measured times so progress is visible.
