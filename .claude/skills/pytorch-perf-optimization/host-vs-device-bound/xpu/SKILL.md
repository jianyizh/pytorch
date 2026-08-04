---
name: host-vs-device-bound-xpu
description: Intel XPU-specific steps for classifying a PyTorch workload as host-bound or device-bound using unitrace.
---

# Host-vs-Device Bound -- Intel XPU

XPU-specific addendum for `host-vs-device-bound`.

## Prerequisites

```bash
source /opt/intel/oneapi/setvars.sh
which unitrace
echo 0 > /proc/sys/dev/xe/observation_paranoid 2>/dev/null
```

## Profiling with unitrace

### Get kernel names and t_dev

Run the benchmark under unitrace with `-d` (device timeline):

```bash
unitrace -d python $RUN_DIR/bench_<op>.py profile > $RUN_DIR/unitrace_timeline.log 2>&1
```

The output contains per-kernel entries:

```
=== Device Timing Summary ===
Kernel, Calls, Time (ns), Time (%), Average (ns), Min (ns), Max (ns)
```

And kernel properties (Compiled=AOT/JIT, SIMD width, etc).

### Parse t_dev

1. Filter to the kernel(s) belonging to the op.
2. Skip the first instance as warmup.
3. Take the min or median of remaining instances as t_dev (min is closest to steady-state).
4. If multiple kernels per op call, sum them.
5. Record kernel properties (Compiled, SIMD, num args).

### Grep for source

```bash
grep -rn "KernelBaseName" aten/src/ATen/native/
grep -rn "KernelBaseName" third_party/torch-xpu-ops/src/ATen/native/xpu/sycl/
```

## Measuring t_op

Run the benchmark in `bench` mode WITHOUT unitrace:

```bash
python $RUN_DIR/bench_<op>.py bench
```

## XPU-specific host-bound levers

| Symptom | Likely cause | Lever |
|---------|--------------|-------|
| Many tiny kernels, gaps | Python/ATen overhead | `torch.compile`, op fusion |
| Big idle before GPU work | H2D data upload | `pin_memory`, `non_blocking=True` |
| Idle after GPU work | D2H sync | Defer sync, overlap compute and copy |
