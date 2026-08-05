---
name: kernel-arithmetic-intensity
description: Compute the theoretical upper-bound arithmetic intensity (AI = FLOPs / Bytes) of a GPU kernel from its shape and dtype. Device-agnostic and implementation-agnostic.
---

# Kernel Arithmetic Intensity

**Goal:** Compute the **theoretical upper-bound** Arithmetic Intensity (AI = FLOPs / Bytes) from op shape and dtype only. The result is implementation-agnostic and hardware-agnostic.

## REQUIRED INPUTS

| Input | Source | Notes |
|-------|--------|-------|
| `$RUN_DIR/01_kernel_profiler_setup.json` | Step 1 | run_dir |
| `$RUN_DIR/02_host_vs_device_bound.json` | Step 2 | classification must be Device Bound or Mixed |
| `$RUN_DIR/03_kernel_profiler_parser.json` | Step 3 | dominant_kernel_name, median_gpu_time_ns |
| Op config | User / orchestrator | op_name, shapes, dtype |
| PyTorch source tree | Local repo | For inspecting kernel source if needed |

Read all prior JSON files. Fail if classification is Host Bound (AI is not meaningful).

## Execution context

| Action | Where |
|--------|-------|
| Read prior step JSON files | REMOTE |
| Read kernel source code (grep, inspect `.cpp`) | LOCAL (PyTorch source tree) |
| Compute FLOPs/Bytes/AI | LOCAL (arithmetic from shapes) |
| Write step JSON/log to `$RUN_DIR` | REMOTE |
| Verification | REMOTE |

This step is mostly local computation. Only reading prior JSON and writing outputs touches the remote machine.

## Procedure

### 1. Compute Total FLOPs

Count multiplies + adds in the math definition. Each FMA = 2 FLOPs.

| Op | FLOPs |
|----|-------|
| GEMM `[M,K] x [K,N]` | `2 * M * K * N` |
| Batched GEMM `[B,M,K] x [B,K,N]` | `2 * B * M * K * N` |
| Conv2D in `[N,Ci,H,W]`, wt `[Co,Ci,kH,kW]`, out `[N,Co,Ho,Wo]` | `2 * N * Co * Ho * Wo * Ci * kH * kW` |
| AvgPool2D out `[N,C,Ho,Wo]`, kernel `[kH,kW]` | `N * C * Ho * Wo * kH * kW` (sum + divide) |
| Attention SDPA Q,K,V `[B,H,S,D]` | `4 * B * H * S^2 * D` |
| Element-wise unary `[N]` | `N` |
| Element-wise binary `[N]` | `N` |
| Reduction over `[N]` | `N` |
| Softmax `[M,N]` | `5 * M * N` |
| LayerNorm `[M,N]` + affine `[N]` | `8 * M * N` |

For ops not listed, derive from first principles.

### 2. Compute Total Bytes

Count **math-defined** input bytes read + output bytes written. Intermediates are assumed on-chip (0 bytes).

| Op | Bytes |
|----|-------|
| GEMM | `(M*K + K*N + M*N) * sizeof(dtype)` |
| Conv2D | `(N*Ci*H*W + Co*Ci*kH*kW + N*Co*Ho*Wo) * sizeof(dtype)` |
| AvgPool2D | `(N*C*H*W + N*C*Ho*Wo) * sizeof(dtype)` |
| SDPA | `4 * B * H * S * D * sizeof(dtype)` |
| Element-wise unary | `2 * N * sizeof(dtype)` |
| Element-wise binary | `3 * N * sizeof(dtype)` |
| Softmax `[M,N]` | `2 * M * N * sizeof(dtype)` |
| LayerNorm `[M,N]` | `(2*M*N + 2*N) * sizeof(dtype)` |

If output dtype differs from input dtype, split accordingly.

### 3. Compute AI

```
AI = Total_FLOPs / Total_Bytes   [FLOP/Byte]
```

### 4. Determine compute_path

| Pattern | compute_path |
|---------|--------------|
| GEMM, batched GEMM, Conv, attention | `matrix` (if dtype supported by matrix engine) |
| Element-wise, reductions, softmax, layernorm, pooling, index ops | `vector` |

## REQUIRED OUTPUTS

### `$RUN_DIR/04_kernel_arithmetic_intensity.json`

```json
{
  "step": "04_kernel_arithmetic_intensity",
  "op": "<op_name>",
  "shapes_summary": "<concise shape description>",
  "dtype": "<dtype>",
  "sizeof_dtype": <int>,
  "total_flops": <int>,
  "total_bytes": <int>,
  "AI": <float>,
  "compute_path": "<matrix|vector>",
  "flops_breakdown": "<description>",
  "bytes_breakdown": "<description>",
  "run_dir": "<$RUN_DIR>"
}
```

### `$RUN_DIR/04_kernel_arithmetic_intensity.log`

Human-readable report with full calculation breakdown.

## VERIFICATION

**Run all verification commands via SSH on the target machine (they access `$RUN_DIR` which is remote). Write any output files locally to `/tmp/opencode/` first, then SCP to `$RUN_DIR`.**

```bash
test -f $RUN_DIR/04_kernel_arithmetic_intensity.json && echo "JSON OK" || echo "JSON MISSING"
python3 -c "
import json
d = json.load(open('$RUN_DIR/04_kernel_arithmetic_intensity.json'))
required = ['total_flops', 'total_bytes', 'AI', 'compute_path']
missing = [k for k in required if k not in d]
assert not missing, f'Missing fields: {missing}'
assert d['AI'] > 0, 'AI must be positive'
assert d['compute_path'] in ('matrix', 'vector'), f'Bad compute_path: {d[\"compute_path\"]}'
print(f'VERIFICATION PASSED: AI={d[\"AI\"]:.2f} FLOP/Byte, path={d[\"compute_path\"]}')
"
```

## Vendor-specific details

See the vendor sub-skill (e.g., `xpu/SKILL.md`) for device-specific dtype/compute_path guidance.
