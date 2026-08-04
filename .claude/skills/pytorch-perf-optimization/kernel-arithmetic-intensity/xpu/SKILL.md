---
name: kernel-arithmetic-intensity-xpu
description: Intel XPU-specific notes for choosing compute_path and dtype when computing kernel arithmetic intensity for Xe matrix/vector engines.
---

# Kernel Arithmetic Intensity -- Intel XPU Notes

XPU-specific addendum for `kernel-arithmetic-intensity`.

## Choosing `compute_path` on XPU

XPU devices expose two compute paths with very different peak throughput:

| Path | XPU engine | Typical kernels |
|------|------------|-----------------|
| `matrix` | XMX / DPAS | GEMM, batched GEMM, Conv, attention (if dtype supported) |
| `vector` | XVE vector engines | Element-wise, reductions, softmax, layernorm, pooling, index ops |

Use `matrix` only if the kernel pattern matches AND the device's matrix engine supports the execution dtype.

## dtype considerations

- FP16 and BF16 are the most common matrix-engine dtypes on recent Xe GPUs.
- FP32 may fall back to vector engines unless the SKU explicitly supports FP32 matrix.
- INT8 can sometimes use DPAS for quantized GEMM/conv.

When in doubt, check `XVE_INST_EXECUTED_ALU2_ALL` (DPAS counter) from VectorEngineProfile. If it is zero, the matrix path is not active.

## Watch-outs

- Fused elementwise epilogues (GEMM + bias + ReLU) may use matrix path for matmul but add vector-pipe work.
- Mixed-precision kernels: use the dtype actually executed on the accelerator, not the Python dtype.
