---
name: kernel-memory-compute-bound-xpu
description: Intel XPU-specific guidance for sourcing peak FLOPS and peak memory bandwidth for the Roofline model.
---

# Kernel Memory-vs-Compute Bound -- Intel XPU Notes

XPU-specific addendum for `kernel-memory-compute-bound`.

## Identifying the device

```python
import torch
device_name = torch.xpu.get_device_name(device=None)
print(device_name)
```

## XPU spec table

Match `device_name` by substring against this table:

| Identifier | Xe cores | XVEs | XMXs | Clock | Memory BW | FP16 matrix | FP16 vector | FP32 vector |
|------------|----------|------|------|-------|-----------|-------------|-------------|-------------|
| `B580` | 20 | 160 | 160 | 2850 MHz | 456 GB/s | 116 TFLOPS | 14 TFLOPS | 7 TFLOPS |

Notes:
- FP32 does NOT use the matrix engine on B580; use the vector-engine peak.
- FP16 matmul-family kernels use matrix-engine peak only when XMX/DPAS is actually active.
- These are datasheet peaks. Prefer measured achievable peaks when available.

If the device is NOT in the table, query or measure:

```python
props = torch.xpu.get_device_properties(device)
print(props.name, props.total_memory, props.max_compute_units)
```

## Measuring achievable peak bandwidth

See `memory-load-store-measurement/xpu` for the copy benchmark. Use the measured value for the Roofline ceiling.

## Peak slot rate (for instruction analysis)

```
peak_slots_per_second = num_XVEs * GPU_frequency
```

For B580: 160 XVEs * 2.85 GHz = 456 G-slots/s per pipe.
