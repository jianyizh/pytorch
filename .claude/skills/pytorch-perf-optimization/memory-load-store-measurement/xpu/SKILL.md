---
name: memory-load-store-measurement-xpu
description: Intel XPU-specific memory load/store measurement using ComputeBasic counters, LOAD_STORE_CACHE events, and stall sampling.
---

# XPU Memory Load/Store Measurement

XPU-specific addendum for `memory-load-store-measurement`.

## XPU counter names (ComputeBasic group)

| Counter | Meaning |
|---------|---------|
| `GPU_MEMORY_BYTE_READ[bytes]` | DRAM bytes read |
| `GPU_MEMORY_BYTE_WRITE[bytes]` | DRAM bytes written |
| `GpuTime[ns]` | Kernel execution time |
| `LOAD_STORE_CACHE_BYTE_READ[bytes]` | L1/LSC bytes read |
| `LOAD_STORE_CACHE_BYTE_WRITE[bytes]` | L1/LSC bytes written |
| `LOAD_STORE_CACHE_ACCESS[events]` | L1/LSC access count |
| `LOAD_STORE_CACHE_HIT[events]` | L1/LSC hit count |
| `XVE_ACTIVE[%]` | % time XVEs are active |
| `XVE_STALL[%]` | % time XVEs are stalled |

## Measuring achievable peak bandwidth on XPU

```python
import torch, time, statistics

device = "xpu"
total_mem = torch.xpu.get_device_properties(device).total_memory
n = int(total_mem * 0.8 / 3 / 4)
a = torch.randn(n, device=device)
b = torch.randn(n, device=device)
c = torch.randn(n, device=device)

for _ in range(10):
    a.neg_(); b.neg_(); c.neg_()
torch.xpu.synchronize()

results = []
for trial in range(5):
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    reps = 30
    for _ in range(reps):
        a.neg_(); b.neg_(); c.neg_()
    torch.xpu.synchronize()
    t1 = time.perf_counter()
    total_bytes = (a.nbytes + b.nbytes + c.nbytes) * reps * 2
    results.append(total_bytes / (t1 - t0) / 1e9)

peak_bw_gbps = statistics.median(results)
print(f"peak_bw_gbps={peak_bw_gbps:.2f}")
```

## XPU stall interpretation for memory analysis

From VectorEngineStalls group:

| Stall | Meaning | How to distinguish |
|-------|---------|-------------------|
| `XVE_STALL_SBID` | Scoreboard: XVE waiting on an in-flight memory operation to complete | If BW near peak: normal latency cost in a BW-bound kernel. If BW well below peak: **insufficient MLP** (too few concurrent requests) or **poor compute-memory overlap** (compute is blocking new requests from issuing). Distinguish by checking ALU pipe utilization in Step 7. |
| `XVE_STALL_SENDWR` | SEND pipe write-back conflict: memory return path is congested | Often seen with uncoalesced access patterns that generate many small transactions |
| `XVE_STALL_ALUWR` | ALU write-back dependency: waiting for an ALU result | Not directly memory, but when ALU is saturated (e.g., index math), it prevents new memory requests from being issued, causing **poor compute-memory overlap** |

## Intel XPU specifics

- **Compression**: hardware compressor sits between L2/L3 and DRAM. Random data gives incompressible traffic (ratio ~1.0); zero/constant tensors compress and under-report. Always use random data.
- **Coalescing**: a coalesced load needs contiguous addresses and payload >= 4 bytes. FP16/BF16 should be packed into d32 or wider.
- **Load width note**: when DRAM utilization is already low (clean, coalesced traffic, T_actual / T_mem > 1.4), wider loads (e.g., float4) mainly reduce instruction count and amortize index math -- they do NOT increase peak DRAM bandwidth.
