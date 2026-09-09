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
| `XVE_STALL_SBID` | Scoreboard: XVE waiting on an in-flight memory operation to complete | If BW near peak: normal latency cost in a BW-bound kernel. If BW well below peak: **insufficient MLP** (too few concurrent requests) or **poor compute-memory overlap** (compute is blocking new requests from issuing). Distinguish by checking ALU pipe utilization in Step 9. |
| `XVE_STALL_SENDWR` | SEND pipe write-back conflict: memory return path is congested | Often seen with uncoalesced access patterns that generate many small transactions |
| `XVE_STALL_ALUWR` | ALU write-back dependency: waiting for an ALU result | Not directly memory, but when ALU is saturated (e.g., index math), it prevents new memory requests from being issued, causing **poor compute-memory overlap** |

## Load/store cache (L1 / LSC) on XPU

### Background

The load/store cache sits between the execution units and the rest of the
memory hierarchy. On Intel XPU it is called **LSC**. It is **on-chip and
private to a core** (one LSC per Xe-core). L1 caches are typically **not
coherent** across cores; coherence is handled by the last level cache (L2/L3)
and the memory subsystem.

Its main job is coalescing: it turns the SIMD load/store requests issued by a
wave into cache-line transactions.

A well-behaved kernel saturates the load/store cache without producing excess
traffic to the last level cache (L2/L3) and DRAM.

### How it differs from the last level cache (L2/L3)

| Property | Load/store cache (L1 / LSC) | Last level cache (L2/L3) |
|----------|----------------------------|--------------------------|
| Location | On-chip, close to execution units | Usually larger, closer to memory |
| Sharing | Per-core / per-SM | Shared across cores |
| Coherence | Generally not coherent across cores | Maintains coherence across cores |
| Main role | Coalescing SIMD requests | Capacity / reuse, shared layer |

### Two key questions

1. **Coalescing**: For each send, how many requests are generated, and how many
   bytes are moved per event?
2. **Traffic sanity**: Does the total number of bytes read from the LSC match the
   op semantics and projection? If not, look at amplification (partial writes,
   repeated reads, prolog effects).

A coalesced load may touch one or two cache lines to move 128 bytes; a scattered
load may touch 32 lines and move only 128 bytes total.

### Metrics and formulas

What to measure:
- **Access count**: number of requests (`LOAD_STORE_CACHE_ACCESS`).
- **Byte read**: bytes read from the LSC data array into registers (`LOAD_STORE_CACHE_BYTE_READ`).
- **Hit / miss counters**: reuse within the LSC (`LOAD_STORE_CACHE_HIT` / `LOAD_STORE_CACHE_MISS`).
- **Bytes per access** = `byte_read / requests`.

High request counts with low bytes per access indicate poor coalescing. High
bytes per access with low hit rate usually means the working set does not fit,
but the access pattern is at least contiguous.

#### Cache-line utilization

```
bytes_per_access = byte_read / requests
```

For a fully coalesced SIMD32 load of 16B per lane: 32 x 16B = 512B payload
across 8 cache lines, plus a prolog; observed bytes per access ~= 32-60
depending on the prolog and store mix.

For a scattered load where each lane touches a different cache line: 32 x 16B
payload but 32 requests; bytes per access ~= 2.5.

#### Hit / miss

```
hit_rate = hits / requests
misses   = requests - hits
```

A low hit rate is not automatically bad: streaming kernels are supposed to
miss. A low hit rate *combined with* low bytes per access points to double
damage: scattered accesses that also thrash the cache.

#### Load/store-cache bandwidth

```
read_bw  = byte_read / GpuTime
write_bw = byte_write / GpuTime
```

Compare to the peak load/store-cache bandwidth of the device.

### XPU counter names (LSC)

| Counter | Meaning |
|---------|---------|
| `LOAD_STORE_CACHE_ACCESS[events]` | LSC cache-line requests |
| `LOAD_STORE_CACHE_BYTE_READ[bytes]` | Bytes read from LSC to register |
| `LOAD_STORE_CACHE_BYTE_WRITE[bytes]` | Unusual to see non-zero; XPU stores bypass L1 data by default |
| `LOAD_STORE_CACHE_HIT[events]` | LSC hits (can include hits-in-flight) |

Note: `HIT` can include hits-in-flight. If a single load instruction fetches
multiple cache lines, the first may be a miss while the subsequent lines are
counted as hits because they are part of the same in-flight request.

### B/acc model (B580)

A practical model for `LOAD_STORE_CACHE_ACCESS` per thread group on B580:

```
ACCESS    = 1 + sum(load_CLs_per_send) + sum(store_CLs_per_send)
BYTE_READ = 32 + sum(SIMD_width x bytes_per_lane)
```

- The `1` is the per-thread-group prolog CTI load (32B).
- `load_CLs_per_send` = number of distinct 64B cache lines touched by that send.
- `store_CLs_per_send` = same for stores.

Verified against microbenchmarks: CL = 64B predicts all tested access counts.

### Store behavior on XPU

Xe LSC is a cache for loads but **stores write through to the last level cache**.
Therefore:
- `LOAD_STORE_CACHE_BYTE_WRITE` non-zero is unusual and suggests the store path
  is not bypassing as expected.
- Store efficiency is better diagnosed at the last level cache (`L3_WRITE`,
  partial-write counts).

### LSC workflow

1. **Collect counters**:
   ```bash
   unitrace -q -i 20 -g ComputeBasic ./bench > compute_basic.csv
   ```
   `ComputeBasic` contains `LOAD_STORE_CACHE_BYTE_READ`, `LOAD_STORE_CACHE_ACCESS`, `LOAD_STORE_CACHE_HIT`.

2. **Compute bytes per access and hit rate**:
   ```python
   bytes_per_access = LOAD_STORE_CACHE_BYTE_READ / LOAD_STORE_CACHE_ACCESS
   hit_rate = LOAD_STORE_CACHE_HIT / LOAD_STORE_CACHE_ACCESS
   ```

3. **Compare with the instruction pattern**: disassemble the kernel and classify
   each load/store send by width and address pattern. Count total load/store
   sends per work-item, their payload widths, and how many distinct cache lines
   the addresses span. This gives expected `ACCESS` and `BYTE_READ` per thread group.

4. **Decide the lever**:
   1. **Traffic sanity**: compare measured `byte_read`/`byte_write` against the
      expected minimum traffic. If far off, look for extra loads/stores,
      alignment padding, or an unexpected compiler transform.
   2. **Coalescing**: check `bytes_per_access`. Far below ideal -> scattered
      addresses -> rewrite loop order / use vector loads. Close to ideal -> address
      pattern is fine; the bottleneck is elsewhere.
   3. **If coalescing is good**: look at stall sampling.
      - High memory/SBID stalls -> LSC is fast enough; latency not hidden ->
        increase occupancy or add prefetch.
      - High compute-pipeline / control stalls -> LSC is not the issue; the
        kernel is stalled on computation.

### LSC cautions

- `LOAD_STORE_CACHE_ACCESS` counts one request per cache line, including a prolog.
  Compute bytes per access rather than comparing raw request counts across
  different kernel designs.
- A streaming kernel is *supposed* to have a low LSC hit rate. The goal is high
  bytes per access, not high hit rate.
- On XPU the cache line is 64B; do not assume a larger line just because SIMD32
  wide sends move 512B.
- Comparing `LOAD_STORE_CACHE_BYTE_READ` across shapes requires normalizing by
  the number of thread groups.

---

## Last level cache (L2/L3) on XPU

### Background

The last level cache is shared across the GPU and sits between the per-core LSC
(L1) and the DRAM controllers. Unlike the load/store cache, which is primarily
about coalescing, the last level cache is about:

1. **Capacity** -- does the working set fit?
2. **Reuse / data partitioning** -- is the current data partition keeping data in
   the cache long enough to benefit from reuse? Norm-like operations that load
   the same input several times are a common example.
3. **Miss traffic** -- how much data reaches DRAM because it was not found in the
   last level cache?

The last-level-cache question is:

> Is the last level cache generating more traffic or stalls than the access
> pattern should require, and is the working set partitioned to maximize reuse?

### Data partitioning and reuse

A key device-agnostic consideration is reuse across the chosen data partition.
For example, a normalization kernel may load the same input tensor two times
along different reduction dimensions. If the tile that contains those values
fits in the last level cache and is kept resident, the repeated loads become
cheap cache hits instead of multiple DRAM fetches.

When the number of observed last-level-cache read events is much larger than
the unique cache lines the partition needs, the first lever to revisit is
usually the data partition, the tile size, or the loop order.

### Metrics and formulas

#### Last-level-cache traffic

The two traffic quantities are the bytes **read from** the last level cache and
the bytes **stored to** the last level cache. Counters report cache-line events;
multiply by the cache-line size (64B) to get bytes.

#### Hit / miss

```
hit_rate = hits / (hits + misses)
```

#### Reuse / capacity

Compare the amount of data one wave (or thread group) needs against the last
level cache capacity. If the working set is larger than the cache, reuse is
impossible and you will see capacity misses. If the working set fits but is
still being refetched, the data partition or loop order is not keeping the data
resident.

### XPU counter mapping

The generic variables map to Intel XPU unitrace `ComputeBasic` counters as:

- `read_events` -> `L3_READ`
- `write_events` -> `L3_WRITE`
- `hits` / `misses` -> `L3_HIT` / `L3_MISS`

Additional useful counters: `LOAD_STORE_CACHE_ACCESS`, `LOAD_STORE_CACHE_HIT`
(L1-level), plus `L3_STALL[%]`.

### Partial writes and overfetch

- **Partial writes**: when a write does not cover a full cache line, the hardware
  performs a partial write. If the cache line is evicted before it is fully
  written, a read-modify-write is triggered, causing extra memory traffic.
- **Overfetch**: the last level cache fetches 256 bytes from global memory by
  default. A load/store request smaller than 256 bytes can therefore generate
  extra memory traffic.

### L3 workflow

1. **Collect counters**:
   ```bash
   unitrace -q -i 20 -g ComputeBasic ./bench > compute_basic.csv
   ```
   `ComputeBasic` includes `L3_READ`, `L3_WRITE`, `L3_HIT`, `L3_MISS`.

2. **Compute last-level-cache read/write traffic**:
   ```python
   L3_read_bytes  = L3_READ * 64
   L3_write_bytes = L3_WRITE * 64
   ```

3. **Compare with projection**: compute expected full writes from the kernel's
   logical output bytes:
   ```python
   expected_write_events = total_bytes_written / 64
   amplification = L3_WRITE / expected_write_events
   ```
   - `amplification ~= 1` -> clean vectorized stores.
   - `amplification > 1` -> non-coalesced stores, or the projection / implementation
     assumption is wrong.

4. **Decide the lever**:

   | Observation | Interpretation | Lever |
   |-------------|----------------|-------|
   | `L3_WRITE` > expected | Non-coalesced stores, or projection / implementation wrong | Vectorize stores or check the implementation |
   | `L3_READ` >> projected read | Last-level-cache miss / thrashing or read-modify-write from partial stores | Improve locality or fix stores first |
   | `L3_HIT` very high but BW low | Last-level-cache latency not hidden | Increase occupancy or prefetch |

### Common causes and levers (L3)

| Cause | Metrics signature | Lever |
|-------|-------------------|-------|
| Capacity thrashing | `L3_HIT` low, `L3_READ` and DRAM reads both high | Tile the working set to fit the last level cache |
| Write-through pressure | `L3_WRITE` high even with full writes | Reduce write traffic or coalesce stores |

### L3 cautions

- `L3_WRITE` counts **events**, not bytes. Always multiply by 64B to compare
  with byte-level projections.
- The last level cache is write-through from the load/store cache, so every
  store reaches it even if the same data is later overwritten.
- The hardware compressor between the last level cache and DRAM can reduce DRAM
  traffic for compressible patterns. Use random data in microbenchmarks if you
  want to measure the uncompressed worst case.
- A small number of partial writes at boundaries is normal after peeling or for
  tail loops. Worry about the ratio, not isolated boundary cases.

## Intel XPU specifics

- **Compression**: hardware compressor sits between L2/L3 and DRAM. Random data gives incompressible traffic (ratio ~1.0); zero/constant tensors compress and under-report. Always use random data.
- **Coalescing**: a coalesced load needs contiguous addresses and payload >= 4 bytes. FP16/BF16 should be packed into d32 or wider.
- **Load width note**: when DRAM utilization is already low (clean, coalesced traffic, T_actual / T_mem > 1.4), wider loads (e.g., float4) mainly reduce instruction count and amortize index math -- they do NOT increase peak DRAM bandwidth.
