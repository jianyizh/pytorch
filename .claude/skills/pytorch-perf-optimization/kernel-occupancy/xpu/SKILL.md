---
name: kernel-occupancy-xpu
description: Intel XPU-specific guidance for computing kernel occupancy from launch-time resources (work-group size, subgroup size, GRF register usage, SLM usage) on Xe architecture.
---

# Kernel Occupancy Analysis -- Intel XPU Notes

XPU-specific addendum for `kernel-occupancy`.

The algorithm below matches the official **Intel GPU Occupancy Calculator**
(source: `oneapi-src/oneAPI-samples` -> `Tools/GPU-Occupancy-Calculator/index.html`,
functions `compute_ss_occupancy` / `compute_gpu_occupancy`).

## Xe-core device parameters

Occupancy is computed per **sub-slice (Xe-core)**. Each field maps to the tool's
`device_info` object. Match the device to the closest Xe generation:

| Device family | EU/Xe-core | Threads/EU | Max threads/Xe-core | SLM/Xe-core (KB) | Max WGs | Max barrier regs | Subgroup sizes |
|---------------|-----------|-----------|--------------------|------------------|---------|------------------|----------------|
| Xe2 HPG/BMG (Arc B580) | 8 | 8 | 64 | 128 | 64 | 64 | 32, 16 |
| Xe2 LPG (LNL) | 8 | 8 | 64 | 128 | 64 | 64 | 32, 16 |
| Xe HPG (DG2 Arc) | 16 | 8 | 128 | 128 | 128 | 128 | 32, 16, 8 |
| Xe HPC (PVC / Max) | 8 | 8 | 64 | 128 | 64 | 64 | 32, 16 |
| Xe LP (DG1 / Iris Xe Max) | 16 | 7 | 112 | 64 | 112 | 64 | 32, 16, 8 |

Notes:
- **Max threads/Xe-core == `Max_Threads_Per_Sub_Slice`**; each thread slot is one subgroup (hardware thread). This is the occupancy denominator.
- **Large-GRF is a per-kernel compiler decision, not a device attribute.** Whether a given kernel runs in large-GRF mode is fixed at compile time by register-pressure analysis, and it varies kernel by kernel -- some kernels in the same binary may use it while others do not. It is therefore NOT listed in the device table. Read it from the profile output for the specific dominant kernel (see launch parameters below). By default one subgroup fits in 128 GRF; if a kernel needs more, the compiler enables large-GRF, which merges two hardware-thread slots into one so `Max_Threads_Per_SubSlice` halves (64 -> 32) and occupancy halves. A kernel that exceeds 128 GRF but cannot use large-GRF (e.g. on B580) instead spills.
- SLM and WG caps are per Xe-core.
- Input SLM is rounded up to the nearest allowed value in `TG_SLM_Sizes` (B580: [0,1,2,4,8,16,24,32,48,64,96,128] KB) before the SLM limit is computed.

## Getting the launch parameters

The Step 1 timeline log produced with the verbosity flag captures these for the dominant kernel:

```bash
unitrace -d -v ./bench
```

Extract for the dominant kernel: `workgroup_size`, `subgroup_size`, registers/thread (+ large-GRF flag), SLM per WG, number of WGs, barrier usage.

The verbose timeline's **Register File Size Per Thread** is directly readable in unitrace. Interpret it as:
- `128` -> default GRF mode, one subgroup per hardware-thread slot, thread slots NOT halved.
- `256` -> large-GRF mode, two hardware-thread slots merge into one, `Max_Threads_Per_SubSlice` halves (64 -> 32), occupancy halves.

Also confirm via `Spill Memory Per Thread == 0` (spills mean the kernel exceeded its register budget without large-GRF).

## Occupancy formula (matches the official calculator)

```
MAX_THREADS_PER_CORE = Max_Threads_Per_Sub_Slice      # 64 for B580/BMG
if large_grf: MAX_THREADS_PER_CORE /= 2               # per-kernel; see note above

threads_per_wg = ceil(workgroup_size / subgroup_size) # thread slots one WG needs;
                                                      # subgroup_size is SIMD width
limit_threads = floor(MAX_THREADS_PER_CORE / threads_per_wg)  # WGs by thread slots

max_num_wg = Max_Num_Of_Workgroups                    # 64 for B580/BMG
if uses_barriers: max_num_wg = Max_Num_Of_Barrier_Registers   # 64 for B580/BMG

if slm_per_wg == 0:
    limit_slm = max_num_wg
else:
    limit_slm = floor(SLM_Size_Per_Sub_Slice / slm_per_wg)    # SLM KB / SLM per WG, in same units

num_wg = min(limit_threads, limit_slm, max_num_wg, total_WGs_launched)

occupancy = threads_per_wg * num_wg / Max_Threads_Per_Sub_Slice   # 0.0 to 1.0
```

The `total_WGs_launched` term (global_size / workgroup_size) only matters for small workloads; for steady-state kernels it is large and does not bind.

### Worked example -- B580 (BMG), no large-GRF, no barrier

Inputs: `wg=256`, `sg=32`, `slm_per_wg=64 KB`, `global=1M`:

```
MAX_THREADS_PER_CORE = 64
threads_per_wg  = ceil(256/32)  = 8
limit_threads   = floor(64/8)   = 8 WGs
max_num_wg      = 64
limit_slm       = floor(128/64) = 2 WGs      # SLM binds here
num_wg          = min(8, 2, 64, 3906) = 2
occupancy       = 8 * 2 / 64 = 0.25   # 25% -- SLM-limited
```

If instead `slm_per_wg=16 KB`: `limit_slm=floor(128/16)=8`, `num_wg=min(8,8,64)=8`, occupancy = `8*8/64 = 1.0` (100%).

## Work-group size recommendation

Pick `workgroup_size` so that:
- `threads_per_wg = workgroup_size / subgroup_size` is a **power of two**
- `1 <= threads_per_wg <= 32` (leaves room for at least 2 resident WGs on a 64-slot Xe-core; a single WG with >32 subgroups under-uses slot granularity)
- `64 % threads_per_wg == 0` so an integer number of WGs fills the 64-slot budget (`64 / threads_per_wg` WGs resident)

Too small a WG dispatches too many WGs (dispatch/sync overhead); WG larger than 32 subgroups cannot co-reside cleanly across Xe-cores.

## Caveat

Higher occupancy is not always faster -- a kernel that lowers occupancy by using more SLM for data reuse (or fewer, larger WGs) can beat a naive high-occupancy version. Use occupancy to explain a measured bound, not as a blind target.

## Output

Write the same `04_kernel_occupancy.json` / `.log` schema as the generic skill. XPU's `SLM`
maps to the generic `smem` fields: record `smem_per_workgroup_bytes` in `launch_params`, and
`limit_smem` (= the `limit_slm` computed above) in `resource_limits`. Also record `device`
(Xe generation) and `max_threads_per_core` (= `Max_Threads_Per_SubSlice`, 64 on BMG) in
`resource_limits`, plus XPU-specific `large_grf` in `launch_params`.
