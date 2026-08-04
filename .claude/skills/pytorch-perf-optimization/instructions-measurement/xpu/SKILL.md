---
name: instructions-measurement-xpu
description: Intel XPU-specific instruction/pipeline measurement using VectorEngineProfile counters and XVE stall events.
---

# XPU Instructions Measurement

XPU-specific addendum for `instructions-measurement`.

## Intel XPU pipeline model (Xe2 / B580)

| Pipe | Operations |
|------|------------|
| ALU0 | FP8/FP16 SIMD32, FP32 SIMD16, BF16 SIMD16 |
| ALU1 | INT8/INT16 SIMD32, INT32 SIMD16, INT64 SIMD8, FP64 SIMD1, MATH (div/sqrt SIMD4), control (JEU) |
| ALU2 | DPAS (systolic matrix ops) |
| SEND | Memory loads/stores |

ALU0 and ALU1 can execute concurrently.

## Peak slot rate

```
peak_slots_per_second = num_XVEs * GPU_frequency
```

For B580: 160 XVEs * 2.85 GHz = 456 G-slots/s per pipe.

## Key VectorEngineProfile counters

```
XVE_INST_EXECUTED_ALU0_ALL      # float ALU
XVE_INST_EXECUTED_ALU1_ALL      # int ALU (does NOT include MATH)
XVE_INST_EXECUTED_MATH          # divide, sqrt, rsqrt (MUST be added to ALU1)
XVE_INST_EXECUTED_SEND_ALL      # memory loads/stores
XVE_INST_EXECUTED_ALU2_ALL      # DPAS
XVE_INST_EXECUTED_CONTROL_ALL   # branches/jumps
XVE_INST_EXECUTED_FP16          # sub-counter of ALU0
XVE_INST_EXECUTED_FP32          # sub-counter of ALU0
XVE_INST_EXECUTED_INT32         # sub-counter of ALU1
XVE_INST_EXECUTED_INT64         # sub-counter of ALU1
XVE_INST_EXECUTED_BITCONV       # type conversions
```

## Computing per-pipe time

```python
peak = num_xves * freq_hz   # slots/s

T_alu0 = ALU0_ALL / peak
T_alu1 = (ALU1_ALL + MATH) / peak   # MATH is NOT included in ALU1_ALL
T_dpas = ALU2_ALL / peak
T_send = SEND_ALL / peak

T_instruction = max(T_alu0, T_alu1, T_dpas, T_send)
```

**Critical: MATH is NOT included in ALU1_ALL. Always add it separately.**

## VectorEngineStalls counters

From the VectorEngineStalls raw log (collected in Step 3):

| Counter | Meaning |
|---------|---------|
| `XVE_STALL_ALUWR` | ALU write-back dependency |
| `XVE_STALL_SBID` | Scoreboard: waiting on in-flight memory |
| `XVE_STALL_PIPESTALL` | Pipe conflict (pipe is busy) |
| `XVE_STALL_INSTFETCH` | Instruction cache miss |
| `XVE_STALL_CONTROL` | Control flow stall |
| `XVE_STALL_BARRIER` | Barrier synchronization |
| `XVE_STALL_SENDWR` | SEND write-back conflict |

## Typical levers

| Hot pipe | Typical cause | Levers |
|----------|---------------|--------|
| ALU1 (INT + MATH) | Index decomposition, 64-bit pointers, runtime division/modulo | 1. IntDivider (mul_hi + shift) 2. Narrow 64-bit to 32-bit 3. Vectorize to amortize index math 4. Reparameterize grid |
| ALU0 (FP) | Excess FP math | Remove redundancy, lower precision, native math |
| DPAS | GEMM/Conv tile mismatch | Tune tile/DPAS parameters |
| SEND | Memory instruction count | Re-evaluate with memory-load-store; may be memory-bound |
