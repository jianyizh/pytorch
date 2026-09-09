---
name: xpu-stall-sampling
description: Reference for interpreting Intel XPU XVE stall-sampling results at per-instruction (IP) granularity. Explains each stall reason and what it means when it appears at a specific instruction address.
---

# XPU Stall-Sampling Reference (per-IP)

Stall sampling attributes stalled cycles to **specific instruction addresses (IPs)**,
unlike the aggregate `XVE_STALL_*` percentages (from VectorEngineStalls) which cover the
whole kernel. Use this to explain *why an individual instruction stalls* and to identify
the source line that causes each stall type.

Effective use requires the ISA dump (see `SKILL.md`): for each hot IP you read the stall
type from the sampling log, then look up that IP in the assembly to find the instruction
and its source line. The hot IPs and their stall counts come from Step 3.

Per-IP stalls split into two families: **memory** stalls (waiting for data) and
**non-memory** stalls (waiting for instructions, ALU results, or pipe resources). Two
rules of thumb:

- The sampled IP is where the stall is *recorded*, which is not always the instruction
  that *caused* it. For fetch stalls it is the jump/branch **target** (the destination's
  first instruction), not the jump itself; for pipe stalls it is often the *victim*
  instruction that could not issue, not the *perpetrator* that held the pipe.
- A stall type in the `Disagregation` column of the sampling log is the *dominant* stall
  at that IP; the same IP can accumulate several stall types across samples.

---

## Memory stalls

### SbidStall — wait for an in-flight memory operation (scoreboard)

**Meaning:** the XVE is waiting on a `send` (load) whose scoreboard tag has not completed.
The sampled IP is typically the **first instruction that consumes the load result**, right
after the `send` + `sync.nop`.

**What it looks like** (scatter load example):

```asm
mov (16|M0)  r7.0<1>:uq  r64.0:uq        // per-lane 64-bit address
send.ugm (32|M0) r12 r7 load.d16u32.a64  // scatter load
sync.nop
mov r67.0<1>:w r12.0<2;1,0>:w {$11.dst}  // <- SbidStall: waiting on scoreboard $11
```

`{$11.dst}` means the instruction depends on the `send` tag `$11` writing its target.

**Diagnosis:**
- **High SbidStall + DRAM BW near peak** -> this is the *normal* latency cost of a
  memory-bound kernel; it is the productive wait that keeps the bus busy. Not a problem.
- **High SbidStall + DRAM BW well below peak** -> data is late but the bus is idle:
  either **insufficient MLP** (too few independent loads in flight) or **poor
  compute-memory overlap** (ALU work blocks new sends). Distinguish via ALU pipe
  utilization: high ALU% alongside -> compute is blocking memory; low ALU% -> too few
  outstanding requests.

**Root cause to look for:** a hot, dependent chain of `send` -> consumer with few
independent loads between consumer instructions.

---

## Non-memory stalls

### InstrFetchStall — waiting for the next instruction (I-cache / control flow)

**Meaning:** the XVE cannot get the next instruction. The sampled IP is usually the
**target of a jump/branch** (the first instruction fetched after a control transfer),
because that is where the fetch wait is recorded.

Two patterns:
- **Scattered stalls across the binary** -> binary larger than what the I-cache working
  set can keep resident; regions keep evicting each other (capacity / thrashing).
- **Stalls clustered at branch/join targets** -> control-flow driven; many target
  fetches, not binary size per se.

**Healthy vs unhealthy:**
- Healthy: InstrFetchStall is a small fraction of total stalls; binary small relative to
  the I-cache.
- Unhealthy: `Active%` is low and `InstrFetchStall` is high (roughly > 20% of stalls),
  usually meaning the binary is too large or has too many active code paths.

**Hot-IP pattern -> cause:**

| Pattern | Likely cause |
|---------|--------------|
| IPs spread across most of the kernel | Binary too large for the I-cache working set |
| IPs clustered at `jmpi` / `join` targets | Branchy control flow, many join points |
| High InstrFetchStall with high ControlStall | Large kernel plus branchy code |

**Levers:** reduce binary size (specialize over dtype/generic dispatch instead of a
runtime switch), reduce join points, lower unroll factors. Fixing I-cache pressure often
also drops secondary stalls (PipeStall, DistStall, ControlStall) because instruction flow
restores interleaving.

**Cautions:** report the *target* address of the fetch wait, not the jump; do not optimize
the wrong basic block. A high InstrFetchStall usually co-occurs with ControlStall /
PipeStall, so treat it as the primary cause to fix first.

### PipeStall — ready thread but the pipe cannot accept an issue

**Official definition:** "XVE has >= 1 ready thread, but pipe cannot accept issue (GRF
conflict / send holds / ALU hold)."

**Perpetrator/Victim model:**
- **Perpetrator**: the instruction that occupies the pipe for an extra cycle (64-bit ALU
  hold, GRF write-port hold, GRF read-port conflict), blocking subsequent threads from
  issuing.
- **Victim**: any ready thread's instruction (cmp/mul/mov...) that cannot issue that cold
  cycle; the stall is recorded at the **victim IP**.

Because the stall is recorded at the victim, PipeStall IPs are heavily scattered and no
single IP dominates. To find the real culprit you must look at the *preceding* instruction
(the perpetrator), not the sampled victim.

**Common perpetrators** (from a 64-bit-add + strided-write cast kernel):

| Perpetrator | Mechanism |
|-------------|-----------|
| `add(16):q` (64-bit ptr add) | ALU hold 2 cycles |
| `mov(16)<stride>:uw` (widening) | GRF write port hold |
| Dual full-width SIMD32 sources | GRF read port conflict |

**Key insight:** PipeStall is often *exposed* by a larger problem. At full occupancy, 8
threads/XVE rotate and interleaving hides these pipe holds. If something else (e.g. I-cache
thrashing from a large binary) reduces the number of schedulable threads, PipeStall becomes
visible. Fixing the bigger problem first (e.g. shrinking the binary) can collapse PipeStall.

### DistStall — ALU write-back / dependency-distance stall (`XVE_STALL_ALUWR`)

**Meaning:** `DistStall` is the same stall as the aggregate `XVE_STALL_ALUWR`: an instruction
is waiting on an **ALU producer-to-consumer dependency** -- the value produced by a preceding
instruction (e.g. the result of a division, a widened move, or an accumulation) is not yet
written back when this instruction needs it.

**What it looks like:** the DistStall IP is the *consumer* instruction right after a
long-latency ALU producer:

```asm
add (16|M0) r20.0<1>:d r10.0<0;1,0>:d  r12.0<0;1,0>:d   // producer
cmp (16|M0) f0.0<1>:ud r20.0<0;1,0>:d r5.0<0;1,0>:d     // <- DistStall: waits for ALUWR of r20
```

**Diagnosis:** correlated with the ALU1/INT pipe (index math, 64-bit arithmetic). A high
DistStall/ALUWR share alongside high ALU1 utilization means dependent ALU chains are exposing
producer latency -- reduce dependency chains, expose more independent work, or (when caused by
long-latency ops like division) strength-reduce the op.

**Levers:** shorten dependent ALU chains, interleave independent computations across the
dependency, replace long-latency ops (div/sqrt) with cheaper forms, unroll to expose ILP.
Splitting 3-source ALU patterns or reducing 64-bit ops also lowers the producer latency that
triggers DistStall.

Note: DistStall here is a *data-dependency* stall (~`XVE_STALL_ALUWR`), distinct from
`ControlStall` (branch resolution) and from a branch-flag-distance pattern. They are separate
stalls; do not conflate them.

### ControlStall — branch resolution delay

**Meaning:** the `jmpi` branch-condition evaluation latency. Branch resolution is on the
order of ~23 cycles (IGC BRANCH latency). A kernel with many `jmpi` accumulates a few
cycles each.

**Diagnosis:** smallest contributor in most kernels; proportional to the number of `jmpi`.
Treat it as a byproduct of branchy / switch-based code, and usually fixed as a side effect
of removing the branchiness (e.g. replacing a runtime switch with compile-time dispatch).

### SendStall — SEND backpressure

**Meaning:** the send pipe cannot accept a new store/load because of backpressure (rare on
modern setups with no backpressure on small stores). If non-zero, check for uncoalesced
stores generating many small transactions.

---

## Aggregate-column cross-reference

| Sampling column | Related `XVE_STALL_*` (VectorEngineStalls) | Family |
|-----------------|-------------------------------------------|--------|
| `SbidStall` | `XVE_STALL_SBID` | memory (load result) |
| `SendStall` | `XVE_STALL_SENDWR`, `XVE_STALL_SBID` | memory (store/send backpressure) |
| `InstrFetchStall` | `XVE_STALL_INSTFETCH` | non-memory (I-cache/control) |
| `PipeStall` | `XVE_STALL_PIPESTALL` | non-memory (pipe/GRF) |
| `DistStall` | `XVE_STALL_ALUWR` (ALU write-back dependency) | non-memory (data dependency distance) |
| `ControlStall` | `XVE_STALL_CONTROL` | non-memory (branch resolution) |
| `BarrierStall` | `XVE_STALL_BARRIER` | synchronization |

Correspondence notes:
- **`DistStall` == `XVE_STALL_ALUWR`.** The sampling-column `DistStall` is the same stall as
  aggregate `XVE_STALL_ALUWR` (waiting on an ALU write-back / a producer-to-consumer dependency
  distance); it is a data-dependency stall, NOT a branch-distance or control-flow stall.
- **`SbidStall` == `XVE_STALL_SBID`** (waiting on an in-flight memory / scoreboard).
- The two families must NOT be compared directly across sampling-vs-aggregate: aggregates are
  `%` of total stall *time*, sampling counts are event counts per IP, and they sample different
  windows, so a given stall may appear at different relative magnitudes.

The aggregate `XVE_STALL_*` percentages (instructions-measurement/xpu) tell you *how much*
of each stall across the whole kernel; the sampling data tells you *which specific
instruction address* each stall came from. Combine them: use the aggregate to prioritize
which stall family matters, use the sampling to locate the responsible instruction and
its source line.