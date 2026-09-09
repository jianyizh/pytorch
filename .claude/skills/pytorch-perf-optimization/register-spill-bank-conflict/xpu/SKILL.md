---
name: register-spill-bank-conflict-xpu
description: Intel XPU-specific guidance for analyzing register file pressure, spills, and GRF bank/bundle conflicts on Intel GPUs. Covers the IGC register-file model, bank/bundle formulas, IGC dump setup ({BC=N}), and spill detection via unitrace.
---

# Register Spill and Bank-Conflict -- Intel XPU Notes

XPU-specific addendum for `register-spill-bank-conflict`.

## Background

SIMD / SIMT execution units issue one instruction across many lanes at once. A
single ALU instruction can read several source operands, so the register file
must supply multiple operands per cycle.

The hardware has a finite number of read ports. When two or more operands need
the same port in the same cycle, the reads are serialized and the instruction
stalls. In the absence of such conflicts, the hardware can typically read one
register operand per cycle.

The exact layout of the register file (whether it is split into banks, bundles,
or other structures) is hardware-specific. The sections below describe the model
used by IGC for Intel GPUs.

---

## Intel GPU (XPU) architecture and IGC model

All of the bank/bundle terminology in this section comes from the IGC
implementation, especially the code in `visa/G4_BB.cpp`, `visa/GraphColor.h`,
`visa/LocalScheduler/LocalScheduler_G4IR.cpp`, and `visa/Passes/AccSubstitution.cpp`.

### GRF per thread

On Intel GPUs, each thread owns a private **General Register File (GRF)**.

- The GRF size is **64 bytes** on newer platforms such as PVC and Xe2+.
- The number of GRFs available to a thread is configurable; common values are
  128, 256, or 512, depending on the platform and driver settings.
- A kernel refers to GRFs by their index `r0, r1, r2, ...`. The physical bank
  and bundle of a register are derived from that index.

### Bank and bundle

To support multiple simultaneous operand reads, the GRF is divided into
**banks**, and each bank is further divided into **bundles**. A bundle is a set
of registers that share the same narrow read resource inside a bank.

The rule of thumb in IGC is:

- **Different banks** can usually be read in parallel without conflict.
- **Same bank, different bundles** can be read in parallel for up to two sources,
  but not three.
- **Same bank and same bundle** cannot be read in parallel -- this is the strongest
  conflict.

### How IGC computes bank and bundle IDs

The exact formula is **platform-specific**. Representative mappings used by IGC
are:

| Platform flavor | Bank ID | Bundle ID | Notes |
|---|---|---|---|
| XeLP (default) | `reg % 2` | `(reg % 16) / 2` | 2 banks (even/odd); 8 bundles per 16 registers |
| TGL / XeHP (`hasTwoGRFBank16Bundles`) | `(reg % 4) / 2` | `(reg % 64) / 4` | bank toggles every 2 GRFs; bundle covers 4 GRFs |
| ARL (`has64bundleSize2GRFPerBank`) | `(reg % 4) / 2` | `(reg % 32) / 4` | 64-byte GRF, 2 GRFs per bank slice |
| Xe2 / PVC 64-byte (`has64bundleSize`) | `reg % 2` | `(reg % 16) / 2`; if 512 GRF then `(reg % 32) / 2` | bundle widens for the largest register mode |

The XeLP mapping is the easiest to visualize:

```text
register index : 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
bank           : 0 1 0 1 0 1 0 1 0 1  0  1  0  1  0  1
bundle         : 0 0 1 1 2 2 3 3 4 4  5  5  6  6  7  7
```

So `r0` and `r2` are in the same bank but different bundles; `r0` and `r1` are
in different banks; `r0` and `r16` are in the same bank and the same bundle,
because the pattern repeats every 16 registers.

### Conflict types in IGC

IGC models two kinds of conflicts for 3-source instructions such as `mad`,
`add3`, and `dpas`:

1. **Bundle conflict** -- two or more source registers fall in the **same bank
and the same bundle**. This always adds a conflict cycle.
2. **Bank conflict** -- all three source registers fall in the **same bank**,
even if they are in different bundles. This also counts as a conflict.

A bundle conflict is strictly worse than a bank conflict because it implies the
bank conflict already. This is expressed explicitly in
`visa/Passes/AccSubstitution.cpp`:

```cpp
/*
 * Bank conflict types:
 *  1. any two from same bundle and same bank
 *  2. all three from same bank
 */
```

and the same model appears in the scheduler conflict check in
`visa/LocalScheduler/LocalScheduler_G4IR.cpp`.

In short, a 3-source instruction counts as a bank conflict on XPU when either:
1. **two or more source registers fall in the same bank AND the same bundle**
   (a *bundle* conflict -- always adds a conflict cycle), or
2. **all three source registers fall in the same bank**, even if in different
   bundles (a *bank* conflict).

A bundle conflict implies the bank conflict, so it is strictly worse. The fix here is NOT
simply splitting the 3-source op into two 2-source instructions; IGC instead mitigates with
register-allocation and scheduling passes (`BankConflictPass`, `EnableBCR`,
`EnableGroupScheduleForBC`), read-suppression awareness, and HW swap for `src1`/`src2` (see
below) -- trust the `{BC=N}` compiler annotation as authoritative.

### Hardware helpers: read suppression and HW swap

IGC models two hardware mechanisms that reduce the number of actual conflicts:

- **Read suppression:** if a GRF was read by the previous instruction, the value
  may be kept in a small per-source cache and reused without another GRF read.
  There is both *inter-instruction* suppression (between instructions) and
  *intra-instruction* suppression (the same register used twice inside one
  instruction).
- **HW swap (XeLP):** for a 3-source instruction, the default read order is
  `src0` first, then `src1` and `src2`. If `src1` and `src2` conflict, the
  hardware can swap the read order to avoid the stall.

### How IGC mitigates bank conflicts

IGC uses several techniques to reduce bank conflicts:

- **Register allocation (RA):** allocate live ranges so that values that are read
  together land in different banks or bundles. This is the main job of the
  `BankConflictPass`.
- **Instruction scheduling:** reorder independent instructions so that registers
  read in adjacent cycles do not share the same bundle.
- **Pattern-match heuristics:** avoid converting code into 3-source instructions
  if the transformation increases register pressure and makes bank conflicts more
  likely.
- **Read suppression awareness:** take the hardware suppression cache into
  account so that repeated reads are not counted as conflicts.
- **Bank/bundle-aware spilling and layout:** choose spill slots and variable
  offsets to spread accesses across banks.

### IGC flags that influence bank-conflict handling

| Flag | Description |
|---|---|
| `EnableBCR` | Enable bank-conflict reduction during register allocation. |
| `ForceBCR` | Force bank-conflict reduction even when spilling would increase. |
| `EnableGroupScheduleForBC` | Enable bank-conflict reduction in the scheduler. |
| `AvoidSrc1Src2Overlap` | Avoid `src1`/`src2` GRF overlap when read suppression cannot help. |
| `AvoidDstSrcGRFOverlap` | Avoid destination/source GRF overlap for wide SIMD instructions. |
| `TotalGRFNum` | Total GRF setting used by both the LLVM and vISA phases. |

These are documented in `documentation/configuration_flags.md`.

---

## Using IGC dumps to spot bank conflicts

### Enabling the dump

The key vISA option is `-dumpAllBCInfo`. Without it, IGC only prints `{BC=N}`
when `N > 0`. With it, every instruction gets a `{BC=...}` annotation, so you
can also confirm that nearby instructions are clean.

#### ocloc / OpenCL path

```bash
ocloc compile ... -options "-igc_opts 'VISAOptions=-dumpAllBCInfo -asmToConsole'"
```

#### SYCL / Level Zero / DPC++ (JIT or AOT)

You cannot pass `VISAOptions` through the SYCL front-end flags directly. Use the
IGC regkey mechanism instead:

```bash
export IGC_ShaderDumpEnable=1
export IGC_VISAOptions="-dumpAllBCInfo"

# run your SYCL application, or AOT-compile with icpx
icpx -fsycl -fsycl-targets=spir64_gen -Xs "-device pvc" source.cpp -o a.out
```

For JIT, set the variables before launching the application; for AOT, set them
before the compile step.

### What the dump looks like

A typical annotated line looks like this:

```text
mad (8) r10 r20 r30 r40 {E:0,O:1, E:2} R{} IR{} {BC=0}
mad (8) r11 r20 r30 r50 {E:0,O:1, E:0} R{} IR{} {BC=1}
```

Fields before `{BC=N}` include:

- `{E:..., O:..., ...}` -- the bank and bundle membership of each source after
  read suppression. `E` = even bank, `O` = odd bank; the number is the bundle ID.
- `R{...}` -- inter-instruction read suppression: registers that did not need to
  be re-read because they were cached from the previous instruction.
- `IR{...}` -- intra-instruction read suppression: registers reused within the
  same instruction.

### Interpreting `BC=N`

- `{BC=0}` -- no modeled bank conflict on that instruction.
- `{BC=1}` -- one conflict cycle. In a hot loop, this usually means one extra
  cycle of latency for that instruction.
- Higher numbers appear when compressed SIMD16/SIMD32 instructions are split
  into multiple physical SIMD8/SIMD16 reads and each part has conflicts.

### Practical workflow

1. **Find the hot kernel.** Use PyTorch Profiler (or equivalent) to identify
   which kernel/loop dominates the runtime.
2. **Confirm the stall is a pipe stall.** Because a GRF conflict causes a pipe
   stall, use a tool such as **unitrace stall sampling** to check whether the
   hot kernel shows stalls at the relevant execution point.
3. **Get the IGC assembly dump.** Re-compile / re-run the kernel with
   `IGC_VISAOptions="-dumpAllBCInfo"` and `IGC_ShaderDumpEnable=1`.
4. **Search for conflicts.** Open the generated `.asm` dump and search for
   `{BC=`. Cross-reference the conflicting instructions with the stall locations
   from unitrace.
5. **Decide whether the conflict is fixable:**
   - If the conflicting sources are algorithmically independent, a different
     register assignment may solve it -- try enabling `EnableBCR` or
     `EnableGroupScheduleForBC`.
   - If the same value is read twice, the compiler may already be able to use
     read suppression; check whether better scheduling helps.
   - If the conflict comes from a 3-source pattern (e.g., `add3`, `bfn`),
     sometimes splitting the pattern back into two 2-source instructions
     reduces conflicts at the cost of extra instructions.

---

## Quick reference: does my access conflict?

Use the platform-specific formulas from the table above. For the common XeLP
case:

```text
bank(reg)   = reg % 2
bundle(reg) = (reg % 16) / 2
```

Then:

| Access pattern | Conflict? |
|---|---|
| 2 sources, different banks | No |
| 2 sources, same bank, different bundles | No (in the 3-source ALU model) |
| 2 sources, same bank, same bundle | Yes (bundle conflict) |
| 3 sources, all same bank | Yes (bank conflict) |
| 3 sources, all same bank and same bundle | Yes, multiple bundle conflicts |

Remember that **read suppression** can remove a source from the conflict check
if it was just read, and **HW swap** can sometimes hide conflicts between
`src1` and `src2` on older platforms. The dump already accounts for these
mechanisms, so `{BC=0}` is the authoritative IGC model output.

---

## Detecting register spill on XPU

From the Step 1 verbose timeline (`unitrace -d -v`), read the per-kernel
properties row: **Spill Memory Per Thread** and **Private Memory Per Thread**.

- `Spill Memory Per Thread == 0` -> no spilling.
- `Spill Memory Per Thread > 0` -> live values spilled to local memory. This
  adds hidden SEND (load/store) traffic and ALU/address instructions, and often
  shows up as elevated SBID stalls in Step 3. Cross-check: increased SEND pipe
  utilization and memory-scoreboard stalls without matching DRAM traffic are a
  symptom of spills.
- **Register File Size Per Thread** (>= unitrace 2.4.0) tells the compile-time
  register budget: 128 (default) vs 256 (large-GRF). A spilled kernel that stays
  under budget may instead force large-GRF, halving thread slots -> tie back to
  Step 4 occupancy.

When the kernel is AOT-compiled (framework-native SYCL), IGC dumps are typically
not available at runtime; rely on the unitrace spill counters plus the 
bank/bundle model derived from the known register indices in the source/assembly.