---
name: register-spill-bank-conflict
description: Analyze GPU kernel register pressure, spills, and register-file bank/bundle conflicts. Determines whether a kernel spills registers to memory (capacity limit) or stalls on conflicting register reads (port limit). Use after occupancy analysis and before arithmetic intensity / Roofline.
---

# Register Spill and Bank-Conflict Analysis

**Goal:** Determine whether the dominant kernel is limited by register-file pressure. Two distinct failure modes:

1. **Register spill (capacity limit):** the kernel needs more physical registers than the thread's register-file budget, so the compiler spills live values to local memory. Spawning extra memory traffic and (on some devices) disabling/increasing a large-register-file mode or halving occupancy.
2. **Bank/bundle conflict (port limit):** even with no spilling, a single instruction that reads 2-3 operands can be serialized when the operands map to the same register-file bank/bundle, adding stall cycles per instruction.

Run this after Step 4 (occupancy) because register pressure can also change occupancy (a large-register-file mode halving thread slots) -- tie those results together.

## REQUIRED INPUTS

| Input | Source | Notes |
|-------|--------|-------|
| `$RUN_DIR/01_kernel_profiler_setup.json` | Step 1 | device, vendor_tool |
| `$RUN_DIR/03_kernel_profiler_parser.json` | Step 3 | dominant_kernel_name, stalls, pipe util |
| `$RUN_DIR/04_kernel_occupancy.json` | Step 4 | registers_per_thread, large_register_file, spill |
| Register / spill / private-memory values for the dominant kernel | Vendor profiler (see vendor sub-skill) | register file size per thread, spill bytes |
| Stall data | Step 3 | distinguish pipe stall from data dependency |

Read all prior JSON files.

## Execution context

| Action | Where |
|--------|-------|
| Read prior step JSON files | REMOTE |
| Read kernel source code | LOCAL (PyTorch source tree) |
| Enable compiler dump and parse register/bank-conflict annotations (vendor-specific) | REMOTE |
| Compute spill/conflict impact | LOCAL |
| Write step JSON/log to `$RUN_DIR` | REMOTE |
| Verification | REMOTE |

## Procedure

### 1. Detect register spill

From the **vendor profiler** output (e.g. the per-kernel properties it reports), get:
- `registers_per_thread` (register-file size per thread)
- `spill_memory_per_thread` (bytes spilled)

The exact counter / field names and where to read them are vendor-specific -- follow the
vendor sub-skill.

Rules:
- `spill_memory == 0` -> no spilling; capacity is fine.
- `spill_memory > 0` -> the kernel spilled. Spilling adds hidden local-memory traffic and instruction overhead (spill stores/reloads), which shows up as extra memory/ALU instructions and memory-scoreboard stalls. This is a strong signal to reduce live register pressure (smaller WG, less unrolling, fewer simultaneous accumulators).

Also note: high register usage that stays under the budget may still force a **large-register-file**
mode (e.g. so called large-GRF on Intel, or similar large-register configs elsewhere), which halves
thread slots and lowers occupancy (cross-check with Step 4).

### 2. Detect register-file read conflicts

Register-file read conflicts, when many operands share the same read port, cause **pipe
stalls** (not dependency stalls). The exact mechanism (bank/bundle layout, port sharing,
read-suppression behavior) is **vendor-specific** -- see the vendor sub-skill for the model
and how to detect conflicts (e.g. compiler bank-conflict annotations, or mapped formulas).

### 3. Correlate with measured stalls

Compare the register/pipe-stall findings with Step 3 measured stalls to see whether the
conflict explains a meaningful share of the pipe stall. The specific stall counters to compare
and how to attribute them are **vendor-specific** -- follow the vendor sub-skill's guidance.

### 4. Decide fixability

- **Algorithmically independent sources** landing in the same bank -> a different register assignment may fix it; enable the vendor's register-allocation / bank-conflict-reduction passes.
- **Same value read twice** -> hardware read suppression should already handle it; check scheduling.
- **Spill** -> reduce register pressure (smaller WG, split loops, fewer accumulators).

The precise definition of a conflict (including any multi-source / 3-source cases) and the
specific fixes for it are **vendor-specific** -- see the vendor sub-skill.

## REQUIRED OUTPUTS

### `$RUN_DIR/05_register_bank_conflict.json`

```json
{
  "step": "05_register_bank_conflict",
  "dominant_kernel_name": "<kernel_name>",
  "launch_params": {
    "registers_per_thread": <int>,
    "spill_memory_per_thread_bytes": <int>,
    "private_memory_per_thread_bytes": <int>,
    "large_register_file": <bool>
  },
  "spill_analysis": {
    "spilled": <bool>,
    "spill_bytes_per_thread": <int>,
    "impact": "<none|minor|spill-limited>",
    "recommendation": "<string>"
  },
  "conflict_analysis": {
    "bank_conflict_present": <bool>,
    "conflict_instructions_found": <int or null>,
    "max_bc_cycles_per_instr": <int or null>,
    "biases": "<register-layout/conflict model summary>",
    "pipe_stall_correlation_pct": <float or null>
  },
  "overall_assessment": "<string>",
  "run_dir": "<$RUN_DIR>"
}
```

### `$RUN_DIR/05_register_bank_conflict.log`

Human-readable breakdown: register budget, spill bytes, register-layout/conflict model, any compiler-dump summary.

## VERIFICATION

**Run all verification commands via SSH on the target machine (they access `$RUN_DIR` which is remote).**

```bash
test -f $RUN_DIR/05_register_bank_conflict.json && echo "JSON OK" || echo "JSON MISSING"
python3 -c "
import json
d = json.load(open('$RUN_DIR/05_register_bank_conflict.json'))
required = ['launch_params', 'spill_analysis', 'conflict_analysis', 'overall_assessment']
missing = [k for k in required if k not in d]
assert not missing, f'Missing fields: {missing}'
assert 'registers_per_thread' in d['launch_params']
assert 'spilled' in d['spill_analysis']
print(f'VERIFICATION PASSED: spilled={d[\"spill_analysis\"][\"spilled\"]}, bc_present={d[\"conflict_analysis\"][\"bank_conflict_present\"]}')
"
```

## Vendor-specific details

See the vendor sub-skill (e.g., `xpu/SKILL.md`) for the register-file bank/bundle formulas, compiler-dump setup, and bank-conflict annotation interpretation.