---
name: asm-source-mapping
description: Device-agnostic workflow for mapping hot GPU assembly or stall/sample IPs back to source code. Use after profiling has identified a dominant kernel/pipe/stall.
---

# Assembly-to-Source Mapping

**Goal:** Find the source code that corresponds to the hot instructions of a GPU kernel.

## REQUIRED INPUTS

| Input | Source | Notes |
|-------|--------|-------|
| `$RUN_DIR/01_kernel_profiler_setup.json` | Step 1 | vendor_tool |
| `$RUN_DIR/02_host_vs_device_bound.json` | Step 2 | dominant_kernel_name, kernel_source_file |
| `$RUN_DIR/03_kernel_profiler_parser.json` | Step 3 | raw logs |
| `$RUN_DIR/04_kernel_arithmetic_intensity.json` | Step 4 | op context |
| `$RUN_DIR/05_kernel_memory_compute_bound.json` | Step 5 | bound_type |
| `$RUN_DIR/06_memory_load_store.json` | Step 6 | memory metrics |
| `$RUN_DIR/07_instructions_measurement.json` | Step 7 | dominant_pipe, stall_breakdown |
| PyTorch source tree | Local repo | For source file inspection |

Read ALL prior JSON files.

## Execution context

| Action | Where |
|--------|-------|
| Read prior step JSON files | REMOTE |
| Read kernel source code (`.cpp`/`.h` files) | LOCAL (PyTorch source tree) |
| Grep for kernel name in source tree | LOCAL |
| Dump ISA / run `ocloc disasm` (if needed) | REMOTE |
| Parse stall-sampling IP data | REMOTE |
| Write step JSON/log to `$RUN_DIR` | REMOTE |
| Verification | REMOTE |

## Procedure

### 1. Use the already-known kernel name

The dominant kernel name was identified in Step 2. Use it directly; do not re-profile.

### 2. Classify the codegen path

| Kernel signal | Path |
|---------------|------|
| `gemm_kernel` / `gen_conv_kernel` / `mkldnn::*` | oneDNN ngen |
| `triton_*` / `triton_per_fused_*` | Triton through IGC |
| `_ZTS...` AND Compiled = AOT | SYCL AOT |
| `_ZTS...` AND Compiled = JIT | SYCL JIT |

Use the `kernel_properties.compiled` field from Step 2 JSON.

### 3. Inspect the kernel source code

Using the `kernel_source_file` from Step 2 (or by grepping the source tree), read the kernel source code and identify:
- The inner loop structure
- Index decomposition patterns (division/modulo from linear index)
- Memory access patterns (strided, coalesced, random)
- Any obvious inefficiencies

### 4. Map hot pipe/stall to source patterns

Using the dominant pipe and stall breakdown from Step 7:

| Hot pattern | Likely source-level cause | Lever |
|-------------|--------------------------|-------|
| INT div/mod in inner loop | Index decomposition from linear index | IntDivider, reparameterize grid, vectorize |
| 64-bit pointer arithmetic | 64-bit indices where not needed | Narrow to 32-bit |
| Many FP math ops | Redundant elementwise math | Remove redundancy, lower precision |
| Uncoalesced/scattered loads | Bad memory access pattern | Improve locality, pack, tile |
| Branches clustered | Divergent control flow | Predicate, unify branches |

### 5. Optionally extract and disassemble ISA

If source-level inspection is insufficient, dump the GPU ISA:
- On XPU: use `IGC_ShaderDumpEnable=1` or `DumpZEBin=1` + `ocloc disasm`
- On NVIDIA: compile with `-lineinfo`, use `cuobjdump -sass`
- On AMD: `llvm-objdump -l`

### 6. Correlate stall-sampling IPs (if available)

If `--stall-sampling` data was collected:
1. Aggregate stall events by IP for the hot kernel.
2. Look up each IP in the ISA dump from step 5.
3. Focus on the loop/instruction accumulating the most samples.

## REQUIRED OUTPUTS

### `$RUN_DIR/08_asm_source_mapping.json`

```json
{
  "step": "08_asm_source_mapping",
  "dominant_kernel_name": "<kernel_name>",
  "scenario": "<onednn|triton|sycl-aot|sycl-jit>",
  "kernel_source_file": "<file:line>",
  "hot_source_locations": ["<file:line description>"],
  "hot_patterns_found": ["<pattern description>"],
  "recommended_levers": ["<lever 1>", "<lever 2>"],
  "mapping_confidence": "<high|medium|low>",
  "asm_file": "<path to ISA dump or null>",
  "run_dir": "<$RUN_DIR>"
}
```

### `$RUN_DIR/08_asm_source_mapping.log`

Human-readable source mapping report with the kernel source code excerpts showing the hot patterns.

## VERIFICATION

**Run all verification commands via SSH on the target machine (they access `$RUN_DIR` which is remote). Write any output files locally to `/tmp/opencode/` first, then SCP to `$RUN_DIR`.**

```bash
test -f $RUN_DIR/08_asm_source_mapping.json && echo "JSON OK" || echo "JSON MISSING"
test -f $RUN_DIR/08_asm_source_mapping.log && echo "LOG OK" || echo "LOG MISSING"
python3 -c "
import json
d = json.load(open('$RUN_DIR/08_asm_source_mapping.json'))
required = ['scenario', 'hot_source_locations', 'recommended_levers', 'mapping_confidence']
missing = [k for k in required if k not in d]
assert not missing, f'Missing fields: {missing}'
print(f'VERIFICATION PASSED: scenario={d[\"scenario\"]}, confidence={d[\"mapping_confidence\"]}')
"
```

## Cautions

- Optimized code may inline, unroll, or reorder; source lines are approximate.
- For library kernels (oneDNN, cuDNN), mapping may require library build artifacts.
- Only optimize addresses that accumulate the most samples, not every address.
