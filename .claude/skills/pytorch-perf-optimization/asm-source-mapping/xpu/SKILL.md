---
name: asm-source-mapping-xpu
description: Intel XPU-specific assembly extraction and source mapping. Classifies kernel codegen path, extracts GPU ISA, and maps instruction addresses back to source.
---

# XPU Assembly Extraction and Source Mapping

XPU-specific addendum for `asm-source-mapping`.

## Classifying the codegen path

| Kernel signal | Path | How to extract ISA |
|---------------|------|-------------------|
| `gemm_kernel` / `gen_conv_kernel` | oneDNN ngen | `ONEDNN_VERBOSE=2` + ngen debug |
| `triton_*` | Triton through IGC | `IGC_ShaderDumpEnable=1` |
| `_ZTS...` AND Compiled = AOT | SYCL AOT | `DumpZEBin=1` + `ocloc disasm` |
| `_ZTS...` AND Compiled = JIT | SYCL JIT | `IGC_ShaderDumpEnable=1` |

For `_ZTS...` kernels, check the `kernel_properties.compiled` field from Step 2.

## Probing with IGC (for JIT kernels)

```bash
rm -rf /tmp/igc_probe && mkdir -p /tmp/igc_probe
IGC_ShaderDumpEnable=1 IGC_ShaderDumpPidDisable=1 IGC_DumpToCustomDir=/tmp/igc_probe python $RUN_DIR/bench_<op>.py profile 2>/dev/null
ls /tmp/igc_probe/*.asm
```

## Extracting ISA from AOT zebin

```bash
# Find the zebin in the PyTorch installation
find $(python -c "import torch; print(torch.__path__[0])") -name "*.spv" -o -name "*.bin" | head

# Disassemble
DumpZEBin=1 python $RUN_DIR/bench_<op>.py profile
ocloc disasm -file <zebin_path> -dump /tmp/isa_dump
```

## Source mapping methods (priority order)

1. **Source code inspection**: Read the kernel source file identified in Step 2. Match patterns in the source to the pipe/stall breakdown from Step 7. This is often sufficient for PyTorch native kernels.

2. **Zebin .debug_line**: For AOT/JIT zebin ELFs compiled with `-g`, use standard DWARF line tables.

3. **IGC inline comments**: In JIT dumps, look for `// Line N:` or `// file:line` annotations.

4. **Pattern recognition**: For oneDNN ngen or `-g`-less builds, match instruction patterns to known code structures.

## Correlating stall-sampling IPs

If stall-sampling data was collected in Step 3:

1. Parse the stall-sampling raw log for the dominant kernel.
2. Aggregate stall events by `IP[Address]`.
3. Map the top IPs to assembly instructions in the ISA dump.
4. Map assembly back to source using the method above.
5. Focus on the hottest loop.

## Common hot patterns on XPU

| Assembly pattern | Source cause | Lever |
|-----------------|-------------|-------|
| `rem`, `div` in inner loop | `index % N`, `index / N` | IntDivider (mul_hi + shift) |
| 64-bit `add`/`mul` for addresses | `int64_t` index, pointer math | Narrow to 32-bit |
| Dense `dpas` chain | GEMM tile body | Tune tile parameters |
| Scattered `send.ugm` | Uncoalesced memory access | Improve locality, pack types |
| Many `cmp`/`jmp` | Divergent branches | Predicate, unify |

## Cautions

- oneDNN ngen kernels have no DWARF; source mapping is pattern-only.
- Only optimize addresses that accumulate the most samples.
- Optimizations may inline/unroll/reorder; source lines are approximate.
