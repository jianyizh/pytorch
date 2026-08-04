# XPU ASM → Source Mapping Methods

Given one or more GPU instruction addresses (IPs) and an ASM/zebin directory, resolve each IP to the original source file:line.

## Inputs/Outputs

- **Input:** IP offsets from `unitrace --stall-sampling` or a profile, plus the zebin/asm directory produced by IGC/SYCL dumps.
- **Output:** Mapped source file:line entries, including the method used (`debug-line`, `inline-comment`, or `pattern-recognition`).

## Method priority (strict waterfall)

```text
Step 1: Try zebin .debug_line          → method = "debug-line"
         ↓ (not found)
Step 2: Try IGC ASM inline comments    → method = "debug-line"
         ↓ (not found)
Step 3: Fallback: pattern recognition  → method = "pattern-recognition"
```

Rule: if Method 1 succeeds, STOP. Only run pattern recognition if no debug info exists.

## Method 1 — Zebin `.debug_line` (primary)

Requires the kernel compiled with `-g` (not `-gline-tables-only` for AOT `spir64_gen`).

1. Locate the dumped zebin ELF (e.g. `dumped_zebin_module_N.elf` from `DumpZEBin=1`).
2. Verify the section exists:
   ```bash
   readelf -S "$ZEBIN" | grep -q .debug_line || { echo "NO debug_line"; exit 1; }
   ```
3. Decode the line table:
   ```bash
   readelf --debug-dump=decodedline "$ZEBIN" 2>/dev/null
   ```
   Ignore warnings about unknown reloc types for `e_machine=205`.
4. Build a sorted lookup `(offset, file, line)` and floor-lookup the IP offset.
5. The `.asm` file uses labels `L<N>` where `N` is the decimal byte offset in `.text`. Map IP offset → label → ASM line.

## Method 2 — IGC ASM inline comments (secondary)

JIT-compiled kernels (`IGC_ShaderDumpEnable=1`) produce `.asm` with inline source comments:

```asm
// Format A (newer IGC, requires -g):
// Line 8:  int val = buf[it.get_global_id(0)];
(W) send.ugm (32|M0) r16 r14 ...

// Format B (older IGC):
(W) add (M1, 16) r14.0<1>:d ...  // shift_reduce.cpp:93
```

Walk backward from the instruction IP to the nearest annotated line.

## Method 3 — Pattern recognition (fallback only)

Use only when `-g` is unavailable or for oneDNN ngen (no DWARF).

| Instruction pattern | Source construct |
|---|---|
| Dense `dpas` chain, 8× repeat | GEMM tile (matmul accumulate) |
| `mul :f` + broadcast src1 | Scalar rescale |
| `exp2 :f` / `log2 :f` | Softmax numerics |
| `mov :bf :f` + `store` | BF16 epilogue / output write |
| `send.slm` | SLM load/store |
| `send.slm` store + `fence.slm` + `send.slm` load + `add :f` | Workgroup reduction tree |
| `send.ugm` | Global memory load/store |
| `send.gtwy` + `sync.bar` | Barrier / sync |
| VxH indirect `mov r[a0.0]` | Sub-group shuffle |

## Output format

```json
[
  {
    "ip": "0x238",
    "asm_offset": 568,
    "asm_label": "L568",
    "sycl_file": "shift_reduce.cpp",
    "sycl_line": 93,
    "source_construct": "ALU compute (value += other)",
    "method": "debug-line"
  }
]
```
