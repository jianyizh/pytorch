# Extract ASM from JIT-Compiled SYCL Kernels

Same stack as AOT (`SYCL → LLVM IR → SPIR-V → IGC → zebin`), but the zebin is generated at first launch and only exists in memory. Capture it with `IGC_ShaderDumpEnable=1`.

## When to use

- Kernel is `_ZTS…` AND binary has no AOT bundle for the current device (e.g. built for PVC, running on BMG → runtime JIT).
- `torch-xpu-ops` built with `TORCH_XPU_ARCH_LIST=none`.
- Standalone DPC++ without `-fsycl-targets=spir64_gen`.

## Steps

1. **Pin the launched kernel**:
   ```bash
   unitrace -d <repro_cmd>
   # or fallback:
   SYCL_UR_TRACE=-1 <repro_cmd> 2>&1 | grep -oP 'pKernelName = 0x[0-9a-f]+ \(\K[^)]+' | sort -u
   ```

2. **Re-run with IGC dump (cold cache)**:
   ```bash
   OUT="<workdir>/jit_dump"
   mkdir -p "$OUT/igc"
   NEO_CACHE_PERSISTENT=0 \
   IGC_ShaderDumpEnable=1 \
   IGC_DumpToCustomDir="$OUT/igc" \
   ONEAPI_DEVICE_SELECTOR=level_zero:0 \
     <repro_cmd> 2>&1 | tee "$OUT/run.log"
   ```

3. **Match kernel name to `.asm` file**:
   ```bash
   MATCH=$(grep -l "$KERNEL" "$OUT/igc"/OCL_asm*_simd*_entry_*.asm | head -1)
   echo "asm-file: $MATCH"
   ```

## Source mapping

JIT dump `.asm` files usually contain inline `// file:line` comments (Method 2 in `source-mapping-methods.md`). If not, use Method 1 (zebin `.debug_line`) or Method 3.
