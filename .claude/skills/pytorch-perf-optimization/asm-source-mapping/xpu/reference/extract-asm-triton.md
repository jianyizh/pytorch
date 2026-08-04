# Extract ASM from Triton Kernels on XPU

Triton on XPU uses the same IGC backend as SYCL JIT:
`Triton IR → ttgir → SPIR-V → IGC (JIT) → zebin`.
Extraction is mechanically identical to SYCL JIT: use `IGC_ShaderDumpEnable=1` at runtime.

## When to use

- Hot kernel is `triton_per_fused_*`, `triton_poi_*`, `triton_red_*` from `torch.compile` / Inductor.
- Standalone `@triton.jit` kernel in a Python script.

## Steps

1. **Identify the kernel name**.

   For Inductor fusion:
   ```bash
   TRITON_CACHE="${TRITON_CACHE_DIR:-$HOME/.triton/cache}"
   rm -rf "$TRITON_CACHE" /tmp/torchinductor_$USER
   TORCH_LOGS=output_code python <repro.py> 2>&1 \
     | grep -oE 'triton_(poi|per|red)_fused_[A-Za-z0-9_]+' | sort -u
   ```

   For standalone `@triton.jit`:
   ```bash
   unitrace -d python <repro.py>
   ```

2. **Re-run with IGC dump (cold cache)**:
   ```bash
   TRITON_CACHE="${TRITON_CACHE_DIR:-$HOME/.triton/cache}"
   rm -rf "$TRITON_CACHE" /tmp/torchinductor_$USER
   OUT="<workdir>/triton_dump"
   mkdir -p "$OUT/igc"

   IGC_ShaderDumpEnable=1 \
   IGC_DumpToCustomDir="$OUT/igc" \
   ONEAPI_DEVICE_SELECTOR=level_zero:0 \
     python <repro.py> 2>&1 | tee "$OUT/run.log"
   ```

3. **Match kernel name to `.asm`**:
   ```bash
   MATCH=$(grep -l "$NAME" "$OUT/igc"/OCL_asm*_simd*_entry_*.asm | head -1)
   echo "asm-file: $MATCH"
   ```

   Cross-check: `.zeinfo` reports `simd_size`; confirm it matches `num_warps * 32`.

## Source mapping

Same as SYCL JIT: inline `// file:line` comments in the IGC dump usually work (Method 2); otherwise Method 1 or Method 3.
