# Extract ASM from AOT-Compiled SYCL Binaries

GPU ISA is compiled at build time by IGC and embedded in the host binary. `IGC_ShaderDumpEnable` does **not** work because IGC is not invoked at runtime.

## When to use

Kernel is a mangled SYCL symbol (`_ZTS…`) from an AOT binary whose AOT target matches the current device:
- `libtorch_xpu.so` / `libtorch_xpu_ops.so`
- SYCL-TLA FMHA `.so`
- Standalone DPC++ executable built with `-fsycl-targets=spir64_gen -Xs "-device <gpu>"`

## Steps

1. **Verify `ocloc`**:
   ```bash
   command -v ocloc >/dev/null || { echo "source oneapi-vars.sh"; exit 1; }
   ```

2. **Dump zebin ELFs at runtime**:
   ```bash
   OUT="<workdir>/aot_dump"
   mkdir -p "$OUT" && cd "$OUT"
   DumpZEBin=1 NEOReadDebugKeys=1 <repro_cmd>
   ls *.elf
   ```

3. **Disassemble**:
   ```bash
   DEVICE=$(sycl-ls 2>/dev/null | grep -oP '(?<=\[)[^]]+' | head -1 | awk '{print tolower($NF)}')
   DEVICE=${DEVICE:-bmg}
   for elf in *.elf; do
     name=$(basename "$elf" .elf)
     ocloc disasm -file "$elf" -dump "${name}_dump" -device "$DEVICE"
   done
   ```

4. **Identify the target kernel**:
   ```bash
   for f in *_dump/.text._ZTS*.asm; do
     mangled=$(basename "$f" .asm | sed 's/^\.text\.//')
     echo "$f  →  $(c++filt "$mangled")"
   done
   ```

## Source mapping

AOT zebin may contain `.debug_line` (if built with `-g`). Use Method 1 in `source-mapping-methods.md`; otherwise Method 2/3.
