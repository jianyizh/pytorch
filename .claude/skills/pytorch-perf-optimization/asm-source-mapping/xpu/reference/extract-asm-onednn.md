# Extract ASM from oneDNN ngen-JIT Kernels

**Only codegen path that bypasses SYCL/SPIR-V/IGC.** oneDNN uses its own native generator (ngen) that directly emits GPU ISA bytes. No zebin ELF, no `.debug_line`.

## When to use

Op dispatched through oneDNN (`mkldnn::*`): `linear`, `matmul`, `mm`, `bmm`, `conv*`, `_scaled_dot_product_attention`.

## Locate `libiga64.so`

```bash
ONEAPI=${ONEAPI_ROOT:-${CMPLR_ROOT:+${CMPLR_ROOT%/*}}}
if [ -z "$ONEAPI" ]; then
  for d in /opt/intel/oneapi ~/intel/oneapi /usr/local/oneapi; do
    [ -d "$d" ] && ONEAPI="$d" && break
  done
fi
IGA_LIB=$(find "$ONEAPI" -name 'libiga64.so' 2>/dev/null | head -1)
```

## Steps

1. **Dump raw ISA**:
   ```bash
   OUT="<workdir>/onednn_dump"
   mkdir -p "$OUT" && cd "$OUT"
   ONEDNN_JIT_DUMP=1 ONEAPI_DEVICE_SELECTOR=level_zero:0 <repro_cmd>
   ls dnnl_dump_gpu_*.bin
   ```

2. **Disassemble with IGA ctypes**:
   ```bash
   for bin in dnnl_dump_gpu_*.bin; do
     name=$(basename "$bin" .bin)
     python3 -c "
import ctypes, sys, pathlib
iga = ctypes.CDLL('$IGA_LIB')
raw = pathlib.Path('$bin').read_bytes()
buf = ctypes.create_string_buffer(1 << 20)
# Platform IDs: BMG/Xe2=0x2000000, PVC=0x30000, DG2=0x30004
iga.iga_disassemble(0x2000000, raw, len(raw), buf, len(buf))
sys.stdout.write(buf.value.decode())
" > "${name}.asm"
   done
   ```

3. **Pin the invoked kernel**:
   ```bash
   ONEDNN_VERBOSE=1 <repro_cmd> 2>&1 | grep -E '^onednn_verbose.*exec' | nl -ba
   # Nth (1-indexed) exec line → dnnl_dump_gpu_*_kernel.<N-1>.bin
   ```

   The largest `.bin` is typically the GEMM kernel. Cross-check with `grep -c dpas <asm>`.

## Source mapping

No DWARF; use pattern recognition only (see `source-mapping-methods.md`, Method 3).
