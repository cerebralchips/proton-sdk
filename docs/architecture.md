# Architecture

## Placement and ownership

`proton-sdk` is the software product repository under Cerebral Chips. `proton-npu`
continues to own hardware RTL, hardware bare-metal smoke tests, ISA documentation,
and the simulator. This separates compiler dependencies and model artifacts from
RTL development while making their compatibility explicit in a target manifest.
The repositories are published under the
[Cerebral Chips organization](https://github.com/cerebralchips). Hardware/software
compatibility is determined by the pinned revision, not by the latest branch tip.

## First deployment

Host: pinned checkpoint -> independent reference + quantization -> MLIR -> IREE
compiler -> VMFB and statically linked RISC-V dispatch code -> bare-metal ELF.
Target: application/token loop -> IREE VM + synchronous local HAL -> dispatches
-> versioned kernel ABI -> Proton matrix MMIO and custom-1 instructions.

The Linux VM hosts build tools only. The target runs bare metal on the real CVA6
pipeline with Ara and the matrix extension enabled. Linux on Proton is deferred.
The first model has batch size one and a bounded KV cache. FP32 scalar code handles
operations not supported by the current INT8 matrix engine. The first integration
uses explicit custom CPU dispatches, not a new MLIR dialect or a claim of automatic
mapping of arbitrary models. General graph import and pattern-based rewrites can
grow independently of the target kernel contract.

## Frontend compiler boundary

New model frontends produce **Torch MLIR** as input to `iree-compile`. ONNX is
optional at the SDK boundary. The first implemented frontend exports Stories260K
to ONNX, imports it with `iree-import-onnx`, and qualifies FP32/W8A8 host execution
with explicit KV inputs/outputs. PyTorch/IREE Turbine can be a later frontend.
See [frontends](frontends.md) for the measured host-only contract. This path does
not yet invoke Proton custom dispatches or establish new RTL execution evidence.

## Extension boundaries

- `targets/`: hardware capabilities, memory, ISA and compatibility pins.
- `kernels/`: semantic kernel contracts and target implementations.
- `compiler/`: model import, MLIR generation and IREE lowering integration.
- `runtime/`: allocator, startup, linker layout, IREE embedding and token loop.
- `models/`: checkpoint/tokenizer acquisition and reproducible reference execution.
- `scripts/`: reproducible environment/build/run/evidence commands.
- `tests/`: independent numerical and failure-path checks.
- `verification/`: compact evidence, with raw artifacts excluded from Git.

Future target revisions get a new manifest/implementation. Future dialects lower
to the same semantic operations or introduce a versioned contract. DMA/offload
would require a new completion and buffer-ownership contract; today's matrix unit
is synchronous, CPU-fed and single-hart. Do not represent it as a DMA device.

## Memory

The original SRAM-only target has 16 MiB of behavioral SRAM main memory. The 1 GiB AXI decode window
is larger than physical storage and aliases; it is not installed DDR. The SDK
must constrain its ELF, heap, KV cache and stack to [0x80000000, 0x81000000).
The upstream generic linker script advertises 32 MiB and startup takes its stack
from the decode-window end; the SDK supplies an explicit layout instead.

The simulator's named RAM ELF loader (`-l ram,...,elf`) is used by the hardware
runner. Its behavior must be verified for the SDK ELF; its separate 1 MiB region
metadata is not evidence that the hardware has 1 MiB or that an arbitrary ELF fits.

The [DDR profile](ddr.md) adds a separately pinned 4 GiB region at
`0x100000000`, with exact SRAM decode and an address-aware ELF loader. Its linker
places packed projection weights in external memory and working storage in SRAM.
The allocator remains SRAM-only; weight placement is static and explicit.

## Sources

- https://iree.dev/guides/deployment-configurations/bare-metal/
- https://github.com/iree-org/iree/tree/main/samples/static_library
- https://github.com/iree-org/iree/tree/main/samples/custom_dispatch/cpu/embedded
- https://github.com/karpathy/llama2.c
