# Torch MLIR frontend: one-token RTL baseline

This runner compiles the imported Stories260K Torch MLIR into RISC-V code and
executes it through bare-metal IREE on the CVA6/Ara RTL. Each requested precision
runs **exactly one decoder invocation**, generating one token from BOS at position
zero. There is no generation loop or multi-token option in this runner.

The flow is checkpoint → ONNX → Torch MLIR → `iree-compile` → RISC-V object +
VMFB → bare-metal ELF → Verilator. Torch MLIR remains the compiler input boundary;
ONNX is this model's implemented frontend, not an SDK-wide requirement.

## Run

From a provisioned SDK and hardware lab:

```sh
./scripts/sdk onnx-bootstrap
/usr/bin/time -p ./scripts/frontend-rtl        # one FP32 token, then one W8A8 token
# To run only one precision instead:
/usr/bin/time -p ./scripts/frontend-rtl fp32
/usr/bin/time -p ./scripts/frontend-rtl w8a8
```

The wrapper uses the existing SDK staging lock and hardware lab execution lock.
The separate build directory is `artifacts/build-frontend/`. The existing model
runtime, compiler graph and qualified source files are preserved.

The [frontend CPU target](../targets/proton_frontend_cpu.json) pins the hardware
revision and qualified simulator hash. The runner checks the hardware checkout
is clean and its sources match the published DMA-system qualification evidence.
It also checks the exporter/reference sources and compiler match the existing
host frontend qualification. A mismatch fails before simulation; do not bypass
these checks by changing a pin without qualification.

All program constants and working buffers are in 16 MiB SRAM. The simulation
configuration also contains 4 GiB functional DDR, DMA and the matrix engine,
but this baseline does not use them. Compiler CPU features are RV64 scalar
`+m,+a,+f,+d,+c`; automatic RVV generation is disabled by this target choice.
This is a compiler-generated scalar baseline, not the existing accelerated
custom-dispatch application.

## What is checked

The only model inputs are token 1, position 0 and zero FP32 KV caches with shape
`[5,64,32]`. A separate NumPy reference performs just one matching step per
precision. The target checks output shapes and types and compares:

- All 512 vocabulary logits.
- All 10,240 key-cache elements.
- All 10,240 value-cache elements.
- The greedy output token, bounded heap behavior and stack canary.

Every output element uses the existing tolerance
`abs(error) <= 1e-4 + 1e-5 * abs(reference)`. FP32 is compared with the FP32 oracle;
W8A8 is compared with the W8A8 oracle. Quantization differences between those
precisions are not treated as implementation errors.

The target emits FP32 bit patterns and explicit zero runs covering every output
index. The host reconstructs all three tensors, rejects missing or overlapping
ranges, independently repeats the numerical comparison and reconciles target
error summaries. Successful simulator exit alone is insufficient: unique target
PASS, simulator SUCCESS, exactly one token and positive cycle counts are required.

Negative checks mutate the captured log to introduce corruption, incomplete
cache coverage, missing completion, a second token or a timeout. They must fail
validation and **do not launch additional model simulations**.

The scope is one token at position zero. Repeated decoding, cache feedback,
nonzero-position RTL execution, matrix/DMA lowering and DDR placement remain
unqualified for this imported graph. The broader cache-position and recurrent
checks in [host validation](frontends.md) are host results only.

## Runtime alignment fix

The first FP32 attempt trapped during IREE deferred command replay, before token
completion. In the pinned IREE source, the dispatch binding array could be placed
immediately after four-byte constants with byte alignment. Its entries contain
RV64 pointers requiring natural alignment.

[The alignment patch](../patches/iree/0001-align-dispatch-bindings.patch) uses the
natural alignment of `iree_hal_buffer_ref_t`. The runner verifies the original
source hash and applies the patch to a generated copy in the run directory.
The frontend CMake project replaces the deferred-command-buffer **object library**
source with that copy. No downloaded dependency file or historical runtime build
is edited. The patch retains the upstream license and is confined to this build.

An initial build override targeted the static archive instead of its object
library, so it still linked the original object and reproduced the trap. That
failed attempt is retained too. The final object-library replacement is visible
in the generated Ninja build graph. A follow-up attempt used the default field
macro, which inspection showed was also unaligned; the final patch explicitly
uses `iree_alignof(iree_hal_buffer_ref_t)`.

A small host probe reproduces four misaligned original layouts and verifies nine
corrected layouts without running another model:

```sh
cc -std=c11 -O2 -DIREE_STATUS_FEATURES=0 \
  -I artifacts/deps/iree/runtime/src tests/frontend_alignment.c \
  -o artifacts/frontend_alignment
./artifacts/frontend_alignment
```

This probe expects a 64-bit host, matching the RV64 binding layout.

## Artifacts and timing

Each invocation creates `artifacts/frontend-rtl-runs/<UTC-run-id>/`. Each precision
retains ONNX, Torch MLIR, RISC-V dispatch object, VMFB, ELF, disassembly, full UART
output, one-step reference tensors, provenance and a checked result. Generated
files and simulator instruction traces remain ignored. Failed runs retain
`failure.json` and their original logs.

The result distinguishes invocation cycles (around the single IREE call), total
RTL cycles (including setup and validation), and simulator process wall time.
They are functional simulation measurements; they do not establish silicon
frequency or production throughput.

## Measured result — 7 October 2026

Both precisions passed on the pinned DMA-capable hardware configuration, using
scalar CPU code and SRAM only. Each generated **token 403, `Once`**, from one
invocation. Each compared all **20,992 output elements** on target and host.

| Precision | Invocation cycles | Total RTL cycles | Simulator elapsed, monotonic | Heap high-water |
| --- | ---: | ---: | ---: | ---: |
| FP32 | 4,699,835 | 12,170,347 | 301.384 s | 2,448,640 bytes |
| W8A8 | 6,704,657 | 13,965,920 | 352.177 s | 910,336 bytes |

| Precision | Maximum logits error | Maximum keys error | Maximum values error |
| --- | ---: | ---: | ---: |
| FP32 | 5.245209e-6 | 3.814698e-6 | 5.364418e-7 |
| W8A8 | 0 (bit-identical logits) | 1.907349e-6 | 2.384186e-7 |

Invocation cycles enclose one IREE invocation, including its runtime dispatch
work. Total cycles additionally include initialization, transfers, complete
numerical checks and UART evidence. The process timer excludes compilation and
postprocessing. GNU `/usr/bin/time -p` reported `real` values of 328.69 s and
353.81 s respectively; those differ from the monotonic timer on this VM and are
also retained in the records. Use RTL cycle counts for reproducible target
comparisons, and keep both host timer measurements when estimating run duration.

An independent audit counted 6,994,155 retired instructions for FP32 and
8,044,759 for W8A8. Both had **zero RVV instructions, zero matrix instructions,
zero matrix command events, and zero DDR reads/writes**. This confirms the stated
scalar/SRAM execution scope. These are baseline measurements, not a claim that
W8A8 is faster before accelerator mapping.

Evidence:

- [FP32 one-step record](../verification/frontend-rtl-fp32.json)
- [W8A8 one-step record](../verification/frontend-rtl-w8a8.json)
- [Retirement and memory-traffic audit](../verification/frontend-rtl-retirement.json)

Run `python3 tests/frontend_retirement.py` to re-audit the retained local traces
without executing the model again. Each model record also includes seven rejected
evidence mutations. The source/dependency hashes bind these results to the
executed implementation. These checks are functional evidence, not a formal proof.
