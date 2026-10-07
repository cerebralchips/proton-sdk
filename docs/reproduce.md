# Reproduce locally

For the latest SRAM + DDR target, use the [DDR procedure](ddr.md). The original
commands below reproduce the SRAM-only baseline on the older hardware revision
pinned in `targets/proton_v1.json`; they deliberately reject a different checkout.

Use the existing provisioned proton-npu lab and its ARM64 Ubuntu VM. Set
`PROTON_HW_ROOT` to the hardware checkout if it is not the sibling
`../Hardware/ara-lab`. Commands below run from the SDK root on the Mac.

The launcher stages SDK sources into the hardware checkout's ignored
`artifacts/sdk-workspace/src`, reuses the existing Linux mount, and acquires the
hardware lab's run lock. SDK `artifacts/` is a local symlink to that workspace.
No VM reconfiguration, hardware edits or remote repository are needed.

While a long run is active, `python3 scripts/status.py` reads the latest results
and completed-token lines without taking the simulator lock.

```sh
./scripts/sdk bootstrap
./scripts/sdk model
./scripts/sdk host-model
./scripts/sdk smoke
./scripts/sdk kernels
./scripts/sdk graph
./scripts/sdk model-run model_step
./scripts/sdk model-run model_matrix
./scripts/sdk model-run model_scalar_step
./scripts/sdk wave
./scripts/sdk negative
./scripts/sdk report
```

`bootstrap` verifies source/checkpoint/tokenizer archive hashes and compiler/source
revision agreement. It uses pinned Python packages. `model` emits an independent
NumPy FP32/quantized reference plus packed weights. `host-model` checks the portable
C semantic kernels against NumPy; this is host validation, not hardware proof.
`smoke` establishes VMFB invocation on RTL. `kernels` checks all 36 actual model
weight matrices, including K=172 padding, exact INT32 results and FP32 rescaling.
`graph` emits and compiles the explicit decoder graph. Model runs retain actual
logits and greedy token IDs and require numerical, completion and trace evidence.

Raw records live in `artifacts/runs/<UTC timestamp>-<test>/`. Each successful run
has `result.json`, `evidence.json`, `provenance.json`, `program.elf`, disassembly,
UART log, retired matrix CSV, full instruction trace and issue/done/response CSV.
Compilation commands and failures are preserved under `artifacts/attempts/`.
Generated VMFB, static dispatch object/header, MLIR and weights live under
`artifacts/generated/`. Build maps are under `artifacts/build/`.

`wave` runs a representative model query projection in the existing tracing RTL
simulator. It checks every completed tile directly from FST signals with the
hardware repository's independent arithmetic checker. `negative` executes an
intentionally corrupted kernel result and a forced 20-cycle timeout, then corrupts
copies of positive model evidence to test the independent checker. `report` copies
compact, portable summaries back into the SDK's `verification/` directory.

The simulator is a functional model. Wall time depends on host load and trace I/O;
only target cycle counters are used for comparisons. Keep simulation runs serial.
Allow several GiB of disk for traces and preserve the failed runs when diagnosing.

The pinned hardware repository must be clean and at the target revision. Changes
require deliberately qualifying a new target manifest, rather than disabling the
preflight checks. The ELF loader verifies segments against actual 16 MiB RAM and
reserved stack space. The application checks heap exhaustion and a stack canary.

This procedure currently reuses the provisioned hardware lab toolchain and RTL
simulator. A fresh machine must first provision and verify proton-npu. SDK source
bootstrap and cross-build are separate from provisioning the hardware tools.

## Stories260K Torch MLIR frontend (host only)

In the provisioned lab, run `./scripts/sdk onnx-bootstrap` followed by
`./scripts/sdk onnx-validate`. This uses an isolated host environment and does
not run Verilator. See the [frontend guide](frontends.md) for generated ONNX,
Torch MLIR and VMFB files, full-output checks and measured limitations.

## Imported Torch MLIR on RTL — one token only

After frontend bootstrap, run `/usr/bin/time -p ./scripts/frontend-rtl` for
one FP32 token and one W8A8 token, serially. Use `fp32` or `w8a8` as an argument
to select just one precision. The [runner guide](frontend-rtl.md) describes the
separate target pin, runtime alignment patch and full logits/cache comparison.
No 16-token workload is launched by this command.
