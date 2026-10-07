# Model frontends and Torch MLIR

The frontend contract is a **Torch MLIR file consumed by `iree-compile`**.
Models may originate in ONNX or a framework such as PyTorch. ONNX import is the
first implemented frontend; PyTorch/IREE Turbine is a possible later route to
this boundary, not yet implemented or qualified here.

The implemented path is:

```text
Pinned Stories260K checkpoint
  -> standard ONNX decoder equations (FP32 or W8A8)
  -> iree-import-onnx
  -> stories260k.<precision>.torch.mlir
  -> iree-compile (LLVM CPU backend)
  -> stories260k.<precision>.vmfb
  -> IREE host CPU execution + independent full-output comparison
```

The imported file actually contains Torch tensors and `torch.operator` ONNX
operations. IREE legalizes those operations during compilation; `.torch.mlir`
is a descriptive filename, not a different serialization format. This follows
[IREE's ONNX import workflow](https://iree.dev/guides/ml-frameworks/onnx/).
The exported graph contains standard mathematical operations and explicit cache
updates, without calls to the existing `proton_op` custom dispatch.

## Reproduce

Use the existing Mac/ARM64 Ubuntu lab described in [setup](reproduce.md). From
`proton-sdk`:

```sh
./scripts/sdk onnx-bootstrap
/usr/bin/time -p ./scripts/sdk onnx-validate
python3 tests/repository.py
```

The wrapper stages SDK sources in the hardware lab's Linux mount and serializes
commands with the existing locks. These commands use the Linux **host CPU**;
they do not build or run Verilator and do not require the hardware checkout to
match a target execution pin. They do require the lab VM/shell launcher.

Bootstrap creates a separate `artifacts/onnx-venv/`, installs the versions in
[onnx.lock.json](../models/stories260k/onnx.lock.json), and checks/downloads the
checkpoint/tokenizer using the existing immutable revision and SHA-256 hashes.
It preserves the original SDK environment and model/math sources. Compiler and
runtime are IREE 3.11.0; the compiler revision must match the SDK dependency lock.

Validation creates `artifacts/onnx-runs/<UTC-run-id>/` containing:

- FP32 and W8A8 `.onnx`, `.torch.mlir` and host `.vmfb` files.
- A small quantization probe in the same three formats.
- `outputs.npz`: actual and independent reference logits and full caches for
  every generation and boundary case.
- Commands, logs, stage durations, `host-results.json` and `result.json`.
- Separate failure-test logs for corrupted logits, keys and values.

A successful run copies the compact record to
[verification/onnx-stories260k.json](../verification/onnx-stories260k.json).
Failures retain their logs and `failure.json` without promoting a new PASS.
Generated model files and raw tensors stay outside Git.

To inspect or recompile a generated file inside the guest, set `RUN` to one of
its run directories and use the isolated environment:

```sh
cd /ara-workspace/artifacts/sdk-workspace
# Set RUN to onnx-runs/<run-id> from the validation output.
onnx-venv/bin/iree-import-onnx "$RUN/stories260k.w8a8.onnx" \
  -o "$RUN/stories260k.w8a8.torch.mlir"
onnx-venv/bin/iree-compile "$RUN/stories260k.w8a8.torch.mlir" \
  --iree-hal-target-device=local \
  --iree-hal-local-target-device-backends=llvm-cpu \
  --iree-llvmcpu-target-cpu=host \
  -o "$RUN/stories260k.w8a8.vmfb"
```

This is host machine code. It is not the RISC-V bare-metal compilation command.

## Decoder-step interface

The exported function is `stories260k_fp32` or `stories260k_w8a8`.

| Tensor | Type and shape | Meaning |
| --- | --- | --- |
| `token` | INT64 `[1]` | Token ID, 0–511 |
| `position` | INT64 `[1]` | Cache position, 0–63 |
| `keys`, `values` | FP32 `[5,64,32]` each | Explicit incoming KV state |
| `logits` | FP32 `[512]` | Full vocabulary output |
| `new_keys`, `new_values` | FP32 `[5,64,32]` each | Full updated KV state |

A caller starts with zero caches, token 1 (BOS), and position 0. It feeds each
returned cache into the next invocation, increments position and chooses the
next token from the full logits. The host test performs 16 such invocations.
No tokenizer or generation loop is hidden inside the compiled decoder step.

Inputs must have the stated types/shapes, valid token/position bounds and finite
cache values. The host harness enforces this contract before invocation; the
exported graph itself does not implement invalid-input exceptions. Positions
0–63 are supported; sliding-window decoding beyond that bound is not implemented.

Each layer updates only its current cache slot. Attention masks future slots,
including when they contain nonzero finite values. RoPE uses fixed FP32 tables
for the 64 supported positions, computed from the same definition as the reference.

## Quantization contract

The W8A8 export preserves the existing SDK arithmetic:

- All 36 linear-projection weight tensors are symmetric INT8, scaled per output
  channel. Activations are dynamically quantized per input row.
- Rounding is `sign(x) * floor(abs(x) + 0.5)` in FP32, including its FP32 boundary
  behavior. A zero activation scale is replaced with one. ONNX's ties-to-even
  `Round` is not used.
- INT8 operands are sign-extended and multiplied with standard ONNX INT32
  `MatMul`; sums are exact within the model's bounded reduction sizes. Rescaling
  applies the activation scale and then the weight scale in FP32.
- Embeddings, normalization, RoPE, attention, nonlinear functions, KV caches
  and logits remain FP32. This is not an entirely INT8 transformer.

INT32 MatMul is a portable representation of W8A8 arithmetic, not evidence of an
optimized INT8 host kernel or Proton matrix lowering. A future compiler pass
can recognize this pattern. The current compiler may fold/expand weight casts;
ONNX initializer storage does not establish SRAM/DDR allocation at runtime.

## Measured host correctness — 7 October 2026

For each precision and each backend (ONNX Runtime CPU and IREE `local-sync`),
validation checks 16 recurrent generation steps and four seeded-cache cases at
positions 0, 1, 31 and 63. Boundary cases include token IDs 0 and 511, nonzero
future slots and exact preservation of every non-current cache slot.

**All 80 cases passed: 1,679,360 output elements compared**, including all logits
and the entire returned caches on every invocation. Every generated token matches
the corresponding independent reference. All four generation runs produced:

> Once upon a time, there was a little girl named Lily. She

| Precision / backend | Max logits error | Max keys error | Max values error |
| --- | ---: | ---: | ---: |
| FP32 / ONNX Runtime | 1.0491e-5 | 9.5368e-6 | 1.1325e-6 |
| FP32 / IREE | 9.5368e-6 | 9.5368e-6 | 2.6823e-6 |
| W8A8 / ONNX Runtime | 3.8147e-6 | 5.7221e-6 | 4.7684e-7 |
| W8A8 / IREE | 6.6758e-6 | 7.6294e-6 | 5.9605e-7 |

The per-element limit is `abs(error) <= 1e-4 + 1e-5 * abs(reference)`;
nonfinite values and shape/dtype mismatches fail. The existing NumPy decoder
is the independent mathematical oracle. The exporter shares its checkpoint
reader but independently expresses inference and weight quantization.

A separate probe checks 516 quantized activation values and 15 INT32 sums per
backend with **exact equality**, including zero input, positive/negative ties,
neighboring FP32 values, signed extrema and reduction size 172. Three separate
process tests corrupt real logits, key-cache or value-cache output and require
nonzero exit with a numerical mismatch. Input-bound violations are also rejected
by the host harness.

Quantization error is measured separately against FP32 with the same reference
input-token path: maximum logit difference **0.47903**, RMS **0.14223**, and
16/16 top-1 agreement. That quantization difference is not the implementation
error reported in the table.

The recorded run used ARM64 Linux. Export, both model imports/compiles and the
quantization probe import/compile took about 4.7 seconds in total; the host
validation subprocess took about 0.49 seconds. These measurements exclude
installation, wrapper overhead and separate deliberate-failure runs. They are
not inference benchmarks, Verilator estimates or hardware cycle measurements.

## Limits and retained bring-up evidence

This milestone qualifies the frontend and host arithmetic. It does not qualify
this imported graph on RTL, provide automatic Proton matrix instruction selection,
or integrate DMA, double buffering or SRAM/DDR placement. Historical RTL results
in the SDK belong to the existing custom-dispatch flow.

The first host attempt imported and compiled successfully but invoked `main`;
the importer exports the ONNX graph name. That failure is retained locally under
run `20261007T060658.645091Z`; the runner now uses the declared graph name.
A later successful attempt exposed IREE Python binding keep-alive warnings from
`to_host()` at shutdown. The harness uses an owned copy through the mapped buffer
protocol with the pinned runtime; the final run has no such leak warning. ONNX
Runtime still reports an informational unknown-CPU-vendor warning on this VM.

Other pretrained families and a PyTorch/Turbine frontend remain separate tasks.
[Stories260K model entry](../models/stories260k/README.md) ·
[Verification record](../verification/onnx-stories260k.json)
