# Proton SDK

Stories260K now generates **16 tokens end to end on the existing CVA6 + Ara +
INT8 matrix RTL in Verilator**, through IREE's bare-metal VM and synchronous HAL.
Target kernels execute the matrix instructions for all 36 linear projections.

Actual target output:

> Once upon a time, there was a little girl named Lily. She

| Measured result | Evidence |
| --- | --- |
| 8,192 target logits checked; maximum absolute error 0.00000763 against independent W8A8 reference | [Model record](verification/model_matrix.json) |
| 65,152 mmacc and 14,048 mzero retired; all 79,200 commands reconciled | [Target output](verification/model-output.txt) |
| First decoder invocation: 2,327,901 matrix cycles vs 6,713,729 scalar cycles; bit-identical logits | [2.88× cycle comparison](verification/comparison.json) |
| All 36 weight matrices and K tails checked independently | [Kernel record](verification/kernels.json) |
| 160 actual matrix commands checked from FST tile signals | [Waveform record](verification/waveform.json) |
| Wrong result, forced timeout and corrupted evidence rejected | [Failure tests](verification/negative.json) |

The original SRAM-only deployment used **16 MiB of behavioral main-memory SRAM**.
Its model ELF uses 830,080
bytes of allocated address span; measured heap high-water is 185,728 bytes, with a
256 KiB reserved stack. See the [memory record](verification/memory.json).

The 53,512,271-cycle complete run includes initialization, buffer transfers,
reference checks and UART logit output. The comparison above times only the first
decoder graph invocation. These are functional RTL cycle measurements, not a
silicon frequency or production throughput claim.

The SDK is a sibling repository to
[proton-npu](https://github.com/cerebralchips/proton-npu). Hardware tests and RTL
remain in proton-npu; compiler integrations, runtime, kernels, model import, and
deployment tests live here.

This is a qualified tiny-model deployment: batch one, context bound 64, greedy
generation from BOS, W8A8 linear operations and FP32 cache/nonlinear operations.
Attention scores/softmax/value aggregation remain scalar. The graph is explicitly
generated for this model; arbitrary framework-model import is a later milestone. A separate
[Stories260K frontend](docs/frontends.md) now qualifies ONNX → Torch MLIR →
IREE execution on the host CPU. Its separate
[one-token RTL runner](docs/frontend-rtl.md) provides a compiler-generated scalar
baseline alongside the original accelerated deployment.
Compiler-generated RVV is disabled after a verifier stall was observed and
retained in the [bring-up notes](docs/bringup-notes.md).

## DDR deployment

A separate [DDR profile](docs/ddr.md) targets the latest qualified hardware with
**16 MiB SRAM + 4 GiB simulated DDR**. All 36 packed INT8 projection matrices
(260,608 bytes) live in DDR; activations, KV cache, heap and stack remain in SRAM.
The CPU reads weights from DDR while executing the same IREE graph and matrix
kernels. This is functional external-memory support, not a physical DDR PHY or
a general large-model memory planner.

```sh
./scripts/sdk ddr-run model_step
./scripts/sdk ddr-run model_matrix
./scripts/sdk ddr-run model_scalar_step
./scripts/sdk ddr-report
```

The [DDR verification record](verification/ddr.json) passes all 16 tokens and
8,192 logits, bit-identical to the SRAM result. The trace records 4,169,728 DDR
reads and all 79,200 expected matrix commands; the scalar first-step comparison
also passes. The full DDR run took 91,466,323 RTL cycles.

See the guide for hardware provisioning, revision pinning and generated-model
prerequisites. The measurements in the first table describe the original SRAM profile.

## Torch MLIR frontend

The compiler input boundary is **Torch MLIR**; ONNX is one supported source
format, and future PyTorch/IREE Turbine frontends can use the same boundary.
The first qualified frontend exports the pretrained Stories260K model as FP32
and W8A8 ONNX, imports it with `iree-import-onnx`, then runs the compiled result
on the host CPU. All 80 cases pass full logits and KV-cache comparison against
an independent reference, with deliberate corruption rejected.

```sh
./scripts/sdk onnx-bootstrap
./scripts/sdk onnx-validate
```

See the [model entry](models/stories260k/README.md),
[frontend guide and measured results](docs/frontends.md), and
[host verification record](verification/onnx-stories260k.json).
The [one-token RTL baseline](docs/frontend-rtl.md) runs the imported graph on
CVA6 through bare-metal IREE, with full logits and KV-cache comparison:

```sh
/usr/bin/time -p ./scripts/frontend-rtl  # one token per FP32/W8A8 variant
```

Matrix code generation, DMA and DDR placement for this imported graph remain
later gates. The new runner has no multi-token generation option.

## Documentation

Start with the [documentation index](docs/README.md) and
[implementation walkthrough](docs/implementation.md) to understand what runs,
how IREE reaches the matrix instructions, and the current limits. Documentation
is Markdown rendered directly on GitHub and readable in a local checkout.

Contributors should read [CONTRIBUTING.md](CONTRIBUTING.md) for setup, code
ownership, validation requirements and the change review checklist.

- [Architecture and extension boundaries](docs/architecture.md)
- [Verification gates and acceptance criteria](docs/verification.md)
- [Evidence index](verification/README.md)
- [Hardware target](targets/proton_v1.json)
- [Third-party sources](THIRD_PARTY.md)
- [Run and reproduce](docs/reproduce.md)
- [Operator mapping](docs/operator-mapping.md)
- [Bring-up findings](docs/bringup-notes.md)
- [Extending targets and compiler support](docs/extending.md)

## Run and verify

The source/documentation check needs only Python's standard library:

```sh
python3 tests/repository.py
./scripts/sdk --help
```

Hardware execution requires the provisioned proton-npu lab; follow the
[reproduction guide](docs/reproduce.md) in gate order. `./scripts/sdk status`
shows local run results without interrupting a simulation. Model downloads,
build products and raw traces are excluded from Git; compact measured results
are available in [verification/](verification/README.md).
