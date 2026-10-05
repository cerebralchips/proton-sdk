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

No RTL or memory-size changes were needed. The target still has **16 MiB of
behavioral main-memory SRAM**, not a DDR controller. The model ELF uses 830,080
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
generated for this model; arbitrary framework-model import is a later milestone.
Compiler-generated RVV is disabled after a verifier stall was observed and
retained in the [bring-up notes](docs/bringup-notes.md).

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
