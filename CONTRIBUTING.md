# Contributing to Proton SDK

Start with the [documentation index](docs/README.md),
[implementation walkthrough](docs/implementation.md),
[architecture](docs/architecture.md) and [verification gates](docs/verification.md).
The current qualified deployment is Stories260K on the pinned Proton v1 RTL.
The [Stories260K frontend](docs/frontends.md) additionally qualifies ONNX →
Torch MLIR → IREE on the host CPU. General model import, other model families
and future hardware revisions remain extension work.

## Set up a checkout

```sh
git clone https://github.com/cerebralchips/proton-sdk.git
cd proton-sdk
python3 tests/repository.py
./scripts/sdk --help
```

The repository check uses only Python's standard library and needs no simulator.
It validates documentation links, syntax, JSON records and the committed evidence
against selected source hashes. It does not execute the model.

For target builds and execution, first provision the
[proton-npu hardware lab](https://github.com/cerebralchips/proton-npu) at the
revision in [targets/proton_v1.json](targets/proton_v1.json). The supported
launcher currently uses the lab's Mac host and ARM64 Ubuntu VM. A fresh machine
needs that lab's RISC-V toolchain and Verilator simulator before SDK bootstrap.

If the hardware checkout is not at the default `../Hardware/ara-lab`, set its
location in your shell:

```sh
export PROTON_HW_ROOT=/absolute/path/to/proton-npu
```

Follow [reproduce locally](docs/reproduce.md) for the ordered bootstrap, reference,
IREE smoke, kernel, graph, model and failure-test commands. Runs are serialized by
the hardware lab lock. Do not bypass a lock or run overlapping simulations.

## Choose the right layer

| Change | Main locations | Contract to preserve |
| --- | --- | --- |
| Import or graph lowering | `models/`, `compiler/` | Model identity, graph semantics, supported shapes and quantization |
| Operator or matrix implementation | `kernels/` | Numeric results, packing, tail padding, ordering and command counts |
| Runtime or platform support | `runtime/`, `cmake/` | Bare-metal ABI, buffer lifetime, physical memory bounds and traps |
| New hardware target | `targets/` plus kernel/platform adapters | Explicit revision, ISA, MMIO, memory and completion semantics |
| Verification tooling | `scripts/`, `tests/` | Independent result checks and rejection of invalid evidence |
| Explanation or measured results | `docs/`, `verification/` | Distinguish implemented behavior, measured scope and future work |

RTL and ISA implementation changes belong in proton-npu. A new target manifest
alone is insufficient if tile dimensions, layout, memory or execution semantics
change; update and qualify the corresponding adapter. See
[extending the SDK](docs/extending.md).

## Validate changes

For documentation-only changes, run:

```sh
python3 tests/repository.py
git diff --check
```

For implementation changes, repeat the affected gates and their dependent gates
in the [reproduction procedure](docs/reproduce.md). A matrix-kernel change needs
independent arithmetic checks, target execution, model numerical checks and
retired instruction/counter evidence. A model or quantization change also needs
fresh reference results. Verification-tool changes need the failure-path tests.

The repository check deliberately rejects changes to qualified model/math sources
when the committed execution evidence still refers to old hashes. Re-run the
necessary validation and use `./scripts/sdk report` to generate updated records.
Never hand-edit a hash or PASS result to make a check succeed. Keep failed runs
when investigating; simulator exit status alone is not evidence of correctness.

Use the same quantization when comparing scalar and matrix kernels. State exactly
which cycle-counter boundaries were measured. Verilator wall time and functional
RTL cycles do not establish silicon frequency or production throughput.

## Prepare a contribution

- Explain the problem, resulting behavior and affected target/model contract.
- List the checks actually run and link their compact evidence; identify any
  unverified behavior explicitly.
- Update the relevant documentation and add new pages to the documentation index.
- Keep downloads, model binaries, generated weights, objects, builds and raw
  traces under ignored artifact locations. Commit concise records in
  `verification/` with source fingerprints and run identities.
- Retain dependency pins and applicable notices when changing third-party code;
  update [THIRD_PARTY.md](THIRD_PARTY.md) when needed.
- Keep unrelated hardware and user changes out of the SDK contribution.

Open a pull request against `main` with the above context and validation. The SDK
uses [Apache-2.0](LICENSE); third-party components retain their own licenses.

[Documentation index](docs/README.md)
