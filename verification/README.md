# Reading the evidence

`scripts/sdk report` creates the original SRAM-only compact records only after the required local
runs and failure tests pass. Raw artifacts stay outside version control under
`artifacts/`. Each record identifies its original run, source provenance and ELF.

| Record | What it establishes |
| --- | --- |
| [gates.json](gates.json) | Explicit outcomes and evidence for G0 through G6 |
| [hardware-baseline.json](hardware-baseline.json) | The original hardware integration passed on the same normal simulator binary |
| [hardware-units.json](hardware-units.json) | Independent PE, tile, bus and router gates |
| [dependencies.json](dependencies.json) | Compiler/source agreement, archive/model/tokenizer hashes and extracted source integrity |
| [reference.json](reference.json) | FP32 vs W8A8 tokens and quantization error |
| [smoke.json](smoke.json) | Actual IREE VMFB execution on CVA6 |
| [kernels.json](kernels.json) | All 36 model matrices, including K tails, match the numerical oracle |
| [model_matrix.json](model_matrix.json) | Complete 16-token graph, 8,192 logits, matrix retirement and command reconciliation |
| [model-output.txt](model-output.txt) | Actual target token IDs, pieces, cycle counters and per-projection command counts |
| [model-provenance.json](model-provenance.json) | Executed SDK and hardware source fingerprints and actual ELF segments |
| [memory.json](memory.json) | ELF segments, state size, reserved stack and measured heap high-water mark |
| [model_scalar_step.json](model_scalar_step.json) | The same first decoder step using scalar INT8 arithmetic |
| [comparison.json](comparison.json) | Target cycle comparison, requiring bit-identical scalar/matrix logits |
| [waveform.json](waveform.json) | Actual FST tile arithmetic and pump count checks for a model projection |
| [negative.json](negative.json) | Deliberate arithmetic error, timeout and corrupted-evidence rejection |

For a model run, open `artifacts/<run>/program.elf`, `program.dump`, `rtl.log`,
`trace_hart_0.dasm`, `retired-matrix.csv`, and `matrix-events.csv`. The generated
graph and VMFB are in `artifacts/generated/model/`. The waveform run also retains
its FST, extracted `matrix.vcd`, and all independently checked tile snapshots.

The independent checker requires clean RTL completion, positive cycles, reference
logits and greedy tokens, per-model command totals, and matching issue, completion,
response and retirement counts. The model also checks real heap exhaustion and
stack canaries. The final repository check binds current model/math sources to
the executed source hashes.

This evidence qualifies the pinned tiny-model deployment, not arbitrary models,
full ISA conformance, all RVV sequences, Linux support, silicon frequency or DDR
bandwidth. See [operator coverage](../docs/operator-mapping.md) and
[bring-up findings](../docs/bringup-notes.md) for the exact boundaries.

## DDR deployment

`./scripts/sdk ddr-report` publishes the separately pinned SRAM + DDR deployment.
See the [DDR guide](../docs/ddr.md) for commands and placement details. These
records supplement the historical SRAM-only G0–G6 results above.

| Record | What it establishes |
| --- | --- |
| [ddr.json](ddr.json) | 16-token matrix run, scalar/matrix first-step comparison, reference checks and DDR reads |
| [ddr-provenance.json](ddr-provenance.json) | Hardware pin, SDK/RTL hashes and separate SRAM/DDR ELF segments |
| [ddr-output.txt](ddr-output.txt) | Runtime placement, tokens, memory guards, cycle and DDR access counts |
| [ddr-negative.json](ddr-negative.json) | Corrupted DDR placement and traffic evidence rejected |

The host-only `python3 tests/ddr_preflight.py` checks both valid DDR edges and
rejects 12 malformed or out-of-range memory contracts. To repeat the evidence
negative test inside the provisioned Linux VM, run its Python with NumPy against
a passing `*-model_step-ddr` run:

```sh
cd /ara-workspace/artifacts/sdk-workspace
venv/bin/python src/tests/ddr_evidence_negative.py runs/<passing-model-step-ddr> ddr-negative.json
```

Raw DDR runs retain instruction and matrix event traces, but no full-model FST.
All emitted logits must match the independent reference. Local SRAM baseline
logits, when present, are also compared bit for bit. This deployment reads about
255 KiB of distinct DDR weights; it does not establish a full-capacity sweep or
a model larger than SRAM.

## Stories260K frontend — host only

[onnx-stories260k.json](onnx-stories260k.json) records FP32 and W8A8 export through
ONNX → Torch MLIR → IREE, 80 full-output host cases, exact quantization checks,
negative gates, source/model/artifact hashes and stage durations. It establishes
no new RTL or matrix-acceleration result. Reproduce with
`./scripts/sdk onnx-bootstrap` and `./scripts/sdk onnx-validate`; see the
[frontend guide](../docs/frontends.md).
