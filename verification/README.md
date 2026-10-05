# Reading the evidence

`scripts/sdk report` creates these compact records only after the required local
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
