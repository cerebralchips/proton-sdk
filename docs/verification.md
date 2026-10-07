# Verification gates

All seven gates passed for the pinned 16-token Stories260K deployment. The
machine-readable outcomes are in [gates.json](../verification/gates.json); the
[evidence index](../verification/README.md) links each result to its raw run.

Every gate has an explicit result. Failures are retained and explained; a later
pass does not erase them. Do not move a dependent gate to PASS until prerequisites
pass. Source inspection and host computation do not constitute target execution.

| Gate | Required evidence |
| --- | --- |
| G0 hardware baseline | Pinned unmodified RTL; existing matrix unit/integration gates pass |
| G1 model and memory | Checkpoint/config/tokenizer hashes; independent FP32 and INT8 reference; bounded RAM map |
| G2 bare-metal IREE | Real VMFB invoked through IREE on CVA6; output checked; clean completion and positive cycles |
| G3 matrix kernels | Exact INT32 oracle for all model linear shapes and tails; quantized FP32 tolerances; matrix retirement/counters |
| G4 graph integration | IREE dispatch calls target kernels; inspectable IR/object/ELF; graph output matches reference |
| G5 language model | Target attention, FFN, KV cache and logits; multi-token generation; tokens/logits compared with independent reference |
| G6 evidence and failure paths | Timeout, numerical mismatch and memory exhaustion rejected; scalar/matrix comparisons; provenance and reproduction guide |

Acceptance for quantization is reported against the FP32 reference rather than
assuming quantized greedy tokens always match. Hardware arithmetic must match the
quantized oracle. Greedy token agreement alone is insufficient: preserve logits,
intermediate checks where practical, and numerical error measurements. Report
operator coverage: matrix-backed projections vs scalar attention/normalization.

The target acceptance envelope is `abs(error) <= 1e-4 + 1e-5 * abs(reference)`
for every logit, with exact greedy token agreement. The observed maximum error
was 7.6294e-6 across 8,192 logits. Kernel INT32 outputs require exact equality;
kernel FP32 rescaling uses `2e-5 + 2e-5 * abs(reference)`. The matrix/scalar target
comparison additionally requires bit-identical first-step logits.

Proof of acceleration means that the intended operations execute matrix commands.
A speedup is a separate measured claim requiring the same arithmetic/model and
cycle-counter boundaries. Verilator wall time is not device throughput.

Retain raw logs, model/compiler hashes, generated MLIR, VMFB, object, ELF,
disassembly, instruction/event traces and result JSON. A representative FST can
establish tile activity without recording an entire language-model run waveform.

## DDR profile

The historical G0–G6 records above belong to the original SRAM-only target.
The separately pinned [DDR deployment](ddr.md) adds ELF-region rejection tests,
runtime weight placement checks, nonzero external reads, full-model reference
comparison, and scalar/matrix agreement. It retains the same numerical envelope
and architectural trace requirements. A DDR model pass covers the exercised
weights and program; it does not qualify every external address or larger models.

## Torch MLIR frontend host gate

The [Stories260K frontend](frontends.md) has a separate
[host qualification record](../verification/onnx-stories260k.json). FP32 and W8A8
ONNX Runtime/IREE results are compared against the independent NumPy decoder,
including every logit and KV-cache element, cache boundaries, an exact integer
quantization probe and deliberate corruption. This record does not promote or
replace any target G0–G6 or DDR gate. Torch MLIR is the compiler input boundary;
ONNX is the implemented frontend for this workload.
