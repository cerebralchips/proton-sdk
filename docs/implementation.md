# What is implemented

The qualified deployment generates 16 greedy tokens with Stories260K on the
existing CVA6 + Ara + INT8 matrix RTL in Verilator. All model computation runs on
the simulated target. The host prepares weights and independent reference
results, builds the application and checks the target's output.

Stories260K is a pretrained Llama-style language model with roughly 260,000
parameters, five decoder layers, hidden dimension 64, FFN dimension 172 and a
512-token vocabulary. See the pinned
[upstream model description](https://huggingface.co/karpathy/tinyllamas/blob/0bd21da7698eaf29a0d7de3992de8a46ef624add/stories260K/readme.md).

## Deployment path

```text
Host preparation and compilation
  stories260K.bin + tok512.bin
    -> checkpoint validation, NumPy reference and INT8 weight packing
    -> model-specific MLIR graph + generated C weight constants
    -> IREE compiler: VMFB + static RISC-V dispatch object
    -> link with Proton kernels, bare-metal IREE runtime and startup
    -> RISC-V ELF

Target execution in Verilator
  application greedy token loop
    -> IREE VM + synchronous local HAL
    -> 68 semantic dispatches per decoder step
    -> Proton C operator kernels
    -> scalar FP32 operations or matrix MMIO + mzero/mmacc
    -> logits, next token and updated KV cache
```

The Linux VM hosts build tools and the simulator. The target itself runs bare
metal. Weights are immutable constants linked into the ELF; they are not loaded
from a target filesystem or streamed from external storage.

## Source map

| Stage | Entry point | Responsibility |
| --- | --- | --- |
| Dependency acquisition | [scripts/bootstrap.py](../scripts/bootstrap.py), [dependency lock](../dependencies.lock.json) | Fetch pinned sources/model assets and verify hashes |
| Model preparation | [models/stories.py](../models/stories.py) | Validate the checkpoint/tokenizer, compute independent reference results, pack weights |
| Graph generation | [compiler/model_graph.py](../compiler/model_graph.py) | Emit the explicit decoder graph and imported kernel call |
| Build orchestration | [scripts/build.py](../scripts/build.py), [CMakeLists.txt](../CMakeLists.txt) | Compile MLIR, runtime and kernels; link and inspect the ELF |
| Target application | [runtime/iree_runner.c](../runtime/iree_runner.c) | Initialize IREE, invoke the decoder, choose tokens and emit verification output |
| Model state | [runtime/model_state.h](../runtime/model_state.h) | Define the bounded scratch, activations, logits and KV cache |
| Operator implementations | [kernels/model_ops.c](../kernels/model_ops.c) | Implement decoder operations and route linear projections |
| Matrix adapter | [kernels/linear.c](../kernels/linear.c), [kernel ABI](../kernels/linear.h) | Quantize activations, submit tiles, execute instructions and rescale outputs |
| Target contract | [targets/proton_v1.json](../targets/proton_v1.json) | Pin hardware revision, physical RAM, ISA and matrix capabilities |
| Result checking | [scripts/evidence.py](../scripts/evidence.py) | Check numerical results, completion, cycles and hardware command reconciliation |

## How IREE reaches the matrix engine

The graph contains 68 explicit dispatches: embedding, 13 operations for each of
five layers, final normalization and classifier. A tied tensor buffer carries
state between dispatches. Static RISC-V wrappers call `proton_op` through
`hal.import.static` and `llvm.bareptr`.

For a linear projection, the call path is:

```text
IREE flow.dispatch -> compiled wrapper -> proton_op
  -> proton_linear -> proton_qmatvec
  -> CPU MMIO tile writes -> mzero / mmacc -> accumulator reads
```

This is IREE's custom CPU dispatch route. It is not an integration with IREE's
standard ukernel/mmt4d selection, a new MLIR dialect, or automatic instruction
selection for arbitrary imported matrix operations. IREE compiles the dispatch
wrappers and orchestrates execution; handwritten C kernels implement the math.

The matrix adapter uses signed INT8 operands and INT32 accumulators. One command
multiplies a 4-by-16 activation tile by a 16-by-4 weight tile. Batch-one decode
uses the first activation row and zeros the remaining three rows. Weight packing
and dynamic activation quantization allow larger projections to use repeated
tiles. The CPU transfers operands through MMIO; commands complete synchronously.
There is no DMA submission queue in this implementation.

All 36 linear projections per token use the matrix engine: Q/K/V, attention
output, FFN up/gate/down and the final vocabulary projection. Attention scores,
softmax, value aggregation, normalization, RoPE, SiLU and residual operations
remain scalar FP32. See [operator mapping](operator-mapping.md) for exact shapes,
quantization rules and instruction counts.

## Supported scope and limits

| Area | Implemented and qualified | Work needed to extend it |
| --- | --- | --- |
| Model input | Pinned llama2.c Stories260K checkpoint and tokenizer | Framework graph import and model descriptors |
| IREE integration | Explicit model-specific custom dispatches | General operator matching/lowering; evaluate standard ukernels |
| Generation | Batch one, context bound 64, 16 checked greedy tokens from BOS | Prompt handling, larger contexts, batching and additional validation |
| Linear kernel | W8A8, dimensions N/K bounded to 512 | Generalized shapes/layouts, batching and new acceptance cases |
| Memory | 16 MiB behavioral main-memory SRAM; linked weights | Parameter loading/placement and a qualified platform for larger memory |
| Vector execution | Current model compiled without RVV | Resolve and requalify the compiler-generated RVV sequence in the bring-up notes |
| Other model families | Reusable runtime and matrix adapter | CNN convolution lowering and other operators; ViT patch embedding, LayerNorm, GELU and attention support |

There is no ONNX or PyTorch importer in the current SDK. The model graph is
explicitly generated from the known architecture, not recovered from the binary
checkpoint. Supporting another model requires its operators, layouts, memory
budget and numerical behavior to be qualified; replacing the weight file is not
sufficient.

The hardware's tile dimensions do not cap the size of a complete matrix. The
current kernel bounds are software limits. Memory must accommodate the runtime,
weights, activations, KV cache, heap and stack together. The SoC's 1 GiB decoded
address window is not installed RAM, and the current platform has no DDR
controller. See [architecture](architecture.md) and [extension guidance](extending.md).

## Evidence and interpretation

The committed [evidence index](../verification/README.md) records 8,192 checked
logits, a maximum absolute error of 0.00000763 against the independent W8A8
reference, and 65,152 `mmacc` plus 14,048 `mzero` instructions. These establish
numerical execution through the matrix engine on the simulated hardware.

The measured 2.88x cycle comparison covers only the first decoder invocation,
using identical W8A8 arithmetic and bit-identical logits for scalar and matrix
paths. It is not a full-generation speedup, silicon throughput measurement, or a
claim about FP32-to-INT8 accuracy. Quantization accuracy is recorded separately
in [reference.json](../verification/reference.json).

Read [verification gates](verification.md) before changing acceptance criteria,
and [bring-up findings](bringup-notes.md) for retained failures and their impact.

[Documentation index](README.md)
