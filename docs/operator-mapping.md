# Model and matrix mapping

Stories260K is a trained five-layer decoder with dimension 64, FFN dimension 172,
eight query heads, four KV heads, head dimension 8, and vocabulary 512. The first
deployment limits context to 64 and demonstrates 16 greedy tokens starting at BOS.
The original checkpoint supports a longer context; this is a software bound.

| Operation | Per-layer output × input | Initial implementation | mmacc per token |
| --- | --- | --- | --- |
| Query projection | 64 × 64 | INT8 matrix | 64 |
| Key projection | 32 × 64 | INT8 matrix | 32 |
| Value projection | 32 × 64 | INT8 matrix | 32 |
| Attention output projection | 64 × 64 | INT8 matrix | 64 |
| FFN up projection | 172 × 64 | INT8 matrix | 172 |
| FFN gate projection | 172 × 64 | INT8 matrix | 172 |
| FFN down projection | 64 × 172 | INT8 matrix, K padded to 176 | 176 |
| Final classifier, once after all layers | 512 × 64 | INT8 matrix | 512 |
| QK scores, softmax, probability × V | Bounded causal attention | Scalar FP32 | 0 |
| Embedding, RoPE, RMSNorm, SiLU, residual | Model-specific dimensions | Scalar FP32 | 0 |

There are 36 linear projections per decoder step. Each uses one activation row;
the other three A rows in the 4×4 tile are zero. The compiler does not invent a
matrix instruction feature flag: the target C kernel emits the documented custom-1
encodings explicitly. The input and weight tiles are transferred by CPU MMIO.

Weights are symmetric signed INT8 per output channel, rounded halfway away from
zero, with zero padding. Activations are dynamically quantized per input row.
INT32 products accumulate before multiplication by the activation and weight
scales. Other math, including KV cache storage, remains FP32. This is W8A8 for
linear operations, not a claim that every operator or the whole model uses INT8.

For N output channels and K input channels:

```
mzero = ceil(N / 4)
mmacc = ceil(N / 4) * ceil(K / 16)
```

This gives 878 mzero and 4,072 mmacc per complete token step, including the final
classifier. Five layers contribute 3,560 mmacc; the classifier contributes 512.
The evidence checker derives totals and compares hardware counters, retired
instructions, and issue/done/response sequences. Instrumented model builds also
record actual counts for each of the 36 projections.

IREE owns VM execution, dispatch scheduling and buffer lifetime. The emitted MLIR
contains an explicit chain of 68 semantic dispatches: embedding, 13 operations
per layer, final normalization and classifier. The application owns the outer
greedy token loop. A tied tensor buffer carries activations and the KV cache;
weights are immutable linked constants. Static RISC-V wrappers call `proton_op`
through IREE's `hal.import.static` and bare-pointer interface.

The graph is deliberately inspectable but model-specific. It does not currently
import arbitrary PyTorch/ONNX models, automatically recognize GEMM patterns, fuse
operators or tile a batch larger than one. This first integration validates the
runtime/kernel boundary. Later work can represent individual tensors and weights
directly in MLIR, add pattern rewrites or ukernel lowering, and improve memory
planning without replacing the hardware kernel contract.

Next performance opportunities are activation/weight packing, fused QKV dispatch,
using all four tile rows during prompt prefill, and reducing CPU MMIO transfers.
Accelerating attention's dynamic matrix products requires a separately validated
quantization strategy. A new matrix dialect becomes useful when it expresses
scheduling/layout semantics that existing Linalg and kernel interfaces cannot.
