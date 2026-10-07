# Stories260K frontend workload

This entry exports the existing pinned pretrained Stories260K checkpoint into
FP32 and W8A8 decoder-step ONNX models, imports each into **Torch MLIR**, compiles
with IREE, and validates execution on the host CPU.

**Torch MLIR is the SDK compiler input boundary.** ONNX is the frontend used for
this workload. A future PyTorch/IREE Turbine frontend can produce that same
boundary; ONNX is not a universal model-format requirement.

```sh
./scripts/sdk onnx-bootstrap
./scripts/sdk onnx-validate
```

Run these commands from the SDK root in the existing provisioned lab environment.
See the [frontend guide](../../docs/frontends.md) for inputs, quantization,
acceptance criteria, generated files and measured results.

| Item | Location |
| --- | --- |
| Checkpoint/tokenizer revision and SHA-256 | [SDK dependency lock](../../dependencies.lock.json) |
| Host frontend package versions | [ONNX environment lock](onnx.lock.json) |
| Model exporter | [export_onnx.py](export_onnx.py) |
| Independent inference reference | [existing NumPy decoder](../stories.py) |
| Host full-output checks | [onnx_host.py](../../tests/onnx_host.py) |
| Measured evidence | [verification record](../../verification/onnx-stories260k.json) |

Generated `.onnx`, `.torch.mlir`, `.vmfb`, reference tensors and logs live under
ignored `artifacts/onnx-runs/<run-id>/`. The pretrained checkpoint and tokenizer
live in `artifacts/models/`. Source recipes and compact evidence are committed;
model binaries and build artifacts are regenerated locally.

This workload retains all five pretrained layers, the original 512-token
vocabulary, grouped-query attention, RoPE, RMS normalization and gated FFN.
Batch size is one and the exported cache capacity is 64. Restricting the cache
capacity does not remove layers or truncate the vocabulary.

The checkpoint metadata declares MIT; see [third-party sources](../../THIRD_PARTY.md).
This frontend has **host numerical qualification only**. It does not yet select
Proton matrix instructions, DMA, SRAM/DDR placement, or a bare-metal RTL runner.
