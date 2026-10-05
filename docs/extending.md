# Extending the SDK

## Hardware revisions

Keep the old target and its evidence. Add a new manifest for a changed ISA, tile
shape, MMIO layout, memory capacity or numeric format, and implement its kernel
adapter. Verify tile arithmetic and ordering before testing a model. The current
adapter and linker describe Proton v1 specifically; editing only the manifest is
not sufficient to support a different board. Pin the new hardware revision and
rebuild its simulator, then repeat the SDK gates with the new binary fingerprint.

Treat these as independent contracts:

1. Numeric semantics: signed operands, accumulator width, overflow and scale rules.
2. Layout: tile dimensions, weight packing, alignment and tail padding.
3. Execution: command submission, completion, ownership and memory visibility.
4. Platform: physical RAM, stack/heap bounds, startup, traps and console.

The initial linear kernel has a batch-one interface and bounded N/K of 512. The
qualified model uses N in {32,64,172,512} and K in {64,172}. Other shapes, datatypes
or batch sizes require their own acceptance cases. The initial implementation
does not claim to exploit all four matrix rows during decode.

## Compiler development

The current input is a pinned llama2.c checkpoint and an explicit decoder-graph
generator. Keep model import separate from kernel selection. A next compiler
milestone can import a framework graph into StableHLO/Linalg, recognize supported
linear operations, validate quantization/layout requirements, and lower them to
the proven kernel interface with a scalar fallback. Save both original and lowered
IR and test numerical behavior at the rewrite boundary.

An IREE ukernel route is appropriate for regular supported tiled matmuls. The
current external dispatch route is useful for proving a complete implementation
without maintaining an IREE fork. A dedicated dialect should be introduced when
there are concrete layout/scheduling/async semantics to preserve across passes;
it is not required just to emit two custom instructions.

Model weights are currently linked immutable constants. Larger models should move
to separately managed parameter storage, with explicit loading and placement.
The model descriptor, packed layout version and target must be validated together.
Changing a tokenizer must also change its identity in the model manifest.

## Performance and memory

Measure prefill and decode separately. First improvements can reuse packed
activations across Q/K/V, fuse appropriate dispatches, expose individual tensors
to IREE's memory planner, and use four tile rows during prefill. Keep the scalar
comparison on the same W8A8 arithmetic. Compare logits before accepting a faster
kernel, especially when changing reduction order or quantization.

The current 16 MiB simulated RAM is sufficient for this milestone. A few GiB of
future DDR would require a real memory interface/backing model, a compatible
address map and loader, and a new memory budget. The 1 GiB decoded address range
in the present SoC must never be mistaken for installed memory. Quantization does
not eliminate activation, scratch, runtime or KV-cache storage.

## RVV and Linux

Re-enable compiler-generated RVV only after reducing and resolving the verifier
stall described in [bring-up notes](bringup-notes.md). Repeat both the original
hardware RVV tests and the failing workload; a small passing vector smoke test
does not establish every compiler-generated sequence.

Linux is a separate platform milestone. It needs a qualified CPU/platform port,
memory and boot flow, and an ownership/context policy for the custom matrix state.
The current unit is synchronous and CPU-fed; treating it as an asynchronous DMA
offload device would require a different runtime/driver contract. Keep that future
work behind the target adapter so the model/compiler code can be reused.
