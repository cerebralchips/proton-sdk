# Stories260K with external DDR weights

The DDR profile runs the existing IREE Stories260K example on Proton's **16 MiB
SRAM + 4 GiB functional DDR** Verilator target. It changes weight placement and
the ELF loader; model arithmetic, quantization, kernels and reference stay the same.
The original SRAM-only profile and its historical evidence remain available.

## Memory contract

| Contents | Region | Placement |
| --- | --- | --- |
| 36 packed INT8 projection matrices, including output classifier | DDR `[0x100000000, 0x200000000)` | 260,608 initialized bytes in `.ddr_weights` |
| Code, FP32 embedding/norms/scales, tokenizer and reference data | SRAM `[0x80000000, 0x81000000)` | Linked normally |
| Activations, KV cache, IREE runtime, heap and stack | SRAM | Existing allocator; 256 KiB reserved stack |

The DDR region is uncached, non-executable data memory. These weights are
read-only to the application. The hardware itself supports DDR reads and writes;
its separate bare-metal tests verify both directions and boundary responses.

`targets/proton_v1_ddr.json` pins the hardware commit and installed capacities.
`link-ddr.ld` puts the packed arrays in DDR. The preparation script adds section
attributes to the generated weight declarations and checks that removing those
attributes reconstructs the original source exactly. Weight descriptors in SRAM
contain 64-bit pointers; this avoids trying to address far-away DDR symbols using
RISC-V's limited-range PC-relative code addressing.

Startup validates all 36 weight pointers and SRAM scale pointers. Preflight
rejects out-of-range ELF segments, writable/executable DDR weight segments,
missing external weights and overlap with the reserved SRAM stack. The physical
address-aware `--load-elf=FILE` loader initializes both regions, including
addresses above 32 bits. The old named-RAM loader is only for the original profile.

During inference the CPU reads packed tiles directly from DDR and feeds the
matrix unit through MMIO. Working buffers stay in SRAM. There is **no DMA,
compiler-managed DDR tiling, or automatic spilling to DDR** in this change.
Compiler-generated RVV remains disabled under the previously qualified profile.

## Run from the SDK checkout

First provision the hardware lab as described in its README. Keep its checkout
clean at the commit in `targets/proton_v1_ddr.json`; SDK preflight enforces the pin.
With the usual sibling checkout layout:

```sh
../Hardware/ara-lab/scripts/ara ddr-build
./scripts/sdk bootstrap
./scripts/sdk model
./scripts/sdk graph
./scripts/sdk ddr-run model_step
/usr/bin/time -p ./scripts/sdk ddr-run model_matrix
./scripts/sdk ddr-run model_scalar_step
./scripts/sdk ddr-report
python3 tests/ddr_preflight.py
python3 tests/repository.py
```

If the model, reference and graph are already generated, start at `ddr-run`.
Set `PROTON_HW_ROOT` for a different hardware checkout location. Commands acquire
the hardware lab lock; do not run simulations concurrently. `ddr-run` defaults to
latency 1 and no periodic stall. An optional delayed-response run is:

```sh
./scripts/sdk ddr-run model_step 7 3
```

Builds use `artifacts/build-ddr/`. Runs use `artifacts/runs/<timestamp>-<name>-ddr/`,
separate from SRAM runs. Each retains the ELF, disassembly, UART log, instruction
trace, matrix event trace, source fingerprints, numerical results and DDR access
counts. Full-model FST recording is disabled; pass/fail does not require a waveform.
`./scripts/sdk status` inspects progress without taking the simulation lock.

`ddr-report` requires a passing 16-token matrix run and passing matrix/scalar
first-step runs. Every run checks all emitted logits against the independent
NumPy W8A8 reference, exact token IDs, positive cycle counts, clean completion,
and retirement/event/counter agreement. DDR evidence requires initialized external
ELF data and actual DDR reads, with no writes to the weights. When a previous SRAM
run is locally available, logits must also agree bit for bit with that run.

## Verified result — 6 October 2026

The [DDR record](../verification/ddr.json) reports a passing **16-token** matrix
run with **8,192 logits** checked against the independent W8A8 reference. Maximum
absolute error is **7.6294e-6**, and every logit is bit-identical to the original
SRAM run. The scalar DDR first step is also bit-identical to the matrix first step.

> Once upon a time, there was a little girl named Lily. She

| Run | Total RTL cycles | Decoder invocation cycles | DDR read transactions |
| --- | ---: | ---: | ---: |
| Matrix first step | 6,946,217 | 4,706,513 | 260,608 |
| Matrix, 16 tokens | 91,466,323 | 77,516,295 | 4,169,728 |
| Scalar first step | 11,168,930 | 8,942,117 | 259,328 |

All runs report zero DDR writes. The complete matrix run reconciles 65,152
`mmacc` and 14,048 `mzero` instructions with hardware events and counters. SRAM
heap/stack guards pass; its measured heap peak is 187,904 bytes. The simulator
reported 2,014.55 seconds (about 33.6 minutes) for the complete run. This excludes
build and host evidence processing; host sleep/load can change elapsed time.

These cycle counts use the functional latency-1/no-stall DDR configuration. They
do not predict physical DDR throughput. Distinct external storage remains
260,608 bytes even though each token reads the weights again. See the
[provenance](../verification/ddr-provenance.json),
[target output](../verification/ddr-output.txt), and
[negative evidence checks](../verification/ddr-negative.json).

## Scope

This tiny model occupies about 255 KiB of DDR; it does **not** prove that a model
larger than SRAM runs, nor does it sweep 4 GiB. A larger-model qualification and
compiler memory planning remain follow-up work. The 4 GiB capacity is an address
space backed by sparse simulator storage; fully populating it needs considerably
more host memory than 4 GiB.

DDR delays are configurable functional delays, not physical DDR timing or
bandwidth. FPGA/ASIC deployment still needs a controller/PHY and platform
integration. See the hardware [DDR guide](https://github.com/cerebralchips/proton-npu/blob/hardware/docs/ddr-simulation.md)
for the memory tests, open RVV initializer issue and CPU bus-error limitation.
