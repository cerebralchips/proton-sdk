# Proton SDK documentation

Proton SDK runs the Stories260K language model through bare-metal IREE and the
Proton matrix instructions on the existing CVA6 + Ara RTL in Verilator. This
documentation describes the implemented deployment, its measured evidence and
the contracts contributors must preserve when extending it.

## Start here

1. Read the [implementation walkthrough](implementation.md) for the complete
   model-to-hardware path and the distinction between custom dispatch and a
   general compiler backend.
2. Read [architecture](architecture.md) for repository ownership, target
   boundaries and the physical memory map.
3. Use [operator mapping](operator-mapping.md) to see which computations execute
   matrix instructions, their numeric formats and expected command counts.
4. Inspect the [verification evidence](../verification/README.md) and
   [acceptance gates](verification.md) before making performance claims.
5. Follow the [contributor guide](../CONTRIBUTING.md) and
   [reproduction procedure](reproduce.md) to make and validate a change.

## Documentation map

| Document | Questions it answers |
| --- | --- |
| [Implementation walkthrough](implementation.md) | What is implemented? How do model weights, IREE and matrix instructions connect? |
| [Model frontends](frontends.md) | How does Stories260K reach Torch MLIR through ONNX, and what is qualified on the host? |
| [DDR deployment](ddr.md) | How are weights placed in external memory and checked on RTL? |
| [Architecture](architecture.md) | What belongs in each repository? What are the runtime and memory contracts? |
| [Operator mapping](operator-mapping.md) | Which operations are accelerated? What are the packing, scaling and tile-count rules? |
| [Reproduce locally](reproduce.md) | What environment is required? Which commands run each gate? Where are artifacts stored? |
| [Verification gates](verification.md) | What constitutes a pass? Which numerical and hardware evidence is mandatory? |
| [Evidence index](../verification/README.md) | Where are the measured results, source fingerprints and limitations? |
| [Bring-up findings](bringup-notes.md) | Which failures were observed, and why are certain features disabled? |
| [Extension guide](extending.md) | How should new hardware targets, models, compiler lowering and memory support be added? |
| [Contributing](../CONTRIBUTING.md) | How do I set up a checkout, choose checks and prepare a reviewable change? |
| [Third-party sources](../THIRD_PARTY.md) | Which upstream projects are used and where are their licenses? |

## Reading without a simulator

All pages are ordinary Markdown, rendered by GitHub. No documentation generator,
HTML build or provisioned hardware lab is needed to browse them. From the SDK
root, run the existing documentation/source check with Python's standard library:

```sh
python3 tests/repository.py
```

It validates relative Markdown links, Python syntax, JSON records, all seven
recorded gate outcomes and selected executed-source hashes. It does not rerun the
model or establish that a changed kernel works on hardware.

Raw traces and model downloads are intentionally absent from the repository.
The committed evidence identifies the original runs and source fingerprints;
the reproduction guide explains how to generate a new set locally.

[Back to the SDK overview](../README.md)
