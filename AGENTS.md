# Proton SDK development

This repository owns software; proton-npu owns RTL, ISA implementation and its
integration tests. Read README.md, docs/architecture.md and docs/verification.md.
Use a pinned target manifest and explicit kernel ABI. Do not silently assume a
new hardware revision is compatible. Never treat the AXI address window as RAM.

Keep downloads, models, generated objects, raw traces and builds ignored. Record
source hashes, compiler versions, commands and failed attempts in artifacts.
Keep concise, portable measured results in verification/. Results must identify
whether execution was host reference, host emulation or actual CVA6/Ara RTL.

Advance gates only on successful completion, numerical checks, positive cycle
counts and, for acceleration, retired custom instructions and hardware counters.
Simulator exit code alone is insufficient. Compare matrix and scalar kernels
with identical quantization. Do not claim throughput from a functional simulation.

Run simulations serially under the hardware lab lock. Begin with at most four
build jobs. Preserve baseline RTL and existing user changes. Push or publish only
when requested by the user. Publish source, documentation and compact verification
records; keep downloaded models, build products and raw traces outside Git. Do not
send external messages unless explicitly requested.
