# Bring-up findings

These are observations from this local implementation, not hardware conformance
claims. Failed builds and runs are retained under `artifacts/attempts` and `runs`.

1. The pinned IREE wheel reports 3.11.0rc20260316, source revision
   `e4a3b0405d7d23554da26403658d0e8c3c5ecf25`, although its PyPI package version is
   3.11.0. Runtime source is pinned to that exact revision.
2. The runtime-only cross build needs host flatcc, the printf dependency, and
   `CheckCSourceCompiles` explicitly imported by the outer CMake project. The
   existing lab compiler wrapper supplies link flags on compile invocations;
   only the unused-command-line-argument warning is suppressed for that wrapper.
3. Bare-metal time hooks have different macro contracts: `IREE_TIME_NOW_FN` is
   a function body, while `IREE_WAIT_UNTIL_FN` is an expression. This deployment
   has no target wall clock; deadline waits report unsupported/false. Execution
   is synchronous and single-threaded. The simulator enforces execution limits.
4. Disabling all IREE status features triggered upstream unused-variable errors.
   The SDK retains status features 3, including messages and source locations.
5. The existing Newlib/libgcc installation contains medlow-only helper objects
   that cannot address RAM at positive 0x80000000. SDK console output uses the
   pinned embedded printf, with floating formatting disabled, and provides a
   small medany CLZ helper. FP32 evidence is printed as exact hexadecimal bits.
   Arithmetic still uses hardware FP32 and the existing Newlib math functions.
6. An initial IREE run with RVV enabled stopped retiring instructions in the
   bytecode verifier after a compiler-generated fractional-LMUL vector sequence
   (last retired PC 0x8001c4a6). The 5,000,000-cycle timeout rejected that run.
   This is an unresolved RVV integration observation, not a diagnosed root cause.
   The initial SDK profile therefore compiles runtime and kernels as RV64GC,
   plus explicit matrix custom-1 instructions. An independent trace check rejects
   unexpected vector retirement. Existing hardware RVV smoke results remain
   valid within their original scope. No RTL was modified to bypass the issue.
7. The control peripheral takes a raw exit code and adds the simulator valid bit
   itself. Encoding a tohost bit in startup incorrectly reported failure despite
   correct IREE outputs. SDK startup now matches the hardware BSP contract.
8. Clang eliminated a malloc/free exhaustion probe when the pointer did not
   escape, so the test initially failed without calling the allocator. A volatile
   function pointer now forces the real allocation call. The positive RTL run
   verifies rejection of 17 MiB, then successfully runs IREE with the same heap.
9. Scalar RV64GC selects a Clang multilib include directory missing from the
   existing toolchain. The SDK explicitly selects the installed Newlib headers;
   it does not change the hardware lab wrapper or toolchain installation.

The compiler-generated model graph currently invokes semantic C kernels through
IREE static dispatch. It is not an automatic arbitrary-model lowering pipeline.
All decoder operations run on target, but only the linear projections and final
classifier use the INT8 matrix unit. Attention scores, softmax, value aggregation,
RoPE, RMSNorm, activation and residual additions use scalar FP32.

## DDR placement qualification

The DDR profile preserves the model and kernel math sources and changes only
placement, linker/loader integration and checks. The first cross-link attempt
rejected the linker length suffix `4G`; specifying `0x100000000` fixed that syntax
error. The failed build log is retained in local artifacts. The first successful
model step checked all 512 logits, all 36 projections and 260,608 external reads.
Its logits were bit-identical to the original SRAM deployment.

This does not resolve the earlier compiler-generated RVV limitation. DDR weights
are read by the existing CPU-fed matrix kernels; vector compilation remains off.
