// SPDX-License-Identifier: Apache-2.0
.section .text.init
.globl _start
_start:
  .option push
  .option norelax
  la gp, __global_pointer$
  .option pop
  la sp, __stack_top
  csrwi mie, 0
  csrwi medeleg, 0
  csrwi mideleg, 0
  li t0, 0x2200 // Initial FS and VS, stay in machine mode.
  csrw mstatus, t0
  csrwi fcsr, 0
  la t0, trap_entry
  csrw mtvec, t0
  la t0, __bss_start
  la t1, __bss_end
1:
  bgeu t0, t1, 2f
  sd zero, 0(t0)
  addi t0, t0, 8
  j 1b
2:
  call main
  // The control peripheral adds its own valid bit; write the raw exit code.
  li t0, 0xd0000000
  sd a0, 0(t0)
3: j 3b
.balign 4
trap_entry:
  csrr a0, mcause
  csrr a1, mepc
  csrr a2, mtval
  call proton_trap
  j 3b
