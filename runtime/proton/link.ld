OUTPUT_ARCH("riscv")
ENTRY(_start)
MEMORY { RAM (rwx) : ORIGIN = 0x80000000, LENGTH = 16M }
SECTIONS {
  .text : { KEEP(*(.text.init)) *(.text .text.*) } > RAM
  .rodata : ALIGN(16) { *(.rodata .rodata.*) } > RAM
  .data : ALIGN(16) { *(.data .data.*) } > RAM
  .sdata : ALIGN(16) { __global_pointer$ = . + 0x800; *(.sdata .sdata.* .srodata .srodata.*) } > RAM
  .init_array : ALIGN(8) { __init_array_start = .; *(.init_array .init_array.*) __init_array_end = .; } > RAM
  .fini_array : ALIGN(8) { __fini_array_start = .; *(.fini_array .fini_array.*) __fini_array_end = .; } > RAM
  .bss (NOLOAD) : ALIGN(16) {
    __bss_start = .; *(.bss .bss.* .sbss .sbss.* COMMON) . = ALIGN(16); __bss_end = .;
  } > RAM
  __heap_start = ALIGN(16);
  __heap_end = ORIGIN(RAM) + LENGTH(RAM) - 256K;
  __stack_bottom = __heap_end;
  __stack_top = ORIGIN(RAM) + LENGTH(RAM);
  ASSERT(__heap_start + 2M <= __heap_end, "ELF leaves less than 2 MiB heap")
  /DISCARD/ : { *(.comment) *(.eh_frame .eh_frame.*) }
}
