// SPDX-License-Identifier: Apache-2.0
#include <stdint.h>
#include <stddef.h>
#include <sys/stat.h>
#include <errno.h>
#include <unistd.h>
#include <stdlib.h>
#include "platform.h"

extern char __heap_start[], __heap_end[], __stack_bottom[], __stack_top[];
static char *heap_break;
uintptr_t proton_heap_peak;
int proton_memory_begin(void) {
  uintptr_t sp; __asm__ volatile("mv %0, sp":"=r"(sp));
  if(sp<(uintptr_t)__stack_bottom+4096 || sp>(uintptr_t)__stack_top) return 1;
  volatile uint64_t *guard=(volatile uint64_t *)__stack_bottom;
  for(unsigned i=0;i<32;++i) guard[i]=0x50524f544f4e0000ULL+i;
  // Deliberately exceed physical RAM: the bounded allocator must reject this.
  // Prevent allocation-elision optimizations from removing this failure probe.
  void *(*volatile allocate)(size_t)=malloc;
  void *p=allocate(17U*1024U*1024U);
  if(p) { free(p); return 2; }
  return 0;
}
int proton_memory_end(void) {
  volatile uint64_t *guard=(volatile uint64_t *)__stack_bottom;
  for(unsigned i=0;i<32;++i) if(guard[i]!=0x50524f544f4e0000ULL+i) return 1;
  uintptr_t sp; __asm__ volatile("mv %0, sp":"=r"(sp));
  return sp<(uintptr_t)__stack_bottom+4096 || sp>(uintptr_t)__stack_top;
}
void _putchar(char c) { *(volatile uint8_t *)0xc0000000UL = c; }
int _write(int fd, const void *buf, size_t n) {
  (void)fd;
  const char *p = buf;
  for (size_t i = 0; i < n; ++i) _putchar(p[i]);
  return (int)n;
}
void *_sbrk(ptrdiff_t increment) {
  if (!heap_break) heap_break = __heap_start;
  if (increment < 0 || (uintptr_t)increment > (uintptr_t)(__heap_end - heap_break)) {
    errno = ENOMEM; return (void *)-1;
  }
  char *old = heap_break;
  heap_break += increment;
  proton_heap_peak = (uintptr_t)(heap_break - __heap_start);
  return old;
}
int _close(int fd) { (void)fd; return -1; }
int _fstat(int fd, struct stat *st) { (void)fd; st->st_mode = S_IFCHR; return 0; }
int _isatty(int fd) { (void)fd; return 1; }
off_t _lseek(int fd, off_t o, int w) { (void)fd; (void)o; (void)w; return 0; }
int _read(int fd, void *b, size_t n) { (void)fd; (void)b; (void)n; return 0; }
int _getpid(void) { return 1; }
int _kill(int pid, int sig) { (void)pid; (void)sig; return -1; }
__attribute__((noreturn)) void _exit(int code) {
  *(volatile uint64_t *)0xd0000000UL = (uint64_t)(unsigned)code;
  for (;;) __asm__ volatile("nop");
}
extern int printf(const char *, ...);
void proton_trap(uintptr_t cause, uintptr_t pc, uintptr_t value) {
  printf("TRAP cause=%lx pc=%lx value=%lx\n", cause, pc, value);
  _exit(2);
}
