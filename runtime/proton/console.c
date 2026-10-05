// SPDX-License-Identifier: Apache-2.0
// Route formatted output through IREE's pinned embedded printf dependency.
// Newlib's full printf pulls medlow-only quad conversion helpers in the lab's
// existing GCC installation; no quad formatting is needed by this deployment.
#include <stdio.h>
#include <stdarg.h>
#include "printf/printf.h"
extern void _putchar(char);
static void emit(char c,void *unused) { (void)unused; _putchar(c); }
int printf(const char *fmt,...) { va_list a; va_start(a,fmt); int n=vfctprintf(emit,NULL,fmt,a); va_end(a); return n; }
int fprintf(FILE *f,const char *fmt,...) { (void)f; va_list a; va_start(a,fmt); int n=vfctprintf(emit,NULL,fmt,a); va_end(a); return n; }
int vfprintf(FILE *f,const char *fmt,va_list a) { (void)f; return vfctprintf(emit,NULL,fmt,a); }
int puts(const char *s) { while(*s) _putchar(*s++); _putchar('\n'); return 0; }
int fputs(const char *s,FILE *f) { (void)f; while(*s) _putchar(*s++); return 0; }
int putchar(int c) { _putchar((char)c); return c; }
// Portable medany implementation for the lab's medlow libgcc helper.
__attribute__((optnone)) int __clzdi2(unsigned long long x) {
  int n=0; if(!x) return 64;
  while(!(x & (1ULL<<63))) { ++n; x<<=1; } return n;
}
void __libc_fini(void) {}
