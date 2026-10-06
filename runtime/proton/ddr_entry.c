// Placement checks run before the unchanged IREE model entry point.
#include <stdint.h>
#include <stdio.h>
#include "model_data.h"

int __real_main(void);
int __wrap_main(void) {
  unsigned total = 0;
  for (unsigned i = 0; i < 36; ++i) {
    const proton_weight_t *w = &weights[i];
    uintptr_t p = (uintptr_t)w->packed, s = (uintptr_t)w->scale;
    unsigned bytes = ((w->n + 3) / 4) * ((w->k + 15) / 16) * 64;
    if (p < UINT64_C(0x100000000) || p + bytes > UINT64_C(0x200000000) ||
        s < UINT64_C(0x80000000) || s + w->n * 4 > UINT64_C(0x81000000)) {
      printf("FAIL DDR placement id=%u\n", i);
      return 1;
    }
    printf("DDR_WEIGHT id=%u address=%lx bytes=%u\n", i, (unsigned long)p, bytes);
    total += bytes;
  }
  printf("DDR_PLACEMENT weights=36 bytes=%u working_memory=SRAM\n", total);
  return __real_main();
}
