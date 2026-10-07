// SPDX-License-Identifier: Apache-2.0
// Host-side probe for the pinned IREE layout API; no model/RTL invocation.
#include <stdio.h>
#include "iree/base/allocator.h"
#include "iree/hal/command_buffer.h"

int main(void) {
  unsigned old_misaligned=0;
  for(unsigned bytes=0;bytes<=32;bytes+=4) {
    iree_host_size_t total=0,offset=0,old_offset=0;
    iree_status_t status=IREE_STRUCT_LAYOUT(128,&total,
      IREE_STRUCT_FIELD_ALIGNED(bytes,uint8_t,iree_max_align_t,NULL),
      IREE_STRUCT_FIELD_ALIGNED(3,iree_hal_buffer_ref_t,
        iree_alignof(iree_hal_buffer_ref_t),&offset));
    if(!iree_status_is_ok(status) || offset%iree_alignof(iree_hal_buffer_ref_t) ||
       offset<128+bytes || total<offset+3*sizeof(iree_hal_buffer_ref_t)) return 1;
    status=IREE_STRUCT_LAYOUT(128,&total,
      IREE_STRUCT_FIELD_ALIGNED(bytes,uint8_t,iree_max_align_t,NULL),
      IREE_STRUCT_FIELD_ALIGNED(3,iree_hal_buffer_ref_t,1,&old_offset));
    if(!iree_status_is_ok(status)) return 2;
    old_misaligned+=old_offset%iree_alignof(iree_hal_buffer_ref_t)!=0;
  }
  if(sizeof(void*)!=8 || old_misaligned!=4) return 3;
  printf("ALIGNMENT PASS: 9 corrected layouts; 4 original misaligned layouts reproduced\n");
  return 0;
}
