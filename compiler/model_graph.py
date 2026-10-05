#!/usr/bin/env python3
"""Explicit lowered decoder graph. No compiler or hardware source modification.

Each dispatch has one workgroup and a tied read/write state buffer. This is a
first integration of semantic kernels, not a general PyTorch importer.
"""
from pathlib import Path
import sys
size=4+64*4+32*2+172*2+512+2*5*64*32
ops=[(0,0)]+[(op,l) for l in range(5) for op in range(1,14)]+[(14,0),(15,0)]
ir=['module @proton_model {', '''
  stream.executable private @ops {
    stream.executable.export public @apply workgroups() -> (index,index,index) {
      %one = arith.constant 1 : index
      stream.return %one, %one, %one : index,index,index
    }
    builtin.module {
      func.func private @proton_op(%base: memref<f32>, %offset: index, %op: i32, %layer: i32)
        attributes {hal.import.static, llvm.bareptr = true}
      func.func @apply(%binding: !stream.binding, %op: i32, %layer: i32) {
        %zero = arith.constant 0 : index
''',f'        %mem = stream.binding.subspan %binding[%zero] : !stream.binding -> memref<{size}xf32>',
f'        %base, %offset, %sz, %stride = iree_codegen.extract_strided_metadata %mem : memref<{size}xf32> -> memref<f32>, index, index, index',
'''        func.call @proton_op(%base, %offset, %op, %layer) : (memref<f32>,index,i32,i32) -> ()
        return
      }
    }
  }
''',f'  func.func @step(%state: tensor<{size}xf32>) -> tensor<{size}xf32> {{']
for i in range(16): ir.append(f'    %c{i} = arith.constant {i} : i32')
current='%state'
for i,(op,l) in enumerate(ops):
    nxt=f'%s{i}'
    ir.append(f'    {nxt} = flow.dispatch @ops::@apply[]({current}, %c{op}, %c{l}) : (tensor<{size}xf32>,i32,i32) -> {current}')
    current=nxt
ir.extend([f'    return {current} : tensor<{size}xf32>','  }','}'])
Path(sys.argv[1]).write_text('\n'.join(ir)+'\n')
