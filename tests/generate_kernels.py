#!/usr/bin/env python3
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'models'))
from stories import Model,quantize
m=Model(sys.argv[1]); dest=Path(sys.argv[2]); dest.mkdir(parents=True,exist_ok=True)
rows=[(m.qw[name][0][l],m.qw[name][1][l]) for l in range(5) for name in ['q','k','v','o','up','gate','down']]
rows.append(m.qw['cls'])
data=['#pragma once']
def emit(ctype,name,a):
    v=[float(x).hex()+'f' for x in a] if ctype=='float' else [str(int(x)) for x in a]
    data.append(f'static const {ctype} test_{name}[]={{'+','.join(v)+'};')
for i,(w,scale) in enumerate(rows):
    n,k=w.shape
    q=((np.arange(k,dtype=np.int32)*17+i*13)%256-128).astype(np.int8)
    x=np.sin(np.arange(k,dtype=np.float32)*np.float32(.71)+np.float32(i))*np.float32(2)
    xq,s=quantize(x)
    exact=w.astype(np.int32)@q.astype(np.int32)
    fp=((w.astype(np.int32)@xq.astype(np.int32)).astype(np.float32)*s)*scale
    for ctype,name,a in [('int8_t','q',q),('float','x',x),('int32_t','acc',exact),('float','y',fp)]: emit(ctype,f'{name}{i}',a)
data.append('static const struct { const int8_t *q; const float *x; const int32_t *acc; const float *y; } cases[]={'+
            ','.join('{test_q%d,test_x%d,test_acc%d,test_y%d}'%(i,i,i,i) for i in range(len(rows)))+'};')
(dest/'kernel_cases.h').write_text('\n'.join(data)+'\n')
