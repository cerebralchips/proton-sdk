#!/usr/bin/env python3
"""Independent host checks of target results and architectural execution."""
import csv
import hashlib
import json
from pathlib import Path
import re
import struct
import sys
import numpy as np

def check(directory,reference=None):
    d=Path(directory); log=(d/'rtl.log').read_text()
    if '*** SUCCESS ***' not in log or 'RESULT: PASS' not in log or any(x in log for x in ['FAIL','TRAP','Simulation timeout','%Error']):
        raise ValueError('Missing clean RTL completion or numerical PASS')
    cycles=re.search(r'Executed cycles:\s*(\d+)',log)
    if not cycles or int(cycles[1])<=0: raise ValueError('Invalid cycle count')
    expected=[0,0]
    if match:=re.search(r'MATRIX_COUNTS zero=(\d+) macc=(\d+)',log): expected=list(map(int,match.groups()))
    if 'PASS kernels' in log or 'PASS kernel_tile' in log:
        cases=[tuple(map(int,x)) for x in re.findall(r'KERNEL PASS id=(\d+) N=(\d+) K=(\d+)',log)]
        count=1 if 'PASS kernel_tile' in log else 36
        if [c[0] for c in cases]!=list(range(count)): raise ValueError('Missing kernel cases')
        calculated=[sum(2*((n+3)//4) for _,n,k in cases),sum(2*((n+3)//4)*((k+15)//16) for _,n,k in cases)]
        if expected!=calculated: raise ValueError('Kernel counter formula mismatch')
    counts={'retired':0,'mzero':0,'mmacc':0,'vector':0}
    with (d/'retired-matrix.csv').open('w') as out:
        out.write('cycle,pc,encoding,operation\n')
        with (d/'trace_hart_0.dasm').open() as stream:
            for line in stream:
                m=re.match(r'\s*(\d+) 0x([0-9a-f]+) ([MSU]) \(0x([0-9a-f]+)\)',line)
                if not m: continue
                counts['retired']+=1; cyc,pc,mode,raw=m.groups(); insn=int(raw,16)
                if (insn&127)==0x57 or ((insn&127) in [0x07,0x27] and ((insn>>12)&7) in [0,5,6,7]): counts['vector']+=1
                if (insn&127)==0x2b:
                    masked=insn&~(31<<7)
                    if masked not in [0x2b,0x0200002b]: raise ValueError('Reserved matrix opcode retired')
                    op='mmacc' if masked==0x0200002b else 'mzero'; counts[op]+=1
                    out.write(f'{cyc},0x{pc},0x{raw},{op}\n')
    if [counts['mzero'],counts['mmacc']]!=expected: raise ValueError('Retirement/counter mismatch')
    if counts['vector']: raise ValueError('Unexpected RVV in scalar runtime profile')
    event_counts=[0,0]; group=[]
    with (d/'matrix-events.csv').open() as stream:
        for row in csv.DictReader(stream):
            group.append(row)
            if len(group)==3:
                if [r['event'] for r in group]!=['issue','done','response']: raise ValueError('Command event ordering')
                if len({r['transaction'] for r in group})!=1: raise ValueError('Command transaction mismatch')
                times=[int(r['cycle']) for r in group]; op=int(group[0]['operation'])
                if op not in [0,1] or not times[0]<times[1]<times[2]: raise ValueError('Invalid event')
                if times[1]-times[0]!=(12 if op else 1): raise ValueError('Matrix command latency')
                event_counts[op]+=1; group=[]
    if group or event_counts!=expected: raise ValueError('Issue/done/response counts mismatch')
    result={'result':'PASS','rtl_cycles':int(cycles[1]),'instructions':counts,'matrix_counts':expected,'matched_commands':sum(event_counts)}
    if 'PASS iree_model' in log:
        if reference is None: raise ValueError('Model reference required')
        ref=np.load(reference)
        tokens=[tuple(map(int,m)) for m in re.findall(r'TOKEN pos=(\d+) id=(\d+) invoke_cycles=(\d+)',log)]
        header=re.search(r'MODEL steps=(\d+) backend=(\d+) invoke_cycles=(\d+) max_error_bits=([0-9a-f]+)',log)
        if not header: raise ValueError('Missing model metadata')
        steps,backend,invocations=map(int,header.groups()[:3])
        if len(tokens)!=steps or [t[0] for t in tokens]!=list(range(steps)): raise ValueError('Token count/order')
        if expected!=[878*steps*backend,4072*steps*backend]: raise ValueError('Model matrix count formula')
        projections=[tuple(map(int,x)) for x in re.findall(r'PROJECTION id=(\d+) N=(\d+) K=(\d+) calls=(\d+) zero=(\d+) macc=(\d+)',log)]
        if projections:
            if [x[0] for x in projections]!=list(range(36)): raise ValueError('Projection coverage incomplete')
            for _,n,k,c,z,m in projections:
                ez=((n+3)//4)*steps*backend
                if c!=steps or z!=ez or m!=ez*((k+15)//16): raise ValueError('Projection count mismatch')
            result['projections']=projections
        logits=np.full((steps,512),np.nan,np.float32); seen=set()
        for p,i,raw in re.findall(r'LOGITS (\d+) (\d+) ([0-9a-f ]+)',log):
            p,i=int(p),int(i)
            if (p,i) in seen or p>=steps or i%8 or i>=512: raise ValueError('Duplicate/out-of-range logits')
            seen.add((p,i)); values=[struct.unpack('<f',struct.pack('<I',int(x,16)))[0] for x in raw.split()]
            if len(values)!=8: raise ValueError('Incomplete logit row')
            logits[p,i:i+8]=values
        if not np.isfinite(logits).all(): raise ValueError('Missing/nonfinite logits')
        oracle=ref['q_logits'][:steps]
        np.testing.assert_allclose(logits,oracle,atol=.0001,rtol=.00001)
        if [t[1] for t in tokens]!=ref['q_tokens'][:steps].tolist(): raise ValueError('Reference token mismatch')
        if np.argmax(logits,axis=1).tolist()!=[t[1] for t in tokens]: raise ValueError('Tokens are not argmax of target logits')
        if sum(t[2] for t in tokens)!=invocations: raise ValueError('Invocation cycle mismatch')
        result['model']={'steps':steps,'backend':backend,'tokens':[t[1] for t in tokens],
            'invoke_cycles':invocations,'max_abs_logit_error':float(np.max(np.abs(logits-oracle))),
            'rms_logit_error':float(np.sqrt(np.mean((logits-oracle)**2)))}
        np.save(d/'target-logits.npy',logits)
    result['sha256']={name:hashlib.sha256((d/name).read_bytes()).hexdigest() for name in ['program.elf','program.dump','rtl.log','matrix-events.csv']}
    (d/'evidence.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in {'projections','sha256'}},indent=2))
    return result

if __name__=='__main__': check(sys.argv[1],sys.argv[2] if len(sys.argv)>2 else None)
