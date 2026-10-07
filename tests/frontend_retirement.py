#!/usr/bin/env python3
"""Audit existing single-token RTL traces; never launches a simulation."""
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): digest.update(chunk)
    return digest.hexdigest()

def main():
    records={}
    for mode in ['fp32','w8a8']:
        result_path=ROOT/'verification'/f'frontend-rtl-{mode}.json'
        result=json.loads(result_path.read_text())
        if result['result']!='PASS' or result['steps']!=1: raise ValueError('Expected passing one-step run')
        run=ROOT/'artifacts/frontend-rtl-runs'/result['run_id']/mode
        counts={'retired':0,'vector':0,'matrix':0}
        with (run/'trace_hart_0.dasm').open() as f:
            for line in f:
                match=re.match(r'\s*(\d+) 0x[0-9a-f]+ [MSU] \(0x([0-9a-f]+)\)',line)
                if not match: continue
                counts['retired']+=1
                insn=int(match[2],16)
                op=insn&127
                if op==0x57 or (op in [0x07,0x27] and ((insn>>12)&7) in [0,5,6,7]): counts['vector']+=1
                if op==0x2b: counts['matrix']+=1
        if counts['retired']<=0 or counts['vector'] or counts['matrix']:
            raise ValueError('Unexpected instruction counts: '+str(counts))
        log=(run/'rtl.log').read_text()
        ddr=re.findall(r'DDR_ACCESSES reads=(\d+) writes=(\d+) allocated_words=(\d+)',log)
        if ddr!=[('0','0','0')]: raise ValueError('Unexpected external-memory activity')
        events=(run/'matrix-events.csv').read_text().strip().splitlines()
        if len(events)!=1: raise ValueError('Unexpected matrix command events')
        records[mode]={'result':'PASS','run_id':result['run_id'],'instructions':counts,
            'ddr_reads':0,'ddr_writes':0,'matrix_events':0,
            'result_sha256':sha(result_path),'trace_sha256':sha(run/'trace_hart_0.dasm'),
            'matrix_events_sha256':sha(run/'matrix-events.csv')}
        print(mode,counts)
    output={'result':'PASS','runs':records,'checker_sha256':sha(Path(__file__)),
            'additional_model_invocations':0}
    (ROOT/'verification/frontend-rtl-retirement.json').write_text(json.dumps(output,indent=2)+'\n')

if __name__=='__main__': main()
