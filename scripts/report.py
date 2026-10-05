#!/usr/bin/env python3
"""Create compact, portable records from completed local verification only."""
from pathlib import Path
import hashlib
import json
import shutil
import sys
import numpy as np

root=Path(__file__).resolve().parents[1]; work=root.parent
out=root/'verification'; out.mkdir(exist_ok=True)
names=['smoke','kernels','model_matrix','model_scalar_step','kernel_tile']
records={}
selected={}
hw=Path('/ara-workspace')
baseline=sorted(hw.glob('artifacts/*-matrix-run-scalar_vector_matrix-*/result.json'))
baseline=[p for p in baseline if json.loads(p.read_text()).get('result')=='PASS']
if not baseline: raise RuntimeError('Missing hardware baseline')
b=json.loads(baseline[-1].read_text())
record={'result':b['result'],'run':str(baseline[-1].parent.relative_to(hw)),
        'rtl_cycles':b['rtl_cycles'],'elf_sha256':b['elf_sha256'],'simulator_sha256':b['simulator_sha256'],
        'execution':b['execution'],'sources':b['sources'],'rtl_sha256':b['rtl_sha256'],'patch_sha256':b['patch_sha256']}
(out/'hardware-baseline.json').write_text(json.dumps(record,indent=2)+'\n')
units=sorted(hw.glob('artifacts/*-matrix-unit-*/result.json'))
units=[p for p in units if json.loads(p.read_text()).get('result')=='PASS']
if not units: raise RuntimeError('Missing hardware unit gates')
unit=json.loads(units[-1].read_text()); unit['run']=str(units[-1].parent.relative_to(hw))
(out/'hardware-units.json').write_text(json.dumps(unit,indent=2)+'\n')
if not (work/'bootstrap.json').exists(): raise RuntimeError('Missing source integrity verification')
shutil.copy2(work/'bootstrap.json',out/'dependencies.json')
for name in names:
    candidates=sorted((work/'runs').glob('*-'+name))
    good=[p for p in candidates if (p/'result.json').exists() and json.loads((p/'result.json').read_text())['result']=='PASS']
    if not good: raise RuntimeError(f'No passing run: {name}')
    p=good[-1]; result=json.loads((p/'result.json').read_text())
    selected[name]=p
    evidence=json.loads((p/'evidence.json').read_text())
    if name=='model_matrix' and evidence['model']['steps']!=16: raise RuntimeError('Require 16 model steps')
    if name=='kernel_tile' and result.get('waveform')!='PASS': raise RuntimeError('Require checked FST')
    records[name]={'run':str(p.relative_to(work)),'evidence':evidence,
                   'simulator_sha256':result['simulator_sha256']}
    records[name]['trace_sha256']={}
    for filename in ['trace_hart_0.dasm','retired-matrix.csv']:
        with (p/filename).open('rb') as stream:
            records[name]['trace_sha256'][filename]=hashlib.file_digest(stream,'sha256').hexdigest()
    if name!='kernel_tile' and result['simulator_sha256']!=record['simulator_sha256']:
        raise RuntimeError('SDK simulator differs from verified hardware baseline')
    (out/(name+'.json')).write_text(json.dumps(records[name],indent=2)+'\n')
    if name=='model_matrix':
        shutil.copy2(p/'provenance.json',out/'model-provenance.json')
        # Keep an easy-to-read UART record, without absolute loader/build paths.
        log=(p/'rtl.log').read_text()
        lines=[s for s in log.splitlines() if s.startswith(('TOKEN','PROJECTION','MATRIX_COUNTS','MODEL ','MEMORY_GUARD','RESULT:','Executed cycles:'))]
        (out/'model-output.txt').write_text('\n'.join(lines)+'\n')
        matrix_first=int(__import__('re').search(r'TOKEN pos=0 id=\d+ invoke_cycles=(\d+)',log)[1])
        from_data={'run':str(p.relative_to(work)),'heap_peak':int(__import__('re').search(r'heap_peak=(\d+)',log)[1]),
          'state_bytes':86640,'reserved_stack_bytes':262144}
        from_data['elf_segments']=json.loads((p/'provenance.json').read_text())['segments']
        (out/'memory.json').write_text(json.dumps(from_data,indent=2)+'\n')
    if name=='kernel_tile':
        wave=json.loads((p/'matrix-wave-check.json').read_text())
        (out/'waveform.json').write_text(json.dumps({'result':wave['result'],'commands':wave['commands'],
          'run':str(p.relative_to(work)),'timescale':wave['timescale'],'first_tiles':wave['tiles'][:5]},indent=2)+'\n')
negatives=sorted((work/'attempts').glob('*/negative-results.json'))
if not negatives: raise RuntimeError('Missing failure-path verification')
neg=json.loads(negatives[-1].read_text())
if neg['result']!='PASS': raise RuntimeError('Negative tests failed')
shutil.copy2(negatives[-1],out/'negative.json')
shutil.copy2(work/'generated/model/reference.json',out/'reference.json')
scalar=records['model_scalar_step']['evidence']['model']['invoke_cycles']
matrix_logits=np.load(selected['model_matrix']/'target-logits.npy')[0]
scalar_logits=np.load(selected['model_scalar_step']/'target-logits.npy')[0]
if not np.array_equal(matrix_logits.view(np.uint32),scalar_logits.view(np.uint32)):
    raise RuntimeError('Matrix/scalar target logits differ; investigate before performance comparison')
comparison={'scope':'First BOS decoder step, same W8A8 arithmetic and IREE graph, instrumented builds',
 'matrix_invoke_cycles':matrix_first,'scalar_invoke_cycles':scalar,'scalar_divided_by_matrix':scalar/matrix_first,
 'target_logits_bitwise_identical':True,
 'excludes':'runtime setup, input/output copies, numerical checks and UART evidence output',
 'limitations':'Functional RTL cycles only. No silicon frequency, wall-time throughput, or general model speedup claim.'}
(out/'comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
gates={
 'G0':{'result':'PASS','evidence':['hardware-baseline.json','hardware-units.json']},
 'G1':{'result':'PASS','evidence':['dependencies.json','reference.json','memory.json']},
 'G2':{'result':'PASS','evidence':['smoke.json']},
 'G3':{'result':'PASS','evidence':['kernels.json','waveform.json']},
 'G4':{'result':'PASS','evidence':['model_matrix.json','model-provenance.json']},
 'G5':{'result':'PASS','evidence':['model_matrix.json','model-output.txt']},
 'G6':{'result':'PASS','evidence':['negative.json','comparison.json']}}
(out/'gates.json').write_text(json.dumps(gates,indent=2)+'\n')
print(json.dumps({'result':'PASS','runs':{k:v['run'] for k,v in records.items()},'comparison':comparison},indent=2))
