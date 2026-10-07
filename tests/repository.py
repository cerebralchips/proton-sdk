#!/usr/bin/env python3
"""Check documentation links and bind final source to executed model evidence."""
from pathlib import Path
import hashlib
import json
import py_compile
import re
root=Path(__file__).resolve().parents[1]
skip={'.git','artifacts','__pycache__','.venv','build','third_party'}
files=[]
for p in root.rglob('*'):
    if set(p.relative_to(root).parts)&skip or not p.is_file(): continue
    files.append(p)
    if p.suffix=='.py': py_compile.compile(p,doraise=True)
    if p.suffix=='.json': json.loads(p.read_text())
    if p.suffix=='.md':
        for dest in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            if '://' in dest or dest.startswith('#'): continue
            dest=dest.split('#')[0]
            if not (p.parent/dest).exists(): raise ValueError(f'Broken link: {p.relative_to(root)} -> {dest}')
gates=json.loads((root/'verification/gates.json').read_text())
assert set(gates)=={f'G{i}' for i in range(7)}
for gate in gates.values():
    assert gate['result']=='PASS'
    for name in gate['evidence']: assert (root/'verification'/name).is_file()
provenance=json.loads((root/'verification/model-provenance.json').read_text())
math_sources=['kernels/linear.c','kernels/model_ops.c','runtime/iree_runner.c','runtime/model_state.h',
              'runtime/proton/platform.c','runtime/proton/start.S','runtime/proton/link.ld',
              'runtime/proton/console.c','runtime/iree_config.h','models/stories.py','compiler/model_graph.py']
for name in math_sources:
    actual=hashlib.sha256((root/name).read_bytes()).hexdigest()
    assert actual==provenance['sdk_sha256'][name],f'Executed model source differs: {name}'
model=json.loads((root/'verification/model_matrix.json').read_text())['evidence']
assert model['model']['steps']==16
assert model['matrix_counts']==[14048,65152]
assert len(model['projections'])==36
print(f'PASS: {len(files)} local source/record files, links, seven gates and executed model source hashes')
ddr=json.loads((root/'verification/ddr.json').read_text())
ddr_provenance=json.loads((root/'verification/ddr-provenance.json').read_text())
assert ddr['result']=='PASS'
assert ddr_provenance['target']==json.loads((root/'targets/proton_v1_ddr.json').read_text())
for name in math_sources+['CMakeLists.txt','cmake/proton.cmake','runtime/proton/ddr_entry.c',
                         'runtime/proton/link-ddr.ld','scripts/ddr.py','scripts/build.py',
                         'scripts/preflight.py','targets/proton_v1_ddr.json']:
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==ddr_provenance['sdk_sha256'][name],name
for name,run in ddr['runs'].items():
    assert run['result']=='PASS' and run['ddr']['result']=='PASS'
    assert run['ddr']['reads']>0 and run['ddr']['writes']==0
    assert run['ddr']['packed_weight_bytes']==260608
full=ddr['runs']['model_matrix']['evidence']
assert full['model']['steps']==16 and full['matrix_counts']==[14048,65152]
assert ddr['scalar_matrix_comparison']=='PASS bit-identical first-step logits'
assert json.loads((root/'verification/ddr-negative.json').read_text())['result']=='PASS'
print('PASS: DDR deployment, memory traffic, source hashes and scalar/matrix comparison')

# Host frontend evidence is independent of the historical RTL/DDR gates.
frontend=json.loads((root/'verification/onnx-stories260k.json').read_text())
assert frontend['result']=='PASS'
assert frontend['execution'].endswith('not RTL')
for name, expected in frontend['sdk_sha256'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected, name
assert set(frontend['runs'])=={'fp32','w8a8'}
compared=0
for mode, backends in frontend['runs'].items():
    assert set(backends)=={'onnxruntime','iree'}
    for backend, result in backends.items():
        assert result['result']=='PASS' and result['steps']==16
        assert len(result['tokens'])==16 and len(result['cases'])==20
        for case in result['cases']:
            for name, elements in [('logits',512),('keys',10240),('values',10240)]:
                assert case[name]['elements']==elements
                compared+=elements
assert compared==1679360
assert frontend['quantization_probe']['result']=='PASS'
assert frontend['quantization_probe']['comparison']=='exact equality'
assert frontend['process_failure_checks']=={'result':'PASS','rejected_outputs':['logits','keys','values']}
for mode in ['fp32','w8a8']:
    for suffix in ['onnx','torch.mlir','vmfb']:
        artifact=frontend['artifacts'][f'stories260k.{mode}.{suffix}']
        assert artifact['bytes']>0 and len(artifact['sha256'])==64
print('PASS: Stories260K Torch MLIR host frontend evidence and source fingerprints (not RTL)')

retirement=json.loads((root/'verification/frontend-rtl-retirement.json').read_text())
assert retirement['result']=='PASS' and retirement['additional_model_invocations']==0
assert hashlib.sha256((root/'tests/frontend_retirement.py').read_bytes()).hexdigest()==retirement['checker_sha256']
for mode in ['fp32','w8a8']:
    path=root/'verification'/f'frontend-rtl-{mode}.json'
    result=json.loads(path.read_text())
    assert result['result']=='PASS' and result['steps']==1
    assert result['compared_elements']==20992 and result['token']==403
    assert 0<result['invoke_cycles']<result['rtl_cycles']
    assert result['target']==json.loads((root/'targets/proton_frontend_cpu.json').read_text())
    for name,digest in result['sdk_sha256'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
    assert result['negative_checks']['result']=='PASS'
    assert result['negative_checks']['additional_rtl_invocations']==0
    assert len(result['negative_checks']['rejected'])==7
    audit=retirement['runs'][mode]
    assert audit['result_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
    assert audit['instructions']['retired']>0
    assert audit['instructions']['matrix']==audit['instructions']['vector']==0
    assert audit['ddr_reads']==audit['ddr_writes']==audit['matrix_events']==0
print('PASS: single-token FP32/W8A8 frontend RTL results, source hashes and retirement audit')
