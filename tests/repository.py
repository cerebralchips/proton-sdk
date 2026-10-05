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
