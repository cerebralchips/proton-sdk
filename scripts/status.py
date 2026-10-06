#!/usr/bin/env python3
"""Read progress without acquiring the simulator/build lock."""
from pathlib import Path
import json
root=Path(__file__).resolve().parents[1]
work=root/'artifacts'
for p in sorted((work/'runs').glob('*')):
    if not p.is_dir(): continue
    result=p/'result.json'
    if result.exists():
        d=json.loads(result.read_text())
        print(p.name,d['result'],f"cycles={d.get('rtl_cycles','unavailable')}")
    else:
        print(p.name,'INCOMPLETE (running or interrupted)')
logs=sorted((work/'attempts').glob('*/model*-rtl.log'))
if logs:
    p=logs[-1]
    tokens=[s for s in p.read_text().splitlines() if s.startswith('TOKEN ')]
    print(f'Latest model log: {p.parent.name}/{p.name}; {len(tokens)} completed tokens')
    for s in tokens: print(s)
