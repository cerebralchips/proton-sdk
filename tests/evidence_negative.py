#!/usr/bin/env python3
"""Ensure the independent result checker rejects corrupted model evidence."""
from pathlib import Path
import contextlib
import io
import json
import shutil
import sys
import tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from evidence import check

source=Path(sys.argv[1]); reference=Path(sys.argv[2]); results=[]
mutations={
 'missing-completion':lambda s:s.replace('*** SUCCESS ***','completion removed'),
 'wrong-logit':lambda s:__import__('re').sub(r'(LOGITS 0 0 )([0-9a-f]{8})',r'\g<1>7fc00000',s,count=1),
 'wrong-token':lambda s:__import__('re').sub(r'(TOKEN pos=0 id=)\d+',r'\g<1>0',s,count=1),
 'wrong-counter':lambda s:s.replace('MATRIX_COUNTS zero=878','MATRIX_COUNTS zero=879',1),
}
for name,mutate in mutations.items():
    with tempfile.TemporaryDirectory(prefix='proton-negative-') as td:
        dest=Path(td)
        for p in source.iterdir():
            if p.is_file(): (dest/p.name).symlink_to(p.resolve())
        log=(source/'rtl.log').read_text(); changed=mutate(log)
        if changed==log: raise ValueError(f'Injection did not alter input: {name}; use the one-step matrix run')
        (dest/'rtl.log').unlink(); (dest/'rtl.log').write_text(changed)
        # Checker output files must never alias the positive evidence.
        for p in ['retired-matrix.csv','evidence.json','target-logits.npy']:
            (dest/p).unlink(missing_ok=True)
        try:
            with contextlib.redirect_stdout(io.StringIO()): check(dest,reference)
        except (ValueError,AssertionError) as exc: results.append({'test':name,'result':'PASS rejected','reason':str(exc)[:250]})
        else: raise ValueError(f'Corruption accepted: {name}')
print(json.dumps({'result':'PASS','checks':results},indent=2))
