"""Reject corrupted copies of a passing DDR run; never modify its evidence."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from ddr import check_run

positive=Path(sys.argv[1]).resolve()
work=positive.parent.parent
assert check_run(positive,work)['result']=='PASS'
log=(positive/'rtl.log').read_text()
cases={
    'missing DDR reads':lambda s:s.replace('DDR_ACCESSES reads=', 'REMOVED reads='),
    'writes to weights':lambda s:s.replace(' writes=0 ', ' writes=1 '),
    'weight in SRAM':lambda s:s.replace('address=100000000', 'address=80000000'),
    'weight beyond DDR':lambda s:s.replace('address=100000000', 'address=200000000'),
    'missing projection':lambda s:s.replace('DDR_WEIGHT id=0 ', 'REMOVED id=0 '),
    'wrong weight size':lambda s:s.replace('address=100000000 bytes=4096', 'address=100000000 bytes=4097'),
}
outcomes=[]
with tempfile.TemporaryDirectory(prefix='ddr-evidence-',dir=work) as tmp:
    dest=Path(tmp)
    for name in ['provenance.json','target-logits.npy']:
        shutil.copy2(positive/name,dest/name)
    for name,mutate in cases.items():
        modified=mutate(log)
        assert modified!=log,name
        (dest/'rtl.log').write_text(modified)
        try: check_run(dest,work)
        except ValueError as e: outcomes.append({'test':name,'result':'PASS rejected','reason':str(e)})
        else: raise AssertionError('Accepted '+name)
    (dest/'rtl.log').write_text(log)
    provenance=json.loads((dest/'provenance.json').read_text())
    provenance['segments']=[s for s in provenance['segments'] if s['region']!='ddr']
    (dest/'provenance.json').write_text(json.dumps(provenance))
    try: check_run(dest,work)
    except ValueError as e: outcomes.append({'test':'missing loaded DDR segment','result':'PASS rejected','reason':str(e)})
    else: raise AssertionError('Accepted missing DDR segment')
record={'result':'PASS','positive_run':positive.name,'checks':outcomes}
print(json.dumps(record,indent=2))
if len(sys.argv)>2: Path(sys.argv[2]).write_text(json.dumps(record,indent=2)+'\n')
