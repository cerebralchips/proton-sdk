"""DDR placement, execution evidence and portable report for Stories260K."""
import hashlib
import json
from pathlib import Path
import re
import shutil

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def prepare_weights(directory):
    original = directory / 'model_data.c'
    source = original.read_text()
    pattern = r'(const int8_t (w_\w+)\[(\d+)\] __attribute__\(\(aligned\(16\)\)\))='
    found = list(re.finditer(pattern, source))
    if len(found) != 36 or len({m[2] for m in found}) != 36:
        raise ValueError('Expected exactly 36 packed projection matrices')
    modified = re.sub(pattern, r'\1 __attribute__((section(".ddr_weights")))=', source)
    if modified.replace(' __attribute__((section(".ddr_weights")))', '') != source:
        raise ValueError('Placement transformation changed model data')
    dest = directory / 'model_data_ddr.c'
    dest.write_text(modified)
    record = {'source_sha256': sha(original), 'ddr_source_sha256': sha(dest),
              'packed_bytes': sum(int(m[3]) for m in found),
              'weights': [{'name': m[2], 'bytes': int(m[3])} for m in found]}
    (directory / 'ddr-placement.json').write_text(json.dumps(record, indent=2)+'\n')
    return record

def check_run(directory, work):
    import numpy as np
    log = (directory / 'rtl.log').read_text()
    placements = [(int(i), int(p, 16), int(n)) for i,p,n in
                  re.findall(r'DDR_WEIGHT id=(\d+) address=([0-9a-f]+) bytes=(\d+)', log)]
    declared = json.loads((work/'generated/model/ddr-placement.json').read_text())
    if [p[0] for p in placements] != list(range(36)):
        raise ValueError('Missing DDR weight placement checks')
    provenance = json.loads((directory/'provenance.json').read_text())
    external = [s for s in provenance['segments'] if s['region']=='ddr']
    for (_,address,size), weight in zip(placements, declared['weights']):
        if size != weight['bytes'] or not any(int(s['address'],16)<=address and
                address+size<=int(s['address'],16)+s['file_bytes'] for s in external):
            raise ValueError('Weight is outside initialized DDR ELF segments')
    if sum(s['file_bytes'] for s in external) != declared['packed_bytes']:
        raise ValueError('Unexpected external data size')
    match = re.search(r'DDR_ACCESSES reads=(\d+) writes=(\d+) allocated_words=(\d+)', log)
    if not match or int(match[1])<=0 or int(match[2])!=0:
        raise ValueError('Missing external reads or writes to read-only weights')
    result = {'result':'PASS', 'packed_weight_bytes':declared['packed_bytes'],
              'weights':36, 'reads':int(match[1]), 'writes':int(match[2]),
              'allocated_words':int(match[3]), 'placement':declared}
    # Additional comparison when the original SRAM result is available locally.
    candidates = sorted((work/'runs').glob('*-model_matrix'))
    baseline = next((p for p in reversed(candidates) if (p/'target-logits.npy').exists()), None)
    if baseline:
        actual = np.load(directory/'target-logits.npy')
        original = np.load(baseline/'target-logits.npy')[:len(actual)]
        if not np.array_equal(actual.view(np.uint32), original.view(np.uint32)):
            raise ValueError('DDR and SRAM baseline logits differ bitwise')
        result['sram_comparison']={'result':'PASS bit-identical', 'logits':int(actual.size),
                                 'baseline_result_sha256':sha(baseline/'result.json')}
    return result

def report(root, work):
    import numpy as np
    selected = {}
    for name in ['model_step','model_matrix','model_scalar_step']:
        candidates = sorted((work/'runs').glob('*-'+name+'-ddr'))
        p = next((p for p in reversed(candidates) if (p/'result.json').exists() and
                  json.loads((p/'result.json').read_text())['result']=='PASS'), None)
        if p is None: raise ValueError('Missing passing DDR run: '+name)
        selected[name] = p
    matrix=np.load(selected['model_step']/'target-logits.npy')
    scalar=np.load(selected['model_scalar_step']/'target-logits.npy')
    if not np.array_equal(matrix.view(np.uint32),scalar.view(np.uint32)):
        raise ValueError('DDR scalar and matrix logits differ bitwise')
    records={}
    for name,p in selected.items():
        result=json.loads((p/'result.json').read_text())
        provenance=json.loads((p/'provenance.json').read_text())
        # Bind executing sources to current source without including generated reports.
        for source in ['CMakeLists.txt','cmake/proton.cmake','runtime/proton/ddr_entry.c',
                       'runtime/proton/link-ddr.ld','scripts/ddr.py','scripts/build.py','scripts/preflight.py']:
            if sha(root/source)!=provenance['sdk_sha256'].get(source):
                raise ValueError('Changed source after DDR execution: '+source)
        records[name]={'run':p.name, **result}
    full=records['model_matrix']['evidence']['model']
    if full['steps']!=16: raise ValueError('Full 16-token run required')
    out=root/'verification'
    (out/'ddr.json').write_text(json.dumps({'result':'PASS', 'runs':records,
        'scalar_matrix_comparison':'PASS bit-identical first-step logits',
        'reference_sha256':sha(work/'generated/model/reference.npz')},indent=2)+'\n')
    shutil.copy2(selected['model_matrix']/'provenance.json',out/'ddr-provenance.json')
    log=(selected['model_matrix']/'rtl.log').read_text()
    (out/'ddr-output.txt').write_text('\n'.join(line for line in log.splitlines()
        if any(word in line for word in ['DDR_','TOKEN ','MODEL ','MATRIX_COUNTS','MEMORY','RESULT:','SUCCESS','Executed cycles:']))+'\n')
    print('PASS: DDR matrix 16 tokens, scalar/matrix first step, external reads and SRAM placement')
