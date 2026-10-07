#!/usr/bin/env python3
"""Isolated host-only Stories260K ONNX qualification; no RTL gate promotion."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT.parent
VENV = WORK/'onnx-venv'
LOCK = json.loads((ROOT/'models/stories260k/onnx.lock.json').read_text())
SDK_LOCK = json.loads((ROOT/'dependencies.lock.json').read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args, dest, name, timeout=600, expect_failure=False):
    args = list(map(str, args))
    print(f'{name} ...', flush=True)
    (dest/(name+'.command.json')).write_text(json.dumps(args, indent=2)+'\n')
    start = time.monotonic()
    with (dest/(name+'.log')).open('w') as out:
        proc = subprocess.run(args, stdout=out, stderr=subprocess.STDOUT, timeout=timeout)
    if bool(proc.returncode) != expect_failure:
        print((dest/(name+'.log')).read_text()[-10000:], flush=True)
        raise RuntimeError(f'{name} failed: exit {proc.returncode}; see {dest}')
    if expect_failure:
        log = (dest/(name+'.log')).read_text()
        if 'Deliberate corruption:' not in log or 'mismatch at' not in log:
            raise RuntimeError(f'{name} failed for an unexpected reason')
    return round(time.monotonic()-start, 3)


def bootstrap(dest):
    if not (VENV/'bin/python').exists():
        run([sys.executable, '-m', 'venv', VENV], dest, 'venv')
    packages = [f'{name}=={version}' for name, version in LOCK['packages'].items()]
    run([VENV/'bin/pip', 'install', *packages], dest, 'pip-install')
    run([VENV/'bin/pip', 'check'], dest, 'pip-check')
    model_dir = WORK/'models'
    model_dir.mkdir(exist_ok=True)
    for name, expected in SDK_LOCK['model_files'].items():
        target = model_dir/name
        if not target.exists():
            pin = SDK_LOCK['model']['revision']
            url = f'https://huggingface.co/karpathy/tinyllamas/resolve/{pin}/stories260K/{name}'
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read()
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError(f'Download hash mismatch: {name}')
            target.write_bytes(data)
        if sha(target) != expected:
            raise ValueError(f'Checkpoint hash mismatch: {name}')
    print('ONNX BOOTSTRAP PASS: isolated environment and pinned model', flush=True)


def validate(dest):
    version = subprocess.check_output([VENV/'bin/iree-compile', '--version'], text=True)
    if SDK_LOCK['iree']['revision'] not in version:
        raise ValueError('ONNX compiler differs from pinned SDK IREE revision')
    freeze = json.loads(subprocess.check_output([VENV/'bin/pip', 'list', '--format=json'], text=True))
    installed = {p['name'].lower().replace('_', '-'): p['version'] for p in freeze}
    for name, value in LOCK['packages'].items():
        if installed.get(name.replace('_', '-')) != value:
            raise ValueError(f'Package mismatch: {name}; run onnx-bootstrap')
    timing = {}
    timing['export'] = run([VENV/'bin/python', ROOT/'models/stories260k/export_onnx.py',
        '--model-dir', WORK/'models', '--output-dir', dest], dest, 'export')
    for mode in ['fp32', 'w8a8', 'quantization_probe']:
        prefix = dest/(f'stories260k.{mode}' if mode != 'quantization_probe' else mode)
        timing[mode+'_import'] = run([VENV/'bin/iree-import-onnx', str(prefix)+'.onnx',
             '-o', str(prefix)+'.torch.mlir'], dest, mode+'-import')
        imported = Path(str(prefix)+'.torch.mlir').read_text()
        if 'torch.vtensor' not in imported or 'torch.operator \"onnx.MatMul\"' not in imported:
            raise ValueError('Expected imported Torch MLIR tensor/matmul operations')
        timing[mode+'_compile'] = run([VENV/'bin/iree-compile', str(prefix)+'.torch.mlir',
             '--iree-hal-target-device=local', '--iree-hal-local-target-device-backends=llvm-cpu',
             '--iree-llvmcpu-target-cpu=host', '-o', str(prefix)+'.vmfb'], dest, mode+'-compile')
    timing['host_validate'] = run([VENV/'bin/python', ROOT/'tests/onnx_host.py',
        '--model-dir', WORK/'models', '--artifacts', dest], dest, 'host-validate')
    record = json.loads((dest/'host-results.json').read_text())
    for output in ['logits', 'keys', 'values']:
        run([VENV/'bin/python', ROOT/'tests/onnx_host.py', '--model-dir', WORK/'models',
             '--artifacts', dest, '--corrupt-output', output], dest, 'reject-'+output, expect_failure=True)
    record['process_failure_checks'] = {'result':'PASS', 'rejected_outputs':['logits', 'keys', 'values']}
    record['run_id'] = dest.name
    record['pipeline'] = 'checkpoint -> ONNX -> iree-import-onnx -> Torch MLIR -> iree-compile -> host IREE'
    record['frontend_contract'] = 'Torch MLIR is the compiler input; ONNX is this model frontend, not an SDK-wide requirement'
    record.update({'execution': 'host CPU: NumPy, ONNX Runtime CPUExecutionProvider, IREE local-sync; not RTL',
        'compiler': version, 'packages': installed, 'timing_seconds': timing,
        'checkpoint': SDK_LOCK['model'], 'checkpoint_files': SDK_LOCK['model_files'],
        'artifacts': {p.name: {'sha256': sha(p), 'bytes': p.stat().st_size} for p in dest.iterdir()
                      if p.suffix in {'.onnx', '.mlir', '.vmfb', '.npz'}},
        'host': {'machine': os.uname().machine, 'system': os.uname().sysname},
        'sdk_sha256': {name: sha(ROOT/name) for name in ['models/stories.py',
             'models/stories260k/export_onnx.py', 'models/stories260k/onnx.lock.json',
             'scripts/onnx.py', 'scripts/sdk', 'tests/onnx_host.py']}})
    (dest/'result.json').write_text(json.dumps(record, indent=2)+'\n')
    (ROOT/'verification/onnx-stories260k.json').write_text(json.dumps(record, indent=2)+'\n')
    print((dest/'host-validate.log').read_text(), flush=True)
    print(f'ONNX HOST VALIDATION PASS: {dest}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['onnx-bootstrap', 'onnx-validate'])
    args = parser.parse_args()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    dest = WORK/'onnx-runs'/stamp
    dest.mkdir(parents=True)
    try:
        (bootstrap if args.command == 'onnx-bootstrap' else validate)(dest)
    except Exception as exc:
        (dest/'failure.json').write_text(json.dumps({'result':'FAIL', 'error':str(exc)}, indent=2)+'\n')
        raise
