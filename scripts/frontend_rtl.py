#!/usr/bin/env python3
"""Build and check exactly one decoder invocation on pinned CVA6 RTL."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT.parent
HW=Path('/ara-workspace')
TOOLS=Path(os.environ['LAB_TOOLS'])
VENV=WORK/'onnx-venv'
TARGET=json.loads((ROOT/'targets/proton_frontend_cpu.json').read_text())
sys.path.insert(0,str(ROOT/'models'))
from stories import Model
from preflight import validate
# Avoid shadowing the ONNX package with scripts/onnx.py when importing exporter.
sys.path = [p for p in sys.path if Path(p).resolve() != ROOT/"scripts"]
sys.path.insert(0,str(ROOT/'tests'))
from frontend_rtl_evidence import check, negative_checks


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args,dest,name,timeout=600,cwd=None):
    args=list(map(str,args))
    (dest/(name+'.command.json')).write_text(json.dumps(args,indent=2)+'\n')
    print(name+' ...',flush=True)
    start=time.monotonic()
    with (dest/(name+'.log')).open('w') as out:
        process=subprocess.Popen(args,cwd=cwd,stdout=out,stderr=subprocess.STDOUT)
        try:
            while True:
                try: code=process.wait(timeout=min(30,max(1,timeout-(time.monotonic()-start)))); break
                except subprocess.TimeoutExpired:
                    elapsed=time.monotonic()-start
                    print(f'{name}: running {elapsed:.0f}s',flush=True)
                    if elapsed>=timeout: raise TimeoutError(f'{name}: wall-time limit {timeout}s')
        except BaseException:
            process.kill(); process.wait(); raise
    if code:
        print((dest/(name+'.log')).read_text()[-7000:],flush=True)
        raise RuntimeError(f'{name}: exit {code}')
    return time.monotonic()-start


def build(mode,dest):
    sys.path.insert(0,str(ROOT/'models/stories260k'))
    from export_onnx import export
    export(WORK/'models',dest/'model.onnx',mode=='w8a8')
    run([VENV/'bin/iree-import-onnx',dest/'model.onnx','-o',dest/'model.torch.mlir'],dest,'import')
    imported=(dest/'model.torch.mlir').read_text()
    if 'torch.vtensor' not in imported or 'onnx.MatMul' not in imported: raise ValueError('Expected Torch MLIR')
    run([VENV/'bin/iree-compile',dest/'model.torch.mlir',
        '--iree-hal-target-device=local','--iree-hal-local-target-device-backends=llvm-cpu',
        '--iree-llvmcpu-target-triple=riscv64-unknown-elf','--iree-llvmcpu-target-cpu=generic-rv64',
        '--iree-llvmcpu-target-cpu-features=+m,+a,+f,+d,+c','--iree-llvmcpu-target-abi=lp64d',
        '--iree-llvmcpu-link-embedded=false','--iree-llvmcpu-link-static',
        '--iree-vm-target-index-bits=32',f'--iree-llvmcpu-static-library-output-path={dest}/dispatch.o',
        '-o',dest/'module.vmfb'],dest,'compile')
    data=(dest/'module.vmfb').read_bytes()
    (dest/'embed.c').write_text('#include <stddef.h>\nconst unsigned char module_data[] __attribute__((aligned(16)))={'+
        ','.join(str(b) for b in data)+'};\nconst size_t module_size=sizeof(module_data);\n')
    symbols=list(dict.fromkeys(re.findall(r'(\w+_library_query)\(',(dest/'dispatch.h').read_text())))
    if not symbols: raise ValueError('No static executable libraries')
    (dest/'libraries.h').write_text('#include "dispatch.h"\nstatic const iree_hal_executable_library_query_fn_t proton_libraries[]={'+','.join(symbols)+'};\n')
    model=Model(WORK/'models')
    logits=model.step(1,0,mode=='w8a8')  # One reference step only.
    arrays={'logits':logits,'keys':model.keys,'values':model.values}
    np.savez(dest/'reference.npz',**arrays)
    token=int(np.argmax(logits))
    lines=['#pragma once',f'#define FRONTEND_PRECISION "{mode}"',
           f'#define FRONTEND_FUNCTION "module.stories260k_{mode}"',f'#define REFERENCE_TOKEN {token}']
    for name,array in arrays.items():
        lines.append(f'static const float reference_{name}[{array.size}]={{'+
                     ','.join(float(x).hex()+'f' for x in array.flatten())+'};')
    (dest/'reference.h').write_text('\n'.join(lines)+'\n')
    return {'token':token,'piece':model.decode([token])}


def main(mode):
    pin=TARGET['hardware_revision']
    if subprocess.check_output(['git','-C',HW,'rev-parse','HEAD'],text=True).strip()!=pin:
        raise ValueError('Hardware revision differs from frontend target')
    if subprocess.check_output(['git','-C',HW,'status','--porcelain'],text=True).strip():
        raise ValueError('Hardware checkout must be clean')
    sim=HW/TARGET['simulator']
    if sha(sim)!=TARGET['qualified_simulator_sha256']:
        raise ValueError('Simulator differs from the qualified DMA-system binary')
    # Bind to the hardware qualification evidence, not just its branch name.
    qualified=json.loads((HW/'verification/2026-10-07/dma.json').read_text())
    for name,value in qualified['sources_sha256'].items():
        if sha(HW/name)!=value: raise ValueError('Hardware qualification source differs: '+name)
    host=json.loads((ROOT/'verification/onnx-stories260k.json').read_text())
    for name,value in host['sdk_sha256'].items():
        if sha(ROOT/name)!=value: raise ValueError('Host frontend qualification source differs: '+name)
    version=subprocess.check_output([VENV/'bin/iree-compile','--version'],text=True)
    if version!=host['compiler']: raise ValueError('Compiler differs from host qualification')
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    dest=WORK/'frontend-rtl-runs'/stamp
    dest.mkdir(parents=True)
    modes=['fp32','w8a8'] if mode=='both' else [mode]
    try:
        original=WORK/'deps/iree/runtime/src/iree/hal/utils/deferred_command_buffer.c'
        if sha(original) != 'c94db82392f9138e8241291189464e9595918a8c18f1716c06c80ffcb9ff3f95':
            raise ValueError('IREE runtime source differs from pinned patch base')
        run(['patch', '-o', dest/'deferred_command_buffer.c', original,
             ROOT/'patches/iree/0001-align-dispatch-bindings.patch'],dest,'runtime-patch')
        # Configure both paths even when building only one precision.
        for name in ['fp32','w8a8']: (dest/name).mkdir()
        refs={name:build(name,dest/name) for name in modes}
        build_dir=WORK/'build-frontend'
        run(['cmake','-G','Ninja','-S',ROOT/'cmake/frontend','-B',build_dir,
             f'-DPROTON_WORK={WORK}',f'-DPROTON_FRONTEND_RUN={dest}',
             f'-DCMAKE_TOOLCHAIN_FILE={ROOT}/cmake/proton.cmake','-DCMAKE_BUILD_TYPE=MinSizeRel'],dest,'configure')
        run(['cmake','--build',build_dir,'--target',*['frontend_'+name for name in modes],'-j4'],dest,'build',timeout=1200)
        for name in modes:
            case=dest/name
            elf=build_dir/('frontend_'+name)
            provenance=validate(ROOT,elf,'proton_frontend_cpu.json')
            (case/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
            shutil.copy2(elf,case/'program.elf')
            run([TOOLS/'llvm/bin/llvm-objdump','-d',elf],case,'disassembly')
            seconds=run(['/usr/bin/time','-p',sim,'-c','30000000',f'--load-elf={elf}',
                         '+ddr_latency=1','+ddr_stall=0'],case,'rtl',timeout=1200,cwd=case)
            result=check((case/'rtl.log').read_text(),case/'reference.npz',name)
            result['negative_checks']=negative_checks((case/'rtl.log').read_text(),case/'reference.npz',name)
            result.update(run_id=stamp,precision=name,steps=1,expected=refs[name],
                runtime_source_sha256=sha(dest/'deferred_command_buffer.c'),
                wall_seconds=round(seconds,3),target=TARGET,compiler=version,
                simulator_sha256=sha(sim),hardware_evidence_sha256=sha(HW/'verification/2026-10-07/dma.json'),
                hardware_sources_sha256=qualified['sources_sha256'],
                artifacts={p.name:sha(p) for p in case.iterdir() if p.is_file() and p.suffix in {'.elf','.onnx','.mlir','.vmfb','.o','.npz','.log'}},
                sdk_sha256={p:sha(ROOT/p) for p in ['runtime/frontend_runner.c','cmake/frontend/CMakeLists.txt',
                  'scripts/frontend-rtl','scripts/frontend_rtl.py','tests/frontend_rtl_evidence.py',
                  'targets/proton_frontend_cpu.json','patches/iree/0001-align-dispatch-bindings.patch','models/stories.py','models/stories260k/export_onnx.py',
                  'cmake/proton.cmake','runtime/iree_config.h','runtime/proton/start.S','runtime/proton/platform.c',
                  'runtime/proton/link.ld','runtime/proton/console.c','CMakeLists.txt']})
            (case/'result.json').write_text(json.dumps(result,indent=2)+'\n')
            (ROOT/'verification'/f'frontend-rtl-{name}.json').write_text(json.dumps(result,indent=2)+'\n')
            print(f'PASS {name}: one token {result["token"]}; {result["rtl_cycles"]} RTL cycles; {seconds:.2f}s',flush=True)
    except BaseException as exc:
        (dest/'failure.json').write_text(json.dumps({'result':'FAIL','error':str(exc)},indent=2)+'\n')
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('precision',nargs='?',choices=['both','fp32','w8a8'],default='both')
    main(parser.parse_args().precision)
