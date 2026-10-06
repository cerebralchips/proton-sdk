#!/usr/bin/env python3
"""Linux-side build and gated RTL execution; called through scripts/sdk."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT.parent
LOCK = json.loads((ROOT / 'dependencies.lock.json').read_text())
TOOLS = Path(os.environ['LAB_TOOLS'])
CLANG = Path(os.environ['LLVM_INSTALL_DIR']) / 'bin/clang'
FLAGS = ['-march=rv64gcv_zfh_zvfh','-mabi=lp64d','-mcmodel=medany','-mno-relax',
         '-O3','-fno-vectorize','-fno-slp-vectorize','-ffp-contract=off']
STAMP = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
LOGS = WORK / 'attempts' / STAMP
LOGS.mkdir(parents=True, exist_ok=True)

def run(args, label, cwd=WORK, timeout=1800):
    args = list(map(str,args))
    print('+', ' '.join(args), flush=True)
    (LOGS / (label+'.command.json')).write_text(json.dumps(args,indent=2)+'\n')
    with (LOGS / (label+'.log')).open('w') as f:
        p = subprocess.run(args,cwd=cwd,stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
    if p.returncode:
        print((LOGS / (label+'.log')).read_text()[-12000:])
        raise RuntimeError(f'{label}: exit {p.returncode}; see {LOGS}')

def fetch(url, dest):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        urllib.request.urlretrieve(url, str(dest)+'.part')
        Path(str(dest)+'.part').rename(dest)
    return hashlib.sha256(dest.read_bytes()).hexdigest()

def prepare():
    for repo,rev,name in [
        ('dvidelabs/flatcc','9362cd00f0007d8cbee7bff86e90fb4b6b227ff3','flatcc'),
        ('eyalroz/printf','f1b728cbd5c6e10dc1f140f1574edfd1ccdcbedb','printf')]:
        dest=WORK/'deps/iree/third_party'/name
        if not (dest/'CMakeLists.txt').exists():
            archive=WORK/'deps'/f'{name}.tar.gz'
            fetch(f'https://github.com/{repo}/archive/{rev}.tar.gz',archive)
            with tarfile.open(archive) as tf:
                for entry in tf.getmembers():
                    entry.name='/'.join(entry.name.split('/')[1:])
                    if entry.name: tf.extract(entry,dest,filter='data')
    run(['cmake','-S',WORK/'deps/iree/third_party/flatcc','-B',WORK/'build-flatcc',
         '-DFLATCC_TEST=OFF','-DFLATCC_INSTALL=OFF','-DCMAKE_BUILD_TYPE=Release'],'flatcc-config')
    run(['cmake','--build',WORK/'build-flatcc','-j4'],'flatcc-build')
    host=WORK/'host-tools'; host.mkdir(exist_ok=True)
    cli=host/'iree-flatcc-cli'
    if not cli.exists(): cli.symlink_to(WORK/'deps/iree/third_party/flatcc/bin/flatcc')

def compile_ir(name, source=None):
    dest=WORK/'generated'/name; dest.mkdir(parents=True,exist_ok=True)
    run([WORK/'venv/bin/iree-compile',source or ROOT/'compiler'/f'{name}.mlir',
         '--iree-hal-target-device=local','--iree-hal-local-target-device-backends=llvm-cpu',
         '--iree-llvmcpu-target-triple=riscv64-unknown-elf','--iree-llvmcpu-target-cpu=generic-rv64',
         '--iree-llvmcpu-target-cpu-features=+m,+a,+f,+d,+c',
         '--iree-llvmcpu-target-abi=lp64d','--iree-llvmcpu-link-embedded=false','--iree-llvmcpu-link-static',
         '--iree-vm-target-index-bits=32',f'--iree-llvmcpu-static-library-output-path={dest}/dispatch.o',
         '-o',dest/'module.vmfb'],name+'-compile')
    data=(dest/'module.vmfb').read_bytes()
    (dest/'embed.c').write_text('#include <stddef.h>\nconst unsigned char module_data[] __attribute__((aligned(16))) = {\n'+
        ','.join(str(b) for b in data)+'};\nconst size_t module_size=sizeof(module_data);\n')
    h=(dest/'dispatch.h').read_text()
    symbols=list(dict.fromkeys(re.findall(r'(\w+_library_query)\(',h)))
    assert symbols, h
    (dest/'libraries.h').write_text('#include "dispatch.h"\nstatic const iree_hal_executable_library_query_fn_t proton_libraries[] = {'+
                                  ','.join(symbols)+'};\n')

def build_smoke():
    prepare()
    compile_ir('smoke')
    run(['cmake','-G','Ninja','-S',ROOT,'-B',WORK/'build',f'-DPROTON_WORK={WORK}',
         f'-DCMAKE_TOOLCHAIN_FILE={ROOT}/cmake/proton.cmake','-DCMAKE_BUILD_TYPE=MinSizeRel'],'sdk-config')
    run(['cmake','--build',WORK/'build','--target','smoke','-j4'],'sdk-build')

def configure(ddr=False):
    run(['cmake','-G','Ninja','-S',ROOT,'-B',WORK/('build-ddr' if ddr else 'build'),
         f'-DPROTON_DDR={"ON" if ddr else "OFF"}',f'-DPROTON_WORK={WORK}',
         f'-DCMAKE_TOOLCHAIN_FILE={ROOT}/cmake/proton.cmake','-DCMAKE_BUILD_TYPE=MinSizeRel'],'sdk-config')

def simulate(name, max_cycles=5000000, trace=False, ddr=False, latency=1, stall=0):
    if not re.fullmatch(r'[a-z][a-z0-9_]*',name) or max_cycles<1:
        raise ValueError('Invalid application name or cycle limit')
    dest=WORK/'runs'/(f'{STAMP}-{name}'+('-ddr' if ddr else '')); dest.mkdir(parents=True)
    elf=WORK/('build-ddr' if ddr else 'build')/name
    from preflight import validate
    (dest/'provenance.json').write_text(json.dumps(validate(ROOT,elf,'proton_v1_ddr.json' if ddr else 'proton_v1.json'),indent=2)+'\n')
    import shutil
    shutil.copy2(elf,dest/'program.elf')
    run([TOOLS/'llvm/bin/llvm-objdump','-d',elf],name+'-dump')
    shutil.copy2(LOGS/(name+'-dump.log'),dest/'program.dump')
    sim=Path('/ara-workspace/upstream/ara/hardware')/('build-ddr-wave' if ddr else ('build-matrix-wave' if trace else 'build-matrix'))/'verilator/Vara_tb_verilator'
    result={'result':'INCOMPLETE','elf_sha256':hashlib.sha256(elf.read_bytes()).hexdigest(),
            'simulator_sha256':hashlib.sha256(sim.read_bytes()).hexdigest()}
    result.update(profile='ddr' if ddr else 'sram', latency=latency if ddr else None, stall=stall if ddr else None)
    try:
        import time
        started=time.monotonic()
        loader=[f'--load-elf={elf}',f'+ddr_latency={latency}',f'+ddr_stall={stall}'] if ddr else ['-l',f'ram,{elf},elf']
        run(['stdbuf','-oL',sim,'-c',str(max_cycles),*(['-t'] if trace else []),*loader],name+'-rtl',cwd=dest,timeout=7200)
        result['wall_seconds']=time.monotonic()-started
        text=(LOGS/(name+'-rtl.log')).read_text()
        shutil.copy2(LOGS/(name+'-rtl.log'),dest/'rtl.log')
        if '*** SUCCESS ***' not in text or 'RESULT: PASS' not in text or any(x in text for x in ['TRAP','FAIL','Simulation timeout','%Error']):
            raise RuntimeError('RTL completion or self-check failed: '+text[-2000:])
        match=re.search(r'Executed cycles:\s*(\d+)',text)
        if not match or int(match[1])<=0: raise RuntimeError('Missing cycle count')
        result.update(result='PASS',rtl_cycles=int(match[1]))
        from evidence import check
        result['evidence']=check(dest, WORK/'generated/model/reference.npz' if name.startswith('model_') else None)
        if ddr:
            from ddr import check_run
            result['ddr']=check_run(dest, WORK)
        if trace:
            waves=list(dest.glob('*.fst'))
            if len(waves)!=1 or not waves[0].stat().st_size: raise RuntimeError('Missing FST')
            (dest/'execution-evidence.json').write_text(json.dumps({'matched_issue_done_response':result['evidence']['matched_commands']})+'\n')
            run([sys.executable,'/ara-workspace/scripts/lab/matrix-wave.py',waves[0]],'wave-check',timeout=1200)
            result['waveform']=json.loads((dest/'matrix-wave-check.json').read_text())['result']
        print(text[-2500:],flush=True)
    except Exception as e:
        result.update(result='FAIL',error=str(e)); raise
    finally:
        if (LOGS/(name+'-rtl.log')).exists(): shutil.copy2(LOGS/(name+'-rtl.log'),dest/'rtl.log')
        (dest/'result.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':
    if sys.argv[1]=='smoke': build_smoke(); simulate('smoke')
    elif sys.argv[1]=='model':
        run([sys.executable, ROOT/'models/stories.py', WORK/'models', WORK/'generated/model'], 'model-reference')
    elif sys.argv[1]=='kernels':
        run([sys.executable,ROOT/'tests/generate_kernels.py',WORK/'models',WORK/'generated/tests'],'kernel-reference')
        configure()
        run(['cmake','--build',WORK/'build','--target','kernels','-j4'],'kernel-build')
        simulate('kernels',15000000)
    elif sys.argv[1]=='graph':
        path=WORK/'generated/model/model.mlir'
        run([sys.executable,ROOT/'compiler/model_graph.py',path],'model-graph')
        compile_ir('model',path)
    elif sys.argv[1]=='host-model':
        run([TOOLS/'llvm/bin/clang','-O3','-ffp-contract=off',ROOT/'tests/model_host.c',ROOT/'kernels/model_ops.c',
             ROOT/'kernels/linear.c',WORK/'generated/model/model_data.c','-I'+str(ROOT/'runtime'),
             '-I'+str(ROOT/'kernels'),'-I'+str(WORK/'generated/model'),'-lm','-o',WORK/'host-model'],'host-model-build')
        run([WORK/'host-model'],'host-model-check')
        print((LOGS/'host-model-check.log').read_text())
        run([TOOLS/'llvm/bin/clang','-O2','-ffp-contract=off',WORK/'models/run.c','-lm','-o',WORK/'models/upstream-reference'],'upstream-reference-build')
        run([WORK/'models/upstream-reference',WORK/'models/stories260K.bin','-z',WORK/'models/tok512.bin','-t','0','-n','16'],'upstream-reference')
        expected=json.loads((WORK/'generated/model/reference.json').read_text())['fp_text']
        actual=(LOGS/'upstream-reference.log').read_text().split('achieved tok/s:')[0]
        if actual.strip()!=expected: raise RuntimeError('Independent upstream FP32 text mismatch')
    elif sys.argv[1]=='model-run':
        name=sys.argv[2] if len(sys.argv)>2 else 'model_step'
        configure()
        run(['cmake','--build',WORK/'build','--target',name,'-j4'],'model-build')
        simulate(name,10000000 if name=='model_step' else 150000000)
    elif sys.argv[1]=='wave':
        configure()
        run(['cmake','--build',WORK/'build','--target','kernel_tile','-j4'],'tile-build')
        simulate('kernel_tile',1000000,trace=True)
    elif sys.argv[1]=='negative':
        configure()
        run(['cmake','--build',WORK/'build','--target','kernels_bad','-j4'],'negative-build')
        outcomes=[]
        for name,cycles,marker in [('kernels_bad',1000000,'FAIL kernel=0 element=0'),('smoke',20,'Simulation timeout')]:
            try: simulate(name,cycles)
            except RuntimeError:
                log=(LOGS/(name+'-rtl.log')).read_text()
                if marker not in log: raise RuntimeError('Unexpected negative-test failure: '+log[-1000:])
                outcomes.append({'test':name,'result':'PASS intended failure rejected','marker':marker})
            else: raise RuntimeError('Negative test incorrectly passed')
        candidates=sorted((WORK/'runs').glob('*-model_step'))
        positive=next(p for p in reversed(candidates) if (p/'evidence.json').exists())
        run([sys.executable,ROOT/'tests/evidence_negative.py',positive,WORK/'generated/model/reference.npz'],'evidence-negative')
        outcomes.append(json.loads((LOGS/'evidence-negative.log').read_text()))
        (LOGS/'negative-results.json').write_text(json.dumps({'result':'PASS','checks':outcomes},indent=2)+'\n')
        print(json.dumps(outcomes,indent=2))
    elif sys.argv[1]=='ddr-run':
        from ddr import prepare_weights
        name=sys.argv[2] if len(sys.argv)>2 else 'model_step'
        if name not in {'model_step','model_matrix','model_scalar_step'}: raise ValueError('Unsupported DDR example')
        latency=int(sys.argv[3]) if len(sys.argv)>3 else 1
        stall=int(sys.argv[4]) if len(sys.argv)>4 else 0
        if not 1<=latency<=100 or not 0<=stall<=100: raise ValueError('Invalid DDR delay')
        prepare_weights(WORK/'generated/model')
        configure(ddr=True)
        run(['cmake','--build',WORK/'build-ddr','--target',name,'-j4'],'ddr-build')
        simulate(name,150000000,ddr=True,latency=latency,stall=stall)
    elif sys.argv[1]=='ddr-report':
        from ddr import report
        report(ROOT,WORK)
    elif sys.argv[1]=='report':
        run([sys.executable,ROOT/'scripts/report.py'],'report')
        print((LOGS/'report.log').read_text())
    elif sys.argv[1]=='run': simulate(sys.argv[2],int(sys.argv[3]) if len(sys.argv)>3 else 5000000)
    else: raise SystemExit('Usage: scripts/sdk smoke | run NAME [CYCLES]')
