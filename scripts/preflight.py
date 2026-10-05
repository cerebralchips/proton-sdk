#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path
import struct
import subprocess

def validate(root,elf):
    target=json.loads((root/'targets/proton_v1.json').read_text())
    hw=Path('/ara-workspace')
    revision=subprocess.check_output(['git','-C',hw,'rev-parse','HEAD'],text=True).strip()
    if revision!=target['hardware_revision']: raise ValueError('Hardware revision differs from target manifest')
    if subprocess.check_output(['git','-C',hw,'status','--porcelain'],text=True).strip():
        raise ValueError('Hardware checkout has tracked/untracked changes: qualify target before execution')
    raw=Path(elf).read_bytes()
    if raw[:6]!=b'\x7fELF\x02\x01': raise ValueError('Expected little-endian ELF64')
    if struct.unpack_from('<H',raw,18)[0]!=243: raise ValueError('Expected RISC-V ELF')
    entry,phoff=struct.unpack_from('<QQ',raw,24)
    phsize,phnum=struct.unpack_from('<HH',raw,54)
    base=int(target['ram_base'],16); end=base+target['ram_bytes']; segments=[]
    for i in range(phnum):
        typ,flags,offset,va,pa,fs,ms,align=struct.unpack_from('<IIQQQQQQ',raw,phoff+i*phsize)
        if typ!=1 or ms==0: continue
        if va!=pa or va<base or va+ms>end-target['stack_bytes'] or fs>ms or offset+fs>len(raw):
            raise ValueError('ELF segment violates physical memory contract')
        segments.append({'address':hex(va),'file_bytes':fs,'memory_bytes':ms,'flags':flags})
    if not any(int(s['address'],16)<=entry<int(s['address'],16)+s['memory_bytes'] and s['flags']&1 for s in segments):
        raise ValueError('ELF entry outside executable memory')
    # Fingerprint owned RTL and integration patches without modifying the lab.
    files=[*hw.glob('hardware/matrix/**/*.sv'),*hw.glob('patches/**/*.patch')]
    return {'target':target,'hardware_revision':revision,'segments':segments,
      'elf_sha256':hashlib.sha256(raw).hexdigest(),
      'hardware_sha256':{str(p.relative_to(hw)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)},
      'sdk_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob('*'))
         if p.is_file() and '__pycache__' not in str(p) and p.suffix in ['.c','.h','.S','.ld','.py','.json','.cmake','.mlir']}}
