"""Exercise installed-memory boundaries without a simulator or third-party tools."""
import json
from pathlib import Path
import struct
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from preflight import validate_elf

root=Path(__file__).resolve().parents[1]
target=json.loads((root/'targets/proton_v1_ddr.json').read_text())

def elf(address=0x100000000, size=16, flags=4, entry=0x80000000, sram=0x80000000):
    raw=bytearray(512)
    raw[:6]=b'\x7fELF\x02\x01'
    struct.pack_into('<H',raw,18,243)
    struct.pack_into('<QQ',raw,24,entry,64)
    struct.pack_into('<HH',raw,54,56,2)
    struct.pack_into('<IIQQQQQQ',raw,64,1,5,256,sram,sram,16,16,16)
    struct.pack_into('<IIQQQQQQ',raw,120,1,flags,272,address,address,16,size,16)
    return raw

for raw in [elf(),elf(address=0x1fffffff0)]:
    assert [s['region'] for s in validate_elf(raw,target)]==['sram','ddr']
cases={
    'DDR lower boundary':elf(address=0xfffffff8),
    'DDR upper crossing':elf(address=0x1fffffff8),
    'DDR beyond installed end':elf(address=0x200000000),
    'DDR executable':elf(flags=5),
    'DDR writable weights':elf(flags=6),
    'entry in DDR':elf(entry=0x100000000),
    'missing DDR segment':elf(address=0x80001000),
    'SRAM reserved stack':elf(sram=0x80fc0000,entry=0x80fc0000),
    'SRAM upper boundary':elf(sram=0x81000000,entry=0x81000000),
    'filesz exceeds memsz':elf(size=8),
    'truncated header':elf()[:48],
    'truncated program table':elf()[:128],
}
for name,raw in cases.items():
    try: validate_elf(raw,target)
    except ValueError: pass
    else: raise AssertionError('Accepted '+name)
print(f'PASS: 2 valid DDR edges and {len(cases)} rejected memory/ELF contracts')
