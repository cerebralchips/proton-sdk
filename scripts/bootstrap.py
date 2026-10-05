#!/usr/bin/env python3
"""Provision only SDK dependencies in the existing hardware Linux workspace."""
from pathlib import Path
import hashlib
import json
import subprocess
import tarfile
import urllib.request

root=Path(__file__).resolve().parents[1]; work=root.parent
lock=json.loads((root/'dependencies.lock.json').read_text())
record={}
integrity={}
def download(url,p,expected):
    p.parent.mkdir(parents=True,exist_ok=True)
    if not p.exists():
        urllib.request.urlretrieve(url,str(p)+'.part'); Path(str(p)+'.part').rename(p)
    sha=hashlib.sha256(p.read_bytes()).hexdigest()
    if sha!=expected: raise ValueError(f'SHA-256 mismatch: {p}')
    record[str(p.relative_to(work))]={'url':url,'sha256':sha}

for name,item in lock['archives'].items():
    archive=work/'deps'/f'{name}.tar.gz'; download(item['url'],archive,item['sha256'])
    dest=work/'deps/iree' if name=='iree' else work/'deps/iree/third_party'/name
    if not (dest/'CMakeLists.txt').exists():
        with tarfile.open(archive) as tf:
            for entry in tf.getmembers():
                entry.name='/'.join(entry.name.split('/')[1:])
                # Website assets are irrelevant to runtime builds; omit them
                # to avoid case/symlink incompatibilities on the macOS mount.
                if entry.name and not entry.name.startswith('docs/'):
                    tf.extract(entry,dest,filter='data')
    checked=0
    with tarfile.open(archive) as tf:
        for entry in tf:
            path='/'.join(entry.name.split('/')[1:])
            if not path or path.startswith('docs/') or not entry.isfile(): continue
            p=dest/path
            if not p.exists() or p.read_bytes()!=tf.extractfile(entry).read():
                raise ValueError(f'Extracted dependency differs from pinned archive: {name}/{path}')
            checked+=1
    integrity[name]={'result':'PASS','source_files_compared':checked}
pin=lock['model']['revision']
for name,sha in lock['model_files'].items():
    download(f'https://huggingface.co/karpathy/tinyllamas/resolve/{pin}/stories260K/{name}',work/'models'/name,sha)
for name,sha in [('run.c','9c4f2d5c6ae01b71726d1cc37530d71e60bff0ec7cc012565f16a43c1ca658bd'),
                 ('LICENSE','e87b912002f04cdfa837eaf7f70548ab79b9cb62a285495edf575f3da875aec4')]:
    download(f'https://raw.githubusercontent.com/karpathy/llama2.c/{lock["llama2_c"]["revision"]}/{name}',work/'models'/name,sha)
if not (work/'venv/bin/python').exists(): subprocess.run(['python3','-m','venv',work/'venv'],check=True)
subprocess.run([work/'venv/bin/pip','install',f'iree-base-compiler=={lock["iree"]["version"]}',f'numpy=={lock["numpy"]}'],check=True)
version=subprocess.check_output([work/'venv/bin/iree-compile','--version'],text=True)
if lock['iree']['revision'] not in version: raise ValueError('Compiler/source revision mismatch')
(work/'bootstrap.json').write_text(json.dumps({'compiler':version,'downloads':record,'source_integrity':integrity},indent=2)+'\n')
print('BOOTSTRAP PASS: hashes and compiler/source revision match')
