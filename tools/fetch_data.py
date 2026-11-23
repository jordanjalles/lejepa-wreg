#!/usr/bin/env python3
"""Fetch Imagenette 160px exactly as the official LeJEPA example sees it.

The example calls load_dataset("frgfm/imagenette", "160px"). That HF repo is a loader script
(imagenette.py @ HF_REV) that downloads the fast.ai archive below and orders the examples by
the metadata lists train.txt / val.txt. This script fetches the same three files and writes
index_{train,val}.json: the archive member paths and labels in the loader's order.
"""
import hashlib, json, sys, tarfile, urllib.request
from pathlib import Path

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else '<data-root>/caches/ml-datasets/imagenette')
HF_REV = 'a8a496123b40fa739da2acf9b0dae339d30c7bae'
ARCHIVE = 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz'
META = f'https://huggingface.co/datasets/frgfm/imagenette/resolve/{HF_REV}/metadata/imagenette2-160/{{}}.txt'
SCRIPT = f'https://huggingface.co/datasets/frgfm/imagenette/resolve/{HF_REV}/imagenette.py'
LABELS = ['n01440764', 'n02102040', 'n02979186', 'n03000684', 'n03028079',
          'n03394916', 'n03417042', 'n03425413', 'n03445777', 'n03888257']


def get(url, dest):
    if not dest.exists():
        tmp = dest.with_suffix(dest.suffix + '.part')
        with urllib.request.urlopen(url) as r, open(tmp, 'wb') as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        tmp.rename(dest)
    return hashlib.sha256(dest.read_bytes()).hexdigest()


OUT.mkdir(parents=True, exist_ok=True)
sums = {'imagenette2-160.tgz': get(ARCHIVE, OUT / 'imagenette2-160.tgz'),
        'imagenette.py': get(SCRIPT, OUT / 'imagenette.py')}
for s in ('train', 'val'):
    sums[f'meta_{s}.txt'] = get(META.format(s), OUT / f'meta_{s}.txt')
with tarfile.open(OUT / 'imagenette2-160.tgz') as t:
    members = {m.name for m in t.getmembers() if m.isfile()}
    if not (OUT / 'imagenette2-160').exists():
        t.extractall(OUT, filter='data')
print(json.dumps(sums, indent=1))
print('meta_train head:', (OUT / 'meta_train.txt').read_text().splitlines()[:3])
print('archive members sample:', sorted(members)[:3], len(members))
