#!/usr/bin/env python3
"""Copy the four Wasserstein regularizers verbatim out of the user's notebook into lejepa_wreg/wreg_nb.py.

The functions are not edited. The header records the notebook path, cell numbers and a sha256 of
each cell's source, so a run config can prove which code it used.
"""
import hashlib, json, sys
from pathlib import Path

NB = Path('<data-root>/workspaces/Jupyter_AI/MaskedAutoencoders/ImageNet-AE-selfteach.ipynb')
NAMES = ['sliced_wasserstein_distance', 'basis_wasserstein_distance',
         'token_sliced_wasserstein_distance', 'channel_sliced_wasserstein_distance']
OUT = Path(__file__).resolve().parents[1] / 'lejepa_wreg' / 'wreg_nb.py'

cells = json.load(open(NB))['cells']
found = []
for i, c in enumerate(cells):
    src = ''.join(c['source'])
    for n in NAMES:
        if src.lstrip().startswith(f'def {n}('):
            found.append((n, i, src))
assert sorted(n for n, _, _ in found) == sorted(NAMES), [n for n, _, _ in found]
head = ['"""VERBATIM copy of the user\'s Wasserstein regularizers. Do not edit: run tools/extract_wreg.py.', '',
        f'source: {NB}']
for n, i, src in found:
    head.append(f'  cell {i:3d}  {n:40s} sha256 {hashlib.sha256(src.encode()).hexdigest()[:16]}')
head += ['"""', 'import torch', 'import torch.nn.functional as F', '']
body = '\n\n'.join(src.rstrip() + '\n' for _, _, src in found)
OUT.write_text('\n'.join(head) + '\n\n' + body)
SOURCES = {n: {'cell': i, 'sha256': hashlib.sha256(src.encode()).hexdigest()} for n, i, src in found}
(OUT.parent / 'wreg_nb_sources.json').write_text(json.dumps({'notebook': str(NB), 'functions': SOURCES}, indent=1))
print(OUT, {n: i for n, i, _ in found})
