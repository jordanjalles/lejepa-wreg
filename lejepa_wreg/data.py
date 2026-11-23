"""Imagenette 160px with the official LeJEPA minimal example's views and transforms.

Example order matches load_dataset("frgfm/imagenette", "160px"): the HF loader walks the archive in
tar order and keeps the files listed in meta_{train,val}.txt (see tools/fetch_data.py).
"""
import io, json, tarfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2

LABELS = ['n01440764', 'n02102040', 'n02979186', 'n03000684', 'n03028079',
          'n03394916', 'n03417042', 'n03425413', 'n03445777', 'n03888257']
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
HOLDOUT_FRAC = 0.1    # share of each train class held out for tuning
HOLDOUT_SEED = 20261109

# Copied from MINIMAL.md (HFDataset.aug / HFDataset.test) at the pinned upstream commit.
AUG = v2.Compose([
    v2.RandomResizedCrop(128, scale=(0.08, 1.0)),
    v2.RandomApply([v2.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
    v2.RandomGrayscale(p=0.2),
    v2.RandomApply([v2.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))]),
    v2.RandomApply([v2.RandomSolarize(threshold=128)], p=0.2),
    v2.RandomHorizontalFlip(),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=MEAN, std=STD),
])
TEST = v2.Compose([
    v2.Resize(128),
    v2.CenterCrop(128),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=MEAN, std=STD),
])


def load_split(root, split):
    """[(jpeg bytes, label)] of 'train' or 'val' in the HF loader's order. Bytes stay in RAM (~100 MB)."""
    root = Path(root)
    keep = set((root / f'meta_{split}.txt').read_text().split('\n'))
    out = []
    with tarfile.open(root / 'imagenette2-160.tgz') as t:
        for m in t:
            if m.isfile() and m.name in keep:
                out.append((t.extractfile(m).read(), LABELS.index(m.name.split('/')[-2])))
    return out


def holdout_mask(labels):
    """Boolean mask of the tuning hold-out: HOLDOUT_FRAC of each class, fixed seed."""
    labels = np.asarray(labels)
    rng = np.random.default_rng(HOLDOUT_SEED)
    mask = np.zeros(len(labels), bool)
    for c in range(len(LABELS)):
        idx = np.flatnonzero(labels == c)
        mask[rng.choice(idx, round(HOLDOUT_FRAC * len(idx)), replace=False)] = True
    return mask


class Views(torch.utils.data.Dataset):
    """V augmented views per image (V > 1, the example's train set) or one test view (V = 1)."""

    def __init__(self, items, V):
        self.items, self.V = items, V

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        data, label = self.items[i]
        img = Image.open(io.BytesIO(data)).convert('RGB')
        tf = AUG if self.V > 1 else TEST
        return torch.stack([tf(img) for _ in range(self.V)]), label


def splits(root, mode):
    """(ssl_train, probe_train, probe_eval) item lists.

    tune:  SSL and probe training on train minus the hold-out; probe eval on the hold-out.
    final: SSL and probe training on all of train; probe eval on the official validation split.
    """
    train = load_split(root, 'train')
    if mode == 'final':
        return train, train, load_split(root, 'val')
    mask = holdout_mask([y for _, y in train])
    rest = [x for x, m in zip(train, mask) if not m]
    return rest, rest, [x for x, m in zip(train, mask) if m]
