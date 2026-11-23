"""Encoder, arm-A head, and the two regularizers.

Native feature (all arms): timm vit_small_patch8_224 built with num_classes=0, so backbone(x) is
forward_head(forward_features(x), pre_logits=True) = the CLS token after the final LayerNorm,
shape (N*V, 384). Arm A adds the example's learned 384 -> 512 mapping (timm's classifier head
with num_classes=512, rebuilt as a separate Linear with timm's head init) and the MLP+BN projector.
"""
import contextlib, hashlib

import timm
import torch
import torch.nn as nn
from torchvision.ops import MLP

from . import wreg_nb

WREG_WEIGHTS = {'swd': 0.05, 'bwd': 0.05}   # notebook cell 0: ww = 0.05 each; c_swd / t_swd removed (user, 2026-11-09)


def swd(z, prior):
    """Notebook SWD on (N, 384): 1024 random directions of the 384-d cloud of N embeddings."""
    return wreg_nb.sliced_wasserstein_distance(z.unsqueeze(1), prior.unsqueeze(1))


def bwd_per_image(z, prior):
    """Basis Wasserstein in the embedding's own basis: each image's 384 values vs N(0, 1) samples
    (mean 0, std 1, Gaussian shape per embedding). The notebook's basis_wasserstein_distance runs its
    active term loss_b over dim 0, so it gets (384, 1, N): dim 0 = coordinates, one sort per image.
    Same code path as the notebook's commented-out loss_e (bulk_sigma weights, outer .square())."""
    return wreg_nb.basis_wasserstein_distance(z.T.unsqueeze(1).contiguous(), prior.T.unsqueeze(1).contiguous())


def bwd_per_coord(z, prior):
    """Monitor only: the notebook's active loss_b, each coordinate across the N images."""
    return wreg_nb.basis_wasserstein_distance(z.unsqueeze(1), prior.unsqueeze(1))


WREG_FNS = {'swd': swd, 'bwd': bwd_per_image}


@contextlib.contextmanager
def private_rng(seed):
    """Run a block on its own CPU+CUDA RNG stream, leaving the global (data / drop-path) streams untouched."""
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        yield


def derived_seed(*parts):
    return int.from_bytes(hashlib.sha256(repr(parts).encode()).digest()[:7], 'little')


class SIGReg(torch.nn.Module):
    """Copied from MINIMAL.md at the pinned commit (17 knots, 256 random directions)."""

    def __init__(self, knots=17):
        super().__init__()
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        A = torch.randn(proj.size(-1), 256, device="cuda")
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


def wreg(z, square=True):
    """SWD + per-image BWD on image-level features, per view, then averaged over views.

    z: (V, N, D). Prior: randn_like (N, D) per view, as in the notebook.
    square=False takes the square root of each notebook term per view, which undoes the notebook's
    outer .square() (test of its dead zone near the target, 2026-11-10); the notebook code stays verbatim.
    Returns (weighted sum averaged over views, {term: unweighted value averaged over views});
    the terms also hold 'bwd_coord' (monitor, no gradient, not in the sum).
    """
    terms = {k: 0.0 for k in (*WREG_FNS, 'bwd_coord')}
    for v in range(z.shape[0]):
        zz = z[v]
        prior = torch.randn_like(zz)
        for k, fn in WREG_FNS.items():
            t = fn(zz, prior)
            if not square:
                t = t.clamp_min(1e-20).sqrt()
            terms[k] = terms[k] + t / z.shape[0]
        with torch.no_grad():
            terms['bwd_coord'] = terms['bwd_coord'] + bwd_per_coord(zz.detach(), prior) / z.shape[0]
    return sum(WREG_WEIGHTS[k] * terms[k] for k in WREG_FNS), terms


class Net(nn.Module):
    def __init__(self, arm, seed, proj_dim=16, final_norm=True):
        super().__init__()
        self.arm = arm
        torch.manual_seed(seed)                      # encoder init: identical in every arm of a seed
        self.backbone = timm.create_model('vit_small_patch8_224', pretrained=False, num_classes=0,
                                          drop_path_rate=0.1, img_size=128)
        if not final_norm:                           # B: features = raw residual stream of the CLS token
            self.backbone.norm = nn.Identity()
        self.feat_dim = self.backbone.num_features
        if arm == 'A':
            with private_rng(derived_seed('head', seed)):
                self.map = nn.Linear(self.feat_dim, 512)
                nn.init.trunc_normal_(self.map.weight, std=0.02)   # timm init_weights_vit_timm for the head
                nn.init.zeros_(self.map.bias)
                self.proj = MLP(512, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)

    def forward(self, x):
        """x: (N, V, 3, H, W) -> native (V, N, 384), loss-path z (V, N, D), and A's 512-d map output or None."""
        N, V = x.shape[:2]
        feats = self.backbone(x.flatten(0, 1))
        native = feats.reshape(N, V, -1).transpose(0, 1)
        if self.arm != 'A':
            return native, native, None
        emb = self.map(feats)
        z = self.proj(emb).reshape(N, V, -1).transpose(0, 1)
        return native, z, emb


def trunk_hash(net):
    """Hash of the encoder weights except the final LayerNorm (absent in B), for pairing checks across arms."""
    h = hashlib.sha256()
    for k, v in net.backbone.state_dict().items():
        if k.startswith('norm.'):
            continue
        h.update(k.encode()); h.update(v.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()[:16]
