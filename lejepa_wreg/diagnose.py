"""Root-cause diagnostics on the tune checkpoints (run inside the container: python -m tools.diagnose).

Per checkpoint, on native features (and A's projector output z):
  inv      within-image variance across 4 augmented views / total variance (scale-free; 0 = fully invariant)
  inst     instance retrieval: a view's nearest other view (cosine, standardized) is from the same image
  knn      20-NN class accuracy (cosine, standardized) hold-out vs probe-train, test transform
  erank    effective rank of the correlation matrix; top1 = largest eigenvalue fraction
  std, |mu|/std, kurt   per-coordinate scale, offset and excess kurtosis across images
Gradients on one training batch (train mode, 128 images x 4 views), backbone only, unweighted terms:
  |g| per term and cosine with -g_align direction (cos < 0: the term pushes against alignment)
Synthetic: restoring force of each regularizer on z = s * randn(256, 384) and on rank-k clouds.
"""
import json, math, sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from lejepa_wreg import data
from lejepa_wreg.model import Net, SIGReg, swd, bwd_per_image, bwd_per_coord, private_rng

CK = Path('/ckpt')
OUT = Path('/ckpt/diagnose.json')
RUNS = ['init-B', 'init-A', 'lejepa-wreg-tune-A-lam0.02-s0', 'lejepa-wreg-tune-C-lam0.02-s0'] + [
    f'lejepa-wreg-tune-B-lam{l}-s0' for l in ('0.9', '0.99', '0.995', '0.999', '0.9995', '0.9999')]
dev = 'cuda'
sig = SIGReg().to(dev)


def load(name):
    if name.startswith('init-'):
        arm = name[-1]
        return Net(arm, 0, final_norm=arm != 'B').to(dev), {'arm': arm, 'lam': None}
    ck = torch.load(CK / name / 'final.pt', map_location='cpu', weights_only=False)
    cfg = ck['cfg']
    net = Net(cfg['arm'], cfg['seed'], cfg['proj_dim'], final_norm=bool(cfg['final_norm'])).to(dev)
    net.load_state_dict(ck['net'])
    return net, cfg


def stdz(f, ref=None):
    ref = f if ref is None else ref
    return (f - ref.mean(0)) / ref.std(0).clamp_min(1e-6)


@torch.inference_mode()
def feats(net, items, V):
    net.eval()
    out_n, out_z, ys = [], [], []
    with private_rng(1234):
        loader = DataLoader(data.Views(items, V), batch_size=256, num_workers=8,
                            generator=torch.Generator().manual_seed(1234))
        for x, y in loader:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                n, z, _ = net(x.to(dev))
            out_n.append(n.float().transpose(0, 1)); out_z.append(z.float().transpose(0, 1)); ys.append(y)
    return torch.cat(out_n), torch.cat(out_z), torch.cat(ys).to(dev)     # (N, V, D)


def view_stats(f):
    """f (N, V, D)."""
    N, V, D = f.shape
    within = f.var(1, unbiased=True).sum(1).mean()
    total = f.flatten(0, 1).var(0).sum()
    g = F.normalize(stdz(f.flatten(0, 1)), dim=1)
    sim = g @ g.T
    sim.fill_diagonal_(-2)
    nn_idx = sim.argmax(1)
    inst = (nn_idx // V == torch.arange(N * V, device=f.device) // V).float().mean()
    return {'inv': (within / total).item(), 'inst': inst.item()}


def spectrum(f):
    """f (N, D) clean features."""
    sd = f.std(0)
    mu = f.mean(0)
    zc = (f - mu) / sd.clamp_min(1e-6)
    ev = torch.linalg.eigvalsh(torch.cov(zc.T.double())).clamp_min(0)
    p = ev / ev.sum()
    kurt = (zc.pow(4).mean(0) - 3)
    return {'std': sd.mean().item(), 'mu_over_std': (mu.abs() / sd.clamp_min(1e-6)).mean().item(),
            'kurt': kurt.median().item(), 'erank': torch.exp(-(p[p > 0] * p[p > 0].log()).sum()).item(),
            'top1': p.max().item(), 'img_mean': f.mean(1).abs().mean().item(), 'img_std': f.std(1).mean().item()}


def knn(ftr, ytr, fev, yev, k=20):
    a, b = F.normalize(stdz(ftr), dim=1), F.normalize(stdz(fev, ftr), dim=1)
    s = b @ a.T
    v, i = s.topk(k, dim=1)
    votes = torch.zeros(len(fev), 10, device=dev).scatter_add_(1, ytr[i], (v / 0.07).exp())
    return (votes.argmax(1) == yev).float().mean().item()


def grads(net, x):
    net.train()
    params = [p for p in net.backbone.parameters() if p.requires_grad]
    with torch.autocast('cuda', dtype=torch.bfloat16):
        native, z, _ = net(x)
    nat = native.float()
    terms = {'align_native': (nat.mean(0) - nat).square().mean()}
    if z is not native:
        zf = z.float()
        terms['align_z'] = (zf.mean(0) - zf).square().mean()
        with private_rng(7):
            terms['sigreg_z'] = sig(zf)
    with private_rng(7):
        terms['sigreg_native'] = sig(nat)
    for nm, fn in (('swd', swd), ('bwd_img', bwd_per_image), ('bwd_coord', bwd_per_coord)):
        with private_rng(7):
            t = 0
            for v in range(nat.shape[0]):
                t = t + fn(nat[v], torch.randn_like(nat[v])) / nat.shape[0]
        terms[nm] = t
    out, gvec = {}, {}
    for k, t in terms.items():
        g = torch.autograd.grad(t, params, retain_graph=True, allow_unused=True)
        gvec[k] = torch.cat([(gi if gi is not None else torch.zeros_like(p)).flatten() for gi, p in zip(g, params)])
        out[k] = {'value': t.item(), 'gnorm': gvec[k].norm().item()}
    ref = gvec['align_native']
    for k in out:
        out[k]['cos_with_align'] = F.cosine_similarity(gvec[k], ref, dim=0).item()
    return out


def synthetic():
    torch.manual_seed(0)
    res = []
    for kind in ('iso', 'rank4', 'rank16', 'shift1'):
        for s in (0.05, 0.2, 0.5, 0.8, 1.0, 1.5, 3.0):
            g = torch.randn(256, 384, device=dev)
            if kind.startswith('rank'):
                k = int(kind[4:])
                g = torch.randn(256, k, device=dev) @ torch.randn(k, 384, device=dev) / math.sqrt(k)
            if kind == 'shift1':
                g = g + 1.0
            sc = torch.tensor(s, device=dev, requires_grad=True)
            row = {'kind': kind, 's': s}
            for nm, fn in (('swd', lambda z: swd(z, torch.randn_like(z))), ('bwd_img', lambda z: bwd_per_image(z, torch.randn_like(z))),
                           ('bwd_coord', lambda z: bwd_per_coord(z, torch.randn_like(z))), ('sigreg', lambda z: sig(z))):
                vals, ds = [], []
                for rep in range(4):
                    with private_rng(rep):
                        L = fn(sc * g)
                    d, = torch.autograd.grad(L, sc)
                    vals.append(L.item()); ds.append(d.item())
                row[nm] = sum(vals) / 4
                row['d' + nm + '/ds'] = sum(ds) / 4
            res.append(row)
    return res


def main():
    ssl_items, ptr_items, pev_items = data.splits('/datasets/imagenette', 'tune')
    torch.manual_seed(0)
    gb = DataLoader(data.Views(ssl_items, 4), batch_size=128, shuffle=True, generator=torch.Generator().manual_seed(5))
    with private_rng(5):
        xb, _ = next(iter(gb))
    xb = xb.to(dev)
    report = {'synthetic': synthetic()}
    print('synthetic done', flush=True)
    for name in RUNS:
        net, cfg = load(name)
        rec = {'arm': cfg['arm'], 'lam': cfg.get('lam')}
        fv_n, fv_z, _ = feats(net, pev_items, 4)
        ftr, ztr, ytr = feats(net, ptr_items, 1)
        fev, zev, yev = feats(net, pev_items, 1)
        rec['native'] = view_stats(fv_n) | spectrum(fev[:, 0]) | {'knn': knn(ftr[:, 0], ytr, fev[:, 0], yev)}
        if cfg['arm'] == 'A':
            rec['z'] = view_stats(fv_z) | spectrum(zev[:, 0]) | {'knn': knn(ztr[:, 0], ytr, zev[:, 0], yev)}
        rec['grads'] = grads(net, xb)
        report[name] = rec
        print(name, json.dumps(rec), flush=True)
        del net
        torch.cuda.empty_cache()
        OUT.write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
